# dspant_neuroproc/src/dspant_neuroproc/processors/spike_analytics/correlogram.py

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import polars as pl

from dspant.core.internals import public_api
from dspant.nodes.sorter import SorterNode

from .base import BaseSpikeTransform

try:
    from dspant_neuroproc._rs import (
        compute_all_cross_correlograms,
        compute_autocorrelogram,
        compute_correlogram,
        compute_jitter_corrected_correlogram,
        compute_spike_time_tiling_coefficient,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    import numba


@public_api
class SpikeCovarianceAnalyzer(BaseSpikeTransform):
    """
    Spike train correlation analysis for neural data.

    Supports:
    - Autocorrelograms
    - Cross-correlograms
    - Jitter-corrected correlations
    """

    def __init__(
        self,
        bin_size_ms: float = 1.0,
        window_size_ms: float = 100.0,
        normalization: str = "rate",
        jitter_correction: bool = False,
        jitter_iterations: int = 100,
    ):
        """
        Initialize correlation analyzer.

        Parameters
        ----------
        bin_size_ms : float
            Size of time bins in milliseconds
        window_size_ms : float
            Total window size for correlogram
        normalization : str
            Normalization method ('rate', 'count', 'probability')
        jitter_correction : bool
            Apply jitter correction to reduce artificial correlations
        jitter_iterations : int
            Number of iterations for jitter correction
        """
        # Ensure parameters are the correct types for Rust
        self.bin_size_ms = np.float32(bin_size_ms)
        self.window_size_ms = np.float32(window_size_ms)
        self.normalization = normalization
        self.jitter_correction = jitter_correction
        self.jitter_iterations = int(jitter_iterations)  # Ensure it's an integer

    @staticmethod
    def _compute_correlogram_py(
        spike_times1: np.ndarray,
        spike_times2: np.ndarray,
        bin_size: float,
        window_size: float,
    ) -> np.ndarray:
        """
        Python fallback for correlogram computation when Rust is not available.

        Uses Numba for performance.
        """
        # Ensure all inputs are the correct types
        spike_times1 = spike_times1.astype(np.float32)
        spike_times2 = spike_times2.astype(np.float32)
        bin_size = np.float32(bin_size)
        window_size = np.float32(window_size)

        if not hasattr(SpikeCovarianceAnalyzer, "_compute_correlogram_numba"):
            # Define the Numba-compiled function if not already defined
            @numba.jit(nopython=True, parallel=True)
            def _compute_correlogram_numba(
                spike_times1: np.ndarray,
                spike_times2: np.ndarray,
                bin_size: float,
                window_size: float,
            ) -> np.ndarray:
                half_window = window_size / 2
                n_bins = int(window_size / bin_size)
                correlogram = np.zeros(n_bins, dtype=np.int32)

                for t1 in spike_times1:
                    for t2 in spike_times2:
                        time_diff = t2 - t1

                        # Skip if time difference is outside window
                        if abs(time_diff) > half_window:
                            continue

                        # Find bin index
                        bin_idx = int((time_diff + half_window) / bin_size)

                        # Ensure the bin index is valid
                        if 0 <= bin_idx < n_bins:
                            correlogram[bin_idx] += 1

                return correlogram

            # Store the compiled function as a static attribute
            SpikeCovarianceAnalyzer._compute_correlogram_numba = (
                _compute_correlogram_numba
            )

        # Call the Numba-compiled function
        return SpikeCovarianceAnalyzer._compute_correlogram_numba(
            spike_times1, spike_times2, bin_size, window_size
        )

    def compute_autocorrelogram(
        self, sorter: SorterNode, unit_id: int
    ) -> Dict[str, Any]:
        """
        Compute autocorrelogram for a specific unit.

        Parameters
        ----------
        sorter : SorterNode
            Sorter containing spike data
        unit_id : int
            Unit ID to analyze

        Returns
        -------
        Dict with correlogram details
        """
        # Always use int32 for spike train acquisition
        spike_train = sorter.get_unit_spike_train(unit_id).astype(np.int32)

        # Convert to seconds with explicit float32 casting
        sampling_frequency = np.float32(sorter.sampling_frequency)
        spike_times = (spike_train / sampling_frequency).astype(np.float32)

        # Convert from ms to seconds with explicit float32 casting
        bin_size_s = np.float32(self.bin_size_ms / 1000.0)
        window_size_s = np.float32(self.window_size_ms / 1000.0)

        if _HAS_RUST:
            # Use Rust implementation
            correlogram, time_bins = compute_autocorrelogram(
                spike_times,
                bin_size_s,
                window_size_s,
                self.normalization if self.normalization != "count" else None,
            )
        else:
            # Fall back to Python implementation
            correlogram = self._compute_correlogram_py(
                spike_times, spike_times, bin_size_s, window_size_s
            )

            # Create time bins with float32 precision
            time_bins = np.linspace(
                -np.float32(self.window_size_ms / 2000.0),
                np.float32(self.window_size_ms / 2000.0),
                len(correlogram),
                dtype=np.float32,
            )

            # Apply normalization if needed
            if self.normalization == "rate":
                # Convert to Hz
                correlogram = correlogram.astype(np.float32) / (
                    len(spike_times) * bin_size_s
                )
            elif self.normalization == "probability":
                # Convert to probability
                total = correlogram.sum()
                if total > 0:
                    correlogram = correlogram.astype(np.float32) / total

        return {
            "unit_id": int(unit_id),  # Ensure int type
            "autocorrelogram": correlogram,
            "time_bins": time_bins,
        }

    def compute_crosscorrelogram(
        self, sorter: SorterNode, unit1: int, unit2: int
    ) -> Dict[str, Any]:
        """
        Compute crosscorrelogram between two units.

        Parameters
        ----------
        sorter : SorterNode
            Sorter containing spike data
        unit1 : int
            First unit ID
        unit2 : int
            Second unit ID

        Returns
        -------
        Dict with correlogram details
        """
        # Ensure units are integer types
        unit1 = int(unit1)
        unit2 = int(unit2)

        # Get spike trains with explicit int32 casting
        spike_train1 = sorter.get_unit_spike_train(unit1).astype(np.int32)
        spike_train2 = sorter.get_unit_spike_train(unit2).astype(np.int32)

        # Convert to seconds with explicit float32 casting
        sampling_frequency = np.float32(sorter.sampling_frequency)
        spike_times1 = (spike_train1 / sampling_frequency).astype(np.float32)
        spike_times2 = (spike_train2 / sampling_frequency).astype(np.float32)

        # Convert from ms to seconds with explicit float32 casting
        bin_size_s = np.float32(self.bin_size_ms / 1000.0)
        window_size_s = np.float32(self.window_size_ms / 1000.0)

        if _HAS_RUST:
            # Use Rust implementation
            if self.jitter_correction:
                # Explicit casting for jitter parameters
                jitter_window = np.float32(window_size_s / 10.0)
                jitter_iterations = int(self.jitter_iterations)

                correlogram, time_bins = compute_jitter_corrected_correlogram(
                    spike_times1,
                    spike_times2,
                    bin_size_s,
                    window_size_s,
                    jitter_window,
                    jitter_iterations,
                )
            else:
                correlogram, time_bins = compute_correlogram(
                    spike_times1,
                    spike_times2,
                    bin_size_s,
                    window_size_s,
                    self.normalization if self.normalization != "count" else None,
                )
        else:
            # Fall back to Python implementation
            correlogram = self._compute_correlogram_py(
                spike_times1, spike_times2, bin_size_s, window_size_s
            )

            # Create time bins with float32 precision
            time_bins = np.linspace(
                -np.float32(self.window_size_ms / 2000.0),
                np.float32(self.window_size_ms / 2000.0),
                len(correlogram),
                dtype=np.float32,
            )

            # Apply normalization if needed
            if self.normalization == "rate":
                # Convert to Hz with float32 precision
                correlogram = correlogram.astype(np.float32) / (
                    len(spike_times1) * bin_size_s
                )
            elif self.normalization == "probability":
                # Convert to probability
                total = np.float32(correlogram.sum())
                if total > 0:
                    correlogram = correlogram.astype(np.float32) / total

        return {
            "unit1": unit1,
            "unit2": unit2,
            "crosscorrelogram": correlogram,
            "time_bins": time_bins,
        }

    def transform(
        self, sorter: SorterNode, unit_ids: Optional[List[int]] = None, **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Compute correlograms for all specified units.

        Parameters
        ----------
        sorter : SorterNode
            Sorter containing spike data
        unit_ids : List[int], optional
            Units to analyze. If None, use all units.

        Returns
        -------
        Dict containing autocorrelograms and crosscorrelograms
        """
        if unit_ids is None:
            unit_ids = [int(uid) for uid in sorter.unit_ids]  # Ensure all IDs are int
        else:
            unit_ids = [int(uid) for uid in unit_ids]  # Ensure all IDs are int

        # Use efficient Rust implementation for all pairs if available
        if _HAS_RUST and not self.jitter_correction:
            # Convert all spike trains to seconds with explicit float32 casting
            sampling_frequency = np.float32(sorter.sampling_frequency)
            spike_trains = [
                (
                    sorter.get_unit_spike_train(unit_id).astype(np.int32)
                    / sampling_frequency
                ).astype(np.float32)
                for unit_id in unit_ids
            ]

            # Convert parameters from ms to seconds with explicit float32 casting
            bin_size_s = np.float32(self.bin_size_ms / 1000.0)
            window_size_s = np.float32(self.window_size_ms / 1000.0)

            # Compute all correlograms in one go
            return compute_all_cross_correlograms(
                spike_trains,
                unit_ids,
                bin_size_s,
                window_size_s,
                self.normalization if self.normalization != "count" else None,
            )
        else:
            # Fall back to computing each correlogram individually
            # Compute autocorrelograms
            autocorrelograms = [
                self.compute_autocorrelogram(sorter, unit_id) for unit_id in unit_ids
            ]

            # Compute crosscorrelograms
            crosscorrelograms = []
            for i, unit1 in enumerate(unit_ids):
                for unit2 in unit_ids[i + 1 :]:
                    crosscorr = self.compute_crosscorrelogram(sorter, unit1, unit2)
                    crosscorrelograms.append(crosscorr)

            return {
                "autocorrelograms": autocorrelograms,
                "crosscorrelograms": crosscorrelograms,
            }

    def compute_spike_time_tiling_coefficient(
        self,
        sorter: SorterNode,
        unit1: int,
        unit2: int,
        delta_t_ms: Optional[float] = None,
    ) -> float:
        """
        Compute the Spike Time Tiling Coefficient (STTC) between two units.

        STTC is a correlation measure less biased by firing rate differences.

        Parameters
        ----------
        sorter : SorterNode
            Sorter containing spike data
        unit1 : int
            First unit ID
        unit2 : int
            Second unit ID
        delta_t_ms : float, optional
            Synchronicity window in ms. If None, uses bin_size_ms.

        Returns
        -------
        float
            STTC value (-1 to 1, where 0 is no correlation)
        """
        # Ensure unit IDs are integers
        unit1 = int(unit1)
        unit2 = int(unit2)

        # Get spike trains with explicit int32 casting
        spike_train1 = sorter.get_unit_spike_train(unit1).astype(np.int32)
        spike_train2 = sorter.get_unit_spike_train(unit2).astype(np.int32)

        # Convert to seconds with explicit float32 casting
        sampling_frequency = np.float32(sorter.sampling_frequency)
        spike_times1 = (spike_train1 / sampling_frequency).astype(np.float32)
        spike_times2 = (spike_train2 / sampling_frequency).astype(np.float32)

        # Use bin size as default delta_t if not specified
        if delta_t_ms is None:
            delta_t_ms = self.bin_size_ms

        # Convert to seconds with explicit float32 casting
        delta_t_s = np.float32(delta_t_ms / 1000.0)

        # Use Rust implementation if available
        if _HAS_RUST:
            return float(
                compute_spike_time_tiling_coefficient(
                    spike_times1, spike_times2, delta_t_s
                )
            )
        else:
            # Python fallback
            if len(spike_times1) == 0 or len(spike_times2) == 0:
                return 0.0

            # Count spikes from train 1 that have at least one spike from train 2 within ±delta_t
            count_1 = sum(
                1
                for t1 in spike_times1
                if any(abs(t2 - t1) <= delta_t_s for t2 in spike_times2)
            )

            # Count spikes from train 2 that have at least one spike from train 1 within ±delta_t
            count_2 = sum(
                1
                for t2 in spike_times2
                if any(abs(t1 - t2) <= delta_t_s for t1 in spike_times1)
            )

            # Calculate the tiling coefficient with explicit float32 precision
            ta = np.float32(count_1) / len(spike_times1)
            tb = np.float32(count_2) / len(spike_times2)

            # Avoid division by zero with explicit float32 precision
            denominator = np.float32(max(1.0 - ta * tb, 1e-10))
            return float((ta + tb - ta * tb) / denominator)


# Factory function for easy initialization
def create_spike_correlogram_analyzer(
    bin_size_ms: float = 1.0, window_size_ms: float = 100.0, **kwargs
) -> SpikeCovarianceAnalyzer:
    """
    Create a spike correlogram analyzer with default parameters.
    """
    # Ensure parameters are properly typed
    bin_size_ms = np.float32(bin_size_ms)
    window_size_ms = np.float32(window_size_ms)

    return SpikeCovarianceAnalyzer(
        bin_size_ms=bin_size_ms, window_size_ms=window_size_ms, **kwargs
    )
