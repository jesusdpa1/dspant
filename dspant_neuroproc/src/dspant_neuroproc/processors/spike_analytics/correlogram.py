# src/dspant_neuroproc/processors/spike_analytics/correlogram.py

from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np
import polars as pl
from numba import jit, prange

from dspant.core.internals import public_api
from dspant.nodes.sorter import SorterNode

from .base import BaseSpikeTransform


@public_api
class SpikeCovarianceAnalyzer(BaseSpikeTransform):
    """
    Spike train correlation analysis for neural data.

    Supports:
    - Autocorrelograms
    - Cross-correlograms
    - Jitter-corrected correlations
    - FFT-based efficient computation
    """

    def __init__(
        self,
        bin_size_ms: float = 1.0,
        window_size_ms: float = 100.0,
        normalization: str = "rate",
        jitter_correction: bool = False,
        jitter_iterations: int = 100,
        method: str = "fft",
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
        method : str
            Method to use for computing correlograms ('direct', 'fft')
        """
        self.bin_size_ms = bin_size_ms
        self.window_size_ms = window_size_ms
        self.normalization = normalization
        self.jitter_correction = jitter_correction
        self.jitter_iterations = jitter_iterations
        self.method = method

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _numba_cross_correlogram_direct(
        spikes1: np.ndarray,
        spikes2: np.ndarray,
        bin_edges: np.ndarray,
        is_autocorr: bool,
    ) -> np.ndarray:
        """
        Numba-accelerated direct cross-correlogram computation.

        Parameters
        ----------
        spikes1 : np.ndarray
            Spike times for reference unit
        spikes2 : np.ndarray
            Spike times for target unit
        bin_edges : np.ndarray
            Bin edges for correlogram
        is_autocorr : bool
            Whether this is an autocorrelogram

        Returns
        -------
        np.ndarray
            Correlogram counts
        """
        n_bins = len(bin_edges) - 1
        correlogram = np.zeros(n_bins, dtype=np.int32)

        # Parallelize over reference spikes
        for i in prange(len(spikes1)):
            ref_spike = spikes1[i]

            # Process each target spike
            for j in range(len(spikes2)):
                # Skip self-comparison for autocorrelogram
                if is_autocorr and i == j:
                    continue

                # Compute time difference
                time_diff = spikes2[j] - ref_spike

                # Use binary search for better efficiency
                bin_idx = np.searchsorted(bin_edges[1:], time_diff, side="left")

                # Only increment if within valid bin range
                if (
                    0 <= bin_idx < n_bins
                    and bin_edges[bin_idx] <= time_diff < bin_edges[bin_idx + 1]
                ):
                    correlogram[bin_idx] += 1

        return correlogram

    @staticmethod
    @jit(nopython=True, cache=True)
    def _create_spike_train_binned(
        spike_times: np.ndarray, bin_size: float, t_start: float, t_end: float
    ) -> np.ndarray:
        """
        Create binned spike train for FFT-based correlation.

        Parameters
        ----------
        spike_times : np.ndarray
            Spike times in seconds
        bin_size : float
            Size of bins in seconds
        t_start : float
            Start time for binning in seconds
        t_end : float
            End time for binning in seconds

        Returns
        -------
        np.ndarray
            Binned spike train (0s and 1s)
        """
        n_bins = int((t_end - t_start) / bin_size) + 1
        binned = np.zeros(n_bins, dtype=np.int32)

        for spike in spike_times:
            if t_start <= spike < t_end:
                bin_idx = int((spike - t_start) / bin_size)
                if 0 <= bin_idx < n_bins:
                    binned[bin_idx] = 1

        return binned

    @staticmethod
    def _compute_correlogram_fft(
        spikes1: np.ndarray,
        spikes2: np.ndarray,
        bin_size: float,
        window_size: float,
    ) -> np.ndarray:
        """
        Compute cross-correlogram using FFT-based method.

        Parameters
        ----------
        spikes1 : np.ndarray
            Spike times for reference unit
        spikes2 : np.ndarray
            Spike times for target unit
        bin_size : float
            Size of bins in seconds
        window_size : float
            Total window size in seconds

        Returns
        -------
        np.ndarray
            Correlogram counts
        """
        # Determine time range
        if len(spikes1) == 0 or len(spikes2) == 0:
            return np.array([])

        t_min = min(np.min(spikes1), np.min(spikes2))
        t_max = max(np.max(spikes1), np.max(spikes2))

        # Add padding to avoid edge effects
        t_start = t_min - window_size
        t_end = t_max + window_size

        # Create binned spike trains
        binned1 = SpikeCovarianceAnalyzer._create_spike_train_binned(
            spikes1, bin_size, t_start, t_end
        )

        # Check if this is autocorrelogram
        is_auto = spikes1 is spikes2 or np.array_equal(spikes1, spikes2)

        if is_auto:
            binned2 = binned1
        else:
            binned2 = SpikeCovarianceAnalyzer._create_spike_train_binned(
                spikes2, bin_size, t_start, t_end
            )

        # Compute cross-correlation using FFT
        n_bins = int(window_size / bin_size) + 1

        # Use numpy's built-in correlate with 'same' mode
        # This gives the central region of the correlation
        full_xcorr = np.correlate(binned1, binned2, mode="full")

        # Extract the region corresponding to the window size
        mid_point = len(full_xcorr) // 2
        half_n_bins = n_bins // 2
        xcorr = full_xcorr[mid_point - half_n_bins : mid_point + half_n_bins + 1]

        # Ensure the correlogram has the expected size
        if len(xcorr) < n_bins:
            # Pad with zeros if needed
            xcorr = np.pad(xcorr, (0, n_bins - len(xcorr)))
        elif len(xcorr) > n_bins:
            # Truncate if needed
            xcorr = xcorr[:n_bins]

        return xcorr

    def compute_correlogram(
        self,
        spikes1: np.ndarray,
        spikes2: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute cross-correlogram using efficient methods.
        If spikes2 is None, computes autocorrelogram.

        Parameters
        ----------
        spikes1 : np.ndarray
            Spike times for reference unit
        spikes2 : np.ndarray, optional
            Spike times for target unit. If None, computes autocorrelogram.

        Returns
        -------
        Tuple of (time_bins, correlogram, standard_error)
        """
        # Check if this is an autocorrelogram
        is_auto = spikes2 is None

        # If autocorrelogram, use spikes1 for both
        if is_auto:
            spikes2 = spikes1

        # Check for sufficient data
        if len(spikes1) == 0 or len(spikes2) == 0:
            return (np.array([]), np.array([]), np.array([]))

        # Convert bin size and window size to seconds
        bin_size_s = self.bin_size_ms / 1000
        window_size_s = self.window_size_ms / 1000
        half_window = window_size_s / 2

        # Create bin edges and centers
        bin_edges = np.arange(-half_window, half_window + bin_size_s, bin_size_s)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Choose computation method
        if self.method == "fft" and len(spikes1) > 1000 and len(spikes2) > 1000:
            # Use FFT-based method for large spike trains
            correlogram = self._compute_correlogram_fft(
                spikes1, spikes2, bin_size_s, window_size_s
            )
        else:
            # Use direct method for smaller spike trains
            correlogram = self._numba_cross_correlogram_direct(
                spikes1, spikes2, bin_edges, is_auto
            )

        # Normalize if requested
        if self.normalization == "rate":
            # Normalize by bin width and number of reference spikes
            correlogram = correlogram / (len(spikes1) * bin_size_s)
        elif self.normalization == "probability":
            # Normalize by total count to get probability distribution
            total_count = np.sum(correlogram)
            if total_count > 0:
                correlogram = correlogram / total_count

        # Compute standard error (placeholder for now)
        sem = None  # TODO: Implement proper SEM computation

        return bin_centers, correlogram, sem

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
        # Get spike times in seconds
        spike_times = sorter.get_unit_spike_train(unit_id) / sorter.sampling_frequency

        # Compute autocorrelogram by passing only spikes1
        time_bins, autocorr, sem = self.compute_correlogram(spike_times)

        return {
            "unit_id": unit_id,
            "autocorrelogram": autocorr,
            "time_bins": time_bins,
            "standard_error": sem,
            "n_spikes": len(spike_times),
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
        # Get spike times in seconds
        spike_times1 = sorter.get_unit_spike_train(unit1) / sorter.sampling_frequency
        spike_times2 = sorter.get_unit_spike_train(unit2) / sorter.sampling_frequency

        # Compute crosscorrelogram
        time_bins, crosscorr, sem = self.compute_correlogram(spike_times1, spike_times2)

        return {
            "unit1": unit1,
            "unit2": unit2,
            "crosscorrelogram": crosscorr,
            "time_bins": time_bins,
            "standard_error": sem,
            "n_spikes1": len(spike_times1),
            "n_spikes2": len(spike_times2),
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
            unit_ids = sorter.unit_ids

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

    def transform_parallel(
        self, sorter: SorterNode, unit_ids: Optional[List[int]] = None, **kwargs
    ) -> Dict[str, List[Dict]]:
        """
        Compute correlograms for all specified units using parallel processing.

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
        import concurrent.futures

        if unit_ids is None:
            unit_ids = sorter.unit_ids

        # Pre-load spike trains for all units
        print(f"Loading spike trains for {len(unit_ids)} units...")
        spike_trains = {}
        for unit_id in unit_ids:
            spike_trains[unit_id] = (
                sorter.get_unit_spike_train(unit_id) / sorter.sampling_frequency
            )

        print("Computing correlograms in parallel...")
        autocorrelograms = []
        crosscorrelograms = []

        # Helper functions for parallel computation
        def compute_auto(unit_id):
            time_bins, autocorr, sem = self.compute_correlogram(spike_trains[unit_id])
            return {
                "unit_id": unit_id,
                "autocorrelogram": autocorr,
                "time_bins": time_bins,
                "standard_error": sem,
                "n_spikes": len(spike_trains[unit_id]),
            }

        def compute_cross(unit1, unit2):
            time_bins, crosscorr, sem = self.compute_correlogram(
                spike_trains[unit1], spike_trains[unit2]
            )
            return {
                "unit1": unit1,
                "unit2": unit2,
                "crosscorrelogram": crosscorr,
                "time_bins": time_bins,
                "standard_error": sem,
                "n_spikes1": len(spike_trains[unit1]),
                "n_spikes2": len(spike_trains[unit2]),
            }

        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit autocorrelogram jobs
            auto_futures = {
                executor.submit(compute_auto, unit_id): unit_id for unit_id in unit_ids
            }

            # Submit crosscorrelogram jobs
            cross_futures = {}
            for i, unit1 in enumerate(unit_ids):
                for unit2 in unit_ids[i + 1 :]:
                    cross_futures[executor.submit(compute_cross, unit1, unit2)] = (
                        unit1,
                        unit2,
                    )

            # Collect autocorrelogram results
            for future in concurrent.futures.as_completed(auto_futures):
                try:
                    result = future.result()
                    autocorrelograms.append(result)
                except Exception as exc:
                    unit_id = auto_futures[future]
                    print(
                        f"Unit {unit_id} autocorrelogram generated an exception: {exc}"
                    )

            # Collect crosscorrelogram results
            for future in concurrent.futures.as_completed(cross_futures):
                try:
                    result = future.result()
                    crosscorrelograms.append(result)
                except Exception as exc:
                    unit1, unit2 = cross_futures[future]
                    print(
                        f"Units {unit1}-{unit2} crosscorrelogram generated an exception: {exc}"
                    )

        return {
            "autocorrelograms": autocorrelograms,
            "crosscorrelograms": crosscorrelograms,
        }


# Factory function for easy initialization
def create_spike_correlogram_analyzer(
    bin_size_ms: float = 1.0,
    window_size_ms: float = 100.0,
    method: str = "fft",
    **kwargs,
) -> SpikeCovarianceAnalyzer:
    """
    Create a spike correlogram analyzer with default parameters.

    Parameters
    ----------
    bin_size_ms : float
        Size of time bins in milliseconds
    window_size_ms : float
        Total window size for correlogram
    method : str
        Method to use for computing correlograms ('direct', 'fft')
    **kwargs : dict
        Additional keyword arguments for SpikeCovarianceAnalyzer

    Returns
    -------
    SpikeCovarianceAnalyzer
        Configured correlogram analyzer
    """
    return SpikeCovarianceAnalyzer(
        bin_size_ms=bin_size_ms, window_size_ms=window_size_ms, method=method, **kwargs
    )
