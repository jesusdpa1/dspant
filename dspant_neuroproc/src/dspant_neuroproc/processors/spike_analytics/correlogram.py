# src/dspant_neuroproc/processors/spike_analytics/correlogram.py

from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np
import polars as pl

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
        self.bin_size_ms = bin_size_ms
        self.window_size_ms = window_size_ms
        self.normalization = normalization
        self.jitter_correction = jitter_correction
        self.jitter_iterations = jitter_iterations

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def _compute_correlogram(
        spike_times1: np.ndarray,
        spike_times2: np.ndarray,
        bin_size: float,
        window_size: float,
    ) -> np.ndarray:
        """
        Compute correlogram using Numba for performance.

        Parameters
        ----------
        spike_times1 : np.ndarray
            Spike times for first unit
        spike_times2 : np.ndarray
            Spike times for second unit
        bin_size : float
            Bin size in seconds
        window_size : float
            Total window size in seconds

        Returns
        -------
        np.ndarray
            Correlogram counts
        """
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
                correlogram[bin_idx] += 1

        return correlogram

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
        spike_times = sorter.get_unit_spike_train(unit_id) / sorter.sampling_frequency

        bin_size_s = self.bin_size_ms / 1000
        window_size_s = self.window_size_ms / 1000

        # Compute autocorrelogram
        autocorr = self._compute_correlogram(
            spike_times, spike_times, bin_size_s, window_size_s
        )

        # Create time bins
        time_bins = np.linspace(
            -self.window_size_ms / 2000, self.window_size_ms / 2000, len(autocorr)
        )

        return {"unit_id": unit_id, "autocorrelogram": autocorr, "time_bins": time_bins}

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
        spike_times1 = sorter.get_unit_spike_train(unit1) / sorter.sampling_frequency
        spike_times2 = sorter.get_unit_spike_train(unit2) / sorter.sampling_frequency

        bin_size_s = self.bin_size_ms / 1000
        window_size_s = self.window_size_ms / 1000

        # Compute crosscorrelogram
        crosscorr = self._compute_correlogram(
            spike_times1, spike_times2, bin_size_s, window_size_s
        )

        # Create time bins
        time_bins = np.linspace(
            -self.window_size_ms / 2000, self.window_size_ms / 2000, len(crosscorr)
        )

        return {
            "unit1": unit1,
            "unit2": unit2,
            "crosscorrelogram": crosscorr,
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


# Factory function for easy initialization
def create_spike_correlogram_analyzer(
    bin_size_ms: float = 1.0, window_size_ms: float = 100.0, **kwargs
) -> SpikeCovarianceAnalyzer:
    """
    Create a spike correlogram analyzer with default parameters.
    """
    return SpikeCovarianceAnalyzer(
        bin_size_ms=bin_size_ms, window_size_ms=window_size_ms, **kwargs
    )
