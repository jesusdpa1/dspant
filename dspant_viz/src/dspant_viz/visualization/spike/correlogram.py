# src/dspant_viz/visualization/spike/crosscorrelogram.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numba as nb
import numpy as np
from numba import jit
from scipy import stats

from dspant_viz.core.data_models import SpikeData
from dspant_viz.core.internals import public_api
from dspant_viz.visualization.spike.base import BaseSpikeVisualization


@public_api(module_override="dspant_viz.visualization")
class CorrelogramPlot(BaseSpikeVisualization):
    """
    Component for computing and visualizing cross-correlograms between neural units.

    Supports both autocorrelogram (single unit) and cross-correlogram (two units) analysis.
    """

    def __init__(
        self,
        data: SpikeData,
        unit01: Optional[int] = None,
        unit02: Optional[int] = None,
        method: str = "standard",
        bin_width: float = 0.01,  # 10 ms bins
        window_size: float = 0.5,  # ±500 ms window
        normalize: bool = True,
        **kwargs,
    ):
        """
        Initialize the crosscorrelogram visualization.

        Parameters
        ----------
        data : SpikeData
            Spike data containing spike times for different units
        unit01 : int, optional
            First unit ID (primary unit, required)
        unit02 : int, optional
            Second unit ID (optional, creates cross-correlogram if provided)
        method : str, optional
            Correlation computation method ('standard', etc.)
        bin_width : float, optional
            Width of time bins in seconds
        window_size : float, optional
            Total window size (±window_size)
        normalize : bool, optional
            Whether to normalize the correlogram
        **kwargs : dict
            Additional configuration parameters
        """
        # Ensure unit01 is set
        if unit01 is None:
            if not data.spikes:
                raise ValueError("No units found in spike data")
            unit01 = list(data.spikes.keys())[0]

        super().__init__(data, **kwargs)

        self.unit01 = unit01
        self.unit02 = unit02
        self.method = method
        self.bin_width = bin_width
        self.window_size = window_size
        self.normalize = normalize

        # Compute correlogram
        (self.time_bins, self.correlogram, self.sem) = self._compute_correlogram()

    def _compute_correlogram(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Compute cross-correlogram using efficient Numba-accelerated method.

        Returns
        -------
        Tuple of (time_bins, correlogram, standard_error)
        """
        # Get spike times for units
        unit01_spikes = self.data.spikes.get(self.unit01, np.array([]))

        # Determine if autocorrelogram or cross-correlogram
        if self.unit02 is None:
            # Autocorrelogram
            unit02_spikes = unit01_spikes
            is_autocorr = True
        else:
            # Cross-correlogram
            unit02_spikes = self.data.spikes.get(self.unit02, np.array([]))
            is_autocorr = False

        # Check for sufficient data
        if len(unit01_spikes) == 0 or len(unit02_spikes) == 0:
            return (np.array([]), np.array([]), np.array([]))

        # Create bin edges
        bin_edges = np.arange(
            -self.window_size, self.window_size + self.bin_width, self.bin_width
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute correlogram
        correlogram = self._numba_cross_correlogram(
            unit01_spikes, unit02_spikes, bin_edges, is_autocorr
        )

        # Normalize if requested
        if self.normalize:
            # Normalize by bin width and number of reference spikes
            correlogram = correlogram / (len(unit01_spikes) * self.bin_width)

        # Compute standard error (placeholder for now)
        sem = None  # TODO: Implement proper SEM computation

        return bin_centers, correlogram, sem

    @staticmethod
    @jit(nopython=True, cache=True)
    def _numba_cross_correlogram(
        unit01_spikes: np.ndarray,
        unit02_spikes: np.ndarray,
        bin_edges: np.ndarray,
        is_autocorr: bool,
    ) -> np.ndarray:
        """
        Numba-accelerated cross-correlogram computation.

        Parameters
        ----------
        unit01_spikes : np.ndarray
            Spike times for first unit
        unit02_spikes : np.ndarray
            Spike times for second unit
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

        for ref_spike in unit01_spikes:
            for comp_spike in unit02_spikes:
                # Skip self-comparison for autocorrelogram
                if is_autocorr and ref_spike == comp_spike:
                    continue

                # Compute time difference
                time_diff = comp_spike - ref_spike

                # Find appropriate bin
                for bin_idx in range(n_bins):
                    if bin_edges[bin_idx] <= time_diff < bin_edges[bin_idx + 1]:
                        correlogram[bin_idx] += 1
                        break

        return correlogram

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare correlogram data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        return {
            "data": {
                "time_bins": self.time_bins,
                "correlogram": self.correlogram,
                "sem": self.sem,
                "unit1": self.unit01,
                "unit2": self.unit02,
            },
            "params": {
                "method": self.method,
                "bin_width": self.bin_width,
                "window_size": self.window_size,
                "normalize": self.normalize,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update correlogram parameters and recompute.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

        # Recompute correlogram
        (self.time_bins, self.correlogram, self.sem) = self._compute_correlogram()

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a plot using the specified backend.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs : dict
            Additional parameters for the backend

        Returns
        -------
        Any
            Plot figure from the specified backend
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.correlogram import render_correlogram
        elif backend == "plotly":
            from dspant_viz.backends.plotly.correlogram import (
                render_correlogram,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_correlogram(self.get_data(), **kwargs)
