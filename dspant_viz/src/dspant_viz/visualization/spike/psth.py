from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.spike.base import BaseSpikeVisualization


class PSTHPlot(BaseSpikeVisualization):
    """Component to compute and render PSTH from spike times"""

    def __init__(
        self,
        data: SpikeData,
        event_times: np.ndarray,  # Required for PSTH
        pre_time: float,  # Required for PSTH
        post_time: float,  # Required for PSTH
        bin_width: float = 0.05,
        line_color: str = "orange",
        line_width: float = 2,
        show_sem: bool = True,
        sem_alpha: float = 0.3,
        unit_id: Optional[int] = None,
        sigma: Optional[float] = None,  # For smoothing
        **kwargs,
    ):
        # PSTHs always require event times
        if event_times is None or pre_time is None or post_time is None:
            raise ValueError("PSTH requires event_times, pre_time, and post_time")

        super().__init__(
            data=data,
            event_times=event_times,
            pre_time=pre_time,
            post_time=post_time,
            **kwargs,
        )

        self.bin_width = bin_width
        self.line_color = line_color
        self.line_width = line_width
        self.show_sem = show_sem
        self.sem_alpha = sem_alpha
        self.unit_id = unit_id or (list(data.spikes.keys())[0] if data.spikes else None)
        self.sigma = sigma

        # Compute PSTH data
        self.time_bins, self.firing_rates, self.sem = self._compute_psth()

    def _compute_psth(self) -> Tuple[List[float], List[float], Optional[List[float]]]:
        """
        Compute PSTH for the specified unit.

        Returns
        -------
        Tuple[List[float], List[float], Optional[List[float]]]
            time_bins, firing_rates, sem
        """
        if self.unit_id is None:
            return [], [], None

        # Get trial data for this unit
        trial_data = self.get_trial_data(self.unit_id)[self.unit_id]

        # Skip if no data
        if not trial_data:
            return [], [], None

        # Create time bins
        bin_edges = np.arange(
            -self.pre_time, self.post_time + self.bin_width, self.bin_width
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Count spikes in each bin for each trial
        trial_counts = []
        for trial_spikes in trial_data.values():
            if trial_spikes:  # Skip empty trials
                hist, _ = np.histogram(trial_spikes, bins=bin_edges)
                trial_counts.append(hist)

        # Skip if no spikes
        if not trial_counts:
            return bin_centers.tolist(), [0] * len(bin_centers), None

        # Convert to numpy array for calculations
        trial_counts = np.array(trial_counts)

        # Calculate mean firing rate
        firing_rates = np.mean(trial_counts, axis=0) / self.bin_width

        # Calculate SEM if there are multiple trials
        sem = None
        if len(trial_counts) > 1:
            sem = (
                np.std(trial_counts, axis=0, ddof=1)
                / np.sqrt(len(trial_counts))
                / self.bin_width
            )

        # Apply smoothing if requested
        if self.sigma is not None:
            from scipy.ndimage import gaussian_filter1d

            # Convert sigma from time to bins
            sigma_bins = self.sigma / self.bin_width
            firing_rates = gaussian_filter1d(firing_rates, sigma_bins)
            if sem is not None:
                sem = gaussian_filter1d(sem, sigma_bins)

        return (
            bin_centers.tolist(),
            firing_rates.tolist(),
            sem.tolist() if sem is not None else None,
        )

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare PSTH data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        # Return empty data if no unit ID or no bins
        if self.unit_id is None or not self.time_bins:
            return {
                "data": {
                    "time_bins": [],
                    "firing_rates": [],
                    "sem": None,
                    "unit_id": self.unit_id,
                },
                "params": {
                    "line_color": self.line_color,
                    "line_width": self.line_width,
                    "show_sem": self.show_sem,
                    "sem_alpha": self.sem_alpha,
                    "bin_width": self.bin_width,
                    **self.config,
                },
            }

        return {
            "data": {
                "time_bins": self.time_bins,
                "firing_rates": self.firing_rates,
                "sem": self.sem,
                "unit_id": self.unit_id,
            },
            "params": {
                "line_color": self.line_color,
                "line_width": self.line_width,
                "show_sem": self.show_sem,
                "sem_alpha": self.sem_alpha,
                "bin_width": self.bin_width,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update PSTH parameters and recompute if necessary.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        needs_recompute = False

        for key, value in kwargs.items():
            if key in [
                "event_times",
                "pre_time",
                "post_time",
                "bin_width",
                "unit_id",
                "sigma",
            ]:
                needs_recompute = True

            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

        # Update trial data if event parameters changed
        if "event_times" in kwargs or "pre_time" in kwargs or "post_time" in kwargs:
            self._trial_data = self._organize_by_trials()

        # Recompute PSTH if necessary
        if needs_recompute:
            self.time_bins, self.firing_rates, self.sem = self._compute_psth()

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a plot using the specified backend.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl', 'plotly')
        **kwargs : dict
            Additional parameters for the backend

        Returns
        -------
        Any
            Plot figure from the specified backend
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.psth import render_psth
        elif backend == "plotly":
            from dspant_viz.backends.plotly.psth import render_psth
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_psth(self.get_data(), **kwargs)
