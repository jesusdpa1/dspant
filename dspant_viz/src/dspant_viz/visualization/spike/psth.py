from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import SpikeData


class PSTHPlot(VisualizationComponent):
    """Component to compute and render PSTH from raw spike times"""

    def __init__(
        self,
        data: SpikeData,
        bin_width: float = 0.05,
        time_window: Tuple[float, float] = (-1.0, 1.0),
        line_color: str = "orange",
        line_width: float = 2,
        show_sem: bool = True,
        sem_alpha: float = 0.3,
        unit_id: Optional[int] = None,  # Add unit_id parameter
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        self.bin_width = bin_width
        self.time_window = time_window
        self.line_color = line_color
        self.line_width = line_width
        self.show_sem = show_sem
        self.sem_alpha = sem_alpha
        self.unit_id = unit_id  # Store which unit to display

        # Compute PSTH
        self.time_bins, self.firing_rates, self.sem = self._compute_psth()

    def _compute_psth(self) -> Tuple[List[float], List[float], Optional[List[float]]]:
        """
        Compute PSTH for the specified unit.

        Returns
        -------
        Tuple[List[float], List[float], Optional[List[float]]]
            time_bins, firing_rates, sem
        """
        # If no unit_id specified and data contains multiple units, use the first one
        if self.unit_id is None:
            available_units = list(self.data.spikes.keys())
            if available_units:
                self.unit_id = available_units[0]
            else:
                # No data available
                return [], [], None

        # Extract trials for this unit
        unit_spikes = self.data.spikes.get(self.unit_id, {})

        # Skip if no data
        if not unit_spikes:
            return [], [], None

        # Extract spike times for each trial
        trial_spikes = list(unit_spikes.values())

        # Create time bins
        start, end = self.time_window
        bins = np.arange(start, end + self.bin_width, self.bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Count spikes in each bin for each trial
        trial_counts = []
        for spikes in trial_spikes:
            if spikes:  # Skip empty trials
                hist, _ = np.histogram(spikes, bins=bins)
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

        return (
            bin_centers.tolist(),
            firing_rates.tolist(),
            sem.tolist() if sem is not None else None,
        )

    def get_data(self) -> Dict[str, Any]:
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
                "time_window": self.time_window,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        recalculate = False

        for key, value in kwargs.items():
            if key in ["bin_width", "time_window", "unit_id"]:
                recalculate = True

            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

        if recalculate:
            self.time_bins, self.firing_rates, self.sem = self._compute_psth()

    def plot(self, backend="mpl", **kwargs):
        if backend == "mpl":
            from dspant_viz.backends.mpl.psth import render_psth
        elif backend == "plotly":
            from dspant_viz.backends.plotly.psth import render_psth
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_psth(self.get_data(), **kwargs)
