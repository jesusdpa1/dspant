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
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        self.bin_width = bin_width
        self.time_window = time_window
        self.line_color = line_color
        self.line_width = line_width
        self.show_sem = show_sem
        self.sem_alpha = sem_alpha

        self.time_bins, self.firing_rates, self.sem = self._compute_psth()

    def _compute_psth(self) -> Tuple[List[float], List[float], Optional[List[float]]]:
        spike_times, y_vals, _ = self.data.flatten()
        label_to_spikes = self.data.spikes

        start, end = self.time_window
        bins = np.arange(start, end + self.bin_width, self.bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        group_hist = [
            np.histogram(spikes, bins=bins)[0] for spikes in label_to_spikes.values()
        ]
        group_hist = np.array(group_hist)

        firing_rates = group_hist.mean(axis=0) / self.bin_width

        sem = (
            group_hist.std(axis=0, ddof=1)
            / np.sqrt(group_hist.shape[0])
            / self.bin_width
            if group_hist.shape[0] > 1
            else None
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
                "unit_id": self.data.unit_id,
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
        updated = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                updated = True
            else:
                self.config[key] = value
        if updated:
            self.time_bins, self.firing_rates, self.sem = self._compute_psth()

    def plot(self, backend="mpl", **kwargs):
        if backend == "mpl":
            from dspant_viz.backends.mpl.psth import render_psth
        elif backend == "plotly":
            from dspant_viz.backends.plotly.psth import render_psth
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_psth(self.get_data(), **kwargs)
