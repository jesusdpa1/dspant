from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import SpikeData


class RasterPlot(VisualizationComponent):
    """Component for spike raster visualization"""

    def __init__(
        self,
        data: SpikeData,
        marker_size: float = 4,
        marker_color: str = "#2D3142",
        marker_alpha: float = 0.7,
        marker_type: str = "|",
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        self.marker_size = marker_size
        self.marker_color = marker_color
        self.marker_alpha = marker_alpha
        self.marker_type = marker_type

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        spike_times, trial_indices, label_map = self.data.flatten()

        return {
            "data": {
                "spike_times": spike_times,
                "y_values": trial_indices,  # Keep y_values for backward compatibility
                "trial_indices": trial_indices,  # Add trial_indices for clarity
                "label_map": label_map,
            },
            "params": {
                "marker_size": self.marker_size,
                "marker_color": self.marker_color,
                "marker_alpha": self.marker_alpha,
                "marker_type": self.marker_type,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

    def plot(self, backend: str = "mpl", **kwargs):
        if backend == "mpl":
            from dspant_viz.backends.mpl.raster import render_raster
        elif backend == "plotly":
            from dspant_viz.backends.plotly.raster import render_raster
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_raster(self.get_data(), **kwargs)
