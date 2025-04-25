from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import SpikeData


class RasterPlot(VisualizationComponent):
    """Component for spike raster visualization with multi-unit support"""

    def __init__(
        self,
        data: SpikeData,
        marker_size: float = 4,
        marker_color: str = "#2D3142",
        marker_alpha: float = 0.7,
        marker_type: str = "|",
        unit_id: Optional[int] = None,  # Add unit_id parameter
        **kwargs,
    ):
        super().__init__(data, **kwargs)

        self.marker_size = marker_size
        self.marker_color = marker_color
        self.marker_alpha = marker_alpha
        self.marker_type = marker_type
        self.unit_id = unit_id  # Store which unit to display

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        # If no unit_id specified and data contains multiple units, use the first one
        if self.unit_id is None:
            available_units = list(self.data.spikes.keys())
            if available_units:
                self.unit_id = available_units[0]
            else:
                # No data available
                return {
                    "data": {
                        "spike_times": [],
                        "y_values": [],
                        "trial_indices": [],
                        "label_map": {},
                        "unit_id": None,
                    },
                    "params": {
                        "marker_size": self.marker_size,
                        "marker_color": self.marker_color,
                        "marker_alpha": self.marker_alpha,
                        "marker_type": self.marker_type,
                        **self.config,
                    },
                }

        # Extract data for the specified unit only
        unit_spikes = self.data.spikes.get(self.unit_id, {})

        # Flatten just this unit's data
        spike_times = []
        y_values = []
        label_map = {}

        for i, (trial_label, spike_list) in enumerate(unit_spikes.items()):
            spike_times.extend(spike_list)
            y_values.extend([i] * len(spike_list))
            label_map[i] = str(trial_label)

        return {
            "data": {
                "spike_times": spike_times,
                "y_values": y_values,
                "trial_indices": y_values,  # For backward compatibility
                "label_map": label_map,
                "unit_id": self.unit_id,
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
