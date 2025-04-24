# components/spike/raster.py
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import SpikeData
from dspant_viz.core.internals import public_api


@public_api
class RasterPlot(VisualizationComponent):
    """Component for spike raster visualization"""

    def __init__(
        self,
        spikes: Dict[Union[str, int], List[float]],
        y_axis_name: Optional[str] = None,
        unit_id: Optional[int] = None,
        marker_size: float = 4,
        marker_color: str = "#2D3142",
        marker_alpha: float = 0.7,
        show_grid: bool = True,
        show_event_onset: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize raster plot component

        Parameters
        ----------
        spikes : Dict[Union[str, int], List[float]]
            Dictionary mapping labels (trial IDs, neuron IDs, etc.) to spike times
        y_axis_name : str, optional
            Name of the y-axis. If None, tries to auto-determine from keys
        unit_id : int, optional
            Unit identifier
        marker_size : float
            Size of spike markers
        marker_color : str
            Color of spike markers
        marker_alpha : float
            Alpha transparency for markers
        show_grid : bool
            Whether to show grid lines
        show_event_onset : bool
            Whether to show vertical line at time=0
        xlim : Tuple[float, float], optional
            X-axis limits as (min, max)
        ylim : Tuple[float, float], optional
            Y-axis limits as (min, max)
        title : str, optional
            Plot title. If None, a default title may be generated
        """
        # Validate and store the spike data
        self.data = SpikeData(spikes=spikes, unit_id=unit_id)

        # Auto-determine y_axis_name if not provided
        if y_axis_name is None:
            # Check first key to guess the appropriate name
            if not spikes:
                y_axis_name = "Index"
            else:
                first_key = next(iter(spikes.keys()))
                if isinstance(first_key, str):
                    if "trial" in first_key.lower():
                        y_axis_name = "Trial"
                    elif "neuron" in first_key.lower() or "unit" in first_key.lower():
                        y_axis_name = "Neuron"
                    else:
                        # Use the prefix common to all keys if possible
                        import re
                        prefix_match = re.match(r'^([a-zA-Z]+)', str(first_key))
                        if prefix_match:
                            y_axis_name = prefix_match.group(1).capitalize()
                        else:
                            y_axis_name = "Group"
                else:
                    y_axis_name = "Index"

        # Store visualization parameters
        self.y_axis_name = y_axis_name
        self.marker_size = marker_size
        self.marker_color = marker_color
        self.marker_alpha = marker_alpha
        self.show_grid = show_grid
        self.show_event_onset = show_event_onset
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.additional_params = kwargs

    def get_data(self) -> Dict[str, Any]:
        """Get raster data in format ready for rendering"""
        # Flatten the spike data
        spike_times, y_values, y_labels = self.data.flatten()

        # Build the data dictionary
        result = {
            "data": {
                "spike_times": spike_times,
                "y_values": y_values,
                "y_labels": y_labels,
                "y_axis_name": self.y_axis_name,
                "unit_id": self.data.unit_id
            },
            "params": {
                "marker_size": self.marker_size,
                "marker_color": self.marker_color,
                "marker_alpha": self.marker_alpha,
                "show_grid": self.show_grid,
                "show_event_onset": self.show_event_onset,
                "title": self.title,
                **self.additional_params
            }
        }

        # Add optional parameters
        if self.xlim is not None:
            result["params"]["xlim"] = self.xlim
        if self.ylim is not None:
            result["params"]["ylim"] = self.ylim

        return result

    def update(self, **kwargs) -> None:
        """Update raster plot parameters"""
        for key, value in kwargs.items():
            if key == "spikes":
                self.data = SpikeData(spikes=value, unit_id=self.data.unit_id)
            elif key == "unit_id":
                self.data.unit_id = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value
