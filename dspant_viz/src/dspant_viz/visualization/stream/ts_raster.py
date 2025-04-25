# src/dspant_viz/visualization/stream/ts_raster.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dspant_viz.core.internals import public_api
from dspant_viz.visualization.stream.base import BaseStreamVisualization


@public_api(module_override="dspant_viz.visualization")
class TimeSeriesRasterPlot(BaseStreamVisualization):
    """Component for time series spike raster visualization with multiple units"""

    def __init__(
        self,
        data: Dict[int, np.ndarray],
        sampling_rate: float,
        units: Optional[List[int]] = None,  # Use units instead of elements or channels
        time_window: Optional[Tuple[float, float]] = None,
        marker_size: float = 4,
        marker_type: str = "|",
        use_colormap: bool = True,
        colormap: str = "viridis",
        alpha: float = 0.7,
        show_legend: bool = True,
        y_spread: float = 1.0,
        y_offset: float = 0.0,
        show_labels: bool = True,
        downsample: bool = False,
        **kwargs,
    ):
        """
        Initialize the time series raster plot for multiple units.
        """
        # Call parent constructor with elements=units
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            elements=units,  # Pass units as elements
            time_window=time_window,
            **kwargs,
        )

        # Create a units attribute that references elements
        self.units = self.elements

        # Store parameters
        self.marker_size = marker_size
        self.marker_type = marker_type
        self.use_colormap = use_colormap
        self.colormap = colormap
        self.alpha = alpha
        self.show_legend = show_legend
        self.y_spread = y_spread
        self.y_offset = y_offset
        self.show_labels = show_labels
        self.downsample = downsample

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.
        """
        # Collect spike times for each unit and apply time window
        unit_spikes = {}
        unit_labels = {}
        y_positions = {}

        for i, unit_id in enumerate(self.units):
            # Get spikes for this unit
            spikes = self._get_element_data(unit_id)

            # Apply time window
            if self.time_window is not None:
                start, end = self.time_window
                mask = (spikes >= start) & (spikes <= end)
                spikes = spikes[mask]

            # Apply downsampling if requested and necessary
            if self.downsample and len(spikes) > 10000:
                # Simple random downsampling for spike data
                downsample_idx = np.random.choice(
                    len(spikes), size=10000, replace=False
                )
                spikes = spikes[downsample_idx]

            # Calculate y-position for this unit (bottom to top)
            y_pos = self.y_offset + (len(self.units) - 1 - i) * self.y_spread

            # Store for rendering
            unit_spikes[unit_id] = (
                spikes.tolist() if hasattr(spikes, "tolist") else spikes
            )
            unit_labels[unit_id] = f"Unit {unit_id}"
            y_positions[unit_id] = y_pos

        # Create list of positions for y-axis
        y_ticks = [y_positions[unit_id] for unit_id in self.units]
        y_tick_labels = [unit_labels[unit_id] for unit_id in self.units]

        return {
            "data": {
                "unit_ids": self.units,
                "unit_spikes": unit_spikes,
                "unit_labels": unit_labels,
                "y_positions": y_positions,
                "y_ticks": y_ticks,
                "y_tick_labels": y_tick_labels,
            },
            "params": {
                "marker_size": self.marker_size,
                "marker_type": self.marker_type,
                "use_colormap": self.use_colormap,
                "colormap": self.colormap,
                "alpha": self.alpha,
                "show_legend": self.show_legend,
                "show_labels": self.show_labels,
                "time_window": self.time_window,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update component parameters.
        """
        # Special handling for units
        if "units" in kwargs:
            units = kwargs.pop("units")
            self.units = units
            self.elements = units  # Keep elements in sync

        # Update time window if provided
        if "time_window" in kwargs:
            self.time_window = kwargs.pop("time_window")

        # Update other parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate the plot using the specified backend.

        Parameters
        ----------
        backend : str
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs
            Additional parameters for the backend

        Returns
        -------
        Figure or plotly.graph_objects.Figure
            Plot figure from the specified backend
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.ts_raster import render_ts_raster
        elif backend == "plotly":
            from dspant_viz.backends.plotly.ts_raster import render_ts_raster
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_ts_raster(self.get_data(), **kwargs)
