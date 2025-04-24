# components/stream/timeseries.py
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import TimeSeriesData, MultiChannelData
from dspant_viz.core.internals import public_api


@public_api
class TimeSeriesPlot(VisualizationComponent):
    """Component for time series visualization"""

    def __init__(
        self,
        times: List[float],
        values: List[float],
        sampling_rate: Optional[float] = None,
        channel_id: Optional[Union[int, str]] = None,
        channel_name: Optional[str] = None,
        line_color: str = "#1f77b4",
        line_width: float = 1.5,
        show_grid: bool = True,
        time_window: Optional[Tuple[float, float]] = None,  # Added time_window
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize time series plot component

        Parameters
        ----------
        times : List[float]
            Time points in seconds
        values : List[float]
            Signal values
        sampling_rate : float, optional
            Sampling rate in Hz
        channel_id : int or str, optional
            Channel identifier
        channel_name : str, optional
            Channel name for display
        line_color : str
            Color for the signal line
        line_width : float
            Width of the signal line
        show_grid : bool
            Whether to show grid lines
        time_window : Tuple[float, float], optional
            Time window to display (start_time, end_time) in seconds
            This filters the data before visualization
        xlim : Tuple[float, float], optional
            X-axis limits as (min, max) - applied after any time_window filtering
        ylim : Tuple[float, float], optional
            Y-axis limits as (min, max)
        title : str, optional
            Plot title. If None, a default title may be generated
        """
        # Validate and store the time series data
        self.data = TimeSeriesData(
            times=times,
            values=values,
            sampling_rate=sampling_rate,
            channel_id=channel_id,
            channel_name=channel_name
        )

        # Store visualization parameters
        self.line_color = line_color
        self.line_width = line_width
        self.show_grid = show_grid
        self.time_window = time_window
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.additional_params = kwargs

    def get_data(self) -> Dict[str, Any]:
        """Get time series data in format ready for rendering"""
        # Apply time window filtering if specified
        if self.time_window is not None:
            start_time, end_time = self.time_window
            times = np.array(self.data.times)
            values = np.array(self.data.values)

            # Find indices within the time window
            mask = (times >= start_time) & (times <= end_time)
            filtered_times = times[mask].tolist()
            filtered_values = values[mask].tolist()
        else:
            filtered_times = self.data.times
            filtered_values = self.data.values

        # Build the data dictionary
        result = {
            "data": {
                "times": filtered_times,
                "values": filtered_values,
                "sampling_rate": self.data.sampling_rate,
                "channel_id": self.data.channel_id,
                "channel_name": self.data.channel_name,
                "time_window": self.time_window
            },
            "params": {
                "line_color": self.line_color,
                "line_width": self.line_width,
                "show_grid": self.show_grid,
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
        """Update time series plot parameters"""
        for key, value in kwargs.items():
            if key in ["times", "values", "sampling_rate", "channel_id", "channel_name"]:
                # Update data attribute directly
                setattr(self.data, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value


@public_api
class MultiChannelPlot(VisualizationComponent):
    """Component for multi-channel time series visualization"""

    def __init__(
        self,
        times: List[float],
        channels: Dict[Union[int, str], List[float]],
        sampling_rate: Optional[float] = None,
        color_mode: str = "colormap",  # "colormap" or "single"
        colormap: str = "viridis",
        color: str = "black",
        line_width: float = 1.0,
        alpha: float = 0.8,
        y_spread: float = 1.0,
        y_offset: float = 0.0,
        normalize: bool = True,
        norm_scale: float = 0.4,
        show_channel_labels: bool = True,
        show_grid: bool = True,
        time_window: Optional[Tuple[float, float]] = None,  # Added time_window
        xlim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize multi-channel plot component

        Parameters
        ----------
        times : List[float]
            Time points in seconds
        channels : Dict[Union[int, str], List[float]]
            Dictionary mapping channel IDs to signal values
        sampling_rate : float, optional
            Sampling rate in Hz
        color_mode : str
            How to color channels: "colormap" or "single"
        colormap : str
            Name of colormap to use when color_mode is "colormap"
        color : str
            Color to use when color_mode is "single"
        line_width : float
            Width of signal lines
        alpha : float
            Transparency of signal lines
        y_spread : float
            Vertical spacing between channels
        y_offset : float
            Baseline offset for all channels
        normalize : bool
            Whether to normalize each channel's amplitude
        norm_scale : float
            Scale factor for normalized signals
        show_channel_labels : bool
            Whether to show channel labels on y-axis
        show_grid : bool
            Whether to show grid lines
        time_window : Tuple[float, float], optional
            Time window to display (start_time, end_time) in seconds
            This filters the data before visualization
        xlim : Tuple[float, float], optional
            X-axis limits as (min, max) - applied after any time_window filtering
        title : str, optional
            Plot title
        """
        # Validate and store multi-channel data
        self.data = MultiChannelData(
            times=times,
            channels=channels,
            sampling_rate=sampling_rate
        )

        # Store visualization parameters
        self.color_mode = color_mode
        self.colormap = colormap
        self.color = color
        self.line_width = line_width
        self.alpha = alpha
        self.y_spread = y_spread
        self.y_offset = y_offset
        self.normalize = normalize
        self.norm_scale = norm_scale
        self.show_channel_labels = show_channel_labels
        self.show_grid = show_grid
        self.time_window = time_window
        self.xlim = xlim
        self.title = title
        self.additional_params = kwargs

    def get_data(self) -> Dict[str, Any]:
        """Get multi-channel data in format ready for rendering"""
        # Apply time window filtering if specified
        if self.time_window is not None:
            start_time, end_time = self.time_window
            times = np.array(self.data.times)

            # Find indices within the time window
            mask = (times >= start_time) & (times <= end_time)
            filtered_times = times[mask].tolist()

            # Filter each channel
            filtered_channels = {}
            for channel_id, values in self.data.channels.items():
                values_array = np.array(values)
                filtered_channels[channel_id] = values_array[mask].tolist()
        else:
            filtered_times = self.data.times
            filtered_channels = self.data.channels

        # Build the data dictionary
        result = {
            "data": {
                "times": filtered_times,
                "channels": filtered_channels,
                "sampling_rate": self.data.sampling_rate,
                "channel_ids": list(filtered_channels.keys()),
                "time_window": self.time_window
            },
            "params": {
                "color_mode": self.color_mode,
                "colormap": self.colormap,
                "color": self.color,
                "line_width": self.line_width,
                "alpha": self.alpha,
                "y_spread": self.y_spread,
                "y_offset": self.y_offset,
                "normalize": self.normalize,
                "norm_scale": self.norm_scale,
                "show_channel_labels": self.show_channel_labels,
                "show_grid": self.show_grid,
                "title": self.title,
                **self.additional_params
            }
        }

        # Add optional parameters
        if self.xlim is not None:
            result["params"]["xlim"] = self.xlim

        return result

    def update(self, **kwargs) -> None:
        """Update multi-channel plot parameters"""
        for key, value in kwargs.items():
            if key in ["times", "channels", "sampling_rate"]:
                # Update data attribute directly
                setattr(self.data, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value
