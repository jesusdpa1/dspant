# src/dspant_viz/visualization/stream/ts_area.py
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numba as nb
import numpy as np

from dspant_viz.core.internals import public_api
from dspant_viz.visualization.stream.base import BaseStreamVisualization


@public_api(module_override="dspant_viz.visualization")
class TimeSeriesAreaPlot(BaseStreamVisualization):
    """
    Component for time series area visualization optimized for dask arrays.

    Designed for visualizing derivative or envelope-like data
    with emphasis on area representation.
    """

    def __init__(
        self,
        data: Union[da.Array, Dict[int, np.ndarray]],
        sampling_rate: float,
        channels: Optional[List[int]] = None,
        time_window: Optional[Tuple[float, float]] = None,
        fill_to: Union[float, str] = "zero",
        fill_alpha: float = 0.3,
        color_mode: str = "colormap",
        colormap: str = "viridis",
        color: str = "black",
        y_spread: float = 1.0,
        y_offset: float = 0.0,
        line_width: float = 1.0,
        alpha: float = 0.8,
        grid: bool = True,
        show_channel_labels: bool = True,
        normalize: bool = True,
        downsample: bool = True,
        max_points: int = 10000,
        **kwargs,
    ):
        """
        Initialize time series area visualization.

        Parameters
        ----------
        fill_to : Union[float, str]
            Value or method to fill area to
            - 'zero': Fill to zero
            - 'min': Fill to minimum value
            - float: Specific value to fill to
        fill_alpha : float
            Transparency of the fill area
        """
        # Call parent constructor
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            elements=channels,
            time_window=time_window,
            **kwargs,
        )

        # Ensure data is at least 2D
        if isinstance(data, da.Array) and data.ndim == 1:
            self.data = data.reshape(-1, 1)
        else:
            self.data = data

        # Store additional parameters
        self.channels = self.elements
        self.fill_to = fill_to
        self.fill_alpha = fill_alpha
        self.color_mode = color_mode
        self.colormap = colormap
        self.color = color
        self.y_spread = y_spread
        self.y_offset = y_offset
        self.line_width = line_width
        self.alpha = alpha
        self.grid = grid
        self.show_channel_labels = show_channel_labels
        self.normalize = normalize
        self.downsample = downsample
        self.max_points = max_points

    def _get_display_data(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Get processed time and signal data ready for display.
        """
        # Get time array
        time = self._get_time_array()

        # Collect data for each channel
        signals = []
        for channel in self.channels:
            # Get channel data
            channel_data = self._get_element_data(channel)

            # Apply time window
            channel_data, filtered_time = self._apply_time_window(channel_data, time)

            # Apply downsampling if needed
            if self.downsample:
                channel_data, downsampled_time = self._downsample_if_needed(
                    channel_data, filtered_time, self.max_points
                )
            else:
                downsampled_time = filtered_time

            # Append to signal list
            signals.append(channel_data)

        return downsampled_time, signals

    def _get_fill_value(self, data: np.ndarray) -> float:
        """
        Determine fill value based on specified method.

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        float
            Value to fill area to
        """
        if isinstance(self.fill_to, (int, float)):
            return float(self.fill_to)

        if self.fill_to == "zero":
            return 0.0

        if self.fill_to == "min":
            return float(np.min(data))

        raise ValueError(f"Invalid fill_to method: {self.fill_to}")

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Processed data and visualization parameters
        """
        time, signals = self._get_display_data()

        # Process data for channels
        channel_info = []
        processed_data = []
        fill_values = []

        # Process each channel for display
        for idx, channel in enumerate(self.channels):
            # Calculate vertical offset for this channel
            channel_offset = (
                self.y_offset + (len(self.channels) - 1 - idx) * self.y_spread
            )

            # Add channel info for y-axis labels
            channel_info.append(f"Channel {channel}")

            # Get signal data
            signal = signals[idx]

            # Normalize if requested
            if self.normalize and len(signal) > 0:
                max_amplitude = np.max(np.abs(signal))
                if max_amplitude > 0:
                    # Normalize and scale by y_spread
                    norm_data = (
                        signal / max_amplitude * (self.y_spread * 0.4)
                    ) + channel_offset
                else:
                    # If flat signal, just offset it
                    norm_data = np.zeros_like(signal) + channel_offset
            else:
                # Just use the raw data with offset
                norm_data = signal + channel_offset

            processed_data.append(norm_data)

            # Determine fill value for this channel
            fill_values.append(self._get_fill_value(signal) + channel_offset)

        # Prepare channel positions for y-axis ticks
        channel_positions = [
            self.y_offset + (len(self.channels) - 1 - idx) * self.y_spread
            for idx in range(len(self.channels))
        ]

        return {
            "data": {
                "time": time,
                "signals": processed_data,
                "fill_values": fill_values,
                "channels": self.channels,
                "channel_info": channel_info,
                "channel_positions": channel_positions,
            },
            "params": {
                "sampling_rate": self.sampling_rate,
                "fill_to": self.fill_to,
                "fill_alpha": self.fill_alpha,
                "color_mode": self.color_mode,
                "colormap": self.colormap,
                "color": self.color,
                "y_spread": self.y_spread,
                "y_offset": self.y_offset,
                "line_width": self.line_width,
                "alpha": self.alpha,
                "grid": self.grid,
                "show_channel_labels": self.show_channel_labels,
                "normalize": self.normalize,
                "time_window": self.time_window,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update plot parameters.
        """
        # Update channels if provided
        if "channels" in kwargs:
            channels = kwargs.pop("channels")
            self.channels = channels
            self.elements = channels  # Keep elements in sync

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
            from dspant_viz.backends.mpl.ts_area import render_ts_area
        elif backend == "plotly":
            from dspant_viz.backends.plotly.ts_area import render_ts_area
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_ts_area(self.get_data(), **kwargs)
