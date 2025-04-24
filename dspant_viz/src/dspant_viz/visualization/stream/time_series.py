# src/dspant_viz/visualization/stream/time_series.py
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dspant_viz.core.base import VisualizationComponent


class TimeSeriesPlot(VisualizationComponent):
    """Component for time series visualization optimized for dask arrays"""

    def __init__(
        self,
        data: da.Array,
        sampling_rate: float,
        channels: Optional[List[int]] = None,
        time_window: Optional[Tuple[float, float]] = None,
        color_mode: str = "colormap",  # "colormap" or "single"
        colormap: str = "Set1",  # Used with color_mode="colormap"
        color: str = "black",  # Used with color_mode="single"
        y_spread: float = 1.0,  # Control channel spacing
        y_offset: float = 0.0,  # Control baseline offset
        line_width: float = 1.0,  # Line width
        alpha: float = 0.8,  # Transparency
        grid: bool = True,  # Show grid
        show_channel_labels: bool = True,  # Label channels on y-axis
        normalize: bool = True,  # Normalize each channel separately
        downsample: bool = True,  # Enable downsampling for large datasets
        max_points: int = 10000,  # Maximum points to display per channel
        **kwargs,
    ):
        """
        Initialize time series visualization for multichannel data.

        Parameters
        ----------
        data : da.Array
            Dask array with data in the format (samples × channels)
        sampling_rate : float
            Sampling frequency in Hz
        channels : list of int, optional
            Specific channels to display. If None, displays all channels.
        time_window : tuple of (float, float), optional
            Start and end times in seconds to display. If None, shows all data.
        color_mode : str
            "colormap" to use different colors from a colormap, or "single" for one color
        colormap : str
            Matplotlib colormap name to use when color_mode is "colormap"
        color : str
            Color to use for all channels when color_mode is "single"
        y_spread : float
            Factor controlling vertical spacing between channels
        y_offset : float
            Baseline vertical offset for all channels
        line_width : float
            Width of signal lines
        alpha : float
            Transparency of signals (0.0-1.0)
        grid : bool
            Whether to show grid lines
        show_channel_labels : bool
            Whether to show channel labels on y-axis
        normalize : bool
            Whether to normalize each channel's amplitude
        downsample : bool
            Whether to downsample signals for display
        max_points : int
            Maximum number of points to display per channel when downsampling
        **kwargs
            Additional configuration parameters
        """
        super().__init__(data, **kwargs)

        # Ensure data is at least 2D
        if data.ndim == 1:
            self.data = data.reshape(-1, 1)
        else:
            self.data = data

        # Store parameters
        self.sampling_rate = sampling_rate
        self.time_window = time_window

        # Validate available channels
        if self.data.ndim < 2:
            available_channels = 1
        else:
            available_channels = self.data.shape[1]

        # Default to all channels if not specified, or validate given channels
        if channels is None:
            self.channels = list(range(available_channels))
        else:
            # Filter out any out-of-range channel indices
            self.channels = [ch for ch in channels if ch < available_channels]
            if len(self.channels) == 0:
                # If no valid channels were specified, use all available
                self.channels = list(range(available_channels))

        # Store other parameters
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

        # Print debug info
        print(f"Initialized TimeSeriesPlot with data shape {self.data.shape}")
        print(f"Using channels: {self.channels}")
        print(f"Available channels: {available_channels}")

    def _get_display_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for display, handling time window selection and downsampling.

        Returns
        -------
        tuple
            (time_values, signal_values) as numpy arrays
        """
        # Calculate full time array
        full_time = np.arange(self.data.shape[0]) / self.sampling_rate

        # Set default time window if not specified
        if self.time_window is None:
            start_idx = 0
            end_idx = self.data.shape[0]
            start_time = full_time[0]
            end_time = full_time[-1]
        else:
            start_time, end_time = self.time_window
            start_idx = max(0, int(start_time * self.sampling_rate))
            end_idx = min(self.data.shape[0], int(end_time * self.sampling_rate))

        # Extract subset of data for the time window
        subset_time = full_time[start_idx:end_idx]

        # Apply downsampling if needed
        if self.downsample and len(subset_time) > self.max_points:
            # Calculate downsampling factor
            downsample_factor = len(subset_time) // self.max_points
            # Downsample time and data
            subset_time = subset_time[::downsample_factor]
            subset_data = self.data[start_idx:end_idx:downsample_factor, :].compute()
        else:
            # Use full resolution data for the time window
            subset_data = self.data[start_idx:end_idx, :].compute()

        return subset_time, subset_data

    def get_data(self) -> Dict:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Data and parameters needed for rendering
        """
        subset_time, subset_data = self._get_display_data()

        # Process data for channels
        channel_info = []
        processed_data = []
        actual_channels = []

        # In case self.channels doesn't match actual data dimensions
        available_channels = min(len(self.channels), subset_data.shape[1])

        for idx in range(available_channels):
            channel = self.channels[idx]
            # Calculate vertical offset for this channel
            channel_offset = (
                self.y_offset + (available_channels - 1 - idx) * self.y_spread
            )

            # Add channel info for y-axis labels
            channel_info.append(f"Channel {channel}")
            actual_channels.append(channel)

            # Get data for this channel
            try:
                channel_data = subset_data[:, idx]

                # Normalize if requested
                if self.normalize:
                    # Find max amplitude for normalization
                    max_amplitude = np.max(np.abs(channel_data))
                    if max_amplitude > 0:
                        # Normalize and scale by y_spread
                        norm_data = (
                            channel_data / max_amplitude * (self.y_spread * 0.4)
                        ) + channel_offset
                    else:
                        # If flat signal, just offset it
                        norm_data = np.zeros_like(channel_data) + channel_offset
                else:
                    # Just use the raw data with offset
                    norm_data = channel_data + channel_offset

                processed_data.append(norm_data)
            except IndexError as e:
                print(f"Error accessing data for channel {idx}: {e}")
                print(f"Channel shape: {subset_data.shape}, Requested index: {idx}")
                # Add a zero signal as a placeholder
                processed_data.append(np.zeros_like(subset_time) + channel_offset)

        # Print debug information
        print(f"Channels: {actual_channels}")
        print(
            f"Processed {len(processed_data)} channels with {len(subset_time)} time points"
        )

        return {
            "data": {
                "time": subset_time,
                "signals": processed_data,
                "channels": actual_channels,
                "channel_info": channel_info,
                "channel_positions": [
                    self.y_offset + (len(processed_data) - 1 - idx) * self.y_spread
                    for idx in range(len(processed_data))
                ],
            },
            "params": {
                "sampling_rate": self.sampling_rate,
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

        Parameters
        ----------
        **kwargs
            Parameters to update
        """
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
            from dspant_viz.backends.mpl.time_series import render_time_series
        elif backend == "plotly":
            from dspant_viz.backends.plotly.time_series import render_time_series
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_time_series(self.get_data(), **kwargs)
