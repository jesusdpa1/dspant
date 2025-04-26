# src/dspant_viz/visualization/stream/time_series.py
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np

from dspant_viz.core.internals import public_api
from dspant_viz.visualization.stream.base import BaseStreamVisualization


@public_api(module_override="dspant_viz.visualization")
class TimeSeriesPlot(BaseStreamVisualization):
    """Component for time series visualization optimized for dask arrays"""

    def __init__(
        self,
        data: da.Array,
        sampling_rate: float,
        channels: Optional[List[int]] = None,
        time_window: Optional[Tuple[float, float]] = None,
        initial_time_window: Optional[Tuple[float, float]] = None,  # Add this parameter
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
        resample_method: str = "lttb",  # Add resampling method parameter
        **kwargs,
    ):
        """
        Initialize time series visualization for multichannel data.

        Parameters
        ----------
        data : da.Array
            Dask array with time series data
        sampling_rate : float
            Sampling frequency in Hz
        channels : List[int], optional
            List of channels to display. If None, displays all channels.
        time_window : Tuple[float, float], optional
            Full time window of data (start_time, end_time) in seconds
        initial_time_window : Tuple[float, float], optional
            Initial time window to display (start_time, end_time) in seconds.
            This is the window shown when the plot is first rendered,
            but the user can zoom out to see the full dataset.
        color_mode : str, optional
            'colormap' or 'single'
        colormap : str, optional
            Colormap name for multiple channels
        color : str, optional
            Color for 'single' color mode
        y_spread : float, optional
            Vertical spread between channels
        y_offset : float, optional
            Vertical offset for all channels
        line_width : float, optional
            Width of lines
        alpha : float, optional
            Transparency of lines
        grid : bool, optional
            Whether to show grid
        show_channel_labels : bool, optional
            Whether to show channel labels on y-axis
        normalize : bool, optional
            Whether to normalize channel amplitudes
        downsample : bool, optional
            Whether to downsample data for display
        max_points : int, optional
            Maximum number of points to display per channel
        resample_method : str
            Resampling method to use: 'minmax', 'lttb', 'minmaxlttb', 'nth', 'overlap'
        **kwargs
            Additional configuration parameters
        """
        # Call parent constructor with elements=channels
        super().__init__(
            data=data,
            sampling_rate=sampling_rate,
            elements=channels,  # Pass channels as elements
            time_window=time_window,
            **kwargs,
        )

        # Ensure data is at least 2D
        if isinstance(data, da.Array) and data.ndim == 1:
            self.data = data.reshape(-1, 1)
        else:
            self.data = data

        # Create a channels attribute that references elements
        self.channels = self.elements

        # Calculate full data duration if not provided
        if time_window is None and isinstance(data, da.Array):
            data_duration = data.shape[0] / sampling_rate
            self.time_window = (0, data_duration)

        # Set initial time window
        self.initial_time_window = initial_time_window

        # If no initial time window is provided but we have a full time window,
        # create a default 10-second window at the start
        if self.initial_time_window is None and self.time_window is not None:
            start = self.time_window[0]
            end = min(
                start + 10.0, self.time_window[1]
            )  # Ensure we don't exceed full window
            self.initial_time_window = (start, end)

        # Store additional parameters
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
        self.resample_method = resample_method

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

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.
        """
        time, signals = self._get_display_data()

        # Process data for channels
        channel_info = []
        processed_data = []

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

        # Prepare channel positions for y-axis ticks
        channel_positions = [
            self.y_offset + (len(self.channels) - 1 - idx) * self.y_spread
            for idx in range(len(self.channels))
        ]

        return {
            "data": {
                "time": time,
                "signals": processed_data,
                "channels": self.channels,
                "channel_info": channel_info,
                "channel_positions": channel_positions,
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
                "initial_time_window": self.initial_time_window,
                "resample_method": self.resample_method,
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

        # Update time windows if provided
        if "time_window" in kwargs:
            self.time_window = kwargs.pop("time_window")

        if "initial_time_window" in kwargs:
            self.initial_time_window = kwargs.pop("initial_time_window")
        elif "time_window" in kwargs and self.time_window is not None:
            # If time_window changed but initial_time_window didn't, update initial_time_window
            start = self.time_window[0]
            end = min(start + 10.0, self.time_window[1])
            self.initial_time_window = (start, end)

        # Update other parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

    def plot(
        self,
        backend: str = "plotly",
        **kwargs,
    ):
        """
        Generate the plot using the specified backend with dynamic loading.

        Parameters
        ----------
        backend : str
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs
            Additional parameters for the backend

        Returns
        -------
        Figure, FigureResampler, or FigureWidgetResampler
            Plot figure from the specified backend
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.time_series import render_time_series

            # Get data for all channels - MPL is for static publication figures
            plot_data = self.get_data()
            return render_time_series(plot_data, **kwargs)

        elif backend == "plotly":
            from dspant_viz.backends.plotly.time_series import render_time_series

            # Use all channels
            plot_data = self.get_data()

            # Set defaults for resampling if not specified
            if "resample_method" not in kwargs and hasattr(self, "resample_method"):
                kwargs["resample_method"] = self.resample_method

            if "max_n_samples" not in kwargs:
                kwargs["max_n_samples"] = 5000

            fig = render_time_series(plot_data, **kwargs)

            # For IPython environments, we need to return the FigureWidgetResampler directly
            # without calling .show() on it, as that would render it statically
            is_resampler = hasattr(fig, "reload_data")

            # For testing if we're in a notebook
            in_notebook = False
            try:
                from IPython import get_ipython

                if get_ipython() is not None:
                    in_notebook = True
            except ImportError:
                pass

            if is_resampler and in_notebook:
                # Return the figure without displaying it
                # (user needs to display it in a cell)
                return fig
            else:
                return fig
        else:
            raise ValueError(f"Unsupported backend: {backend}")
