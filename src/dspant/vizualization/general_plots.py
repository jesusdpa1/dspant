import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure

from dspant.core.internals import public_api


@public_api
def plot_multi_channel_data(
    data,
    channels: list = None,
    fs: float = None,
    figsize: tuple = (15, 6),
    title: str = "Multi-Channel Data",
    color_mode: str = "colormap",  # New parameter: "colormap" or "single"
    colormap: str = "Set1",  # Used when color_mode is "colormap"
    color: str = "black",  # New parameter: Used when color_mode is "single"
    time_window: tuple = None,
    y_spread: float = 1.0,  # Control channel spacing
    y_offset: float = 0.0,  # Control baseline offset
    line_width: float = 1.0,  # Line width for signals
    alpha: float = 0.8,  # Transparency level
    grid: bool = True,  # Show grid lines
    show_channel_labels: bool = True,  # Show channel labels on y-axis
    normalize: bool = True,  # Normalize each channel's amplitude
    norm_scale: float = 0.4,  # Scale factor for normalized signals
) -> Figure:
    """
    Plot multiple channels of time-series data in a single axis with customizable spacing.

    Parameters:
    -----------
    data : numpy array
        Raw signal data (samples × channels)
    channels : list, optional
        List of channel indices to plot. If None, plot all channels.
    fs : float, optional
        Sampling frequency in Hz. Required to convert samples to time.
    figsize : tuple, default (15, 6)
        Figure size (width, height)
    title : str, default "Multi-Channel Data"
        Title for the plot
    color_mode : str, default "colormap"
        How to color channels:
        - "colormap": Use matplotlib colormap to assign different colors
        - "single": Use a single color for all channels
    colormap : str, default "Set1"
        Matplotlib colormap name to use when color_mode is "colormap"
    color : str, default "black"
        Color to use for all channels when color_mode is "single"
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
        If None, plot entire dataset.
    y_spread : float, default 1.0
        Controls vertical spacing between channels (higher values = more spread)
    y_offset : float, default 0.0
        Baseline offset for all channels (shifts entire plot up or down)
    line_width : float, default 1.0
        Width of signal lines
    alpha : float, default 0.8
        Transparency level for signals (0.0 = transparent, 1.0 = opaque)
    grid : bool, default True
        Whether to show grid lines
    show_channel_labels : bool, default True
        Whether to show channel labels on the y-axis
    normalize : bool, default True
        Whether to normalize each channel's amplitude
    norm_scale : float, default 0.4
        Scale factor for normalized signals (relative to y_spread)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    sns.set_theme(style="darkgrid")
    # Validate and prepare inputs
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Default to all channels if not specified
    if channels is None:
        channels = list(range(data.shape[1]))

    # Ensure fs is provided
    if fs is None:
        fs = 1000  # Default to 1000 Hz if not specified
        print("Warning: Sampling frequency not provided. Assuming 1000 Hz.")

    # Prepare figure with a single axis
    fig, ax = plt.subplots(figsize=figsize)

    # Color setup based on color_mode
    import matplotlib.cm as cm

    if color_mode == "colormap":
        cmap = cm.get_cmap(colormap)
    else:  # "single" color mode
        # No need for colormap, will use the single color provided
        pass

    # Compute full time array
    time = np.arange(data.shape[0]) / fs

    # Set up the time bounds for the plot
    start_idx = 0
    end_idx = data.shape[0]
    if time_window is not None:
        start_idx = max(0, int(time_window[0] * fs))
        end_idx = min(data.shape[0], int(time_window[1] * fs))
        ax.set_xlim(time_window)
    else:
        ax.set_xlim(time[start_idx], time[end_idx - 1])

    # Get time segment to plot
    subset_time = time[start_idx:end_idx]

    # Get channel information for labels
    channel_info = []
    channel_positions = []

    # Process each channel
    for idx, channel in enumerate(channels):
        # Calculate vertical offset for this channel
        channel_offset = y_offset + (len(channels) - 1 - idx) * y_spread
        channel_positions.append(channel_offset)

        # Add channel info for y-axis labels
        channel_info.append(f"Channel {channel}")

        # Get data for this channel
        subset_data = data[start_idx:end_idx, channel]

        # Normalize if requested
        if normalize:
            # Find max amplitude for normalization
            max_amplitude = np.max(np.abs(subset_data))
            if max_amplitude > 0:
                # Normalize and scale by y_spread * norm_scale
                norm_data = (
                    subset_data / max_amplitude * (y_spread * norm_scale)
                    + channel_offset
                )
            else:
                # If flat signal, just offset it
                norm_data = np.zeros_like(subset_data) + channel_offset
        else:
            # Just use the raw data with offset
            norm_data = subset_data + channel_offset

        # Choose color based on color_mode
        if color_mode == "colormap":
            plot_color = cmap(idx / max(1, len(channels) - 1))
        else:  # "single" color mode
            plot_color = color

        # Plot the channel data
        ax.plot(
            subset_time,
            norm_data,
            color=plot_color,
            alpha=alpha,
            linewidth=line_width,
            label=f"Channel {channel}",
        )

    # Add y-axis ticks for channel positions if requested
    if show_channel_labels:
        ax.set_yticks(channel_positions)
        ax.set_yticklabels(channel_info)
    else:
        ax.set_yticks([])  # Hide y-axis ticks

    # Customize appearance
    ax.set_xlabel("Time (s)")
    if show_channel_labels:
        ax.set_ylabel("Channels")

    # Set title
    ax.set_title(title)

    plt.tight_layout()

    return fig
