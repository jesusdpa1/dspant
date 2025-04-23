from typing import List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_onset_detection_results(
    data: Union[np.ndarray, da.Array],
    events: pl.DataFrame,
    fs: float,
    time_window: Optional[Tuple[float, float]] = None,
    channels: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (20, 10),
    title: str = "Onset Detection Results",
    highlight_color: str = "red",
    normalize: bool = True,
    y_spread: float = 1.0,
    show_threshold: bool = True,
    threshold: Optional[float] = None,
) -> plt.Figure:
    """
    Plot multi-channel signal with highlighted onset/offset regions.

    Parameters:
    -----------
    data : array
        Raw signal data (samples × channels)
    events : pl.DataFrame
        DataFrame with onset_idx, offset_idx, and channel columns
    fs : float
        Sampling frequency in Hz
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds
    channels : list, optional
        Channels to plot. If None, plot all channels.
    figsize : tuple, default (20, 10)
        Figure size
    title : str
        Plot title
    highlight_color : str
        Color for highlighting detected events
    normalize : bool, default True
        Normalize channel amplitudes
    y_spread : float, default 1.0
        Vertical spacing between channels
    show_threshold : bool, default True
        Show threshold line if provided
    threshold : float, optional
        Threshold value to display

    Returns:
    --------
    matplotlib Figure
    """
    # Set aesthetics
    sns.set_palette("colorblind")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Montserrat"]
    sns.set_theme(style="darkgrid")

    # Font sizes
    TITLE_SIZE = 18
    SUBTITLE_SIZE = 16
    AXIS_LABEL_SIZE = 14
    TICK_SIZE = 12

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Compute data if it's a dask array
    if hasattr(data, "compute"):
        data = data.compute()

    # Default to all channels
    if channels is None:
        channels = list(range(data.shape[1]))

    # Create time array
    time_array = np.arange(len(data)) / fs

    # Apply time window
    start_idx = 0
    end_idx = len(data)
    if time_window is not None:
        start_idx = max(0, int(time_window[0] * fs))
        end_idx = min(len(data), int(time_window[1] * fs))

    # Prepare figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(channels), 1, height_ratios=[1] * len(channels))

    # Process each channel
    for i, channel in enumerate(channels):
        ax = fig.add_subplot(gs[i, 0])

        # Plot channel data
        channel_data = data[start_idx:end_idx, channel]
        channel_time = time_array[start_idx:end_idx]

        # Normalize if requested
        if normalize:
            max_amp = np.max(np.abs(channel_data))
            if max_amp > 0:
                channel_data = channel_data / max_amp

        ax.plot(channel_time, channel_data, color="navy", linewidth=1.5)

        # Filter events for this channel and time window
        channel_events = events.filter(
            (pl.col("channel") == channel)
            & (pl.col("onset_idx") >= start_idx)
            & (pl.col("onset_idx") <= end_idx)
        )

        # Highlight events
        for row in channel_events.iter_rows(named=True):
            onset_time = (row["onset_idx"] - start_idx) / fs
            offset_time = (row["offset_idx"] - start_idx) / fs

            ax.axvspan(onset_time, offset_time, alpha=0.3, color=highlight_color)
            ax.axvline(x=onset_time, color=highlight_color, linestyle="-", linewidth=1)

        # Threshold line
        if show_threshold and threshold is not None:
            ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1)

        # Labeling
        ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, weight="bold")
        ax.set_ylabel(f"Channel {channel}", fontsize=AXIS_LABEL_SIZE, weight="bold")
        ax.set_title(f"Channel {channel} Onsets", fontsize=SUBTITLE_SIZE, weight="bold")
        ax.tick_params(labelsize=TICK_SIZE)

    # Overall title
    plt.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
