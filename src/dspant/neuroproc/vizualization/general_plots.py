import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_spikes(
    data,
    spike_df: pl.DataFrame,
    channels: list = None,
    fs: float = None,
    window_ms: float = 2.0,
    figsize: tuple = None,
    title: str = "Spike Waveforms",
    color_mode: str = "channel",
    max_spikes_per_channel: int = None,
    time_window: tuple = None,
):
    """
    Plot detected spikes across specified channels with optional time windowing.

    Parameters:
    -----------
    data : numpy array
        Raw signal data (samples × channels)
    spike_df : pl.DataFrame
        Polars DataFrame containing spike information with columns:
        - 'index': spike sample index
        - 'time_sec': spike time in seconds
        - 'channel': channel number
        - 'amplitude': spike amplitude
    channels : list, optional
        List of channel indices to plot. If None, plot all channels.
    fs : float, optional
        Sampling frequency in Hz. Required to convert samples to time.
    window_ms : float, default 2.0
        Window size around each spike in milliseconds
    figsize : tuple, optional
        Figure size (width, height)
    title : str, default "Spike Waveforms"
        Title for the plot
    color_mode : str, default "channel"
        Color scheme for spikes. Options:
        - "channel": different color for each channel
        - "amplitude": color based on spike amplitude
    max_spikes_per_channel : int, optional
        Maximum number of spikes to plot per channel. If None, plot all spikes.
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
        If None, plot entire dataset.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
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

    # Compute window samples
    window_samples = int(window_ms / 1000 * fs)
    half_window = window_samples // 2

    # Prepare figure
    if figsize is None:
        figsize = (15, 3 * len(channels))

    fig, axs = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
    if len(channels) == 1:
        axs = [axs]  # Ensure list for single channel

    # Color options
    import matplotlib.cm as cm

    # Compute full time array
    time = np.arange(data.shape[0]) / fs

    # Filter spikes based on time window
    if time_window is not None:
        start_time, end_time = time_window
        spike_df = spike_df.filter(
            (pl.col("time_sec") >= start_time) & (pl.col("time_sec") <= end_time)
        )

    # Filter and process spikes
    for idx, channel in enumerate(channels):
        # Filter spikes for this channel
        channel_spikes = spike_df.filter(pl.col("channel") == channel)

        # Limit number of spikes per channel if needed
        if (
            max_spikes_per_channel is not None
            and len(channel_spikes) > max_spikes_per_channel
        ):
            channel_spikes = channel_spikes.head(max_spikes_per_channel)

        # Create a base plot with full raw signal
        start_idx = 0
        end_idx = data.shape[0]
        if time_window is not None:
            start_idx = max(0, int(time_window[0] * fs))
            end_idx = min(data.shape[0], int(time_window[1] * fs))

        subset_time = time[start_idx:end_idx]
        subset_data = data[start_idx:end_idx, channel]

        axs[idx].plot(
            subset_time,
            subset_data,
            color="lightgray",
            alpha=0.5,
            linewidth=0.5,
            zorder=1,
        )
        axs[idx].set_ylabel(f"Channel {channel}")

        # Plot spikes for this channel
        spike_indices = channel_spikes["index"].to_numpy()
        spike_times = channel_spikes["time_sec"].to_numpy()
        spike_amplitudes = channel_spikes["amplitude"].to_numpy()

        for spike_idx, spike_time, spike_amp in zip(
            spike_indices, spike_times, spike_amplitudes
        ):
            # Determine spike window
            start = max(0, spike_idx - half_window)
            end = min(data.shape[0], spike_idx + half_window)

            # Create spike segment
            spike_segment = data[start:end, channel]
            spike_segment_time = time[start:end]

            # Ensure consistent window size by padding if needed
            if len(spike_segment) < window_samples:
                pad_before = (window_samples - len(spike_segment)) // 2
                pad_after = window_samples - len(spike_segment) - pad_before
                spike_segment = np.pad(
                    spike_segment, (pad_before, pad_after), mode="constant"
                )
                spike_segment_time = time[
                    max(0, start - pad_before) : min(len(time), end + pad_after)
                ]

            # Color selection
            if color_mode == "channel":
                color = cm.Set1(idx / len(channels))
            elif color_mode == "amplitude":
                # Normalize amplitude for color mapping
                norm_amp = (spike_amp - spike_amplitudes.min()) / (
                    spike_amplitudes.max() - spike_amplitudes.min()
                )
                color = cm.viridis(norm_amp)
            else:
                color = "red"

            # Normalize spike time to its actual time point
            if len(spike_segment_time) > 0:
                spike_segment_time = (
                    spike_segment_time - spike_segment_time[0] + spike_time
                )

                # Plot individual spike
                axs[idx].plot(
                    spike_segment_time,
                    spike_segment,
                    color=color,
                    linewidth=0.5,
                    alpha=0.5,
                    zorder=2,
                )

                # Mark spike point
                axs[idx].scatter(
                    spike_time,
                    data[spike_idx, channel],
                    color="red",
                    marker="x",
                    s=50,
                    zorder=3,
                )

    # Finalize plot
    plt.xlabel("Time (s)")
    plt.suptitle(title)
    plt.tight_layout()

    return fig


def plot_spike_raster(
    spike_df: pl.DataFrame,
    num_channels: int = None,
    figsize: tuple = (15, 6),
    title: str = "Spike Raster Plot",
    color_mode: str = "channel",
    time_window: tuple = None,
):
    """
    Create a raster plot of spikes across channels with optional time windowing.

    Parameters:
    -----------
    spike_df : pl.DataFrame
        Polars DataFrame containing spike information with columns:
        - 'index': spike sample index
        - 'time_sec': spike time in seconds
        - 'channel': channel number
    num_channels : int, optional
        Number of channels to plot. If None, use max channel number + 1
    figsize : tuple, default (15, 6)
        Figure size (width, height)
    title : str, default "Spike Raster Plot"
        Title for the plot
    color_mode : str, default "channel"
        Color scheme for spikes. Options:
        - "channel": different color for each channel
        - "time": color based on spike time
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
        If None, plot entire dataset.

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Filter spikes by time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        spike_df = spike_df.filter(
            (pl.col("time_sec") >= start_time) & (pl.col("time_sec") <= end_time)
        )

    # Determine number of channels
    if num_channels is None:
        num_channels = spike_df["channel"].max() + 1

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color options
    import matplotlib.cm as cm

    # Filter spikes
    for channel in range(num_channels):
        # Filter spikes for this channel
        channel_spikes = spike_df.filter(pl.col("channel") == channel)

        # Scatter plot for spikes
        if len(channel_spikes) > 0:
            times = channel_spikes["time_sec"].to_numpy()

            # Color selection
            if color_mode == "channel":
                color = cm.Set1(channel / num_channels)
            elif color_mode == "time":
                # Normalize time for color mapping
                norm_time = (times - times.min()) / (times.max() - times.min())
                color = [cm.viridis(t) for t in norm_time]
            else:
                color = "black"

            ax.scatter(
                times, [channel] * len(times), color=color, marker="|", alpha=0.7
            )

    # Finalize plot
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    ax.set_yticks(range(num_channels))
    ax.set_yticklabels([f"Channel {i}" for i in range(num_channels)])

    # Set x-axis limits if time window is specified
    if time_window is not None:
        ax.set_xlim(time_window)

    plt.tight_layout()

    return fig


def plot_spike_events(
    data,
    spike_df: pl.DataFrame,
    fs: float,
    channels: list = None,
    time_window: tuple = None,
    window_ms: float = 2.0,
    max_spikes_per_channel: int = None,
):
    """
    Convenience function to plot spike events with multiple visualization options.

    Parameters:
    -----------
    data : numpy array
        Raw signal data (samples × channels)
    spike_df : pl.DataFrame
        Spike detection results
    fs : float
        Sampling frequency in Hz
    channels : list, optional
        Channels to plot. If None, use all channels.
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds
    window_ms : float, default 2.0
        Window size around each spike in milliseconds
    max_spikes_per_channel : int, optional
        Maximum number of spikes to plot per channel. If None, plot all spikes.

    Returns:
    --------
    tuple
        (waveform_figure, raster_figure)
    """
    # Default to all channels if not specified
    if channels is None:
        channels = list(range(data.shape[1]))

    # Plot spike waveforms
    waveform_fig = plot_spikes(
        data,
        spike_df,
        channels=channels,
        fs=fs,
        time_window=time_window,
        window_ms=window_ms,
        max_spikes_per_channel=max_spikes_per_channel,
    )

    # Plot raster
    raster_fig = plot_spike_raster(
        spike_df, num_channels=len(channels), time_window=time_window
    )

    return waveform_fig, raster_fig
