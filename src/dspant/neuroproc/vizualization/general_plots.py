import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.figure import Figure


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
    sort_spikes: str = None,  # New parameter: "time" or "amplitude"
    sort_order: str = "ascending",  # New parameter: "ascending" or "descending"
    show_background: bool = True,  # New parameter to control background signal display
) -> Figure:
    """
    Plot detected spikes across specified channels with optional time windowing and sorting.

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
        - "time": color based on spike time (earliest to latest)
    max_spikes_per_channel : int, optional
        Maximum number of spikes to plot per channel. If None, plot all spikes.
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
        If None, plot entire dataset.
    sort_spikes : str, optional
        How to sort spikes for display. Options:
        - "time": Sort by spike time
        - "amplitude": Sort by spike amplitude
        - None: No specific sorting (default)
    sort_order : str, default "ascending"
        Order for sorting spikes:
        - "ascending": Smallest/earliest first
        - "descending": Largest/latest first
    show_background : bool, default True
        Whether to show the background signal or only the spike waveforms

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

        # Sort spikes if requested
        if sort_spikes is not None:
            if sort_spikes == "time":
                channel_spikes = channel_spikes.sort(
                    "time_sec", descending=(sort_order == "descending")
                )
            elif sort_spikes == "amplitude":
                # Handle both positive and negative amplitudes based on absolute value
                if "amplitude" in channel_spikes.columns:
                    if sort_order == "ascending":
                        channel_spikes = channel_spikes.sort(pl.abs("amplitude"))
                    else:
                        channel_spikes = channel_spikes.sort(
                            pl.abs("amplitude"), descending=True
                        )

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

        # Plot background signal if enabled
        if show_background:
            axs[idx].plot(
                subset_time,
                subset_data,
                color="lightgray",
                alpha=0.8,
                linewidth=0.5,
                zorder=1,
            )

        axs[idx].set_ylabel(f"Channel {channel}")

        # Plot spikes for this channel
        spike_indices = channel_spikes["index"].to_numpy()
        spike_times = channel_spikes["time_sec"].to_numpy()

        # Check if amplitude column exists
        if "amplitude" in channel_spikes.columns:
            spike_amplitudes = channel_spikes["amplitude"].to_numpy()
        else:
            # Create dummy amplitudes if the column doesn't exist
            spike_amplitudes = np.ones_like(spike_times)

        # Create a color gradient based on time if color_mode is time
        if color_mode == "time" and len(spike_times) > 0:
            time_norm = (spike_times - spike_times.min()) / (
                spike_times.max() - spike_times.min() + 1e-10
            )
            time_colors = [cm.viridis(t) for t in time_norm]

        for i, (spike_idx, spike_time, spike_amp) in enumerate(
            zip(spike_indices, spike_times, spike_amplitudes)
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
                norm_amp = (abs(spike_amp) - min(abs(spike_amplitudes))) / (
                    max(abs(spike_amplitudes)) - min(abs(spike_amplitudes)) + 1e-10
                )
                color = cm.viridis(norm_amp)
            elif color_mode == "time":
                color = time_colors[i]
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
                    alpha=0.9,
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

    # Update title with sort information
    if sort_spikes is not None:
        title += f" (Sorted by {sort_spikes}, {sort_order})"
    plt.suptitle(title)

    plt.tight_layout()

    return fig


def plot_spike_raster(
    spike_df: pl.DataFrame,
    channels: list = None,
    figsize: tuple = (15, 6),
    title: str = "Spike Raster Plot",
    color_mode: str = "channel",
    time_window: tuple = None,
    cluster_colors: dict = None,
    cluster_column: str = None,
    sort_channels: bool = False,  # New parameter to sort channels by activity
    max_spikes: int = None,  # New parameter to limit total spikes
    sort_spikes: str = None,  # New parameter to sort spikes
    sort_order: str = "ascending",  # New parameter for sort order
) -> Figure:
    """
    Create a raster plot of spikes across channels with optional time windowing and sorting.

    Parameters:
    -----------
    spike_df : pl.DataFrame
        Polars DataFrame containing spike information with columns:
        - 'index': spike sample index
        - 'time_sec': spike time in seconds
        - 'channel': channel number
        - optionally 'cluster' or other cluster column name
    channels : list, optional
        List of channel indices to plot. If None, all channels found in spike_df will be used.
    figsize : tuple, default (15, 6)
        Figure size (width, height)
    title : str, default "Spike Raster Plot"
        Title for the plot
    color_mode : str, default "channel"
        Color scheme for spikes. Options:
        - "channel": different color for each channel
        - "time": color based on spike time
        - "cluster": color based on cluster assignment
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
        If None, plot entire dataset.
    cluster_colors : dict, optional
        Dictionary mapping cluster IDs to colors
    cluster_column : str, optional
        Name of the column containing cluster assignments (default: 'cluster')
    sort_channels : bool, default False
        Whether to sort channels by activity level (most active first)
    max_spikes : int, optional
        Maximum number of total spikes to plot. If None, plot all spikes.
    sort_spikes: str, optional
        How to sort spikes for display:
        - "time": Sort by spike time
        - "amplitude": Sort by spike amplitude (if available)
        - None: No specific sorting (default)
    sort_order : str, default "ascending"
        Order for sorting spikes:
        - "ascending": Smallest/earliest first
        - "descending": Largest/latest first

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Filter spikes based on time window if specified
    if time_window is not None:
        start_time, end_time = time_window
        spike_df = spike_df.filter(
            (pl.col("time_sec") >= start_time) & (pl.col("time_sec") <= end_time)
        )

    # Sort spikes if requested
    if sort_spikes is not None:
        if sort_spikes == "time":
            spike_df = spike_df.sort(
                "time_sec", descending=(sort_order == "descending")
            )
        elif sort_spikes == "amplitude" and "amplitude" in spike_df.columns:
            if sort_order == "ascending":
                spike_df = spike_df.sort(pl.abs("amplitude"))
            else:
                spike_df = spike_df.sort(pl.abs("amplitude"), descending=True)

    # Limit total number of spikes if requested
    if max_spikes is not None and len(spike_df) > max_spikes:
        spike_df = spike_df.head(max_spikes)

    # Determine channels to display
    if channels is None:
        if "channel" in spike_df.columns:
            # Default to all channels present in the data
            channels = list(range(spike_df["channel"].max() + 1))
        else:
            channels = [0]
    num_channels = len(channels)
    # Create a channel mapping if sorting channels by activity
    if sort_channels:
        # Count spikes per channel
        if "channel" in spike_df.columns:
            # Filter to only count spikes from requested channels
            filtered_df = spike_df.filter(pl.col("channel").is_in(channels))
            channel_counts = filtered_df.group_by("channel").count()

            # Sort channels by count (descending)
            sorted_channels = channel_counts.sort("count", descending=True)[
                "channel"
            ].to_numpy()

            # Create a mapping from original channel to y-position
            channel_to_y = {ch: i for i, ch in enumerate(sorted_channels)}

            # Ensure all channels are in the mapping (even if they have no spikes)
            for i, ch in enumerate(channels):
                if ch not in channel_to_y:
                    channel_to_y[ch] = len(channel_to_y)
        else:
            # If no channel column, no sorting needed
            channel_to_y = {0: 0}
    else:
        # No sorting - direct mapping with consecutive y-positions
        channel_to_y = {ch: i for i, ch in enumerate(channels)}

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color options
    import matplotlib.cm as cm

    # Check if clustering is enabled
    clustering_enabled = color_mode == "cluster" and (
        cluster_column is not None or "cluster" in spike_df.columns
    )

    if clustering_enabled:
        # Use provided column name or default to 'cluster'
        cluster_col = cluster_column if cluster_column is not None else "cluster"

        # Ensure the column exists
        if cluster_col not in spike_df.columns:
            print(
                f"Warning: Cluster column '{cluster_col}' not found. Using channel coloring instead."
            )
            clustering_enabled = False
            color_mode = "channel"

    # Plot spikes
    if "channel" in spike_df.columns:
        for channel in channels:
            # Map to y-position based on sorting
            y_pos = channel_to_y.get(channel, channel)

            # Filter spikes for this channel
            channel_spikes = spike_df.filter(pl.col("channel") == channel)

            # Skip if no spikes
            if len(channel_spikes) == 0:
                continue

            # Get spike times
            times = channel_spikes["time_sec"].to_numpy()

            # Color selection
            if color_mode == "channel":
                # Assign colors by channel
                color = cm.Set1(channel / num_channels)
                ax.scatter(
                    times, [y_pos] * len(times), color=color, marker="|", alpha=0.7
                )

            elif color_mode == "time":
                # Color by time
                norm_time = (times - times.min()) / (times.max() - times.min() + 1e-10)
                colors = [cm.viridis(t) for t in norm_time]
                ax.scatter(
                    times, [y_pos] * len(times), color=colors, marker="|", alpha=0.7
                )

            elif clustering_enabled:
                # Color by cluster
                clusters = channel_spikes[cluster_col].to_numpy()
                unique_clusters = np.unique(clusters)

                # Use provided cluster colors or generate new ones
                if cluster_colors is None:
                    cluster_colors = {
                        cluster: cm.tab10(i % 10)
                        for i, cluster in enumerate(unique_clusters)
                    }

                # Plot each cluster separately
                for cluster in unique_clusters:
                    mask = clusters == cluster
                    cluster_times = times[mask]

                    # Get color for this cluster
                    color = cluster_colors.get(cluster, "black")

                    # Plot spikes for this cluster
                    ax.scatter(
                        cluster_times,
                        [y_pos] * len(cluster_times),
                        color=color,
                        marker="|",
                        alpha=0.7,
                    )
            else:
                # Default coloring (plain black)
                ax.scatter(
                    times, [y_pos] * len(times), color="black", marker="|", alpha=0.7
                )
    else:
        # If no channel column, plot all spikes on one line
        times = spike_df["time_sec"].to_numpy()
        y_pos = 0
        if color_mode == "time" and len(times) > 0:
            # Color by time
            norm_time = (times - times.min()) / (times.max() - times.min() + 1e-10)
            colors = [cm.viridis(t) for t in norm_time]
            ax.scatter(times, [y_pos] * len(times), color=colors, marker="|", alpha=0.7)
        else:
            # Default coloring (plain black)
            ax.scatter(
                times, [y_pos] * len(times), color="black", marker="|", alpha=0.7
            )

    # Finalize plot
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")

    # Update title with sort information
    if sort_spikes is not None:
        title += f" (Sorted by {sort_spikes}, {sort_order})"
    if sort_channels:
        title += " (Channels sorted by activity)"
    ax.set_title(title)

    # Set y-ticks based on channel mapping
    if sort_channels:
        # Create sorted y-ticks
        yticks = list(range(len(channel_to_y)))
        yticklabels = [
            f"Channel {channel}"
            for channel, _ in sorted(channel_to_y.items(), key=lambda x: x[1])
        ]
    else:
        yticks = list(range(len(channels)))
        yticklabels = [f"Channel {ch}" for ch in channels]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    # ax.invert_yaxis()
    # Set x-axis limits if time window is specified
    if time_window is not None:
        ax.set_xlim(time_window)

    # Add legend if clustering is enabled
    if clustering_enabled and cluster_colors:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, edgecolor="k", label=f"Cluster {cluster}")
            for cluster, color in cluster_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="best")

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
    figsize: tuple = None,
    cluster_column: str = None,
    sort_spikes: str = None,  # New parameter
    sort_order: str = "ascending",  # New parameter
    sort_channels: bool = False,  # New parameter
    show_background: bool = True,  # New parameter
) -> tuple:
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
    figsize : tuple, optional
        Figure size as (width, height) for each plot. If None, uses defaults.
    cluster_column : str, optional
        Name of column containing cluster assignments for colored raster plot
    sort_spikes : str, optional
        How to sort spikes. Options:
        - "time": Sort by spike time
        - "amplitude": Sort by spike amplitude (if available)
        - None: No specific sorting (default)
    sort_order : str, default "ascending"
        Order for sorting: "ascending" or "descending"
    sort_channels : bool, default False
        Whether to sort channels by activity level
    show_background : bool, default True
        Whether to show background signal in the waveform plot

    Returns:
    --------
    tuple
        (waveform_figure, raster_figure)
    """
    # Default to all channels if not specified
    if channels is None:
        if data.ndim > 1:
            channels = list(range(data.shape[1]))
        else:
            channels = [0]

    # Determine if we have cluster information
    color_mode = "cluster" if cluster_column in spike_df.columns else "channel"

    # Plot spike waveforms
    waveform_fig = plot_spikes(
        data,
        spike_df,
        channels=channels,
        fs=fs,
        time_window=time_window,
        window_ms=window_ms,
        max_spikes_per_channel=max_spikes_per_channel,
        figsize=figsize,
        sort_spikes=sort_spikes,
        sort_order=sort_order,
        show_background=show_background,
        color_mode=color_mode,
    )

    # Plot raster
    raster_figsize = figsize if figsize else (15, 6)
    raster_fig = plot_spike_raster(
        spike_df,
        channels=channels,
        time_window=time_window,
        figsize=raster_figsize,
        color_mode=color_mode,
        cluster_column=cluster_column,
        sort_spikes=sort_spikes,
        sort_order=sort_order,
        sort_channels=sort_channels,
    )

    return waveform_fig, raster_fig
