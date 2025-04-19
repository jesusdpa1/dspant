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
    colormap: str = "Set1",  # New parameter: Colormap name to use
    color: str = "black",  # New parameter: Single color for all spikes
    use_single_color: bool = False,  # New parameter: Whether to use a single color
    max_spikes_per_channel: int = None,
    time_window: tuple = None,
    sort_spikes: str = None,  # "time" or "amplitude"
    sort_order: str = "ascending",  # "ascending" or "descending"
    show_background: bool = True,  # Control background signal display
    background_color: str = "lightgray",  # New parameter: Color for background signal
    background_alpha: float = 0.8,  # New parameter: Transparency for background
    sort_channels: bool = False,  # Sort channels by activity
    line_width: float = 0.5,  # New parameter: Width of spike waveform lines
    spike_alpha: float = 0.9,  # New parameter: Transparency of spike waveforms
    highlight_peaks: bool = True,  # New parameter: Whether to mark spike peaks
    peak_marker: str = "x",  # New parameter: Marker for peaks
    peak_size: float = 50.0,  # New parameter: Size of peak markers
    peak_color: str = "red",  # New parameter: Color of peak markers
    use_dark_grid: bool = False,  # New parameter: Whether to use seaborn's darkgrid
) -> Figure:
    """
    Plot detected spikes across specified channels with optional time windowing, sorting,
    and channel ordering by activity.

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
    colormap : str, default "Set1"
        Matplotlib colormap name to use for coloring
    color : str, default "black"
        Single color to use for all spikes when use_single_color is True
    use_single_color : bool, default False
        Whether to use a single color for all spikes
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
    background_color : str, default "lightgray"
        Color for the background signal
    background_alpha : float, default 0.8
        Transparency level for background signal (0.0 = transparent, 1.0 = opaque)
    sort_channels : bool, default False
        Whether to sort channels by activity level (most active first)
    line_width : float, default 0.5
        Width of spike waveform lines
    spike_alpha : float, default 0.9
        Transparency level for spike waveforms
    highlight_peaks : bool, default True
        Whether to mark spike peaks with a marker
    peak_marker : str, default "x"
        Marker type for spike peaks
    peak_size : float, default 50.0
        Size of peak markers
    peak_color : str, default "red"
        Color of peak markers
    use_dark_grid : bool, default False
        Whether to use seaborn's darkgrid style (better grid visibility)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Apply dark grid style if requested
    if use_dark_grid:
        import seaborn as sns

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

    # Compute window samples
    window_samples = int(window_ms / 1000 * fs)
    half_window = window_samples // 2

    # Filter spikes based on time window
    if time_window is not None:
        start_time, end_time = time_window
        spike_df = spike_df.filter(
            (pl.col("time_sec") >= start_time) & (pl.col("time_sec") <= end_time)
        )

    # Sort channels by activity if requested
    if sort_channels:
        # Count spikes per channel
        if "channel" in spike_df.columns:
            # Filter to only count spikes from requested channels
            filtered_df = spike_df.filter(pl.col("channel").is_in(channels))
            channel_counts = filtered_df.group_by("channel").count()

            # Sort channels by count (descending)
            channel_activity = channel_counts.sort("count", descending=True)
            sorted_channels = channel_activity["channel"].to_list()

            # Ensure all requested channels are included (even if they have no spikes)
            for ch in channels:
                if ch not in sorted_channels:
                    sorted_channels.append(ch)

            # Update the channels list to use the sorted order
            channels = sorted_channels

    # Prepare figure
    if figsize is None:
        figsize = (15, 3 * len(channels))

    fig, axs = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
    if len(channels) == 1:
        axs = [axs]  # Ensure list for single channel

    # Color options - use the specified colormap
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)

    # If using single color, override color_mode
    if use_single_color:
        color_mode = "single"

    # Compute full time array
    time = np.arange(data.shape[0]) / fs

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
                color=background_color,
                alpha=background_alpha,
                linewidth=line_width,
                zorder=1,
            )

        # Add number of spikes to ylabel if sorting by activity
        if sort_channels:
            spike_count = len(channel_spikes)
            axs[idx].set_ylabel(f"Channel {channel}\n({spike_count} spikes)")
        else:
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
            time_min = spike_times.min()
            time_max = spike_times.max()
            time_range = time_max - time_min + 1e-10

        # Plot each spike
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

            # Color selection based on mode
            if color_mode == "single":
                plot_color = color
            elif color_mode == "channel":
                plot_color = cmap(idx / len(channels))
            elif color_mode == "amplitude":
                # Normalize amplitude for color mapping
                amp_min = min(abs(spike_amplitudes))
                amp_max = max(abs(spike_amplitudes))
                amp_range = amp_max - amp_min + 1e-10
                norm_amp = (abs(spike_amp) - amp_min) / amp_range
                plot_color = cmap(norm_amp)
            elif color_mode == "time":
                # Normalize time for color mapping
                norm_time = (spike_time - time_min) / time_range
                plot_color = cmap(norm_time)
            else:
                plot_color = "red"  # Default fallback

            # Normalize spike time to its actual time point
            if len(spike_segment_time) > 0:
                spike_segment_time = (
                    spike_segment_time - spike_segment_time[0] + spike_time
                )

                # Plot individual spike
                axs[idx].plot(
                    spike_segment_time,
                    spike_segment,
                    color=plot_color,
                    linewidth=line_width,
                    alpha=spike_alpha,
                    zorder=2,
                )

                # Mark spike peak if requested
                if highlight_peaks:
                    axs[idx].scatter(
                        spike_time,
                        data[spike_idx, channel],
                        color=peak_color,
                        marker=peak_marker,
                        s=peak_size,
                        zorder=3,
                    )

    # Add grid with higher visibility if not using seaborn's darkgrid
    if not use_dark_grid:
        for ax in axs:
            ax.grid(True, linestyle="-", alpha=0.4, color="gray")

    # Finalize plot
    plt.xlabel("Time (s)")

    # Update title with sort information
    modified_title = title
    if sort_spikes is not None:
        modified_title += f" (Sorted by {sort_spikes}, {sort_order})"
    if sort_channels:
        modified_title += " (Channels sorted by activity)"
    plt.suptitle(modified_title)

    plt.tight_layout()

    return fig


def plot_spike_raster(
    spike_df: pl.DataFrame,
    channels: list = None,
    figsize: tuple = (15, 6),
    title: str = "Spike Raster Plot",
    color_mode: str = "channel",
    colormap: str = "Set1",  # Colormap name when using multiple colors
    color: str = "black",  # New parameter: Single color for all spikes
    use_single_color: bool = False,  # New parameter: Whether to use a single color
    time_window: tuple = None,
    cluster_colors: dict = None,
    cluster_column: str = None,
    sort_channels: bool = False,  # Sort channels by activity
    max_spikes: int = None,  # Limit total spikes
    sort_spikes: str = None,  # Sort spikes
    sort_order: str = "ascending",  # Sort order
    marker_size: float = 20.0,  # Marker size
    marker_width: float = 3.0,  # Marker width
    marker_type: str = "|",  # Marker type
    use_dark_grid: bool = False,  # Whether to use seaborn's darkgrid style
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
    colormap : str, default "Set1"
        Matplotlib colormap name to use for coloring
    color : str, default "black"
        Single color to use for all spikes when use_single_color is True
    use_single_color : bool, default False
        Whether to use a single color for all markers instead of colormap
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
    marker_size : float, default 20.0
        Size of the marker for each spike
    marker_width : float, default 3.0
        Width of the marker for each spike
    marker_type : str, default "|"
        Type of marker to use for spikes
    use_dark_grid : bool, default False
        Whether to use seaborn's darkgrid style (better grid visibility)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Apply dark grid style if requested
    if use_dark_grid:
        import seaborn as sns

        sns.set_theme(style="darkgrid")

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

    # Color options setup
    import matplotlib.cm as cm

    cmap = cm.get_cmap(colormap)

    # If using single color, override color_mode
    if use_single_color:
        # Force simple single color mode
        clustering_enabled = False
        color_mode = "single"
    else:
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

            # Color selection based on mode
            if color_mode == "single":
                # Use the single color specified
                ax.scatter(
                    times,
                    [y_pos] * len(times),
                    color=color,
                    marker=marker_type,
                    alpha=0.8,
                    s=marker_size,
                    linewidth=marker_width,
                )
            elif color_mode == "channel":
                # Assign colors by channel using the specified colormap
                channel_color = cmap(channel / max(num_channels, 1))
                ax.scatter(
                    times,
                    [y_pos] * len(times),
                    color=channel_color,
                    marker=marker_type,
                    alpha=0.8,
                    s=marker_size,
                    linewidth=marker_width,
                )
            elif color_mode == "time":
                # Color by time
                norm_time = (times - times.min()) / (times.max() - times.min() + 1e-10)
                # Use the specified colormap for time-based coloring
                colors = [cmap(t) for t in norm_time]
                ax.scatter(
                    times,
                    [y_pos] * len(times),
                    color=colors,
                    marker=marker_type,
                    alpha=0.8,
                    s=marker_size,
                    linewidth=marker_width,
                )
            elif clustering_enabled:
                # Color by cluster
                clusters = channel_spikes[cluster_col].to_numpy()
                unique_clusters = np.unique(clusters)

                # Use provided cluster colors or generate new ones using the specified colormap
                if cluster_colors is None:
                    cluster_colors = {
                        cluster: cmap(i / max(10, len(unique_clusters)))
                        for i, cluster in enumerate(unique_clusters)
                    }

                # Plot each cluster separately
                for cluster in unique_clusters:
                    mask = clusters == cluster
                    cluster_times = times[mask]

                    # Get color for this cluster
                    cluster_color = cluster_colors.get(cluster, "black")

                    # Plot spikes for this cluster
                    ax.scatter(
                        cluster_times,
                        [y_pos] * len(cluster_times),
                        color=cluster_color,
                        marker=marker_type,
                        alpha=0.8,
                        s=marker_size,
                        linewidth=marker_width,
                    )
    else:
        # If no channel column, plot all spikes on one line
        times = spike_df["time_sec"].to_numpy()
        y_pos = 0

        if color_mode == "single":
            # Use the single color specified
            ax.scatter(
                times,
                [y_pos] * len(times),
                color=color,
                marker=marker_type,
                alpha=0.8,
                s=marker_size,
                linewidth=marker_width,
            )
        elif color_mode == "time" and len(times) > 0:
            # Color by time using the specified colormap
            norm_time = (times - times.min()) / (times.max() - times.min() + 1e-10)
            colors = [cmap(t) for t in norm_time]
            ax.scatter(
                times,
                [y_pos] * len(times),
                color=colors,
                marker=marker_type,
                alpha=0.8,
                s=marker_size,
                linewidth=marker_width,
            )
        else:
            # Default coloring (plain black)
            ax.scatter(
                times,
                [y_pos] * len(times),
                color="black",
                marker=marker_type,
                alpha=0.8,
                s=marker_size,
                linewidth=marker_width,
            )

    # Add grid with higher visibility if not using seaborn's darkgrid
    if not use_dark_grid:
        ax.grid(True, linestyle="-", alpha=0.4, color="gray")

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
    sort_spikes: str = None,
    sort_order: str = "ascending",
    sort_channels: bool = False,
    show_background: bool = True,
    marker_size: float = 10.0,  # New parameter for marker size
    marker_width: float = 2.0,  # New parameter for marker width
    marker_type: str = "|",  # New parameter for marker type
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
    marker_size : float, default 10.0
        Size of the marker for each spike in the raster plot
    marker_width : float, default 2.0
        Width of the marker for each spike
    marker_type : str, default "|"
        Type of marker to use for spikes

    Returns:
    --------
    tuple
        (waveform_figure, raster_figure)
    """
    # Default to all channels if not specified
    sns.set_theme(style="darkgrid")
    if channels is None:
        if data.ndim > 1:
            channels = list(range(data.shape[1]))
        else:
            channels = [0]

    # Determine if we have cluster information for color mode
    color_mode = "cluster" if cluster_column in spike_df.columns else "channel"

    # If sorting channels by activity, we need to determine the channel order once
    # so that both plots use the same order
    sorted_channels = channels
    if sort_channels and "channel" in spike_df.columns:
        # Filter to only count spikes from requested channels
        filtered_df = spike_df.filter(pl.col("channel").is_in(channels))

        # Apply time window filter if specified
        if time_window is not None:
            start_time, end_time = time_window
            filtered_df = filtered_df.filter(
                (pl.col("time_sec") >= start_time) & (pl.col("time_sec") <= end_time)
            )

        # Count spikes per channel
        channel_counts = filtered_df.group_by("channel").count()

        # Sort channels by count (descending)
        channel_activity = channel_counts.sort("count", descending=True)
        sorted_channels = channel_activity["channel"].to_list()

        # Ensure all requested channels are included (even if they have no spikes)
        for ch in channels:
            if ch not in sorted_channels:
                sorted_channels.append(ch)

    # Plot spike waveforms
    waveform_fig = plot_spikes(
        data,
        spike_df,
        channels=sorted_channels,  # Use the potentially sorted channel list
        fs=fs,
        time_window=time_window,
        window_ms=window_ms,
        max_spikes_per_channel=max_spikes_per_channel,
        figsize=figsize,
        sort_spikes=sort_spikes,
        sort_order=sort_order,
        show_background=show_background,
        color_mode=color_mode,
        sort_channels=sort_channels,  # Pass the sort_channels parameter
    )

    # Plot raster
    raster_figsize = figsize if figsize else (15, 6)
    raster_fig = plot_spike_raster(
        spike_df,
        channels=sorted_channels,  # Use the same sorted channel list
        time_window=time_window,
        figsize=raster_figsize,
        color_mode=color_mode,
        cluster_column=cluster_column,
        sort_spikes=sort_spikes,
        sort_order=sort_order,
        sort_channels=sort_channels,
        marker_size=marker_size,  # Add the marker size parameter
        marker_width=marker_width,  # Add the marker width parameter
        marker_type=marker_type,  # Add the marker type parameter
    )

    return waveform_fig, raster_fig


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
