"""
General visualization functions for neural data.

This module provides visualization tools for neural signals, spike data,
and other common neural data visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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
) -> Figure:
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
    cluster_colors: dict = None,
    cluster_column: str = None,
) -> Figure:
    """
    Create a raster plot of spikes across channels with optional time windowing.

    Parameters:
    -----------
    spike_df : pl.DataFrame
        Polars DataFrame containing spike information with columns:
        - 'index': spike sample index
        - 'time_sec': spike time in seconds
        - 'channel': channel number
        - optionally 'cluster' or other cluster column name
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
        - "cluster": color based on cluster assignment
    time_window : tuple, optional
        Time window to plot (start_time, end_time) in seconds.
        If None, plot entire dataset.
    cluster_colors : dict, optional
        Dictionary mapping cluster IDs to colors
    cluster_column : str, optional
        Name of the column containing cluster assignments (default: 'cluster')

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

    # Filter spikes
    for channel in range(num_channels):
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
                times, [channel] * len(times), color=color, marker="|", alpha=0.7
            )

        elif color_mode == "time":
            # Color by time
            norm_time = (times - times.min()) / (times.max() - times.min() + 1e-10)
            colors = [cm.viridis(t) for t in norm_time]
            ax.scatter(
                times, [channel] * len(times), color=colors, marker="|", alpha=0.7
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
                    [channel] * len(cluster_times),
                    color=color,
                    marker="|",
                    alpha=0.7,
                )
        else:
            # Default coloring (plain black)
            ax.scatter(
                times, [channel] * len(times), color="black", marker="|", alpha=0.7
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
    )

    # Plot raster
    raster_figsize = figsize if figsize else (15, 6)
    raster_fig = plot_spike_raster(
        spike_df,
        num_channels=max(channels) + 1,
        time_window=time_window,
        figsize=raster_figsize,
        color_mode=color_mode,
        cluster_column=cluster_column,
    )

    return waveform_fig, raster_fig


def plot_waveform_clusters(
    waveforms: np.ndarray,
    cluster_labels: np.ndarray,
    fs: float = None,
    max_waveforms_per_cluster: int = 100,
    figsize: tuple = (14, 10),
    title: str = "Waveform Clusters",
    plot_mean: bool = True,
    plot_individual: bool = True,
    alpha_individual: float = 0.2,
):
    """
    Plot clustered waveforms with mean waveform for each cluster.

    Parameters:
    -----------
    waveforms : numpy array
        Waveform data (n_waveforms × n_samples × n_channels)
    cluster_labels : numpy array
        Cluster assignments for each waveform
    fs : float, optional
        Sampling frequency in Hz. If provided, x-axis will be in milliseconds
    max_waveforms_per_cluster : int, default 100
        Maximum number of individual waveforms to plot per cluster
    figsize : tuple, default (14, 10)
        Figure size as (width, height)
    title : str, default "Waveform Clusters"
        Title for the plot
    plot_mean : bool, default True
        Whether to plot mean waveform for each cluster
    plot_individual : bool, default True
        Whether to plot individual waveforms
    alpha_individual : float, default 0.2
        Alpha value for individual waveforms

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Handle 3D waveforms (simplify by taking first channel)
    if waveforms.ndim == 3:
        # For multi-channel data, use the first channel
        waveforms_2d = waveforms[:, :, 0]
    else:
        waveforms_2d = waveforms

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Determine subplot layout
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols  # Ceiling division

    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    # Handle single subplot case
    if n_clusters == 1:
        axs = np.array([axs])

    # Flatten axes for easy iteration
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]

    # Create x-axis values
    if fs is not None:
        # Convert to milliseconds
        x_values = np.arange(waveforms_2d.shape[1]) * 1000 / fs
        x_label = "Time (ms)"
    else:
        x_values = np.arange(waveforms_2d.shape[1])
        x_label = "Samples"

    # Get colormap for clusters
    import matplotlib.cm as cm

    colors = [cm.tab10(i % 10) for i in range(n_clusters)]

    # Plot clusters
    for i, cluster in enumerate(unique_clusters):
        # Get waveforms for this cluster
        cluster_mask = cluster_labels == cluster
        cluster_waveforms = waveforms_2d[cluster_mask]

        # Sample waveforms if there are too many
        if len(cluster_waveforms) > max_waveforms_per_cluster and plot_individual:
            sample_indices = np.random.choice(
                len(cluster_waveforms), max_waveforms_per_cluster, replace=False
            )
            plot_waveforms = cluster_waveforms[sample_indices]
        else:
            plot_waveforms = cluster_waveforms

        # Plot individual waveforms
        if plot_individual:
            for wf in plot_waveforms:
                axs[i].plot(
                    x_values, wf, color=colors[i], alpha=alpha_individual, linewidth=0.5
                )

        # Plot mean waveform
        if plot_mean and len(cluster_waveforms) > 0:
            mean_wf = np.mean(cluster_waveforms, axis=0)
            std_wf = np.std(cluster_waveforms, axis=0)

            # Plot mean with std shading
            axs[i].plot(x_values, mean_wf, color=colors[i], linewidth=2, label=f"Mean")
            axs[i].fill_between(
                x_values,
                mean_wf - std_wf,
                mean_wf + std_wf,
                color=colors[i],
                alpha=0.3,
                label="±1 std",
            )

        # Set title and labels
        axs[i].set_title(f"Cluster {cluster} (n={np.sum(cluster_mask)})")
        axs[i].grid(True, linestyle="--", alpha=0.7)

        # Set x and y labels on edge subplots
        if i % n_cols == 0:  # Left edge
            axs[i].set_ylabel("Amplitude")
        if i >= n_clusters - n_cols:  # Bottom edge
            axs[i].set_xlabel(x_label)

    # Hide unused subplots
    for i in range(n_clusters, len(axs)):
        axs[i].set_visible(False)

    # Set common title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    return fig


def plot_firing_rates(
    spike_times: np.ndarray,
    cluster_labels: np.ndarray,
    fs: float = None,
    bin_width_sec: float = 1.0,
    smoothing_window: int = 5,
    figsize: tuple = (14, 6),
    title: str = "Firing Rates by Cluster",
    normalize: bool = False,
):
    """
    Plot firing rates over time for different clusters.

    Parameters:
    -----------
    spike_times : numpy array
        Times of spikes in seconds
    cluster_labels : numpy array
        Cluster assignments for each spike
    fs : float, optional
        Sampling frequency in Hz (used if spike_times are in samples)
    bin_width_sec : float, default 1.0
        Width of time bins in seconds
    smoothing_window : int, default 5
        Size of moving average window for smoothing
    figsize : tuple, default (14, 6)
        Figure size as (width, height)
    title : str, default "Firing Rates by Cluster"
        Title for the plot
    normalize : bool, default False
        Whether to normalize each cluster's firing rate to [0, 1]

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Convert spike times to seconds if needed
    if (
        fs is not None and spike_times.max() > 1000
    ):  # Assume values in samples if very large
        spike_times_sec = spike_times / fs
    else:
        spike_times_sec = spike_times

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create time bins
    min_time = 0
    max_time = spike_times_sec.max() * 1.02  # Add 2% margin
    time_bins = np.arange(min_time, max_time, bin_width_sec)

    # Get colormap for clusters
    import matplotlib.cm as cm

    colors = [cm.tab10(i % 10) for i in range(len(unique_clusters))]

    # Apply simple moving average for smoothing
    def smooth(y, window_size):
        if window_size <= 1:
            return y
        box = np.ones(window_size) / window_size
        return np.convolve(y, box, mode="same")

    # Store rates for legend sorting
    cluster_rates = []

    # Plot firing rate for each cluster
    for i, cluster in enumerate(unique_clusters):
        # Get spike times for this cluster
        cluster_mask = cluster_labels == cluster
        cluster_spike_times = spike_times_sec[cluster_mask]

        # Skip if no spikes
        if len(cluster_spike_times) == 0:
            cluster_rates.append((cluster, 0))
            continue

        # Calculate histogram
        counts, bin_edges = np.histogram(cluster_spike_times, bins=time_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Convert to rates (Hz)
        rates = counts / bin_width_sec

        # Apply smoothing
        if smoothing_window > 1:
            rates = smooth(rates, smoothing_window)

        # Store average rate
        avg_rate = np.mean(rates)
        cluster_rates.append((cluster, avg_rate))

        # Normalize if requested
        if normalize and np.max(rates) > 0:
            rates = rates / np.max(rates)

        # Plot rates
        ax.plot(
            bin_centers, rates, color=colors[i], linewidth=2, label=f"Cluster {cluster}"
        )

    # Sort legend by average firing rate (highest first)
    cluster_rates.sort(key=lambda x: x[1], reverse=True)
    handles, labels = ax.get_legend_handles_labels()
    sorted_idx = [
        next(i for i, l in enumerate(labels) if f"Cluster {cluster}" in l)
        for cluster, _ in cluster_rates
    ]

    # Set labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)" if not normalize else "Normalized Firing Rate")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(
        [handles[i] for i in sorted_idx], [labels[i] for i in sorted_idx], loc="best"
    )

    plt.tight_layout()
    return fig


def plot_spike_scatter_3d(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    figsize: tuple = (10, 8),
    title: str = "3D Spike Clusters",
    alpha: float = 0.7,
    s: int = 20,
):
    """
    Create a 3D scatter plot of spike embeddings colored by cluster.

    Parameters:
    -----------
    embeddings : numpy array
        3D embeddings of spikes (N × 3)
    cluster_labels : numpy array
        Cluster assignments for each spike
    figsize : tuple, default (10, 8)
        Figure size as (width, height)
    title : str, default "3D Spike Clusters"
        Title for the plot
    alpha : float, default 0.7
        Alpha value for scatter points
    s : int, default 20
        Size of scatter points

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    # Check dimensions
    if embeddings.shape[1] < 3:
        raise ValueError("Embeddings must have at least 3 dimensions for 3D plotting")

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)

    # Get colormap for clusters
    import matplotlib.cm as cm

    colors = [cm.tab10(i % 10) for i in range(len(unique_clusters))]

    # Plot each cluster
    for i, cluster in enumerate(unique_clusters):
        # Get points for this cluster
        mask = cluster_labels == cluster
        cluster_points = embeddings[mask]

        # Skip if no points
        if len(cluster_points) == 0:
            continue

        # Plot 3D scatter
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            color=colors[i],
            label=f"Cluster {cluster}",
            alpha=alpha,
            s=s,
        )

    # Set labels and title
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(title)
    ax.legend(loc="best")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.4)

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig
