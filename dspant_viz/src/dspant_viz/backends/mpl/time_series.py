# src/dspant_viz/backends/mpl/time_series.py
from typing import Any, Dict, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_time_series(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render multi-channel time series data using matplotlib.

    Parameters
    ----------
    data : dict
        Data dictionary from TimeSeriesPlot.get_data()
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes with the plot
    """
    # Extract data
    plot_data = data["data"]
    time_values = plot_data["time"]
    signals = plot_data["signals"]
    channels = plot_data["channels"]
    channel_info = plot_data["channel_info"]
    channel_positions = plot_data["channel_positions"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    color_mode = params.get("color_mode", "colormap")
    colormap_name = params.get("colormap", "viridis")
    single_color = params.get("color", "black")
    line_width = params.get("line_width", 1.0)
    alpha = params.get("alpha", 0.8)
    grid = params.get("grid", True)
    show_channel_labels = params.get("show_channel_labels", True)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    else:
        fig = ax.figure

    # Set up color handling
    if color_mode == "colormap":
        cmap = cm.get_cmap(colormap_name)
        n_channels = len(channels)
        colors = [cmap(i / max(1, n_channels - 1)) for i in range(n_channels)]
    else:
        # Use a single color for all channels
        colors = [single_color] * len(channels)

    # Plot each channel
    for idx, channel_idx in enumerate(range(len(channels))):
        # Skip if no data for this channel
        if idx >= len(signals) or len(signals[idx]) == 0:
            continue

        # Plot the channel data
        ax.plot(
            time_values,
            signals[idx],
            color=colors[idx],
            alpha=alpha,
            linewidth=line_width,
            label=f"Channel {channels[idx]}",
        )

    # Set axes labels
    ax.set_xlabel("Time (s)")
    if show_channel_labels:
        ax.set_ylabel("Channels")
        ax.set_yticks(channel_positions)
        ax.set_yticklabels(channel_info)
    else:
        ax.set_yticks([])

    # Add grid if requested
    if grid:
        ax.grid(True, alpha=0.3)

    # Set title if provided
    if "title" in params:
        ax.set_title(params["title"])
    else:
        ax.set_title("Multi-Channel Time Series")

    # Set x limits if provided in time_window
    if "time_window" in params and params["time_window"] is not None:
        ax.set_xlim(params["time_window"])

    # Set y-limits to include all signals with a small margin
    if channel_positions:
        ax.set_ylim(min(channel_positions) - 0.5, max(channel_positions) + 0.5)

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout for better appearance
    plt.tight_layout()

    return fig, ax
