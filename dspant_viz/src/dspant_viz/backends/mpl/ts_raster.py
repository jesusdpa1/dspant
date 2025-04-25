# src/dspant_viz/backends/mpl/ts_raster.py
from typing import Any, Dict, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_ts_raster(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render a time series raster plot using matplotlib.

    Parameters
    ----------
    data : Dict
        Data dictionary from TimeSeriesRasterPlot.get_data()
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
    unit_ids = plot_data["unit_ids"]
    unit_spikes = plot_data["unit_spikes"]
    unit_labels = plot_data["unit_labels"]
    y_positions = plot_data["y_positions"]
    y_ticks = plot_data["y_ticks"]
    y_tick_labels = plot_data["y_tick_labels"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Get plotting parameters
    marker_size = params.get("marker_size", 4)
    marker_type = params.get("marker_type", "|")
    use_colormap = params.get("use_colormap", True)
    colormap_name = params.get("colormap", "viridis")
    alpha = params.get("alpha", 0.7)
    show_legend = params.get("show_legend", True)
    show_labels = params.get("show_labels", True)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Set up colormap if using it
    if use_colormap:
        cmap = cm.get_cmap(colormap_name)
        colors = [cmap(i / max(1, len(unit_ids) - 1)) for i in range(len(unit_ids))]
    else:
        # Use default color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [colors[i % len(colors)] for i in range(len(unit_ids))]

    # Plot each unit
    for i, unit_id in enumerate(unit_ids):
        # Get spikes for this unit
        spikes = unit_spikes.get(unit_id, [])
        if not spikes:
            continue

        # Get y position for this unit
        y_pos = y_positions[unit_id]

        # Get label for this unit
        label = unit_labels.get(unit_id, f"Unit {unit_id}")

        # Plot spikes for this unit
        ax.scatter(
            spikes,
            np.ones_like(spikes) * y_pos,
            marker=marker_type,
            s=marker_size,
            color=colors[i],
            alpha=alpha,
            linewidths=marker_size / 4 if marker_type != "|" else 1,
            label=label,
            rasterized=True,  # Rasterize for better performance with many points
        )

    # Set labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")

    # Set custom y-ticks and labels
    if show_labels and unit_ids:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels)

        # Set y-limits to show all units with a small margin
        if y_ticks:
            ax.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)

    # Add grid if requested
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    # Add title if provided
    if "title" in params:
        ax.set_title(params["title"])
    else:
        ax.set_title("Time Series Raster Plot")

    # Add legend if requested
    if show_legend and len(unit_ids) <= 10:  # Only show legend if not too many units
        ax.legend(loc="upper right", frameon=True, framealpha=0.8)

    # Set axis limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])
    elif "time_window" in params and params["time_window"] is not None:
        ax.set_xlim(params["time_window"])

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout for better appearance
    plt.tight_layout()

    return fig, ax
