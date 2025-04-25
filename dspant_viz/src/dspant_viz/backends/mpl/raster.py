from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_raster(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render a spike raster plot using matplotlib.

    Parameters
    ----------
    data : Dict
        Data dictionary from RasterPlot.get_data()
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
    spike_data = data["data"]
    spike_times = spike_data["spike_times"]
    y_values = spike_data["y_values"]
    unit_id = spike_data.get("unit_id")

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    marker_size = params.get("marker_size", 4)
    marker_color = params.get("marker_color", "#2D3142")
    marker_alpha = params.get("marker_alpha", 0.7)
    marker_type = params.get("marker_type", "|")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Check if we have data to plot
    if not spike_times:
        ax.set_ylabel("Trial")
        if unit_id is not None:
            ax.set_title(f"Unit {unit_id} - No spikes")
        else:
            ax.set_title("No spike data available")

        if params.get("show_event_onset", True):
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

        if params.get("show_grid", True):
            ax.grid(True, alpha=0.3)

        return fig, ax

    # Plot spike times
    ax.scatter(
        spike_times,
        y_values,
        marker=marker_type,
        s=marker_size,
        color=marker_color,
        alpha=marker_alpha,
        linewidths=marker_size / 4 if marker_type != "|" else 1,
    )

    # Set labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")

    # Set title if unit_id is available
    if unit_id is not None:
        trial_count = len(set(y_values))
        ax.set_title(f"Unit {unit_id} - {trial_count} trials")

    # Add grid if requested
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    # Add event onset line
    if params.get("show_event_onset", True):
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    # Set axis limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])

    if "ylim" in params:
        ax.set_ylim(params["ylim"])
    else:
        # Auto-adjust y limits to show all trials with a small margin
        if y_values:
            ax.set_ylim(-0.5, max(y_values) + 0.5)

    # Customize ticks if requested
    if params.get("show_trial_labels", False):
        label_map = spike_data.get("label_map", {})
        if label_map:
            # Get unique y values and their labels
            unique_y = sorted(set(y_values))
            ax.set_yticks(unique_y)
            ax.set_yticklabels([label_map.get(y, str(y)) for y in unique_y])

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax
