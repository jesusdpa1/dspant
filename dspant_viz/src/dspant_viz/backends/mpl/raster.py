# src/dspant_viz/backends/mpl/raster.py
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# needs a way to validate that the data is trail base
def render_raster(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render a trial-based spike raster plot using matplotlib.

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
    trial_indices = spike_data[
        "y_values"
    ]  # In trial-based mode, y values are trial indices
    unit_id = spike_data.get("unit_id")
    n_trials = spike_data.get("n_trials", 0)

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

        # Always show event onset line in trial-based plots
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

        if params.get("show_grid", True):
            ax.grid(True, alpha=0.3)

        return fig, ax

    # Plot spike times - use rasterized for performance with many points
    ax.scatter(
        spike_times,
        trial_indices,
        marker=marker_type,
        s=marker_size,
        color=marker_color,
        alpha=marker_alpha,
        linewidths=marker_size / 4 if marker_type != "|" else 1,
        rasterized=len(spike_times) > 1000,  # Rasterize for large datasets
    )

    # Set labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")

    # Set title if unit_id is available
    if unit_id is not None:
        ax.set_title(f"Unit {unit_id} - {n_trials} trials")
    else:
        ax.set_title("Raster Plot")

    # Add grid if requested
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    # Always show event onset line in trial-based plots
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    # Set axis limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])
    elif "time_window" in params and params["time_window"] is not None:
        ax.set_xlim(params["time_window"])

    if "ylim" in params:
        ax.set_ylim(params["ylim"])
    else:
        # Auto-adjust y limits to show all trials with a small margin
        if trial_indices:
            ax.set_ylim(-0.5, max(trial_indices) + 0.5)

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout for better appearance
    plt.tight_layout()

    return fig, ax
