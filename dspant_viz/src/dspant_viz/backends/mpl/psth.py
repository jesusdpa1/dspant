from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_psth(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render PSTH (Peristimulus Time Histogram) using matplotlib.

    Parameters
    ----------
    data : Dict
        Data dictionary from PSTHPlot.get_data()
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
    time_bins = plot_data["time_bins"]
    firing_rates = plot_data["firing_rates"]
    sem = plot_data.get("sem")
    unit_id = plot_data.get("unit_id")

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    line_color = params.get("line_color", "orange")
    line_width = params.get("line_width", 2)
    show_sem = params.get("show_sem", True)
    sem_alpha = params.get("sem_alpha", 0.3)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Check if we have data to plot
    if not time_bins or not firing_rates:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing rate (Hz)")
        if unit_id is not None:
            ax.set_title(f"Unit {unit_id} PSTH - No data")
        else:
            ax.set_title("PSTH - No data")

        if params.get("show_event_onset", True):
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

        if params.get("show_grid", True):
            ax.grid(True, alpha=0.3)

        return fig, ax

    # Convert to numpy arrays for calculations
    time_bins_array = np.array(time_bins)
    firing_rates_array = np.array(firing_rates)

    # Plot firing rates
    ax.plot(time_bins_array, firing_rates_array, color=line_color, linewidth=line_width)

    # Plot SEM if available and requested
    if show_sem and sem is not None:
        sem_array = np.array(sem)

        # Filter out any NaN or inf values
        valid_mask = np.isfinite(sem_array) & np.isfinite(firing_rates_array)

        if np.any(valid_mask):
            # Use only the valid data points for the fill_between
            valid_times = time_bins_array[valid_mask]
            valid_rates = firing_rates_array[valid_mask]
            valid_sem = sem_array[valid_mask]

            # Only include SEM where we have valid data
            ax.fill_between(
                valid_times,
                valid_rates - valid_sem,
                valid_rates + valid_sem,
                color=line_color,
                alpha=sem_alpha,
                linewidth=0,  # Remove edges for smoother appearance
            )

    # Set axis labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")

    # Set title
    if unit_id is not None:
        ax.set_title(f"Unit {unit_id} PSTH")
    else:
        ax.set_title("PSTH")

    # Add grid if requested
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    # Add event onset line
    if params.get("show_event_onset", True):
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    # Add baseline window if provided
    if "baseline_window" in params:
        start, end = params["baseline_window"]
        ax.axvspan(start, end, color="gray", alpha=0.2)

        # Add text label if there's room
        if end - start > 0.05:
            y_max = max(firing_rates) * 0.9
            ax.text(
                (start + end) / 2,
                y_max,
                "baseline",
                ha="center",
                va="top",
                fontsize=8,
                alpha=0.7,
            )

    # Set axis limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])

    if "ylim" in params:
        ax.set_ylim(params["ylim"])

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax
