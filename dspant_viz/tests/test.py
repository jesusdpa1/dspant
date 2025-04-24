# backends/mpl/renderer.py
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_raster(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render a spike raster plot using matplotlib

    Parameters
    ----------
    data : Dict
        Raster data dictionary from RasterPlot.get_data()
    ax : Axes, optional
        Matplotlib axes to render on. If None, creates a new figure.
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes with the rendered plot
    """
    # Extract data
    spike_data = data["data"]
    spike_times = spike_data["spike_times"]
    trial_indices = spike_data["trial_indices"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with any provided kwargs

    marker_size = params.get("marker_size", 4)
    marker_color = params.get("marker_color", "#2D3142")
    marker_alpha = params.get("marker_alpha", 0.7)
    marker_type = params.get("marker_type", "|")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot raster
    if len(spike_times) > 0:
        ax.scatter(
            spike_times,
            trial_indices,
            marker=marker_type,
            s=marker_size,
            color=marker_color,
            alpha=marker_alpha,
            linewidths=marker_size / 4 if marker_type != "|" else 1,
        )

    # Set labels
    ax.set_ylabel("Trial")

    # Set title if unit_id is provided
    if spike_data["unit_id"] is not None:
        ax.set_title(f"Unit {spike_data['unit_id']}")

    # Apply additional styling from params
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    # Add event onset line at t=0 if requested
    if params.get("show_event_onset", True):
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    # Set x limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])

    # Set y limits if provided
    if "ylim" in params:
        ax.set_ylim(params["ylim"])

    return fig, ax


def render_psth(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render a PSTH plot using matplotlib

    Parameters
    ----------
    data : Dict
        PSTH data dictionary from PSTHPlot.get_data()
    ax : Axes, optional
        Matplotlib axes to render on. If None, creates a new figure.
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : Figure
        Matplotlib figure
    ax : Axes
        Matplotlib axes with the rendered plot
    """
    # Extract data
    psth_data = data["data"]
    time_bins = psth_data["time_bins"]
    firing_rates = psth_data["firing_rates"]
    sem = psth_data["sem"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with any provided kwargs

    line_color = params.get("line_color", "orange")
    line_width = params.get("line_width", 2)
    show_sem = params.get("show_sem", True)
    sem_alpha = params.get("sem_alpha", 0.3)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot PSTH line
    ax.plot(time_bins, firing_rates, color=line_color, linewidth=line_width)

    # Plot SEM if available and requested
    if sem is not None and show_sem:
        sem_array = np.array(sem)
        rates_array = np.array(firing_rates)
        time_array = np.array(time_bins)

        # Make sure SEM values are valid
        valid_sem = np.isfinite(sem_array)
        if np.any(valid_sem):
            ax.fill_between(
                time_array[valid_sem],
                rates_array[valid_sem] - sem_array[valid_sem],
                rates_array[valid_sem] + sem_array[valid_sem],
                color=line_color,
                alpha=sem_alpha,
            )

    # Set labels
    ax.set_xlabel("Time from event onset (s)")
    ax.set_ylabel("Firing rate (Hz)")

    # Set title if unit_id is provided
    if psth_data["unit_id"] is not None:
        ax.set_title(f"Unit {psth_data['unit_id']} PSTH")

    # Apply additional styling from params
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    # Add event onset line at t=0 if requested
    if params.get("show_event_onset", True):
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    # Add baseline window if provided
    if "baseline_window" in params:
        start, end = params["baseline_window"]
        ax.axvspan(start, end, color="gray", alpha=0.2)

        # Add text label if there's room
        if end - start > 0.05:
            ax.text(
                (start + end) / 2,
                ax.get_ylim()[1] * 0.9,
                "baseline",
                ha="center",
                va="top",
                fontsize=8,
                alpha=0.7,
            )

    # Set x limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])

    # Set y limits if provided
    if "ylim" in params:
        ax.set_ylim(params["ylim"])

    return fig, ax
