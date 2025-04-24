# src/dspant_viz/backends/mpl/composite/raster_psth.py
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_raster_psth(
    data: Dict[str, Any], figsize: Optional[Tuple[float, float]] = None, **kwargs
) -> Tuple[Figure, List[Axes]]:
    """
    Render a combined raster plot and PSTH using matplotlib.

    Parameters
    ----------
    data : dict
        Data dictionary from RasterPSTHComposite.get_data()
    figsize : tuple of (float, float), optional
        Figure size in inches. If None, uses (10, 8)
    **kwargs : dict
        Additional parameters to override those in data

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : List[Axes]
        List of matplotlib axes [raster_ax, psth_ax]
    """
    # Extract data
    raster_data = data["raster"]
    psth_data = data["psth"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Extract common parameters
    time_window = params.get("time_window", (-1.0, 1.0))
    title = params.get("title", None)
    show_grid = params.get("show_grid", True)
    ylim_raster = params.get("ylim_raster", None)
    ylim_psth = params.get("ylim_psth", None)
    raster_height_ratio = params.get("raster_height_ratio", 2.0)
    normalize_psth = params.get("normalize_psth", False)
    show_smoothed = params.get("show_smoothed", True)

    # Set up figure
    if figsize is None:
        figsize = (10, 8)

    # Create figure with two subplots (shared x-axis)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [raster_height_ratio, 1]},
    )
    raster_ax, psth_ax = axes

    # Extract spike data
    spike_data = raster_data["data"]
    spike_times = spike_data["spike_times"]

    # Check if we have "y_values" or "trial_indices"
    if "y_values" in spike_data:
        trial_indices = spike_data["y_values"]
    elif "trial_indices" in spike_data:
        trial_indices = spike_data["trial_indices"]
    else:
        # Create default trial indices if neither field exists
        trial_indices = np.zeros(len(spike_times), dtype=int)
        print("Warning: No trial indices found in spike data. Using zeros as default.")

    # Extract marker parameters
    marker_type = raster_data["params"].get("marker_type", "|")
    marker_size = raster_data["params"].get("marker_size", 4)
    marker_color = raster_data["params"].get("marker_color", "#2D3142")
    marker_alpha = raster_data["params"].get("marker_alpha", 0.7)

    # Extract PSTH data
    time_bins = psth_data["data"]["time_bins"]
    firing_rates = psth_data["data"]["firing_rates"]
    sem = psth_data["data"].get("sem", None)

    # Extract PSTH parameters
    psth_color = psth_data["params"].get("line_color", "orange")
    line_width = psth_data["params"].get("line_width", 2)
    show_sem = psth_data["params"].get("show_sem", True)
    sem_alpha = psth_data["params"].get("sem_alpha", 0.3)

    # Plot raster
    if len(spike_times) > 0:
        raster_ax.scatter(
            spike_times,
            trial_indices,
            marker=marker_type,
            s=marker_size,
            color=marker_color,
            alpha=marker_alpha,
            linewidths=marker_size / 4 if marker_type != "|" else 1,
        )

    # Set raster labels
    raster_ax.set_ylabel("Trial")
    if title:
        raster_ax.set_title(title)

    # Set y-limits for raster if provided
    if ylim_raster is not None:
        raster_ax.set_ylim(ylim_raster)
    else:
        # Set y-limits to show all trials, calculate from data
        max_trial = max(trial_indices) if trial_indices else 0
        raster_ax.set_ylim(-0.5, max_trial + 0.5)

    # Plot PSTH
    psth_ax.plot(time_bins, firing_rates, color=psth_color, linewidth=line_width)

    # Plot SEM if requested
    if show_sem and sem is not None:
        # Make sure SEM values are valid
        valid_sem = np.isfinite(sem)
        if np.any(valid_sem):
            psth_ax.fill_between(
                np.array(time_bins)[valid_sem],
                np.array(firing_rates)[valid_sem] - np.array(sem)[valid_sem],
                np.array(firing_rates)[valid_sem] + np.array(sem)[valid_sem],
                color=psth_color,
                alpha=sem_alpha,
            )

    # Set PSTH labels
    psth_ax.set_xlabel("Time (s)")
    psth_ax.set_ylabel("Firing rate (Hz)")

    # Set y-limits for PSTH if provided
    if ylim_psth is not None:
        psth_ax.set_ylim(ylim_psth)

    # Set x-limits if provided
    if time_window is not None:
        for ax in axes:
            ax.set_xlim(time_window)

    # Add vertical line at event onset (time=0)
    for ax in axes:
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)
        if show_grid:
            ax.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    return fig, axes
