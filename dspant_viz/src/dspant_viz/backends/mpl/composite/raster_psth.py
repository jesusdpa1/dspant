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
    unit_id = params.get("unit_id", None)

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

    # Render raster and PSTH
    from dspant_viz.backends.mpl.psth import render_psth
    from dspant_viz.backends.mpl.raster import render_raster

    # Use the raster renderer to populate the top subplot
    render_raster(raster_data, ax=raster_ax)

    # Use the PSTH renderer to populate the bottom subplot
    render_psth(psth_data, ax=psth_ax)

    # Customize title if provided or if we have unit_id
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    elif unit_id is not None:
        fig.suptitle(f"Unit {unit_id}", fontsize=14, fontweight="bold")

    # Remove redundant titles from subplots since we have a figure title
    if title or unit_id is not None:
        raster_ax.set_title("")
        psth_ax.set_title("")

    # Ensure only bottom plot shows x-axis label
    raster_ax.set_xlabel("")
    psth_ax.set_xlabel("Time (s)")

    # Set x-axis limits if provided
    if time_window:
        for ax in axes:
            ax.set_xlim(time_window)

    # Set y-axis limits if provided
    if ylim_raster:
        raster_ax.set_ylim(ylim_raster)

    if ylim_psth:
        psth_ax.set_ylim(ylim_psth)

    # Add event onset line
    for ax in axes:
        if params.get("show_event_onset", True):
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    # Add grid
    if show_grid:
        for ax in axes:
            ax.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    # Adjust for suptitle if present
    if title or unit_id is not None:
        plt.subplots_adjust(top=0.92)

    return fig, axes
