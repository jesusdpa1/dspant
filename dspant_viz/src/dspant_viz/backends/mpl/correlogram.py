# src/dspant_viz/backends/mpl/crosscorrelogram.py
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_correlogram(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render crosscorrelogram using Matplotlib.

    Parameters
    ----------
    data : Dict
        Data dictionary from CrosscorrelogramPlot.get_data()
    ax : Axes, optional
        Matplotlib axes to plot on. If None, creates a new figure.
    **kwargs
        Additional rendering parameters

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
    correlogram = plot_data["correlogram"]
    sem = plot_data.get("sem")
    unit1 = plot_data.get("unit1")
    unit2 = plot_data.get("unit2")

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Check if we have data to plot
    if len(time_bins) == 0 or len(correlogram) == 0:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Correlation")

        # Set title based on units
        if unit2 is None:
            ax.set_title(f"Autocorrelogram for Unit {unit1} - No Data")
        else:
            ax.set_title(
                f"Crosscorrelogram between Units {unit1} and {unit2} - No Data"
            )

        return fig, ax

    # Plot bar histogram
    bar_width = time_bins[1] - time_bins[0]
    ax.bar(
        time_bins,
        correlogram,
        width=bar_width,
        edgecolor="black",
        alpha=0.7,
        label="Correlogram",
    )

    # Add SEM if available
    if sem is not None and len(sem) == len(correlogram):
        ax.fill_between(
            time_bins,
            correlogram - sem,
            correlogram + sem,
            alpha=0.3,
            color="gray",
            label="SEM",
        )

    # Add vertical line at zero
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6)

    # Set labels and title
    ax.set_xlabel("Time Lag (s)")
    ax.set_ylabel("Correlation")

    # Set title based on units
    if unit2 is None:
        ax.set_title(f"Autocorrelogram for Unit {unit1}")
    else:
        ax.set_title(f"Crosscorrelogram between Units {unit1} and {unit2}")

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend if SEM is present
    if sem is not None:
        ax.legend()

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax
