# src/dspant_viz/backends/mpl/waveform.py
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_waveform(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    """
    Render waveform using Matplotlib.

    Parameters
    ----------
    data : Dict
        Data dictionary from WaveformPlot.get_data()
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
    time_values = plot_data["time"]
    waveforms = plot_data["waveforms"]
    sem = plot_data.get("sem")
    unit_id = plot_data.get("unit_id")

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Color handling
    color_mode = params.get("color_mode", "colormap")
    colormap = params.get("colormap", "colorblind")
    line_width = params.get("line_width", 1.0)
    alpha = params.get("alpha", 0.7)

    # Handle color selection
    if color_mode == "colormap":
        # Use seaborn colorblind palette
        if colormap == "colorblind":
            colors = sns.color_palette("colorblind")
        else:
            # Allow custom colormap
            colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, waveforms.shape[1]))
    else:
        # Single color
        colors = [colormap] * waveforms.shape[1]

    # Plot individual waveforms or template
    template = params.get("template", False)

    if not template:
        # Plot individual waveforms
        for i in range(waveforms.shape[1]):
            ax.plot(
                time_values,
                waveforms[:, i],
                color=colors[i % len(colors)],
                linewidth=line_width,
                alpha=alpha,
            )
    else:
        # Plot template with SEM
        ax.plot(
            time_values,
            waveforms,
            color="blue",  # Default template color
            linewidth=line_width,
            label=f"Unit {unit_id} Mean Waveform",
        )

        # Add SEM if available
        if sem is not None:
            ax.fill_between(
                time_values,
                waveforms.ravel() - sem,
                waveforms.ravel() + sem,
                alpha=0.3,
                color="blue",
                label="SEM",
            )
            ax.legend()

    # Set labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    title = f"Waveform for Unit {unit_id}" + (" (Template)" if template else "")
    ax.set_title(title)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Make plot look nice
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax
