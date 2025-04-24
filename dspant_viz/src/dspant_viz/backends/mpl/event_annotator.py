from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def render_event_annotator(
    data: Dict[str, Any], ax: Optional[plt.Axes] = None, **kwargs
) -> plt.Axes:
    """
    Render event annotations on a matplotlib axes.

    Parameters
    ----------
    data : dict
        Event data and rendering parameters
    ax : plt.Axes, optional
        Matplotlib axes to annotate. If None, raises an error.
    **kwargs
        Additional rendering parameters to override those in data

    Returns
    -------
    plt.Axes
        Annotated matplotlib axes
    """
    if ax is None:
        raise ValueError("An existing matplotlib Axes must be provided")

    # Extract event data and parameters
    events_data = data["data"]["events"]
    time_mode = data["data"]["time_mode"]
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Extract rendering parameters
    highlight_color = params.get("highlight_color", "red")
    alpha = params.get("alpha", 0.3)
    marker_style = params.get("marker_style", "line")
    label_events = params.get("label_events", True)

    # Convert events to DataFrame if needed
    starts = events_data.get("start", [])
    ends = events_data.get("end", [])
    labels = events_data.get("label", [None] * len(starts))

    # Handle events with no explicit end time
    if not ends:
        ends = [None] * len(starts)

    # Render each event
    for start, end, label in zip(starts, ends, labels):
        # Render event based on marker style
        if marker_style == "span" and end is not None:
            # Span event with shaded region
            ax.axvspan(start, end, alpha=alpha, color=highlight_color)

        # Always draw a vertical line for each event start
        ax.axvline(x=start, color=highlight_color, linestyle="-", linewidth=1)

        # Add label if requested and label exists
        if label_events and label is not None:
            # Position label just above the vertical line
            ylim = ax.get_ylim()
            y_pos = ylim[1] - 0.05 * (ylim[1] - ylim[0])
            ax.text(
                start,
                y_pos,
                str(label),
                color=highlight_color,
                rotation=45,
                verticalalignment="top",
                fontsize=8,
            )

    return ax
