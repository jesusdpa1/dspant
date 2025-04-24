from typing import Any, Dict, Union

import plotly.graph_objects as go


def render_event_annotator(
    data: Dict[str, Any], ax: Union[go.Figure, None] = None, **kwargs
) -> go.Figure:
    """
    Render event annotations on a Plotly figure.

    Parameters
    ----------
    data : dict
        Event data and rendering parameters
    ax : go.Figure, optional
        Plotly figure to annotate. If None, creates a new figure.
    **kwargs
        Additional rendering parameters to override those in data

    Returns
    -------
    go.Figure
        Annotated Plotly figure
    """
    # If no figure is provided, raise an error
    if ax is None:
        raise ValueError("A Plotly figure must be provided")

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

    # Convert events to lists
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
            ax.add_vrect(
                x0=start,
                x1=end,
                fillcolor=highlight_color,
                opacity=alpha,
                layer="below",
                line_width=0,
            )

        # Always draw a vertical line for each event start
        ax.add_vline(x=start, line_color=highlight_color, line_width=1)

        # Add label if requested and label exists
        if label_events and label is not None:
            # Add text annotation
            ax.add_annotation(
                x=start,
                y=1,  # Top of the plot
                text=str(label),
                showarrow=False,
                font=dict(color=highlight_color, size=10),
                xref="x",
                yref="paper",
                align="left",
                textangle=-45,
            )

    return ax
