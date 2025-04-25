from typing import Any, Dict

import plotly.graph_objects as go


def render_raster(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Render a spike raster plot using Plotly.

    Parameters
    ----------
    data : Dict
        Data dictionary from RasterPlot.get_data()
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : go.Figure
        Plotly figure object
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

    # Create figure
    fig = go.Figure()

    # Set symbol based on marker type
    symbol = "line-ns" if marker_type == "|" else "circle"

    # Add scatter trace for spikes
    if spike_times:
        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=y_values,
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=marker_size,
                    color=marker_color,
                    opacity=marker_alpha,
                    line=dict(width=marker_size / 4 if marker_type != "|" else 1),
                ),
                hovertemplate="Time: %{x:.3f}s<br>Trial: %{y}<extra></extra>",
            )
        )

    # Set title
    title = f"Unit {unit_id}" if unit_id is not None else "Raster Plot"
    if spike_times:
        trial_count = len(set(y_values))
        title += f" - {trial_count} trials"
    else:
        title += " - No spikes"

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Trial",
        hovermode="closest",
    )

    # Show grid if requested
    if params.get("show_grid", True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Add vertical line at event onset (time=0)
    if params.get("show_event_onset", True):
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)

    # Set x-axis limits if provided
    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])

    # Set y-axis limits if provided
    if "ylim" in params:
        fig.update_yaxes(range=params["ylim"])
    else:
        # Auto-adjust y limits with a small margin
        if y_values:
            fig.update_yaxes(range=[-0.5, max(y_values) + 0.5])

    # Set up y-axis ticks if requested to show trial labels
    if params.get("show_trial_labels", False):
        label_map = spike_data.get("label_map", {})
        if label_map:
            unique_y = sorted(set(y_values))
            fig.update_yaxes(
                tickmode="array",
                tickvals=unique_y,
                ticktext=[label_map.get(y, str(y)) for y in unique_y],
            )

    return fig
