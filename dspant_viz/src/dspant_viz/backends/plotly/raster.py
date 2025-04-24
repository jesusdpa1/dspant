from typing import Any, Dict

import plotly.graph_objects as go


def render_raster(data: Dict[str, Any], **kwargs) -> go.Figure:
    spike_data = data["data"]
    spike_times = spike_data["spike_times"]
    trial_indices = spike_data["trial_indices"]
    unit_id = spike_data["unit_id"]

    params = data["params"]
    params.update(kwargs)

    marker_size = params.get("marker_size", 4)
    marker_color = params.get("marker_color", "#2D3142")
    marker_alpha = params.get("marker_alpha", 0.7)
    marker_type = params.get("marker_type", "|")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=spike_times,
            y=trial_indices,
            mode="markers",
            marker=dict(
                symbol="line-ns" if marker_type == "|" else "circle",
                size=marker_size,
                color=marker_color,
                opacity=marker_alpha,
                line=dict(width=marker_size / 4 if marker_type != "|" else 1),
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        yaxis_title="Trial",
        title=f"Unit {unit_id}" if unit_id is not None else "Raster Plot",
        template="simple_white",
        hovermode="closest",
    )

    if params.get("show_grid", True):
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="lightgray")

    if params.get("show_event_onset", True):
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)

    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])

    if "ylim" in params:
        fig.update_yaxes(range=params["ylim"])

    return fig
