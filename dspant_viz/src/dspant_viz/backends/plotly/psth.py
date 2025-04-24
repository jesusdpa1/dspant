from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go


def render_psth(data: Dict[str, Any], **kwargs) -> go.Figure:
    psth_data = data["data"]
    time_bins = psth_data["time_bins"]
    firing_rates = psth_data["firing_rates"]
    sem = psth_data["sem"]
    unit_id = psth_data["unit_id"]

    params = data["params"]
    params.update(kwargs)

    line_color = params.get("line_color", "orange")
    line_width = params.get("line_width", 2)
    show_sem = params.get("show_sem", True)
    sem_alpha = params.get("sem_alpha", 0.3)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=time_bins,
            y=firing_rates,
            mode="lines",
            name="Firing Rate",
            line=dict(color=line_color, width=line_width),
        )
    )

    if sem is not None and show_sem:
        upper = np.array(firing_rates) + np.array(sem)
        lower = np.array(firing_rates) - np.array(sem)

        fig.add_trace(
            go.Scatter(
                x=time_bins + time_bins[::-1],
                y=upper.tolist() + lower[::-1].tolist(),
                fill="toself",
                fillcolor=line_color,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="SEM",
                opacity=sem_alpha,
            )
        )

    if params.get("show_event_onset", True):
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)

    if "baseline_window" in params:
        start, end = params["baseline_window"]
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="gray",
            opacity=0.2,
            layer="below",
            line_width=0,
        )
        if end - start > 0.05:
            y_max = max(firing_rates) * 0.9
            fig.add_annotation(
                x=(start + end) / 2,
                y=y_max,
                text="baseline",
                showarrow=False,
                font=dict(size=10, color="gray"),
                opacity=0.7,
            )

    fig.update_layout(
        xaxis_title="Time from event onset (s)",
        yaxis_title="Firing rate (Hz)",
        title=f"Unit {unit_id} PSTH" if unit_id is not None else "PSTH",
        template="simple_white",
        hovermode="x unified",
    )

    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])
    if "ylim" in params:
        fig.update_yaxes(range=params["ylim"])

    return fig
