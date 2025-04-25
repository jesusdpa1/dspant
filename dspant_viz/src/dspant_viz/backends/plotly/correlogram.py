# src/dspant_viz/backends/plotly/crosscorrelogram.py
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go


def render_correlogram(
    data: Dict[str, Any], use_webgl: bool = True, **kwargs
) -> go.Figure:
    """
    Render crosscorrelogram using Plotly with WebGL acceleration.

    Parameters
    ----------
    data : Dict
        Data dictionary from CrosscorrelogramPlot.get_data()
    use_webgl : bool
        Whether to use WebGL acceleration
    **kwargs
        Additional rendering parameters

    Returns
    -------
    fig : go.Figure
        Plotly figure object
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

    # Create figure
    fig = go.Figure()

    # Check if we have data to plot
    if len(time_bins) == 0 or len(correlogram) == 0:
        # Create an empty figure with informative message
        if unit2 is None:
            title = f"Autocorrelogram for Unit {unit1} - No Data"
        else:
            title = f"Crosscorrelogram between Units {unit1} and {unit2} - No Data"

        fig.update_layout(
            title=title, xaxis_title="Time Lag (s)", yaxis_title="Correlation"
        )
        return fig

    # Choose scatter type based on WebGL option
    ScatterType = go.Scattergl if use_webgl else go.Scatter

    # Add bar trace for correlogram
    bar_width = time_bins[1] - time_bins[0] if len(time_bins) > 1 else 0.01
    fig.add_trace(
        go.Bar(
            x=time_bins,
            y=correlogram,
            name="Correlogram",
            width=bar_width,
            opacity=0.7,
            marker_color="royalblue",
            hovertemplate="Time: %{x:.3f}s<br>Correlation: %{y:.4f}<extra></extra>",
        )
    )

    # Add SEM trace if available
    if sem is not None and len(sem) == len(correlogram):
        # Create fill area for SEM
        upper = correlogram + sem
        lower = correlogram - sem

        # Sort points to ensure proper fill
        x_fill = np.concatenate([time_bins, time_bins[::-1]])
        y_fill = np.concatenate([upper, lower[::-1]])

        fig.add_trace(
            go.Scatter(
                x=x_fill,
                y=y_fill,
                fill="toself",
                fillcolor="rgba(65,105,225,0.2)",  # Soft blue with low opacity
                line=dict(width=0),
                name="SEM",
                hoverinfo="skip",
            )
        )

    # Add vertical line at zero
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)

    # Update layout
    title = (
        f"Autocorrelogram for Unit {unit1}"
        if unit2 is None
        else f"Crosscorrelogram between Units {unit1} and {unit2}"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Lag (s)",
        yaxis_title="Correlation",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Configure grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Add WebGL indicator if used
    if use_webgl:
        fig.add_annotation(
            text="WebGL acceleration enabled",
            xref="paper",
            yref="paper",
            x=1,
            y=0,
            xanchor="right",
            yanchor="bottom",
            showarrow=False,
            font=dict(size=8, color="gray"),
            opacity=0.7,
        )

    return fig
