# src/dspant_viz/backends/plotly/ts_raster.py
from typing import Any, Dict

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go


def render_ts_raster(
    data: Dict[str, Any],
    use_webgl: bool = True,  # Add WebGL option
    **kwargs,
) -> go.Figure:
    """
    Render a time series raster plot using Plotly with WebGL acceleration.

    Parameters
    ----------
    data : Dict
        Data dictionary from TimeSeriesRasterPlot.get_data()
    use_webgl : bool
        Whether to use WebGL acceleration for better performance
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : go.Figure
        Plotly figure object
    """
    # Extract data
    plot_data = data["data"]
    unit_ids = plot_data["unit_ids"]
    unit_spikes = plot_data["unit_spikes"]
    unit_labels = plot_data["unit_labels"]
    y_positions = plot_data["y_positions"]
    y_ticks = plot_data["y_ticks"]
    y_tick_labels = plot_data["y_tick_labels"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Get plotting parameters
    marker_size = params.get("marker_size", 4)
    marker_type = params.get("marker_type", "|")
    use_colormap = params.get("use_colormap", True)
    colormap_name = params.get("colormap", "viridis")
    alpha = params.get("alpha", 0.7)
    show_legend = params.get("show_legend", True)
    show_labels = params.get("show_labels", True)

    # Create figure
    fig = go.Figure()

    # Set symbol based on marker type
    symbol = "line-ns" if marker_type == "|" else "circle"

    # Determine whether to use Scattergl (WebGL) or regular Scatter
    ScatterType = go.Scattergl if use_webgl else go.Scatter

    # Generate colors
    if use_colormap:
        try:
            # Generate colors from colorscale
            colors = pc.sample_colorscale(colormap_name, len(unit_ids))
        except Exception as e:
            # Fallback to generated colors
            print(f"Warning: Color generation failed: {e}, using fallback colors")
            colors = [
                f"rgba({int(220 - i * 30)}, {int(20 + i * 60)}, {int(60 + i * 40)}, {alpha})"
                for i in range(len(unit_ids))
            ]
    else:
        # Use Plotly's default sequential colors
        colors = [
            f"rgba({50 + i * 15}, {100 + i * 10}, {150 + i * 15}, {alpha})"
            for i in range(len(unit_ids))
        ]

    # Plot each unit
    for i, unit_id in enumerate(unit_ids):
        # Get spikes for this unit
        spikes = unit_spikes.get(unit_id, [])
        if not spikes:
            continue

        # Get y position for this unit
        y_pos = y_positions[unit_id]

        # Get label for this unit
        label = unit_labels.get(unit_id, f"Unit {unit_id}")

        # Plot spikes for this unit - use WebGL for better performance
        fig.add_trace(
            ScatterType(
                x=spikes,
                y=[y_pos] * len(spikes),
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=marker_size,
                    color=colors[i % len(colors)],
                    line=dict(width=marker_size / 4 if marker_type != "|" else 1),
                ),
                name=label,
                hovertemplate="Time: %{x:.3f}s<br>Unit: " + label + "<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=params.get("title", "Time Series Raster Plot"),
        xaxis_title="Time (s)",
        yaxis_title="Unit",
        hovermode="closest",
        showlegend=show_legend,
        # Optimizations for large datasets
        uirevision="true",  # Prevent redrawing on UI interactions
    )

    # Set custom y-ticks and labels
    if show_labels and y_ticks:
        fig.update_yaxes(
            tickmode="array",
            tickvals=y_ticks,
            ticktext=y_tick_labels,
            range=[min(y_ticks) - 0.5, max(y_ticks) + 0.5],
        )

    # Show grid if requested
    if params.get("show_grid", True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Set x-axis limits if provided
    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])
    elif "time_window" in params and params["time_window"] is not None:
        fig.update_xaxes(range=params["time_window"])

    # If using WebGL, add a note about it
    if use_webgl:
        fig.add_annotation(
            text="Using WebGL acceleration",
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
