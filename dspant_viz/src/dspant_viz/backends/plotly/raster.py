# src/dspant_viz/backends/plotly/raster.py
from typing import Any, Dict

import plotly.graph_objects as go


def render_raster(
    data: Dict[str, Any],
    use_webgl: bool = True,  # Add WebGL option
    **kwargs,
) -> go.Figure:
    """
    Render a trial-based spike raster plot using Plotly with WebGL acceleration.

    Parameters
    ----------
    data : Dict
        Data dictionary from RasterPlot.get_data()
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
    spike_data = data["data"]
    spike_times = spike_data["spike_times"]
    trial_indices = spike_data[
        "y_values"
    ]  # In trial-based mode, y values are trial indices
    unit_id = spike_data.get("unit_id")
    n_trials = spike_data.get("n_trials", 0)

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    marker_size = params.get("marker_size", 4)
    marker_color = params.get("marker_color", "#2D3142")
    marker_alpha = params.get("marker_alpha", 0.7)
    marker_type = params.get("marker_type", "|")

    # Create figure
    fig = go.Figure()

    # Determine whether to use Scattergl (WebGL) or regular Scatter
    ScatterType = go.Scattergl if use_webgl and len(spike_times) > 1000 else go.Scatter

    # Set symbol based on marker type
    symbol = "line-ns" if marker_type == "|" else "circle"

    # Add scatter trace for spikes
    if spike_times:
        fig.add_trace(
            ScatterType(
                x=spike_times,
                y=trial_indices,
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
    if unit_id is not None:
        title = f"Unit {unit_id} - {n_trials} trials"
    else:
        title = "Raster Plot"

    if not spike_times:
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

    # Always show event onset line in trial-based plots
    fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)

    # Set x-axis limits if provided
    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])
    elif "time_window" in params and params["time_window"] is not None:
        fig.update_xaxes(range=params["time_window"])

    # Set y-axis limits if provided
    if "ylim" in params:
        fig.update_yaxes(range=params["ylim"])
    else:
        # Auto-adjust y limits with a small margin
        if trial_indices:
            fig.update_yaxes(range=[-0.5, max(trial_indices) + 0.5])

    # If using WebGL with many points, add indicator
    if use_webgl and len(spike_times) > 1000:
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
