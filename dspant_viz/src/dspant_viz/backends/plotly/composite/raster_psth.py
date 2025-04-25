# src/dspant_viz/backends/plotly/composite/raster_psth.py
from typing import Any, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_raster_psth(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Render a combined raster plot and PSTH using Plotly.

    Parameters
    ----------
    data : dict
        Data dictionary from RasterPSTHComposite.get_data()
    **kwargs : dict
        Additional parameters to override those in data

    Returns
    -------
    fig : go.Figure
        Plotly figure with both plots
    """
    # Extract data
    raster_data = data["raster"]
    psth_data = data["psth"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Extract common parameters
    time_window = params.get("time_window", (-1.0, 1.0))
    title = params.get("title", None)
    show_grid = params.get("show_grid", True)
    ylim_raster = params.get("ylim_raster", None)
    ylim_psth = params.get("ylim_psth", None)
    raster_height_ratio = params.get("raster_height_ratio", 2.0)
    unit_id = params.get("unit_id", None)

    # Calculate subplot heights
    total_height = 1 + raster_height_ratio
    raster_frac = raster_height_ratio / total_height
    psth_frac = 1 / total_height

    # Create subplot layout
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[raster_frac, psth_frac],
        subplot_titles=["", ""],  # Empty titles, will use main title instead
    )

    # Import renderers
    from dspant_viz.backends.plotly.psth import render_psth
    from dspant_viz.backends.plotly.raster import render_raster

    # Create individual figures
    raster_fig = render_raster(raster_data)
    psth_fig = render_psth(psth_data)

    # Add traces from raster plot to top subplot
    for trace in raster_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Add traces from PSTH plot to bottom subplot
    for trace in psth_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # Set main title
    if title:
        fig.update_layout(title=title)
    elif unit_id is not None:
        fig.update_layout(title=f"Unit {unit_id}")

    # Configure axes
    fig.update_xaxes(title="Time (s)", row=2, col=1)
    fig.update_yaxes(title="Trial", row=1, col=1)
    fig.update_yaxes(title="Firing Rate (Hz)", row=2, col=1)

    # Set axis limits if provided
    if time_window:
        fig.update_xaxes(range=time_window, row=1, col=1)
        fig.update_xaxes(range=time_window, row=2, col=1)

    if ylim_raster:
        fig.update_yaxes(range=ylim_raster, row=1, col=1)

    if ylim_psth:
        fig.update_yaxes(range=ylim_psth, row=2, col=1)

    # Show grid if requested
    if show_grid:
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=1, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=1, col=1
        )
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=2, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="lightgray", row=2, col=1
        )

    # Add vertical line at event onset (time=0)
    if params.get("show_event_onset", True):
        fig.add_vline(
            x=0, line_dash="dash", line_color="red", opacity=0.6, row=1, col=1
        )
        fig.add_vline(
            x=0, line_dash="dash", line_color="red", opacity=0.6, row=2, col=1
        )

    # Update layout for better appearance
    fig.update_layout(
        margin=dict(t=50, b=50, l=50, r=20),
        height=600,  # Set a reasonable default height
        showlegend=False,
    )

    return fig
