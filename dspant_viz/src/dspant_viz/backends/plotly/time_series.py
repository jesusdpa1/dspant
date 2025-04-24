# src/dspant_viz/backends/plotly/time_series.py
from typing import Any, Dict, Optional, Union

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go

# Import plotly-resampler components
try:
    from plotly_resampler import FigureResampler, register_plotly_resampler
    from plotly_resampler.aggregation import MinMaxAggregator

    HAS_RESAMPLER = True
except ImportError:
    HAS_RESAMPLER = False
    print("Warning: plotly-resampler not found. Falling back to standard Plotly.")


def render_time_series(
    data: Dict[str, Any],
    use_resampler: bool = True,
    max_n_samples: int = 10000,
    **kwargs,
) -> Union[go.Figure, "FigureResampler"]:
    """
    Render multi-channel time series data using Plotly with plotly-resampler for large datasets.

    Parameters
    ----------
    data : dict
        Data dictionary from TimeSeriesPlot.get_data()
    use_resampler : bool
        Whether to use plotly-resampler (if available)
    max_n_samples : int
        Maximum number of samples to display without resampling
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : Union[go.Figure, FigureResampler]
        Plotly figure or FigureResampler instance
    """
    # Extract data
    plot_data = data["data"]
    time_values = plot_data["time"]
    signals = plot_data["signals"]
    channels = plot_data["channels"]
    channel_info = plot_data["channel_info"]
    channel_positions = plot_data["channel_positions"]

    # Debug information
    print(
        f"Data summary: {len(time_values)} time points, {len(signals)} signals, {len(channels)} channels"
    )

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    color_mode = params.get("color_mode", "colormap")
    colormap_name = params.get("colormap", "viridis")
    single_color = params.get("color", "black")
    line_width = params.get("line_width", 1.0)
    alpha = params.get("alpha", 0.8)
    grid = params.get("grid", True)
    show_channel_labels = params.get("show_channel_labels", True)

    # Determine if we should use the resampler
    should_use_resampler = (
        use_resampler and HAS_RESAMPLER and len(time_values) > max_n_samples
    )

    # Create the appropriate figure type
    if should_use_resampler:
        # Create a FigureResampler instance
        fig = FigureResampler(
            go.Figure(),
            default_n_shown_samples=max_n_samples,
            default_downsampler=MinMaxAggregator(),
        )
    else:
        # Create a standard Plotly figure
        fig = go.Figure()

    # Prepare color information
    if color_mode == "colormap":
        # Explicitly generate as many colors as we have channels
        num_colors_needed = len(channels)
        try:
            # Generate colors from colorscale
            colorscale = pc.sample_colorscale(colormap_name, num_colors_needed)
            colors = colorscale
        except Exception as e:
            print(f"Warning: Color generation failed: {e}, using default colors")
            # Fallback to basic colors
            colors = [
                f"hsl({h},80%,50%)"
                for h in np.linspace(0, 360, num_colors_needed, endpoint=False)
            ]

        # Double-check we have the right number of colors (debug)
        if len(colors) != num_colors_needed:
            print(
                f"Warning: Expected {num_colors_needed} colors but got {len(colors)}. Using default colors."
            )
            colors = [
                f"hsl({h},80%,50%)"
                for h in np.linspace(0, 360, num_colors_needed, endpoint=False)
            ]

    # Add each channel as a separate trace
    num_traces = min(len(signals), len(channels))
    for idx in range(num_traces):
        # Choose color based on color_mode and ensure we don't go out of bounds
        if color_mode == "colormap" and idx < len(colors):
            plot_color = colors[idx]
        else:  # Single color mode or fallback
            plot_color = single_color

        # Add the trace based on figure type
        channel_idx = channels[idx] if idx < len(channels) else idx

        # Create trace configuration
        trace_kwargs = dict(
            name=f"Channel {channel_idx}",
            line=dict(color=plot_color, width=line_width),
            opacity=alpha,
        )

        # Add the trace based on figure type
        if should_use_resampler:
            # Add trace with resampling
            fig.add_trace(
                go.Scattergl(**trace_kwargs), hf_x=time_values, hf_y=signals[idx]
            )
        else:
            # Add standard trace
            fig.add_trace(
                go.Scatter(x=time_values, y=signals[idx], mode="lines", **trace_kwargs)
            )

    # Set up layout
    fig.update_layout(
        title=params.get("title", "Multi-Channel Time Series"),
        xaxis_title="Time (s)",
        yaxis_title="Channels" if show_channel_labels else None,
        template="plotly_white",  # Clean white template
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Set up grid
    fig.update_xaxes(
        showgrid=grid,
        gridwidth=1,
        gridcolor="lightgray" if grid else None,
    )

    fig.update_yaxes(
        showgrid=grid,
        gridwidth=1,
        gridcolor="lightgray" if grid else None,
    )

    # Add channel labels on y-axis if requested
    if show_channel_labels:
        fig.update_yaxes(
            tickmode="array", tickvals=channel_positions, ticktext=channel_info
        )

    # Set x limits if provided in time_window
    if "time_window" in params and params["time_window"] is not None:
        fig.update_xaxes(range=params["time_window"])

    # For resampler figures, add an indicator
    if should_use_resampler:
        fig.add_annotation(
            text="↔ Zoom/Pan to see more detail (Dynamic Resampling Active)",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="gray"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4,
        )

    return fig
