# src/dspant_viz/backends/plotly/ts_area.py
from typing import Any, Dict

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import seaborn as sns


def _parse_color_with_alpha(color: str, alpha: float) -> str:
    """
    Convert a color to an rgba string with specified alpha.

    Parameters
    ----------
    color : str
        Input color (hex, rgb, name, etc.)
    alpha : float
        Opacity value (0-1)

    Returns
    -------
    str
        RGBA color string
    """
    try:
        # Try Plotly's color parsing
        rgb = pc.label_to_rgb(color)
        # Convert to rgba with specified alpha
        return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"
    except Exception:
        # Fallback to a default color
        return f"rgba(0,0,255,{alpha})"


def _generate_distinct_colors(
    n_colors: int,
    colormap: str = "viridis",
    line_alpha: float = 0.8,
    fill_alpha: float = 0.3,
) -> tuple:
    """
    Generate a set of distinct colors for multiple channels.

    Parameters
    ----------
    n_colors : int
        Number of distinct colors to generate
    colormap : str, optional
        Colormap to use for color generation
    line_alpha : float, optional
        Alpha value for line colors
    fill_alpha : float, optional
        Alpha value for fill colors

    Returns
    -------
    tuple
        (line_colors, fill_colors)
    """
    # Special handling for colorblind palette
    if colormap == "colorblind":
        colors = sns.color_palette("colorblind", n_colors)
        line_colors = [
            f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{line_alpha})"
            for r, g, b in colors
        ]
        fill_colors = [
            f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{fill_alpha})"
            for r, g, b in colors
        ]
        return line_colors, fill_colors

    # Use Plotly colorscales for other cases
    try:
        # Generate line colors
        line_colors = pc.sample_colorscale(colormap, n_colors)

        # Convert to rgba with alphas
        line_colors = [
            _parse_color_with_alpha(color, line_alpha) for color in line_colors
        ]

        # Generate fill colors with different alpha
        fill_colors = [
            _parse_color_with_alpha(color, fill_alpha) for color in line_colors
        ]

        return line_colors, fill_colors

    except Exception:
        # Fallback to default colorscale with distinct colors
        line_colors = pc.sample_colorscale("Viridis", n_colors)
        line_colors = [
            _parse_color_with_alpha(color, line_alpha) for color in line_colors
        ]
        fill_colors = [
            _parse_color_with_alpha(color, fill_alpha) for color in line_colors
        ]
        return line_colors, fill_colors


def render_ts_area(data: Dict[str, Any], use_webgl: bool = True, **kwargs) -> go.Figure:
    """
    Render time series area plot using Plotly with WebGL acceleration.

    Parameters
    ----------
    data : Dict
        Data dictionary from TimeSeriesAreaPlot.get_data()
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
    time_values = plot_data["time"]
    signals = plot_data["signals"]
    channels = plot_data["channels"]
    channel_info = plot_data["channel_info"]
    channel_positions = plot_data["channel_positions"]
    fill_values = plot_data["fill_values"]

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    # Color handling
    color_mode = params.get("color_mode", "colormap")
    colormap = params.get("colormap", "colorblind")
    single_color = params.get("color", "black")
    line_width = params.get("line_width", 1.0)
    alpha = params.get("alpha", 0.8)
    fill_alpha = params.get("fill_alpha", 0.3)
    grid = params.get("grid", True)
    show_channel_labels = params.get("show_channel_labels", True)

    # Create figure
    fig = go.Figure()

    # Determine WebGL or standard scatter
    ScatterType = go.Scattergl if use_webgl else go.Scatter

    # Handle color selection
    if color_mode == "colormap":
        # Generate distinct colors for all channels
        line_colors, fill_colors = _generate_distinct_colors(
            len(channels), colormap, alpha, fill_alpha
        )
    else:
        # Single color
        line_colors = [
            _parse_color_with_alpha(single_color, alpha) for _ in range(len(channels))
        ]
        fill_colors = [
            _parse_color_with_alpha(single_color, fill_alpha)
            for _ in range(len(channels))
        ]

    # Prepare fill traces and line traces for each channel
    traces = []
    for idx, (channel_idx, signal) in enumerate(zip(channels, signals)):
        # Prepare x and y for filling
        fill_x = np.concatenate([time_values, time_values[::-1]])
        fill_y = np.concatenate([signal, np.full_like(signal, fill_values[idx])[::-1]])

        # Fill trace
        traces.append(
            go.Scatter(
                x=fill_x,
                y=fill_y,
                fill="toself",
                fillcolor=fill_colors[idx],
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Line trace
        traces.append(
            ScatterType(
                x=time_values,
                y=signal,
                mode="lines",
                line=dict(color=line_colors[idx], width=line_width),
                name=f"Channel {channel_idx}",
            )
        )

    # Add all traces to figure
    for trace in traces:
        fig.add_trace(trace)

    # Update layout
    fig.update_layout(
        title=params.get("title", "Multi-Channel Area Plot"),
        xaxis_title="Time (s)",
        yaxis_title="Channels" if show_channel_labels else None,
        hovermode="closest",
        showlegend=True,
    )

    # Configure y-axis ticks if channel labels are requested
    if show_channel_labels and channel_positions:
        fig.update_yaxes(
            tickmode="array", tickvals=channel_positions, ticktext=channel_info
        )

    # Show grid if requested
    if grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Set x-axis limits if provided
    if "time_window" in params and params["time_window"] is not None:
        fig.update_xaxes(range=params["time_window"])

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
