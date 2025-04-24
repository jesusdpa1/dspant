# src/dspant_viz/backends/plotly/composite/raster_psth.py
from typing import Any, Dict

import numpy as np
import plotly.colors as pc  # Import the colors module separately
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_raster_psth(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Render a combined raster plot and PSTH using plotly.

    Parameters
    ----------
    data : dict
        Data dictionary from RasterPSTHComposite.get_data()
    **kwargs : dict
        Additional parameters to override those in data

    Returns
    -------
    fig : go.Figure
        Plotly figure
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
    normalize_psth = params.get("normalize_psth", False)
    show_smoothed = params.get("show_smoothed", True)

    # Create subplot layout with appropriate height ratios
    subplot_heights = [raster_height_ratio, 1]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[h / sum(subplot_heights) for h in subplot_heights],
        subplot_titles=["Raster Plot", "PSTH"],
    )

    # Extract spike data
    spike_data = raster_data["data"]
    spike_times = spike_data["spike_times"]

    # Check if we have "y_values" or "trial_indices"
    if "y_values" in spike_data:
        trial_indices = spike_data["y_values"]
    elif "trial_indices" in spike_data:
        trial_indices = spike_data["trial_indices"]
    else:
        # Create default trial indices if neither field exists
        trial_indices = np.zeros(len(spike_times), dtype=int)
        print("Warning: No trial indices found in spike data. Using zeros as default.")

    # Extract marker parameters
    marker_type = raster_data["params"].get("marker_type", "|")
    marker_size = raster_data["params"].get("marker_size", 4)
    marker_color = raster_data["params"].get("marker_color", "#2D3142")
    marker_alpha = raster_data["params"].get("marker_alpha", 0.7)

    # Extract PSTH data
    time_bins = psth_data["data"]["time_bins"]
    firing_rates = psth_data["data"]["firing_rates"]
    sem = psth_data["data"].get("sem", None)

    # Extract PSTH parameters
    psth_color = psth_data["params"].get("line_color", "orange")
    line_width = psth_data["params"].get("line_width", 2)
    show_sem = psth_data["params"].get("show_sem", True)
    sem_alpha = psth_data["params"].get("sem_alpha", 0.3)

    # Add raster plot
    if len(spike_times) > 0:
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
                name="Spikes",
                hovertemplate="Time: %{x:.3f}s<br>Trial: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Add PSTH trace
    fig.add_trace(
        go.Scatter(
            x=time_bins,
            y=firing_rates,
            mode="lines",
            line=dict(color=psth_color, width=line_width),
            name="PSTH",
            hovertemplate="Time: %{x:.3f}s<br>Rate: %{y:.2f} Hz<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add SEM area if requested
    if show_sem and sem is not None and np.any(np.isfinite(sem)):
        # Convert to numpy arrays for calculation
        time_bins_array = np.array(time_bins)
        firing_rates_array = np.array(firing_rates)
        sem_array = np.array(sem)

        # Create a valid mask
        valid_mask = np.isfinite(sem_array)

        # Filter data
        valid_times = time_bins_array[valid_mask]
        valid_rates = firing_rates_array[valid_mask]
        valid_sem = sem_array[valid_mask]

        # Add upper and lower bounds with fill
        x_fill = np.concatenate((valid_times, valid_times[::-1]))
        y_fill = np.concatenate(
            (valid_rates + valid_sem, (valid_rates - valid_sem)[::-1])
        )

        # Create a semi-transparent fill color
        try:
            # Try to convert hex color to RGBA
            rgba_color = f"rgba{tuple(int(c * 255) for c in pc.hex_to_rgb(psth_color)) + (sem_alpha,)}"
        except (ValueError, AttributeError):
            # Fallback: use the original color with added transparency
            rgba_color = f"rgba(220, 20, 60, {sem_alpha})"  # Crimson with alpha

        fig.add_trace(
            go.Scatter(
                x=x_fill,
                y=y_fill,
                fill="toself",
                fillcolor=rgba_color,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    # Add vertical line at event onset (time=0)
    for row in [1, 2]:
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            yref="paper",
            xref=f"x{row}",
            line=dict(color="red", width=2, dash="dash"),
            opacity=0.6,
            row=row,
            col=1,
        )

    # Update layout
    y_axis_titles = ["Trial", "Firing Rate (Hz)"]
    for i, y_title in enumerate(y_axis_titles):
        fig.update_yaxes(
            title_text=y_title,
            row=i + 1,
            col=1,
            gridcolor="lightgray" if show_grid else None,
            showgrid=show_grid,
        )

    # Set x-axis title only on bottom plot
    fig.update_xaxes(
        title_text="Time (s)",
        row=2,
        col=1,
        gridcolor="lightgray" if show_grid else None,
        showgrid=show_grid,
    )

    # Set axis limits if provided
    if time_window is not None:
        for row in [1, 2]:
            fig.update_xaxes(range=time_window, row=row, col=1)

    if ylim_raster is not None:
        fig.update_yaxes(range=ylim_raster, row=1, col=1)
    else:
        # Set y-limits to show all trials
        max_trial = max(trial_indices) if len(trial_indices) > 0 else 0
        fig.update_yaxes(range=[-0.5, max_trial + 0.5], row=1, col=1)

    if ylim_psth is not None:
        fig.update_yaxes(range=ylim_psth, row=2, col=1)

    # Set overall title if provided
    if title:
        fig.update_layout(title=title)

    # Update general layout
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        margin=dict(t=50, b=50, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
