from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go


def render_psth(data: Dict[str, Any], **kwargs) -> go.Figure:
    """
    Render PSTH (Peristimulus Time Histogram) using Plotly.

    Parameters
    ----------
    data : Dict
        Data dictionary from PSTHPlot.get_data()
    **kwargs
        Additional parameters to override those in data

    Returns
    -------
    fig : go.Figure
        Plotly figure object
    """
    # Extract data
    plot_data = data["data"]
    time_bins = plot_data["time_bins"]
    firing_rates = plot_data["firing_rates"]
    sem = plot_data.get("sem")
    unit_id = plot_data.get("unit_id")

    # Extract parameters
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs

    line_color = params.get("line_color", "orange")
    line_width = params.get("line_width", 2)
    show_sem = params.get("show_sem", True)
    sem_alpha = params.get("sem_alpha", 0.3)

    # Create figure
    fig = go.Figure()

    # Add firing rate trace
    if time_bins and firing_rates:
        # Convert to numpy arrays for calculations
        time_bins_array = np.array(time_bins)
        firing_rates_array = np.array(firing_rates)

        fig.add_trace(
            go.Scatter(
                x=time_bins_array,
                y=firing_rates_array,
                mode="lines",
                line=dict(color=line_color, width=line_width),
                name="Firing Rate",
                hovertemplate="Time: %{x:.3f}s<br>Rate: %{y:.2f} Hz<extra></extra>",
            )
        )

        # Add SEM if available and requested
        if show_sem and sem is not None:
            sem_array = np.array(sem)

            # Filter out any NaN or inf values
            valid_mask = np.isfinite(sem_array) & np.isfinite(firing_rates_array)

            if np.any(valid_mask):
                # Use only the valid data points
                valid_times = time_bins_array[valid_mask]
                valid_rates = firing_rates_array[valid_mask]
                valid_sem = sem_array[valid_mask]

                # Create upper and lower bounds
                upper = valid_rates + valid_sem
                lower = valid_rates - valid_sem

                # Sort points by x-value to ensure proper fill
                sort_idx = np.argsort(valid_times)
                valid_times = valid_times[sort_idx]
                upper = upper[sort_idx]
                lower = lower[sort_idx]

                # Create x values for fill (need to go forward then backward)
                x_fill = np.concatenate([valid_times, valid_times[::-1]])
                y_fill = np.concatenate([upper, lower[::-1]])

                # Add fill area for SEM
                fig.add_trace(
                    go.Scatter(
                        x=x_fill,
                        y=y_fill,
                        fill="toself",
                        fillcolor=f"rgba({int(int(line_color[1:3], 16) if line_color.startswith('#') else 220)}, "
                        f"{int(int(line_color[3:5], 16) if line_color.startswith('#') else 20)}, "
                        f"{int(int(line_color[5:7], 16) if line_color.startswith('#') else 60)}, {sem_alpha})",
                        line=dict(width=0),  # No border line
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    # Set title
    title = f"Unit {unit_id} PSTH" if unit_id is not None else "PSTH"
    if not time_bins or not firing_rates:
        title += " - No data"

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Firing rate (Hz)",
        hovermode="x unified",
    )

    # Show grid if requested
    if params.get("show_grid", True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Add vertical line at event onset (time=0)
    if params.get("show_event_onset", True):
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)

    # Add baseline window if provided
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

        # Add text label if there's room
        if end - start > 0.05 and time_bins and firing_rates:
            y_max = max(firing_rates) * 0.9
            fig.add_annotation(
                x=(start + end) / 2,
                y=y_max,
                text="baseline",
                showarrow=False,
                font=dict(size=10, color="gray"),
                opacity=0.7,
            )

    # Set axis limits if provided
    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])

    if "ylim" in params:
        fig.update_yaxes(range=params["ylim"])

    return fig
