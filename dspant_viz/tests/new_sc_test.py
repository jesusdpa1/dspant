# tests/timeseries_test.py
# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# %%
from dspant_viz.components.stream.timeseries import TimeSeriesPlot
from dspant_viz.backends.mpl.themes import Theme

def test_timeseries_mpl():
    """Test plotting a time series using matplotlib backend"""
    # Generate sample data [samples]
    fs = 1000  # Sampling frequency in Hz
    duration = 2.0  # Signal duration in seconds
    t = np.arange(0, duration, 1/fs)  # Time vector
    # Create a signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 5 * t) +          # 5 Hz component
        0.5 * np.sin(2 * np.pi * 15 * t) +   # 15 Hz component
        0.25 * np.sin(2 * np.pi * 40 * t) +  # 40 Hz component
        0.1 * np.random.randn(len(t))        # Noise
    )
    # Create the TimeSeriesPlot component
    ts_plot = TimeSeriesPlot(
        times=t.tolist(),
        values=signal.tolist(),
        sampling_rate=fs,
        channel_name="Test Channel",
        line_color="#1f77b4",
        line_width=1.5,
        show_grid=True,
        time_window=(0.5, 1.5),  # Show only 0.5-1.5 seconds
        title="Time Series Test"
    )
    # Get component data
    ts_data = ts_plot.get_data()
    # Set up matplotlib with neuroscience theme
    Theme.apply("neuroscience")
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    # Extract data and parameters
    data = ts_data["data"]
    params = ts_data["params"]
    # Plot the time series
    ax.plot(
        data["times"],
        data["values"],
        color=params["line_color"],
        linewidth=params["line_width"],
        label=data["channel_name"] or "Signal"
    )
    # Set labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(params["title"] or "Time Series")
    # Add grid if requested
    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)
    # Set axis limits if provided
    if "xlim" in params:
        ax.set_xlim(params["xlim"])
    if "ylim" in params:
        ax.set_ylim(params["ylim"])
    # Add legend
    ax.legend()
    # Show plot
    plt.tight_layout()
    plt.show()
    return fig, ax
# %%
fig, ax = test_timeseries_mpl()

# %%
# Now let's add the Plotly implementation
import plotly.graph_objects as go
from dspant_viz.backends.plotly.themes import Theme as PlotlyTheme

def test_timeseries_plotly():
    """Test plotting a time series using Plotly backend"""
    # Generate the same sample data
    fs = 1000  # Sampling frequency in Hz
    duration = 2.0  # Signal duration in seconds
    t = np.arange(0, duration, 1/fs)  # Time vector

    # Create the same signal with multiple frequency components
    signal = (
        np.sin(2 * np.pi * 5 * t) +          # 5 Hz component
        0.5 * np.sin(2 * np.pi * 15 * t) +   # 15 Hz component
        0.25 * np.sin(2 * np.pi * 40 * t) +  # 40 Hz component
        0.1 * np.random.randn(len(t))        # Noise
    )

    # Create the TimeSeriesPlot component
    ts_plot = TimeSeriesPlot(
        times=t.tolist(),
        values=signal.tolist(),
        sampling_rate=fs,
        channel_name="Test Channel",
        line_color="#1f77b4",
        line_width=1.5,
        show_grid=True,
        time_window=(0.5, 1.5),  # Show only 0.5-1.5 seconds
        title="Time Series Test (Plotly)"
    )

    # Get component data
    ts_data = ts_plot.get_data()

    # Set up plotly with neuroscience theme
    PlotlyTheme.apply("neuroscience")

    # Create a plotly figure
    fig = go.Figure()

    # Extract data and parameters
    data = ts_data["data"]
    params = ts_data["params"]

    # Add the time series trace
    fig.add_trace(
        go.Scatter(
            x=data["times"],
            y=data["values"],
            mode='lines',
            line=dict(
                color=params["line_color"],
                width=params["line_width"]
            ),
            name=data["channel_name"] or "Signal",
            hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
        )
    )

    # Update layout
    fig.update_layout(
        title=params["title"] or "Time Series",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        showlegend=True,
        hovermode="closest"
    )

    # Set axis limits if provided
    if "xlim" in params:
        fig.update_xaxes(range=params["xlim"])
    if "ylim" in params:
        fig.update_yaxes(range=params["ylim"])

    # Add template based on grid preference
    if params.get("show_grid", True):
        fig.update_layout(template="plotly_white")
    else:
        fig.update_layout(template="plotly")

    return fig

# %%
# Run the Plotly test
plotly_fig = test_timeseries_plotly()
plotly_fig.show()  # Display the Plotly figure
