# %%
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from IPython.display import HTML, display

from dspant_viz.visualization.events.events_test import EventAnnotator

# Import visualization components
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot


# %%
# Generate test data (same as previous test)
def generate_test_data(duration=5.0, fs=1000.0, n_channels=4, chunk_size=1000):
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Create signals with different frequencies for each channel
    data = np.zeros((n_samples, n_channels))

    # Generate random frequencies between 5-40 Hz for each channel
    np.random.seed(42)  # For reproducibility
    frequencies = np.random.randint(5, 41, size=n_channels)

    print(f"Generated random frequencies for {n_channels} channels: {frequencies}")

    for i in range(n_channels):
        # Base sine wave with the channel's frequency
        data[:, i] = np.sin(2 * np.pi * frequencies[i] * t)

        # Add harmonics for more complex signals
        data[:, i] += 0.3 * np.sin(2 * np.pi * 2 * frequencies[i] * t)

        # Add some randomness to phase for 3rd harmonic (only for even channels)
        if i % 2 == 0:
            phase = np.random.uniform(0, np.pi)
            data[:, i] += 0.15 * np.sin(2 * np.pi * 3 * frequencies[i] * t + phase)

        # Add noise with varying amplitude
        noise_level = np.random.uniform(0.05, 0.2)
        data[:, i] += np.random.normal(0, noise_level, n_samples)

    # Convert to Dask array with specified chunk size
    dask_data = da.from_array(data, chunks=(chunk_size, n_channels))

    return dask_data, fs


# Generate the test data
data, sampling_rate = generate_test_data()

# %%
# Test EventAnnotator with different input formats

# 1. Dictionary of events
dict_events = {
    "stim1": [0.5, 1.2, 2.3],  # List of event times
    "stim2": [1.5, 3.4],
}

# 2. Polars DataFrame of events
polars_events = pl.DataFrame(
    {
        "start": [0.5, 1.2, 2.3],
        "end": [0.6, 1.3, 2.4],  # Optional end times
        "channel": [0, 1, 2],  # Optional channel specification
    }
)

# %%
# Matplotlib Backend Test

# Create time series plot
ts_plot_mpl = TimeSeriesPlot(
    data=data,
    sampling_rate=sampling_rate,
    title="Neural Signals with Events - Matplotlib",
    color_mode="colormap",
    colormap="viridis",
    normalize=True,
    grid=True,
)

# Plot with Matplotlib backend
fig_mpl, ax_mpl = ts_plot_mpl.plot(backend="mpl")

# Create and apply event annotator
# Test 1: Dictionary events
event_annotator_dict = EventAnnotator(
    dict_events,
    time_mode="seconds",
    highlight_color="red",
    marker_style="line",
    label_events=False,  # This will disable the text labels
)
event_annotator_dict.plot(backend="mpl", ax=ax_mpl)

# Test 2: Polars DataFrame events
event_annotator_df = EventAnnotator(
    polars_events,
    time_mode="seconds",
    highlight_color="green",
    marker_style="span",
    alpha=0.2,
)
event_annotator_df.plot(backend="mpl", ax=ax_mpl)

plt.tight_layout()
plt.show()

# %%
# Plotly Backend Test

# Create time series plot
ts_plot_plotly = TimeSeriesPlot(
    data=data,
    sampling_rate=sampling_rate,
    title="Neural Signals with Events - Plotly",
    color_mode="colormap",
    colormap="Viridis",
    normalize=True,
    grid=True,
)

# Plot with Plotly backend
fig_plotly = ts_plot_plotly.plot(backend="plotly")

# Create and apply event annotator
# Test 1: Dictionary events
event_annotator_dict_plotly = EventAnnotator(
    dict_events,
    time_mode="seconds",
    highlight_color="red",
    marker_style="line",
    label_events=False,  # This will disable the text labels
)
event_annotator_dict_plotly.plot(backend="plotly", ax=fig_plotly)

# Test 2: Polars DataFrame events
event_annotator_df_plotly = EventAnnotator(
    polars_events,
    time_mode="seconds",
    highlight_color="green",
    marker_style="span",
    alpha=0.2,
    label_events=False,  # This will disable the text labels
)
event_annotator_df_plotly.plot(backend="plotly", ax=fig_plotly)

# Show the plot
fig_plotly.show()

# %%
# Large Dataset Test with Dynamic Resampling and Events

# Generate large dataset
large_data, high_fs = generate_test_data(
    duration=60.0,  # 60 seconds
    fs=30000.0,  # 30 kHz sampling rate (typical for neural recordings)
    n_channels=10,  # 10 channels
    chunk_size=30000,  # Chunk size = 1ms of data
)

# Create complex events for large dataset
complex_events = pl.DataFrame(
    {
        "start": [5.0, 15.0, 30.0, 45.0],
        "end": [5.5, 16.0, 31.0, 46.0],
        "label": ["Epoch A", "Epoch B", "Epoch C", "Epoch D"],
        "channel": [0, 3, 6, 9],  # Events on specific channels
    }
)

# Create a TimeSeriesPlot with Plotly backend and plotly-resampler
large_plot = TimeSeriesPlot(
    data=large_data,
    sampling_rate=high_fs,
    title="Large Neural Recording with Dynamic Resampling and Events",
    color_mode="colormap",
    colormap="Viridis",
    normalize=True,
    grid=True,
    downsample=False,
)

# Plot with Plotly backend and dynamic resampling
print("Rendering large dataset with events (zoom/pan to explore dynamically)...")
fig_large = large_plot.plot(
    backend="plotly",
    use_resampler=True,
    max_n_samples=10000,  # Show 10k points at a time
)

# Add complex events to the large dataset plot
large_event_annotator = EventAnnotator(
    complex_events,
    time_mode="seconds",
    highlight_color="red",
    marker_style="span",
    alpha=0.3,
    label_events=True,
)
large_event_annotator.plot(backend="plotly", ax=fig_large)

# Show the plot
fig_large.show()
# %%
