# %%
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

# Import theme manager
from dspant_viz.core.themes_manager import (
    apply_matplotlib_theme,
    apply_plotly_theme,
    theme_manager,
)

# Apply the theme before creating the plot
apply_matplotlib_theme()  # For Matplotlib backend
apply_plotly_theme()  # For Plotly backend

# Assuming the necessary components are installed or in the current path
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot


# %%
# Generate test data
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


# Generate the data
data, sampling_rate = generate_test_data()

# %%
# Create a TimeSeriesPlot instance with Matplotlib backend
ts_plot = TimeSeriesPlot(
    data=data,
    sampling_rate=sampling_rate,
    title="Neural Signals - Matplotlib Backend",
    color_mode="colormap",
    colormap="viridis",
    normalize=True,
    grid=True,
)

# Plot with Matplotlib backend
fig_mpl, ax = ts_plot.plot(backend="mpl")
plt.show()
fig = ts_plot.plot(backend="plotly")
fig.show()
# %%
large_data, high_fs = generate_test_data(
    duration=60.0,  # 60 seconds
    fs=30000.0,  # 30 kHz sampling rate (typical for neural recordings)
    n_channels=10,  # 10 channels
    chunk_size=30000,  # Chunk size = 1ms of data
)
print(f"Generated data shape: {large_data.shape}")
print(f"Memory footprint (estimated): {large_data.nbytes / (1024**3):.2f} GB")

# %%
# Create a TimeSeriesPlot instance with Plotly backend using plotly-resampler
large_plot = TimeSeriesPlot(
    data=large_data,
    sampling_rate=high_fs,
    title="Large Neural Recording with Dynamic Resampling",
    color_mode="colormap",
    colormap="Viridis",
    normalize=True,
    grid=True,
    # No need to manually downsample, plotly-resampler will handle it
    downsample=False,
)

# Plot with Plotly backend
print("Rendering with plotly-resampler (zoom/pan to explore the data dynamically)...")
fig_large = large_plot.plot(
    backend="plotly",
    use_resampler=True,
    max_n_samples=10000,  # Show 10k points at a time
)

# Configure figure to be responsive
fig_large.show()
# %%
