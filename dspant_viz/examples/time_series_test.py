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

# Import components
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot
from dspant_viz.visualization.stream.ts_raster import TimeSeriesRasterPlot


# %%
# 1. Generate test data for TimeSeriesPlot
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


# 2. Generate test data for TimeSeriesRasterPlot - spike data
def generate_spike_data(duration=10.0, n_units=5, rate_range=(1, 15)):
    """Generate spike data for multiple units with varying firing rates"""
    np.random.seed(42)  # For reproducibility

    # Generate units with different firing rates
    unit_spikes = {}

    for unit_id in range(n_units):
        # Random firing rate between rate_range[0] and rate_range[1] Hz
        rate = np.random.uniform(rate_range[0], rate_range[1])

        # Generate spike times using a Poisson process
        n_spikes = np.random.poisson(rate * duration)
        spike_times = np.random.uniform(0, duration, n_spikes)
        spike_times.sort()  # Sort spike times

        # Add to unit dictionary
        unit_spikes[unit_id] = spike_times

        print(f"Unit {unit_id}: {n_spikes} spikes, rate ≈ {rate:.2f} Hz")

    return unit_spikes


# %%
# Test TimeSeriesPlot with Matplotlib and Plotly
print("Testing TimeSeriesPlot...")
data, sampling_rate = generate_test_data(duration=5.0, n_channels=4)

# Create a TimeSeriesPlot instance
ts_plot = TimeSeriesPlot(
    data=data,
    sampling_rate=sampling_rate,
    title="Time Series Plot - Multiple Channels",
    color_mode="colormap",
    colormap="viridis",
    normalize=True,
    grid=True,
)

# Plot with Matplotlib backend
print("Rendering with Matplotlib...")
fig_mpl, ax = ts_plot.plot(backend="mpl")
plt.show()

# Plot with Plotly backend
print("Rendering with Plotly...")
fig_plotly = ts_plot.plot(backend="plotly")
fig_plotly.show()

# %%
# Test TimeSeriesRasterPlot with Matplotlib and Plotly
print("Testing TimeSeriesRasterPlot...")
spike_data = generate_spike_data(duration=10.0, n_units=5, rate_range=(3, 10))

# Create TimeSeriesRasterPlot instance
ts_raster = TimeSeriesRasterPlot(
    data=spike_data,
    sampling_rate=1000.0,  # 1 kHz sampling rate
    title="Time Series Raster Plot - Multiple Units",
    marker_size=5,
    marker_type="|",
    use_colormap=True,
    colormap="plasma",
    alpha=0.8,
    show_legend=True,
    y_spread=1.0,
)

# Plot with Matplotlib backend
print("Rendering with Matplotlib...")
fig_raster_mpl, ax = ts_raster.plot(backend="mpl")
plt.show()

# Plot with Plotly backend
print("Rendering with Plotly...")
fig_raster_plotly = ts_raster.plot(backend="plotly")
fig_raster_plotly.show()

# %%
# Test combining TimeSeriesPlot and TimeSeriesRasterPlot in the same figure
# This demonstrates how to create a custom layout with both components

# Generate data
lfp_data, lfp_fs = generate_test_data(duration=10.0, n_channels=1, chunk_size=1000)
spike_data = generate_spike_data(duration=10.0, n_units=3, rate_range=(5, 15))

# Create separate components
lfp_plot = TimeSeriesPlot(
    data=lfp_data,
    sampling_rate=lfp_fs,
    color="royalblue",
    color_mode="single",
    normalize=True,
    grid=True,
    title=None,  # No title for this subplot
)

spike_plot = TimeSeriesRasterPlot(
    data=spike_data,
    sampling_rate=lfp_fs,  # Same sampling rate as LFP
    marker_type="|",
    marker_size=6,
    use_colormap=True,
    colormap="Dark2",
    title=None,  # No title for this subplot
)

# Create a shared time window for both plots
time_window = (2.0, 8.0)  # Show time from 2s to 8s

# Update both plots to use the same time window
lfp_plot.update(time_window=time_window)
spike_plot.update(time_window=time_window)

# Create a custom matplotlib figure with both plots
fig, axes = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)

# Plot on the custom axes
lfp_fig, lfp_ax = lfp_plot.plot(backend="mpl", ax=axes[0])
raster_fig, raster_ax = spike_plot.plot(backend="mpl", ax=axes[1])

# Add a title to the figure
fig.suptitle("Combined LFP and Spike Raster Visualization", fontsize=16)

# Customize the plot
axes[0].set_ylabel("LFP")
axes[0].set_title("Local Field Potential")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Spike Raster")

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Make room for the suptitle
plt.show()

# %%
# Test with larger dataset using Plotly and WebGL acceleration
print("Testing with larger datasets...")

# Generate large LFP dataset
large_lfp, high_fs = generate_test_data(
    duration=60.0,  # 60 seconds
    fs=30000.0,  # 30 kHz sampling rate
    n_channels=1,  # Single channel for simplicity
    chunk_size=30000,
)

# Generate large spike dataset (sparse data, should be efficient)
large_spike_data = generate_spike_data(
    duration=60.0,  # 60 seconds
    n_units=10,  # 10 units
    rate_range=(5, 20),  # 5-20 Hz firing rates
)

# Create LFP plot with plotly-resampler
large_lfp_plot = TimeSeriesPlot(
    data=large_lfp,
    sampling_rate=high_fs,
    title="Large LFP Dataset with Dynamic Resampling",
    color_mode="single",
    color="royalblue",
    normalize=True,
    grid=True,
    downsample=False,  # Let plotly-resampler handle it
)

# Create raster plot with WebGL acceleration
large_raster_plot = TimeSeriesRasterPlot(
    data=large_spike_data,
    sampling_rate=high_fs,
    title="Large Spike Dataset with WebGL Acceleration",
    marker_size=3,
    marker_type="|",
    use_colormap=True,
    colormap="viridis",
    alpha=0.7,
    show_legend=True,
)

# Render the LFP plot with resampler
print("Rendering large LFP plot with dynamic resampling...")
fig_large_lfp = large_lfp_plot.plot(
    backend="plotly",
    use_resampler=True,
    max_n_samples=10000,
)
fig_large_lfp.show()

# Render the raster plot with WebGL
print("Rendering large raster plot with WebGL acceleration...")
fig_large_raster = large_raster_plot.plot(
    backend="plotly",
    use_webgl=True,
)
fig_large_raster.show()

# %%
# Test time window updating (interactivity simulation)
print("Testing time window updates...")

# Create a plot with the full dataset
window_test_plot = TimeSeriesRasterPlot(
    data=spike_data,
    sampling_rate=1000.0,
    title="Raster Plot with Changing Time Windows",
    use_colormap=True,
    marker_size=5,
)

# Create a figure to compare different time windows
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# Plot full dataset
window_test_plot.update(time_window=None)  # Full view
window_test_plot.plot(backend="mpl", ax=axes[0])
axes[0].set_title("Full Time Range")

# Plot first half
window_test_plot.update(time_window=(0, 5))  # First half
window_test_plot.plot(backend="mpl", ax=axes[1])
axes[1].set_title("First Half (0-5s)")

# Plot second half
window_test_plot.update(time_window=(5, 10))  # Second half
window_test_plot.plot(backend="mpl", ax=axes[2])
axes[2].set_title("Second Half (5-10s)")

# Adjust layout
plt.tight_layout()
plt.show()

# %%
