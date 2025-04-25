# %%
# Time Series Area Chart Visualization
# ====================================
#
# This notebook demonstrates a complete workflow for visualizing
# time series area charts using the dspant_viz package

# Import necessary libraries
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Import dspant_viz components
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.visualization.stream.ts_area import TimeSeriesAreaPlot

# Apply themes
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")

# %%
# Set random seed for reproducibility
np.random.seed(42)


# Generate synthetic time series data with envelope-like characteristics
def generate_area_data(duration=30.0, fs=1000.0, n_channels=3, noise_level=0.1):
    """
    Generate synthetic time series data with envelope characteristics.

    Parameters
    ----------
    duration : float, optional
        Total duration of the recording
    fs : float, optional
        Sampling frequency
    n_channels : int, optional
        Number of channels to generate
    noise_level : float, optional
        Level of noise to add to the signal

    Returns
    -------
    da.Array
        Synthetic time series data
    """
    # Create time array
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Create signals with envelope-like characteristics
    data = np.zeros((n_samples, n_channels))

    for channel in range(n_channels):
        # Create base envelope signal
        base_freq = 2 * np.pi * (channel + 1) / 10  # Different frequencies
        envelope = np.abs(np.sin(base_freq * t)) * (1 + 0.5 * np.sin(base_freq * t / 5))

        # Add harmonic components
        signal = envelope * np.sin(base_freq * t)

        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        data[:, channel] = signal + noise

    # Convert to Dask array
    return da.from_array(data, chunks=(1000, 1))


# %%
# Generate area chart data
area_data = generate_area_data()

# %%
# 1. Test Basic Area Chart Visualization
# --------------------------------------

# Test area chart with default parameters
area_plot_default = TimeSeriesAreaPlot(
    data=area_data,
    sampling_rate=1000.0,
    fill_to="zero",
    fill_alpha=0.3,
    title="Default Area Chart",
)

# Plot with Matplotlib
plt.figure(figsize=(12, 6))
fig_mpl, ax_mpl = area_plot_default.plot(backend="mpl")
plt.show()

# Plot with Plotly
fig_plotly = area_plot_default.plot(backend="plotly", color_mode="colormap")
fig_plotly.show()

# %%
# 2. Test Different Fill Options
# ------------------------------

# Test different fill methods
fill_methods = [0.0, "zero", "min"]

for fill_method in fill_methods:
    area_plot = TimeSeriesAreaPlot(
        data=area_data,
        sampling_rate=1000.0,
        fill_to=fill_method,
        fill_alpha=0.4,
        color_mode="colormap",
        colormap="plasma",
        title=f"Area Chart - Fill to {fill_method}",
    )

    # Plot with Matplotlib
    plt.figure(figsize=(12, 6))
    fig_mpl, ax_mpl = area_plot.plot(backend="mpl")
    plt.show()

# %%
# 3. Test Time Window and Downsampling
# ------------------------------------

# Test time window and downsampling
area_plot_window = TimeSeriesAreaPlot(
    data=area_data,
    sampling_rate=1000.0,
    time_window=(5.0, 15.0),  # Focus on specific time range
    fill_to="zero",
    fill_alpha=0.2,
    downsample=True,
    max_points=5000,
    title="Area Chart with Time Window",
)

# Plot with Matplotlib
plt.figure(figsize=(12, 6))
fig_mpl, ax_mpl = area_plot_window.plot(backend="mpl")
plt.show()

# Plot with Plotly
fig_plotly = area_plot_window.plot(backend="plotly")
fig_plotly.show()

# %%
# Summary
# =======
# We've demonstrated a comprehensive workflow for area chart visualization:
#
# 1. Generated synthetic time series data with envelope characteristics
# 2. Visualized area charts with different fill methods
# 3. Tested both Matplotlib and Plotly backends
# 4. Explored time windowing and downsampling
#
# This showcases the flexibility of the TimeSeriesAreaPlot
# in dspant_viz for visualizing derivative or envelope-like data.
# %%
