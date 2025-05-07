# %%
import time

import dask.array as da
import numpy as np
from IPython.display import display

from dspant_viz.core.themes_manager import apply_plotly_theme

# Import the updated TimeSeriesPlot
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot

# Apply theme
apply_plotly_theme("seaborn")


# %%
# Generate a 30-minute recording at 24kHz with 4 channels
def generate_large_test_data(
    duration=1600.0, fs=24000.0, n_channels=16, chunk_size=500000
):
    print(
        f"Generating a {duration / 60:.1f} minute recording at {fs / 1000:.1f} kHz with {n_channels} channels..."
    )

    # Calculate total samples
    n_samples = int(duration * fs)
    print(f"Total samples: {n_samples:,}")

    # Start timing
    start_time = time.time()

    # Generate frequencies for each channel
    np.random.seed(42)
    frequencies = np.random.randint(5, 41, size=n_channels)
    print(f"Channel frequencies: {frequencies} Hz")

    # Create initial time array (we'll use a small segment approach for memory efficiency)
    segment_size = min(10_000_000, n_samples)  # 10M samples max per segment
    n_segments = (n_samples + segment_size - 1) // segment_size

    # Initialize the Dask array with zeros
    data = da.zeros((n_samples, n_channels), chunks=(chunk_size, n_channels))

    # Generate data in segments to avoid memory issues
    for seg in range(n_segments):
        print(f"Generating segment {seg + 1}/{n_segments}...")

        # Calculate segment boundaries
        start_idx = seg * segment_size
        end_idx = min((seg + 1) * segment_size, n_samples)
        seg_samples = end_idx - start_idx

        # Create time for this segment
        t = np.linspace(start_idx / fs, end_idx / fs, seg_samples, endpoint=False)

        # Create segment data
        seg_data = np.zeros((seg_samples, n_channels))

        for i in range(n_channels):
            # Base sine wave
            seg_data[:, i] = np.sin(2 * np.pi * frequencies[i] * t)

            # Add harmonics
            seg_data[:, i] += 0.3 * np.sin(2 * np.pi * 2 * frequencies[i] * t)

            # Add different features in different time regions
            if i % 2 == 0:
                # Add bursts in some regions
                burst_indices = np.logical_and(t % 60 > 10, t % 60 < 15)
                seg_data[burst_indices, i] *= 2.0

            # Add noise
            noise_level = 0.1 + 0.05 * i
            seg_data[:, i] += np.random.normal(0, noise_level, seg_samples)

        # Assign to the appropriate segment of the dask array
        data[start_idx:end_idx, :] = da.from_array(seg_data)

    end_time = time.time()
    print(f"Data generation completed in {end_time - start_time:.2f} seconds")

    return data, fs


# %%
# Generate the data
data, sampling_rate = generate_large_test_data()

# %%
# Create the TimeSeriesPlot with initial time window
ts_plot = TimeSeriesPlot(
    data=data,
    sampling_rate=sampling_rate,
    title="30-Minute Neural Recording (24 kHz, 4 channels)",
    color_mode="colormap",
    colormap="viridis",
    normalize=True,
    grid=True,
    time_window=(0, 1600),  # Full 30 minutes
    initial_time_window=(0, 10),  # Initial view: first 10 seconds
    resample_method="lttb",  # Use LTTB for better visual quality
)

# Plot with Plotly backend
print("Rendering with Plotly (initial view shows first 10 seconds)...")
fig_plotly = ts_plot.plot(
    backend="plotly",
    use_resampler=True,
    max_n_samples=10000,
    use_webgl=True,
)
# %%
# Show the plot
display(fig_plotly)

# %%
