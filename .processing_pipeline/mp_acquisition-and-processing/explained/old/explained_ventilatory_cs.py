"""
Functions to extract onset detection - Updated with dspant_viz Visualization
Author: Jesus Penaloza
"""

# %%
import os
import time
from pathlib import Path

import dask.array as da
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.fir_filters import create_moving_average
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.double_threshold import (
    create_double_threshold_detector,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.visualization.events.events_test import EventAnnotator

# dspant_viz imports
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot
from dspant_viz.visualization.stream.ts_area import TimeSeriesAreaPlot

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()

# %%
base_path = Path(os.getenv("DATA_DIR"))
data_path = base_path.joinpath(
    r"papers\2025_mp_emg diaphragm acquisition and processing\Sample Ventilator Trace"
)

emg_right_path = data_path.joinpath(r"emg_r.ant")
emg_left_path = data_path.joinpath(r"emg_l.ant")
insp_path = data_path.joinpath(r"insp.ant")

# %%
# Load EMG data
stream_emg_r = StreamNode(str(emg_right_path))
stream_emg_r.load_metadata()
stream_emg_r.load_data()
stream_emg_r.summarize()

stream_emg_l = StreamNode(str(emg_left_path))
stream_emg_l.load_metadata()
stream_emg_l.load_data()
stream_emg_l.summarize()

stream_insp = StreamNode(str(insp_path))
stream_insp.load_metadata()
stream_insp.load_data()
stream_insp.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_emg_l.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)

# %%
# Create processing node with filters
processor_emg_r = create_processing_node(stream_emg_r)
processor_emg_l = create_processing_node(stream_emg_l)
processor_insp = create_processing_node(stream_insp)

# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
lowpass_processor = FilterProcessor(
    filter_func=lowpass_filter.get_filter_function(), overlap_samples=40
)

# %%
# Add processors to the processing node
processor_emg_r.add_processor([notch_processor, bandpass_processor], group="filters")
processor_emg_l.add_processor([notch_processor, bandpass_processor], group="filters")
processor_insp.add_processor([notch_processor, lowpass_processor], group="filters")

# Apply filters and plot results
filtered_emg_r = processor_emg_r.process(group=["filters"]).persist()
filtered_emg_l = processor_emg_l.process(group=["filters"]).persist()
filtered_insp = processor_insp.process(group=["filters"]).persist()

# %%
# Combine filtered data for visualization
filtered_data = da.concatenate([filtered_emg_r, filtered_emg_l, filtered_insp], axis=1)

# %%
# Visualize filtered data with dspant_viz
# Apply themes for better visualization
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")

# Create TimeSeriesPlot
filtered_ts_plot = TimeSeriesPlot(
    data=filtered_data,
    sampling_rate=fs,
    title="Filtered EMG and Inspiration Signals",
    time_window=[100, 110],  # Same as original plot
    normalize=True,
    color_mode="colormap",
    colormap="viridis",
)

# Plot with Matplotlib
plt.figure(figsize=(15, 8))
fig_mpl, ax_mpl = filtered_ts_plot.plot(backend="mpl")
plt.show()

# %%
# TKEO Processing
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
tkeo_data = tkeo_processor.process(filtered_emg_l, fs=fs).persist()

# Normalize TKEO data
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()
zscore_insp = zscore_processor.process(filtered_insp).persist()

# %%
# Activity Detection
st_tkeo_processor = create_double_threshold_detector(
    primary_threshold=0.035,
    secondary_threshold=0.045,
    refractory_period=0.02,
    min_contraction_duration=0.01,
)

st_insp_processor = create_double_threshold_detector(
    primary_threshold=0.1,
    secondary_threshold=0.2,
    secondary_points_required=100,
    refractory_period=0.02,
    min_contraction_duration=0.1,
)

# %%
# Detect epochs
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)

insp_epochs = st_insp_processor.process(zscore_insp, fs=fs).compute()
insp_epochs = st_insp_processor.to_dataframe(insp_epochs)

# %%
# Visualize TKEO with Activity Annotations
# Prepare event annotations
tkeo_events = {
    "start": tkeo_epochs["onset_idx"].to_numpy() / fs,
    "end": tkeo_epochs["offset_idx"].to_numpy() / fs,
    "label": [f"Event {i + 1}" for i in range(len(tkeo_epochs))],
}

# Create TimeSeriesAreaPlot for TKEO envelope
tkeo_area_plot = TimeSeriesAreaPlot(
    data=zscore_tkeo,
    sampling_rate=fs,
    title="TKEO Envelope with Activity Epochs",
    time_window=[100, 110],
    fill_to="zero",
    fill_alpha=0.3,
    color_mode="single",
    color="royalblue",
)

# # Create event annotator
# tkeo_event_annotator = EventAnnotator(
#     events=tkeo_events,
#     highlight_color="red",
#     marker_style="span",
#     alpha=0.2,
#     label_events=True,
# )

# Plot with Matplotlib
plt.figure(figsize=(15, 6))
fig_area_mpl, ax_area_mpl = tkeo_area_plot.plot(backend="mpl")

# Add event annotations
# tkeo_event_annotator.plot(backend="mpl", ax=ax_area_mpl)
# plt.show()

# %%
# Calculate instantaneous frequency
# Calculate the onset times
onset_times = tkeo_epochs["onset_idx"].to_numpy() / fs

# Calculate the time differences between onsets
time_diffs = np.diff(onset_times)

# Calculate the instantaneous frequency (in Hz)
instantaneous_freq = 1 / time_diffs

# Create an array of time points that correspond to the original signal
time_array = np.arange(len(filtered_emg_l)) / fs

# Optional: Visualize instantaneous frequency
plt.figure(figsize=(12, 4))
plt.plot(onset_times[1:], instantaneous_freq, marker="o")
plt.title("Instantaneous Frequency of EMG Activations")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the style
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Simulation parameters
total_duration = 3600  # 1 hour in seconds
time_window = 60  # 60-second windows

# Generate time ranges
time_ranges = list(range(0, total_duration + time_window, time_window))
time_ranges = [
    (time_ranges[i], time_ranges[i + 1]) for i in range(len(time_ranges) - 1)
]


# Create onset frequency data with variation
def generate_onset_frequencies(time_ranges, min_freq=1, max_freq=5):
    frequencies = []
    for start, end in time_ranges:
        # Linear interpolation between min and max frequencies
        base_freq = np.interp(start, [0, total_duration], [min_freq, max_freq])

        # Add random variation
        variation = np.random.normal(0, 0.5)

        # Combine base frequency with variation
        freq = np.clip(base_freq + variation, min_freq, max_freq)

        frequencies.append(
            {
                "start_time": start,
                "end_time": end,
                "open": freq,
                "close": freq + np.random.normal(0, 0.5),
                "high": freq + np.abs(np.random.normal(0, 0.3)),
                "low": freq - np.abs(np.random.normal(0, 0.3)),
            }
        )

    return pd.DataFrame(frequencies)


# Generate onset frequency data
onset_data = generate_onset_frequencies(time_ranges)

# Calculate moving average
window_size = 5  # 5-window moving average
moving_avg = onset_data["open"].rolling(window=window_size, center=True).mean()

# Create figure with careful sizing
fig, ax = plt.subplots(figsize=(16, 8))

# Color selection from palette
up_color = palette[2]  # Green-like color for up candles
down_color = palette[3]  # Red-like color for down candles
moving_avg_color = palette[1]  # Third color for moving average


# Custom candlestick drawing
def draw_candlestick(ax, x, open_price, close_price, high, low, width=30, alpha=0.8):
    # Determine candle color
    if close_price >= open_price:
        color = up_color
        body_low = open_price
        body_high = close_price
    else:
        color = down_color
        body_low = close_price
        body_high = open_price

    # Draw the candle body (thick rectangle)
    body_width = width
    ax.add_patch(
        plt.Rectangle(
            (x - body_width / 2, body_low),
            body_width,
            body_high - body_low,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
    )

    # Draw the wick (thin line)
    ax.plot([x, x], [low, high], color="black", linewidth=1)


# Plot candlesticks
for i, row in onset_data.iterrows():
    draw_candlestick(
        ax, row["start_time"], row["open"], row["close"], row["high"], row["low"]
    )

# Plot moving average
ax.plot(
    onset_data["start_time"],
    moving_avg,
    color=moving_avg_color,
    linewidth=3,
    label=f"{window_size}-window Moving Average",
)

# Customize the plot
ax.set_title("Onset Frequency Variation", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Onset Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

# Set x-axis limits
ax.set_xlim(0, total_duration)

# Adjust tick sizes
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

# Add grid with slight transparency
ax.grid(True, linestyle="--", alpha=0.6)

# Add legend
ax.legend(fontsize=TICK_SIZE)

# Tight layout with some padding
plt.tight_layout()

# Print summary statistics
print("Onset Frequency Data Summary:")
print(f"Total duration: {total_duration} seconds")
print(f"Window size: {time_window} seconds")
print(f"Average onset frequency: {onset_data['open'].mean():.2f} Hz")
print(f"Minimum onset frequency: {onset_data['low'].min():.2f} Hz")
print(f"Maximum onset frequency: {onset_data['high'].max():.2f} Hz")
print(f"Moving Average Window Size: {window_size} windows")

# Show the plot
plt.show()

# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import savgol_filter

# Set the colorblind-friendly palette and styling
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")
sns.set_theme(style="darkgrid")

# Try to set Montserrat font if available
try:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Montserrat", "Arial"]
except:
    pass  # Fall back to default fonts if Montserrat not available

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 10

# Better color selection from palette
up_color = palette[2]  # Green-like color for up candles
down_color = palette[3]  # Red-like color for down candles
avg_line_color = palette[5]  # Third color for average line
trend_line_color = palette[0]  # Fourth color for trendline

# Convert onset_idx to seconds
onset_times = tkeo_epochs["onset_idx"].to_numpy() / fs

# Calculate time differences between consecutive onsets (in seconds)
onset_intervals = np.diff(onset_times)

# Apply Savitzky-Golay filter to smooth the data
window_length = 151  # Must be odd and less than data length
polyorder = 3  # Polynomial order for the filter
if len(onset_intervals) > window_length:
    smoothed_intervals = savgol_filter(onset_intervals, window_length, polyorder)
else:
    # Fall back to simpler moving average for short data
    from scipy.ndimage import uniform_filter1d

    smoothed_intervals = uniform_filter1d(
        onset_intervals, size=min(5, len(onset_intervals))
    )

# Calculate moving average similar to simulation
window_size = 5
moving_avg = []
for i in range(len(smoothed_intervals)):
    start = max(0, i - window_size // 2)
    end = min(len(smoothed_intervals), i + window_size // 2 + 1)
    moving_avg.append(np.mean(smoothed_intervals[start:end]))
moving_avg = np.array(moving_avg)

# For candlestick visualization with reduced frequency
sampling_factor = 25  # Show every 25th point for clearer visualization
candle_data = []

for i in range(sampling_factor, len(smoothed_intervals), sampling_factor):
    # Using current and N samples ago to define open/close for correct directionality
    idx_prev = i - sampling_factor

    # This is the key change - we compare adjacent points to determine up/down
    # Therefore the open is the PREVIOUS value
    open_val = smoothed_intervals[idx_prev]
    # And close is the CURRENT value
    close_val = smoothed_intervals[i]

    # Find min/max within this segment for wicks
    segment = smoothed_intervals[idx_prev : i + 1]
    high = np.max(segment) * 1.02  # Slightly extend to make wicks visible
    low = np.min(segment) * 0.98  # Slightly extend to make wicks visible

    # Use the actual time for the x-coordinate
    x_time = onset_times[i]
    candle_data.append((x_time, open_val, close_val, high, low))

# Create figure and axis with better proportions
fig, ax = plt.subplots(figsize=(16, 8))


# Custom candlestick drawing function with FIXED comparison logic
def draw_candlestick(ax, x, open_price, close_price, high, low, width=0.5, alpha=0.8):
    # Determine candle color correctly
    if close_price > open_price:  # CLOSING HIGHER than opening = green/up
        color = up_color
        body_low = open_price
        body_high = close_price
    else:  # CLOSING LOWER than opening = red/down
        color = down_color
        body_low = close_price
        body_high = open_price

    # Draw the candle body (thick rectangle)
    body_width = width
    ax.add_patch(
        plt.Rectangle(
            (x - body_width / 2, body_low),
            body_width,
            body_high - body_low,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
    )

    # Draw the wick (thin line)
    ax.plot([x, x], [low, high], color="black", linewidth=1)


# Plot smoothed trend line in the background
ax.plot(
    onset_times[sampling_factor::sampling_factor],
    smoothed_intervals[sampling_factor::sampling_factor],
    color=trend_line_color,
    alpha=0.3,
    linewidth=1,
    label="Smoothed Trend",
)

# Plot moving average
ax.plot(
    onset_times[sampling_factor::sampling_factor],
    moving_avg[sampling_factor::sampling_factor],
    color=avg_line_color,
    linewidth=2.5,
    label=f"{window_size}-window Moving Average",
)

# Calculate candlestick width dynamically
# Use the median interval between sampled points as width
candle_times = [x for x, _, _, _, _ in candle_data]
time_intervals = np.diff(candle_times)
typical_width = np.median(time_intervals) * 0.8

# Plot candlesticks with proper comparison logic
for x, open_price, close_price, high, low in candle_data:
    draw_candlestick(ax, x, open_price, close_price, high, low, width=typical_width)

# Calculate average and respiratory rate
avg_interval = np.mean(onset_intervals)
resp_rate = 60 / avg_interval

# Configure the plot with better styling
ax.set_xlim(onset_times[0] - 1, onset_times[-1] + 1)
y_min = max(0, np.min(smoothed_intervals) * 0.9)
y_max = np.max(smoothed_intervals) * 1.1
ax.set_ylim(y_min, y_max)

# Add labeled average line
ax.axhline(
    avg_interval,
    color=palette[4],
    linestyle="dashed",
    linewidth=2,
    label=f"Average: {avg_interval:.2f}s ({resp_rate:.1f} breaths/min)",
)

# Better axis labels and title
ax.set_title(
    "Respiratory Cycle Intervals from EMG Data", fontsize=TITLE_SIZE, fontweight="bold"
)
ax.set_xlabel(
    "Time (seconds)",
    fontsize=AXIS_LABEL_SIZE,
    fontweight="bold",
)
ax.set_ylabel(
    "Interval Duration (seconds)", fontsize=AXIS_LABEL_SIZE, fontweight="bold"
)

# Adjust tick sizes
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

# Add grid with slight transparency
ax.grid(True, linestyle="--", alpha=0.6)

# Add legend with better positioning
ax.legend(loc="upper right", fontsize=TICK_SIZE)

# Add statistics text box
stats_text = (
    f"Statistics:\n"
    f"Total breaths: {len(onset_intervals)}\n"
    f"Avg interval: {avg_interval:.2f}s\n"
    f"Resp. rate: {resp_rate:.1f} breaths/min\n"
    f"Min interval: {np.min(onset_intervals):.2f}s\n"
    f"Max interval: {np.max(onset_intervals):.2f}s"
)

# Add the statistics box with nicer styling
props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
ax.text(
    0.02,
    0.97,
    stats_text,
    transform=ax.transAxes,
    fontsize=TICK_SIZE,
    verticalalignment="top",
    bbox=props,
)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.signal import savgol_filter

# Set the colorblind-friendly palette and styling
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")
sns.set_theme(style="darkgrid")

# Try to set Montserrat font if available
try:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Montserrat", "Arial"]
except:
    pass  # Fall back to default fonts if Montserrat not available

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 10

# Better color selection from palette
up_color = palette[2]  # Green-like color for up candles
down_color = palette[3]  # Red-like color for down candles
avg_line_color = palette[5]  # Third color for average line
trend_line_color = palette[5]  # Fourth color for trendline
tkeo_color = palette[0]  # TKEO color
insp_color = palette[1]  # Inspiration color

# Determine time range
start_time = 800  # seconds
end_time = 1800  # 20-minute window

# Convert to sample indices
start_idx = int(start_time * fs)
end_idx = int(end_time * fs)

# Prepare time array
time_array = np.arange(start_idx, end_idx) / fs - start_time

# Prepare signals
zscore_tkeo_np = (
    zscore_tkeo[start_idx:end_idx].compute()
    if hasattr(zscore_tkeo, "compute")
    else zscore_tkeo[start_idx:end_idx]
)
zscore_insp_np = (
    zscore_insp[start_idx:end_idx].compute()
    if hasattr(zscore_insp, "compute")
    else zscore_insp[start_idx:end_idx]
)

# Ensure 1D arrays
if zscore_tkeo_np.ndim > 1:
    zscore_tkeo_np = zscore_tkeo_np[:, 0]
if zscore_insp_np.ndim > 1:
    zscore_insp_np = zscore_insp_np[:, 0]

# Convert onset_idx to seconds
onset_times = tkeo_epochs["onset_idx"].to_numpy() / fs

# Calculate time differences between consecutive onsets (in seconds)
onset_intervals = np.diff(onset_times)

# Apply Savitzky-Golay filter to smooth the data
window_length = 151  # Must be odd and less than data length
polyorder = 3  # Polynomial order for the filter
if len(onset_intervals) > window_length:
    smoothed_intervals = savgol_filter(onset_intervals, window_length, polyorder)
else:
    # Fall back to simpler moving average for short data
    from scipy.ndimage import uniform_filter1d

    smoothed_intervals = uniform_filter1d(
        onset_intervals, size=min(5, len(onset_intervals))
    )

# Calculate moving average similar to simulation
window_size = 5
moving_avg = []
for i in range(len(smoothed_intervals)):
    start = max(0, i - window_size // 2)
    end = min(len(smoothed_intervals), i + window_size // 2 + 1)
    moving_avg.append(np.mean(smoothed_intervals[start:end]))
moving_avg = np.array(moving_avg)

# For candlestick visualization with reduced frequency
sampling_factor = 25  # Show every 25th point for clearer visualization
candle_data = []

for i in range(sampling_factor, len(smoothed_intervals), sampling_factor):
    # Using current and N samples ago to define open/close for correct directionality
    idx_prev = i - sampling_factor

    # This is the key change - we compare adjacent points to determine up/down
    # Therefore the open is the PREVIOUS value
    open_val = smoothed_intervals[idx_prev]
    # And close is the CURRENT value
    close_val = smoothed_intervals[i]

    # Find min/max within this segment for wicks
    segment = smoothed_intervals[idx_prev : i + 1]
    high = np.max(segment) * 1.02  # Slightly extend to make wicks visible
    low = np.min(segment) * 0.98  # Slightly extend to make wicks visible

    # Use the actual time for the x-coordinate
    x_time = onset_times[i]
    candle_data.append((x_time, open_val, close_val, high, low))

# Create figure with GridSpec
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1])

# Candlestick plot in the top subplot
ax_candle = fig.add_subplot(gs[0])


# Custom candlestick drawing function with FIXED comparison logic
def draw_candlestick(ax, x, open_price, close_price, high, low, width=0.5, alpha=0.8):
    # Determine candle color correctly
    if close_price > open_price:  # CLOSING HIGHER than opening = green/up
        color = up_color
        body_low = open_price
        body_high = close_price
    else:  # CLOSING LOWER than opening = red/down
        color = down_color
        body_low = close_price
        body_high = open_price

    # Draw the candle body (thick rectangle)
    body_width = width
    ax.add_patch(
        Rectangle(
            (x - body_width / 2, body_low),
            body_width,
            body_high - body_low,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
    )

    # Draw the wick (thin line)
    ax.plot([x, x], [low, high], color="black", linewidth=1)


# Calculate candlestick width dynamically
candle_times = [x for x, _, _, _, _ in candle_data]
time_intervals = np.diff(candle_times)
typical_width = np.median(time_intervals) * 0.8

# Plot smoothed trend line in the background
ax_candle.plot(
    onset_times[sampling_factor::sampling_factor],
    smoothed_intervals[sampling_factor::sampling_factor],
    color=trend_line_color,
    alpha=0.3,
    linewidth=1,
    label="Smoothed Trend",
)

# Plot moving average
ax_candle.plot(
    onset_times[sampling_factor::sampling_factor],
    moving_avg[sampling_factor::sampling_factor],
    color=avg_line_color,
    linewidth=2.5,
    label=f"{window_size}-window Moving Average",
)

# Plot candlesticks
for x, open_price, close_price, high, low in candle_data:
    draw_candlestick(
        ax_candle, x, open_price, close_price, high, low, width=typical_width
    )

# Calculate average and respiratory rate
avg_interval = np.mean(onset_intervals)
resp_rate = 60 / avg_interval

# Configure the candlestick plot
ax_candle.set_xlim(start_time, end_time)
y_min = max(0, np.min(smoothed_intervals) * 0.9)
y_max = np.max(smoothed_intervals) * 1.1
ax_candle.set_ylim(y_min, y_max)

# Add labeled average line
ax_candle.axhline(
    avg_interval,
    color=palette[7],
    linestyle="dashed",
    linewidth=2,
    label=f"Average: {avg_interval:.2f}s ({resp_rate:.1f} breaths/min)",
)

ax_candle.set_title(
    "Respiratory Cycle Intervals", fontsize=SUBTITLE_SIZE, fontweight="bold"
)
ax_candle.set_ylabel("Interval Duration (s)", fontsize=AXIS_LABEL_SIZE)
ax_candle.set_xlabel("")
ax_candle.legend(loc="upper right", fontsize=TICK_SIZE)

# Inspiration signal plot
ax_insp = fig.add_subplot(gs[1])
ax_insp.plot(time_array, zscore_insp_np, color=insp_color, linewidth=2)
ax_insp.set_xlim(0, end_time - start_time)
ax_insp.set_title(
    "Inspiratory Pressure [Ventilation]", fontsize=SUBTITLE_SIZE, fontweight="bold"
)
ax_insp.set_ylabel("MinMax Norm", fontsize=AXIS_LABEL_SIZE)
ax_insp.set_xlabel("")

# TKEO signal plot
ax_tkeo = fig.add_subplot(gs[2])
ax_tkeo.plot(time_array, zscore_tkeo_np, color=tkeo_color, linewidth=2)
ax_tkeo.set_xlim(0, end_time - start_time)
ax_tkeo.set_title("EMG TKEO Signal", fontsize=SUBTITLE_SIZE, fontweight="bold")
ax_tkeo.set_ylabel("MinMax Norm", fontsize=AXIS_LABEL_SIZE)
ax_tkeo.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE)

# Add overall title
plt.suptitle(
    "Respiratory and TKEO Signal Analysis", fontsize=TITLE_SIZE, fontweight="bold"
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
# %%
