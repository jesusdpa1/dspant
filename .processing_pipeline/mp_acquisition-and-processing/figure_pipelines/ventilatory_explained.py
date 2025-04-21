"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
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
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_absolute_threshold_detector,
)

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()
# %%
base_path = Path(os.getenv("DATA_DIR"))
data_path = base_path.joinpath(
    r"papers\2025_mp_emg diaphragm acquisition and processing\Sample Ventilator Trace"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_right_path = data_path.joinpath(r"emg_r.ant")
emg_left_path = data_path.joinpath(r"emg_l.ant")
insp_path = data_path.joinpath(r"insp.ant")
# %%
# Load EMG data
stream_emg_r = StreamNode(str(emg_right_path))
stream_emg_r.load_metadata()
stream_emg_r.load_data()
# Print stream_emg summary
stream_emg_r.summarize()


stream_emg_l = StreamNode(str(emg_left_path))
stream_emg_l.load_metadata()
stream_emg_l.load_data()
# Print stream_emg summary
stream_emg_l.summarize()


stream_insp = StreamNode(str(insp_path))
stream_insp.load_metadata()
stream_insp.load_data()
# Print stream_emg summary
stream_insp.summarize()


# %%
# Create and visualize filters before applying them
fs = stream_insp.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)
# %%
# bandpass_plot = bandpass_filter.plot_frequency_response()
# notch_plot = notch_filter.plot_frequency_response()
# lowpass_plot = lowpass_filter.plot_frequency_response()
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
filtered_data = da.concatenate([filtered_emg_r, filtered_emg_l, filtered_insp], axis=1)
# %%
multichannel_fig = plot_multi_channel_data(filtered_data, fs=fs, time_window=[100, 110])

# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=2)
tkeo_data = tkeo_processor.process(filtered_emg_r, fs=fs).persist()
st_processor = create_absolute_threshold_detector(
    tkeo_data.mean().compute(), min_duration=0.05
)

# %%
tkeo_fig = plot_multi_channel_data(tkeo_data, fs=fs, time_window=[100, 110])
# %%
st_epochs = st_processor.process(tkeo_data, fs=fs).compute()
st_processor.to_dataframe(st_epochs)


# %%
"""
time stamps
neural trigger on = 0 to 30
neural trigger off = 365 to 395
neural triggered tidal volume increased = 1090 - 1120
neural triggered tidal volume decreased = 1245 - 1275
"""


# %%

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

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

# Convert st_epochs to DataFrame if not already
if not isinstance(st_epochs, pl.DataFrame):
    epochs_df = st_processor.to_dataframe(st_epochs)
else:
    epochs_df = st_epochs

# Create figure with GridSpec for custom layout
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 1, height_ratios=[3, 1])

# Select time range to visualize (same as your filtered data)
plot_start_time = 1190  # seconds
plot_end_time = 1220  # seconds - adjust as needed

# Convert times to sample indices
plot_start_idx = int(plot_start_time * fs)
plot_end_idx = int(plot_end_time * fs)

# Extract the filtered EMG data and inspiratory data for this time range
emg_r_data = filtered_emg_r[plot_start_idx:plot_end_idx, 0]
emg_l_data = filtered_emg_l[plot_start_idx:plot_end_idx, 0]
insp_data = filtered_insp[plot_start_idx:plot_end_idx, 0]

# Calculate time array
time_array = np.arange(plot_end_idx - plot_start_idx) / fs

# ----- Upper plot: EMG signals with detections -----
ax_signals = fig.add_subplot(gs[0])

# Plot the filtered EMG signals
# Define vertical spacing parameters
y_spread = 1.0
y_offset = 0.0
norm_scale = 0.4

# Store channel positions for tick labels
channel_positions = []

# Normalize and plot each signal (use same approach as your existing code)
signals = [filtered_insp, filtered_emg_r, filtered_emg_l]
signal_labels = ["Insp. Pressure", "Right EMG", "Left EMG"]
colors = [palette[0], palette[1], palette[2]]

for idx, signal_data in enumerate(signals):
    # Calculate vertical offset for this channel
    channel_offset = y_offset + (len(signals) - 1 - idx) * y_spread
    channel_positions.append(channel_offset)

    # Extract data for this signal
    subset_data = signal_data[plot_start_idx:plot_end_idx, 0]

    # Normalize and apply offset
    max_amplitude = np.max(np.abs(subset_data))
    if max_amplitude > 0:
        norm_data = (
            subset_data / max_amplitude * (y_spread * norm_scale) + channel_offset
        )
    else:
        norm_data = np.zeros_like(subset_data) + channel_offset

    # Plot the data
    ax_signals.plot(
        time_array,
        norm_data,
        color=colors[idx],
        linewidth=1.2,
        label=signal_labels[idx],
        alpha=0.8,
    )

# Set up the signal plot
# ax_signals.set_xlim(plot_start_time, plot_end_time)
y_min = 0 - y_spread * 0.6
y_max = channel_positions[0] + y_spread * 0.6
ax_signals.set_ylim(y_min, y_max)
ax_signals.set_ylabel("Normalized Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_signals.tick_params(labelsize=TICK_SIZE)
ax_signals.set_title(
    "EMG Signals with Onset Detections", fontsize=SUBTITLE_SIZE, weight="bold"
)
ax_signals.legend(fontsize=TICK_SIZE, loc="upper right")

# Add y-tick labels at channel positions
ax_signals.set_yticks(channel_positions)
ax_signals.set_yticklabels(signal_labels)

# ----- Lower plot: Raster plot of detected onsets -----
ax_raster = fig.add_subplot(gs[1], sharex=ax_signals)

# Filter onsets within the plotting time range
mask = (epochs_df["onset_idx"] >= plot_start_idx) & (
    epochs_df["onset_idx"] < plot_end_idx
)
filtered_epochs = epochs_df.filter(mask)

# Normalize the amplitudes for visualization
if len(filtered_epochs) > 0:
    max_amp = filtered_epochs["amplitude"].max()
    min_amp = filtered_epochs["amplitude"].min()
    amp_range = max_amp - min_amp if max_amp > min_amp else 1.0

    # Plot scaled bars for each onset
    for i, row in enumerate(filtered_epochs.iter_rows(named=True)):
        onset_time = row["onset_idx"] / fs - plot_start_time
        duration = row["duration"]

        # Normalize amplitude to control bar height (0.3 to 1.0 range)
        normalized_height = (
            0.3 + ((row["amplitude"] - min_amp) / amp_range) * 0.7
            if amp_range > 0
            else 0.7
        )

        # Plot the onset as a vertical line/bar
        ax_raster.plot(
            [onset_time, onset_time],
            [0, normalized_height],
            linewidth=2,
            color=palette[1],
            solid_capstyle="round",
        )

        # Add small marker at the top of each line to make it more visible
        ax_raster.scatter(
            onset_time, normalized_height, color=palette[1], s=20, zorder=10
        )

# Set up the raster plot
ax_raster.set_xlim(plot_start_time, plot_end_time)
ax_raster.set_ylim(0, 1.1)
ax_raster.set_ylabel("Detection\nAmplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_raster.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_raster.tick_params(labelsize=TICK_SIZE)
ax_raster.set_yticks([0, 0.5, 1.0])
ax_raster.set_yticklabels(["Low", "Medium", "High"])
ax_raster.grid(True, axis="x", alpha=0.3)

# Add the detection count
detection_count = len(filtered_epochs)
ax_raster.text(
    0.98,
    0.92,
    f"Detected onsets: {detection_count}",
    transform=ax_raster.transAxes,
    ha="right",
    fontsize=12,
    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
)

# Add overall title
plt.suptitle(
    "EMG Onset Detection Analysis",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
