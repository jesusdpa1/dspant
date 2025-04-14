"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import time

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
from dspant.processors.filters import (
    FilterProcessor,
    create_ffc_notch,
    create_wp_harmonic_removal,
)
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.vizualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\papers\2025_mp_emg diaphragm acquisition and processing"
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_stream_path = base_path + r"/noisy_recording.ant"

# %%
# Load EMG data
stream_emg = StreamNode(emg_stream_path)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

fs = stream_emg.fs  # Get sampling rate from the stream node
# %%
# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
# %%
# Create processing node with filters
processor_emg = create_processing_node(stream_emg)
# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
# %%
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
# %%
# View summary of the processing node
processor_emg.summarize()
# %%
ffc_filter = create_ffc_notch(60)
whp_filter = create_wp_harmonic_removal(60)
# %%
# Apply filters and plot results
raw_data = stream_emg.data.persist()
filter_data = processor_emg.process(group=["filters"]).persist()
ffc_data = ffc_filter.process(filter_data, fs).persist()
whp_data = whp_filter.process(filter_data[0:1000000], fs).persist()
# %%
start = int(fs * 5)
end = int(fs * 10)
base_data = filter_data[start:end, :]

# %%
raw_fig = plot_multi_channel_data(raw_data, fs=fs, time_window=[0, 10])
filtered_fig = plot_multi_channel_data(filter_data, fs=fs, time_window=[0, 10])
ffc_fig = plot_multi_channel_data(ffc_data, fs=fs, time_window=[0, 10])
whp_fig = plot_multi_channel_data(whp_data, fs=fs, time_window=[0, 10])
# %%
import matplotlib.pyplot as plt
import numpy as np
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

# Define data range
start = 0
end = int(10 * fs)
zoom_start = int(1.9 * fs)
zoom_end = int(2.2 * fs)

# Get time values for the zoom highlight box
zoom_start_time = zoom_start / fs
zoom_end_time = zoom_end / fs
zoom_width = zoom_end_time - zoom_start_time

# Create figure with GridSpec for custom layout
# 4 rows, 5 columns with the right side being 1/4 of the left
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1])

# Darker grey with navy tint
dark_grey_navy = "#2D3142"

# Choose a distinct color for the highlight box (orange from the colorblind palette)
highlight_color = palette[0]  # Using a distinct color from the palette

# Calculate time arrays
time_array = np.arange(end - start) / fs
zoom_time_array = np.arange(zoom_end - zoom_start) / fs

# Plot 1: Original Raw Data (spanning first 4 columns)
ax_raw = fig.add_subplot(gs[0, 0:4])
ax_raw.plot(time_array, raw_data[start:end, 0], color=dark_grey_navy, linewidth=2)
ax_raw.set_xlim(0, 10)
ax_raw.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_raw.tick_params(labelsize=TICK_SIZE)
ax_raw.set_title("Raw EMG Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 2: Bandpass Filtered Data
ax_bp = fig.add_subplot(gs[1, 0:4])
ax_bp.plot(time_array, filter_data[start:end, 0], color=palette[0], linewidth=2)
ax_bp.set_xlim(0, 10)
ax_bp.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_bp.tick_params(labelsize=TICK_SIZE)
ax_bp.set_title(
    "Bandpass + Notch Filtered Signal", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Plot 3: FFC Filtered Data
ax_ffc = fig.add_subplot(gs[2, 0:4])
ax_ffc.plot(time_array, ffc_data[start:end, 0], color=palette[1], linewidth=2)
ax_ffc.set_xlim(0, 10)
ax_ffc.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_ffc.tick_params(labelsize=TICK_SIZE)
ax_ffc.set_title("FFC Notch Filtered Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for FFC zoomed region using the zoom variables
y_min, y_max = ax_ffc.get_ylim()
height = y_max - y_min
ffc_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.2,
)
ax_ffc.add_patch(ffc_rect)

# Plot 4: Wavelet Pack Harmonic Removal
ax_whp = fig.add_subplot(gs[3, 0:4])
ax_whp.plot(
    time_array[: len(whp_data)],
    whp_data[start : min(end, len(whp_data)), 0],
    color=palette[2],
    linewidth=2,
)
ax_whp.set_xlim(0, 10)
ax_whp.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_whp.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_whp.tick_params(labelsize=TICK_SIZE)
ax_whp.set_title(
    "Wavelet Packet Harmonic Removal", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Add highlight box for Wavelet zoomed region using the zoom variables
y_min, y_max = ax_whp.get_ylim()
height = y_max - y_min
whp_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.2,
)
ax_whp.add_patch(whp_rect)

# Plot 5: FFC Zoomed (spanning 2 rows on the right)
ax_ffc_zoom = fig.add_subplot(gs[0:2, 4])
ax_ffc_zoom.plot(
    zoom_time_array, ffc_data[zoom_start:zoom_end, 0], color=palette[1], linewidth=2
)
ax_ffc_zoom.set_xlim(0, zoom_width)
ax_ffc_zoom.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_ffc_zoom.tick_params(labelsize=TICK_SIZE)
ax_ffc_zoom.set_title("FFC Filter (Zoomed)", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 6: Wavelet Zoomed (spanning 2 rows on the right)
ax_whp_zoom = fig.add_subplot(gs[2:4, 4])
ax_whp_zoom.plot(
    zoom_time_array[: min(len(zoom_time_array), zoom_end - zoom_start)],
    whp_data[zoom_start : min(zoom_end, len(whp_data)), 0],
    color=palette[2],
    linewidth=2,
)
ax_whp_zoom.set_xlim(0, zoom_width)
ax_whp_zoom.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_whp_zoom.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_whp_zoom.tick_params(labelsize=TICK_SIZE)
ax_whp_zoom.set_title("Wavelet Filter (Zoomed)", fontsize=SUBTITLE_SIZE, weight="bold")

# Add overall title
plt.suptitle(
    "EMG Signal Filtering Comparison", fontsize=TITLE_SIZE, fontweight="bold", y=0.98
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

plt.show()

# %%
