"""
Functions to showcase stim closed loop
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
    r"E:\jpenalozaa\papers\2025_mp_emg diaphragm acquisition and processing\drv_15-40-17_stim"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_path = data_path.joinpath(r"RawG.ant")
stim_path = data_path.joinpath(r"MonA.ant")
# insp_path = data_path.joinpath(r"insp.ant")
# %%
# Load EMG data
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()


stream_stim = StreamNode(str(stim_path))
stream_stim.load_metadata()
stream_stim.load_data()
# Print stream_emg summary
stream_stim.summarize()


# %%
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

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
processor_emg = create_processing_node(stream_emg)
# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters and plot results
filtered_emg = processor_emg.process(group=["filters"]).persist()

# %%
filtered_data = da.concatenate([filtered_emg[:, 1:2], stream_stim.data[:, :1]], axis=1)

# %%

multichannel_fig = plot_multi_channel_data(filtered_data, fs=fs, time_window=[400, 440])

# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()

# %%
tkeo_fig = plot_multi_channel_data(tkeo_data, fs=fs, time_window=[100, 110])
# %%

data_organized = da.concatenate(
    [
        stream_stim.data[:43200500, :1],
        tkeo_data[:, 1:2].compute(),
        filtered_emg[:43200500, 1:2],
    ],
    axis=1,
)
# %%
agg_fig = plot_multi_channel_data(data_organized, fs=fs, time_window=[400, 800])
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

# Assuming we have these variables from the original code:
# stream_stim.data, tkeo_data, filtered_emg, fs

# Define data range for full view and zoomed view
full_duration = 100  # seconds
zoom_duration = 5  # seconds

start = int(400 * fs)
end = int(start + (full_duration * fs))

# The key change: calculate zoom start/end in relation to the time axis, not sample indices
zoom_start_sample = int(430 * fs)  # Start zoom at 440s in absolute time
zoom_end_sample = int(zoom_start_sample + (zoom_duration * fs))

# Convert to relative time for plotting - this is crucial
zoom_start_time = (zoom_start_sample - start) / fs  # Time relative to the plotted start
zoom_end_time = (zoom_end_sample - start) / fs
zoom_width = zoom_end_time - zoom_start_time

# Calculate time arrays
time_array = np.arange(end - start) / fs
zoom_time_array = np.arange(zoom_end_sample - zoom_start_sample) / fs

# Define data arrays for plotting
stim_data_full = stream_stim.data[start:end, 0]
tkeo_data_full = tkeo_data[start:end, 1]
emg_data_full = filtered_emg[start:end, 1]

stim_data_zoom = stream_stim.data[zoom_start_sample:zoom_end_sample, 0]
tkeo_data_zoom = tkeo_data[zoom_start_sample:zoom_end_sample, 1]
emg_data_zoom = filtered_emg[zoom_start_sample:zoom_end_sample, 1]

# Create figure with GridSpec for custom layout
# 6 rows, 6 columns with the right side being 1/3 of the width
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(6, 6, width_ratios=[1, 1, 1, 1, 1, 1])

# Dark grey with navy tint for main lines
dark_grey_navy = "#2D3142"

# Choose a distinct color for the highlight box
highlight_color = palette[3]  # Using a distinct color from the palette

# Left column plots (spans first 4 columns)

# Plot 1: Stimulation Data (top)
ax_stim = fig.add_subplot(gs[0:2, 0:4])
ax_stim.plot(time_array, stim_data_full, color=dark_grey_navy, linewidth=2)
ax_stim.set_xlim(0, full_duration)
ax_stim.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_stim.tick_params(labelsize=TICK_SIZE)
ax_stim.set_title("Stimulation Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for zoomed region - now using correct time positioning
y_min, y_max = ax_stim.get_ylim()
height = y_max - y_min
stim_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.2,
)
ax_stim.add_patch(stim_rect)

# Plot 2: TKEO Data (middle)
ax_tkeo = fig.add_subplot(gs[2:4, 0:4])
ax_tkeo.plot(time_array, tkeo_data_full, color=palette[1], linewidth=2)
ax_tkeo.set_xlim(0, full_duration)
ax_tkeo.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_tkeo.tick_params(labelsize=TICK_SIZE)
ax_tkeo.set_title("TKEO Envelope", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for zoomed region
y_min, y_max = ax_tkeo.get_ylim()
height = y_max - y_min
tkeo_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.2,
)
ax_tkeo.add_patch(tkeo_rect)

# Plot 3: EMG Channel (bottom)
ax_emg = fig.add_subplot(gs[4:6, 0:4])
ax_emg.plot(time_array, emg_data_full, color=palette[2], linewidth=2)
ax_emg.set_xlim(0, full_duration)
ax_emg.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_emg.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_emg.tick_params(labelsize=TICK_SIZE)
ax_emg.set_title("Filtered EMG Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for zoomed region
y_min, y_max = ax_emg.get_ylim()
height = y_max - y_min
emg_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.2,
)
ax_emg.add_patch(emg_rect)

# Right column plots (spans last 2 columns)

# Plot 4: Stimulation Zoomed (top)
ax_stim_zoom = fig.add_subplot(gs[0:2, 4:6])
ax_stim_zoom.plot(zoom_time_array, stim_data_zoom, color=dark_grey_navy, linewidth=2)
ax_stim_zoom.set_xlim(0, zoom_duration)
ax_stim_zoom.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_stim_zoom.tick_params(labelsize=TICK_SIZE)
ax_stim_zoom.set_title(
    "Stimulation Signal (Zoomed)", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Plot 5: TKEO Zoomed (middle)
ax_tkeo_zoom = fig.add_subplot(gs[2:4, 4:6])
ax_tkeo_zoom.plot(zoom_time_array, tkeo_data_zoom, color=palette[1], linewidth=2)
ax_tkeo_zoom.set_xlim(0, zoom_duration)
ax_tkeo_zoom.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_tkeo_zoom.tick_params(labelsize=TICK_SIZE)
ax_tkeo_zoom.set_title("TKEO Envelope (Zoomed)", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 6: EMG Zoomed (bottom)
ax_emg_zoom = fig.add_subplot(gs[4:6, 4:6])
ax_emg_zoom.plot(zoom_time_array, emg_data_zoom, color=palette[2], linewidth=2)
ax_emg_zoom.set_xlim(0, zoom_duration)
ax_emg_zoom.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_emg_zoom.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_emg_zoom.tick_params(labelsize=TICK_SIZE)
ax_emg_zoom.set_title(
    "Filtered EMG Signal (Zoomed)", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Add overall title
plt.suptitle(
    "EMG Signal Processing with Stimulation",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    y=0.98,
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

plt.show()

# %%
