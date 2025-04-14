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

from dspant._rs import compute_tkeo
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
    create_notch_filter,
)
from dspant.vizualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_stream_path = base_path + r"/RawG.ant"
# %%
# Load EMG data
stream_emg = StreamNode(emg_stream_path)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
# %%
bandpass_plot = bandpass_filter.plot_frequency_response()
notch_plot = notch_filter.plot_frequency_response()
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
# Apply filters and plot results
filter_data = processor_emg.process(group=["filters"]).persist()
# %%
multichannel_fig = plot_multi_channel_data(filter_data, fs=fs, time_window=[0, 10])

# %%
start = int(fs * 5)
end = int(fs * 10)
base_data = filter_data[start:end, :]

# %%
abs_processor = RectificationProcessor("abs")
square_processor = RectificationProcessor("square")
hilbert_processor = RectificationProcessor("hilbert")
tkeo_processor = create_tkeo_envelope_rs("modified", rectify=False, smooth=False)

abs_data = abs_processor.process(base_data).compute()
square_data = square_processor.process(base_data).compute()
hilbert_data = hilbert_processor.process(base_data).compute()
tkeo_data = tkeo_processor.process(base_data).compute()
# %%
plt.plot(tkeo_data)
# %%
# use the shape of the tkeo data and only plot one of the channels
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

# Create a figure with a 2-row grid: top row spanning all columns, bottom row split into 4
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, height_ratios=[1, 1.5])

# Select one channel for visualization
channel_to_plot = 0

# Calculate time array for x-axis
time_array = np.arange(len(tkeo_data[:, channel_to_plot])) / fs
len_to_plot = len(tkeo_data[:, channel_to_plot])
max_time = len_to_plot / fs

# Darker grey with navy tint
dark_grey_navy = "#2D3142"

# Plot the original signal spanning the full width (across all 4 columns)
ax_orig = fig.add_subplot(gs[0, :])
ax_orig.plot(
    time_array,
    base_data[:len_to_plot, channel_to_plot],
    color=dark_grey_navy,  # Using darker grey with navy tint
    linewidth=2,
)
ax_orig.set_xlim(0, max_time)
ax_orig.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_orig.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_orig.tick_params(labelsize=TICK_SIZE)
ax_orig.set_title("Original Filtered Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Calculate max values for each dataset with 10% margin
abs_max = np.max(abs_data[:len_to_plot, channel_to_plot]) * 1.1
square_max = np.max(square_data[:len_to_plot, channel_to_plot]) * 1.1
hilbert_max = np.max(hilbert_data[:len_to_plot, channel_to_plot]) * 1.1
tkeo_max = np.max(tkeo_data[:len_to_plot, channel_to_plot]) * 1.1

# Plot 1: Abs Rectification
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(
    time_array,
    abs_data[:len_to_plot, channel_to_plot],
    color=palette[0],
    linewidth=2,
)
ax1.set_xlim(0, max_time)
ax1.set_ylim(0)
ax1.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.tick_params(labelsize=TICK_SIZE)
ax1.set_title("Absolute Rectification", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 2: Square Rectification
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(
    time_array,
    square_data[:len_to_plot, channel_to_plot],
    color=palette[1],
    linewidth=2,
)
ax2.set_xlim(0, max_time)
ax2.set_ylim(0)
ax2.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.tick_params(labelsize=TICK_SIZE)
ax2.set_title("Square Rectification", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 3: Hilbert Transform
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(
    time_array,
    hilbert_data[:len_to_plot, channel_to_plot],
    color=palette[2],
    linewidth=2,
)
ax3.set_xlim(0, max_time)
ax3.set_ylim(0)
ax3.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax3.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax3.tick_params(labelsize=TICK_SIZE)
ax3.set_title("Hilbert Transform Envelope", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 4: TKEO
ax4 = fig.add_subplot(gs[1, 3])
ax4.plot(
    time_array,
    tkeo_data[:len_to_plot, channel_to_plot],
    color=palette[3],
    linewidth=2,
)
ax4.set_xlim(0, max_time)
ax4.set_ylim(0)
ax4.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax4.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax4.tick_params(labelsize=TICK_SIZE)
ax4.set_title("Teager-Kaiser Energy Operator", fontsize=SUBTITLE_SIZE, weight="bold")

# Add overall title
plt.suptitle(
    "EMG Signal Processing Methods", fontsize=TITLE_SIZE, fontweight="bold", y=0.98
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

plt.show()
# %%
