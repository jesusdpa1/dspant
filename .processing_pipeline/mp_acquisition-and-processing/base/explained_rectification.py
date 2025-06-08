"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
Using mp_plotting_utils for standardized publication-quality visualization
"""

# %%
import time

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
from dspant.visualization.general_plots import plot_multi_channel_data

# Set publication style
mpu.set_publication_style()

# %%
# Define font sizes with appropriate scaling - increased for better visibility
TITLE_SIZE = 22
SUBTITLE_SIZE = 20
AXIS_LABEL_SIZE = 18
TICK_SIZE = 16
LEGEND_SIZE = 16

# Define colors - preserving original colors
DARK_GREY_NAVY = "#2D3142"  # Original dark navy blue for time series

# %%
BASE_PATH = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
EMG_STREAM_PATH = BASE_PATH + r"/RawG.ant"

# %%
# Load EMG data
stream_emg = StreamNode(EMG_STREAM_PATH)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

# %%
# Create and visualize filters before applying them
FS = stream_emg.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=60, fs=FS)

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
multichannel_fig = plot_multi_channel_data(filter_data, fs=FS, time_window=[0, 10])

# %%
START = int(FS * 5)
END = int(FS * 10)
base_data = filter_data[START:END, :]

# %%
# Create different EMG envelope methods
abs_processor = RectificationProcessor("abs")
square_processor = RectificationProcessor("square")
hilbert_processor = RectificationProcessor("hilbert")
tkeo_processor = create_tkeo_envelope_rs("modified", rectify=False, smooth=False)

# Process data with different methods
abs_data = abs_processor.process(base_data).compute()
square_data = square_processor.process(base_data).compute()
hilbert_data = hilbert_processor.process(base_data).compute()
tkeo_data = tkeo_processor.process(base_data).compute()

# %%
# Plot EMG Signal Processing Methods

# Select one channel for visualization
CHANNEL_TO_PLOT = 0

# Calculate time array for x-axis
time_array = np.arange(len(tkeo_data[:, CHANNEL_TO_PLOT])) / FS
len_to_plot = len(tkeo_data[:, CHANNEL_TO_PLOT])
max_time = len_to_plot / FS

# Create a figure with a 2-row grid: top row spanning all columns, bottom row split into 4
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, height_ratios=[1, 1.5])

# Plot the original signal spanning the full width (across all 4 columns)
ax_orig = fig.add_subplot(gs[0, :])
ax_orig.plot(
    time_array,
    base_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=DARK_GREY_NAVY,
    linewidth=2,
)

# Format the axis using mpu
mpu.format_axis(
    ax_orig,
    title="Original Filtered Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Prepare colors from the colorblind palette
palette = sns.color_palette("colorblind")

# Plot 1: Abs Rectification
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(
    time_array,
    abs_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=palette[0],
    linewidth=2,
)

# Format the axis using mpu
mpu.format_axis(
    ax1,
    title="Absolute Rectification",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Start from 0 but auto-scale the top
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Plot 2: Square Rectification
ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(
    time_array,
    square_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=palette[1],
    linewidth=2,
)

# Format the axis using mpu
mpu.format_axis(
    ax2,
    title="Square Rectification",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Start from 0 but auto-scale the top
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Plot 3: Hilbert Transform
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(
    time_array,
    hilbert_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=palette[2],
    linewidth=2,
)

# Format the axis using mpu
mpu.format_axis(
    ax3,
    title="Hilbert Transform",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Start from 0 but auto-scale the top
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Plot 4: TKEO
ax4 = fig.add_subplot(gs[1, 3])
ax4.plot(
    time_array,
    tkeo_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=palette[3],
    linewidth=2,
)

# Format the axis using mpu
mpu.format_axis(
    ax4,
    title="TKEO",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Start from 0 but auto-scale the top
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Finalize the figure with mpu
mpu.finalize_figure(
    fig,
    title="EMG Signal Processing Methods",
    title_y=0.98,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

# Add panel labels
mpu.add_panel_label(
    ax_orig,
    "A",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax1,
    "B",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2,
    "C",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3,
    "D",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax4,
    "E",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Save the figure if needed
mpu.save_figure(fig, "emg_signal_processing_methods.png", dpi=600)

plt.show()

# %%
