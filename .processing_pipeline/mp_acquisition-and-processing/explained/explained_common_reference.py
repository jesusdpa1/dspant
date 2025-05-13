"""
Script to showcase CMR
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
Using mp_plotting_utils for standardized publication-quality visualization
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
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
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data

# Set publication style
mpu.set_publication_style()
load_dotenv()
# %%

DATA_DIR = Path(os.getenv("DATA_DIR"))

BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

HD_STREAM_PATH = BASE_PATH.joinpath("HDEG.ant")

# %%
# Load EMG data
stream_hd = StreamNode(str(HD_STREAM_PATH))
stream_hd.load_metadata()
stream_hd.load_data()
# Print stream_emg summary
stream_hd.summarize()

# %%
# Create and visualize filters before applying them
FS = stream_hd.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)
# %%
bandpass_plot = bandpass_filter.plot_frequency_response()
notch_plot = notch_filter.plot_frequency_response()
# %%
# Create processing node with filters
processor_hd = create_processing_node(stream_hd)
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
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
# %%
# View summary of the processing node
processor_hd.summarize()

# %%
# Apply filters and plot results
filter_data = processor_hd.process(group=["filters"]).persist()

# %%
cmr_processor = create_cmr_processor_rs()
cmr_data = cmr_processor.process(filter_data, FS).persist()
cmr_reference = cmr_processor.get_reference(filter_data)

# %%
a = plot_multi_channel_data(
    filter_data, channels=[1, 2, 3, 4], fs=FS, time_window=[0, 5]
)
# %%


# Improved panel label function
def add_panel_label(
    ax,
    label,
    position="top-left",
    offset_factor=0.1,
    fontsize=None,
    fontweight="bold",
    color="black",
):
    """
    Add a panel label (A, B, C, etc.) to a subplot with adaptive positioning.
    """
    # Get the position of the axes in figure coordinates
    bbox = ax.get_position()
    fig = plt.gcf()

    # Set default font size if not specified
    if fontsize is None:
        fontsize = mpu.FONT_SIZES["panel_label"]

    # Calculate offset based on subplot size and offset factor
    x_offset = bbox.width * offset_factor
    y_offset = bbox.height * offset_factor

    # Determine position coordinates based on selected position
    if position == "top-left":
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset
    elif position == "top-right":
        x = bbox.x1 + x_offset
        y = bbox.y1 + y_offset
    elif position == "bottom-left":
        x = bbox.x0 - x_offset
        y = bbox.y0 - y_offset
    elif position == "bottom-right":
        x = bbox.x1 + x_offset
        y = bbox.y0 - y_offset
    else:
        # Default to top-left if invalid position
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset

    # Determine text alignment based on position
    if "left" in position:
        ha = "right"
    else:
        ha = "left"

    if "top" in position:
        va = "bottom"
    else:
        va = "top"

    # Position the label outside the subplot
    fig.text(
        x,
        y,
        label,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha=ha,
        color=color,
    )


# Define data range
START = 0
END = int(5 * FS)  # 5 seconds of data

# Choose channels to display
SELECTED_CHANNELS = [0, 1, 2, 3]
NUM_CHANNELS = len(SELECTED_CHANNELS)

# Define colors for the reference signal
REFERENCE_COLOR = mpu.PRIMARY_COLOR  # Dark navy blue for reference signal

# Create figure with GridSpec for custom layout
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, height_ratios=[2, 1])

# Calculate time array
time_array = np.arange(END - START) / FS

# Calculate vertical offsets for channels
# First, determine the approx amplitude range of signals
filtered_range = np.max(np.abs(filter_data[START:END, SELECTED_CHANNELS])) * 2
cmr_range = np.max(np.abs(cmr_data[START:END, SELECTED_CHANNELS])) * 2

# Calculate offset between channels (make it 1.5x the range for clear separation)
filtered_offset = filtered_range * 1.5
cmr_offset = cmr_range * 1.5

# Plot 1: Filtered Data (top left)
ax_filtered = fig.add_subplot(gs[0, 0])
for i, channel in enumerate(SELECTED_CHANNELS):
    # Add increasing offset for each channel
    offset = i * filtered_offset
    ax_filtered.plot(
        time_array,
        filter_data[START:END, channel] + offset,
        color=mpu.COLORS.get(list(mpu.COLORS.keys())[i % len(mpu.COLORS)]),
        linewidth=1.5,
        label=f"Channel {channel}",
    )

mpu.format_axis(
    ax_filtered,
    title="Bandpass + Notch Filtered Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, (END - START) / FS),
)
# mpu.add_legend(ax_filtered, loc="upper right")

# Plot 2: CMR Data (top right)
ax_cmr = fig.add_subplot(gs[0, 1])
for i, channel in enumerate(SELECTED_CHANNELS):
    # Add increasing offset for each channel
    offset = i * cmr_offset
    ax_cmr.plot(
        time_array,
        cmr_data[START:END, channel] + offset,
        color=mpu.COLORS.get(list(mpu.COLORS.keys())[i % len(mpu.COLORS)]),
        linewidth=1.5,
        label=f"Channel {channel}",
    )

mpu.format_axis(
    ax_cmr,
    title="Common Median Reference (CMR) Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, (END - START) / FS),
)
# mpu.add_legend(ax_cmr, loc="upper right")

# Plot 3: CMR Reference (bottom span)
ax_ref = fig.add_subplot(gs[1, :])
# Get the reference signal and ensure it's the right shape
reference_data = cmr_reference[START:END].compute().flatten()
ax_ref.plot(time_array, reference_data, color=REFERENCE_COLOR, linewidth=2)

mpu.format_axis(
    ax_ref,
    title="Common Median Reference Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, (END - START) / FS),
)

# Finalize the figure using mpu
mpu.finalize_figure(
    fig,
    title="Effect of Common Median Reference (CMR) on Neural Signals",
    title_y=0.98,
)

# Apply tight layout to finalize positions before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

# Add panel labels with adaptive positioning
add_panel_label(ax_filtered, "A", offset_factor=0.05)
add_panel_label(ax_cmr, "B", offset_factor=0.05)
add_panel_label(ax_ref, "C", offset_factor=0.05)

# Save figure if needed
mpu.save_figure(fig, "cmr_comparison.png", dpi=600)

plt.show()

# %%
