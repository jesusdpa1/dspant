"""
Script to showcase CMR
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
Using mp_plotting_utils for standardized publication-quality visualization
Modified to use scientific notation on all y-axes with custom color scheme
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
# Define custom color scheme
NAVY_BLUE = "#2D3142"
FILTERED_COLORS = [
    "#DE8F05",
    "#00A410",
    "#1066DE",
    "#E29ECA",
]  # Different colors for each channel
CMR_COLORS = ["#FFC800", "#90FF4B", "#31B7FA", "#F591B2"]  # Lighter versions for CMR
REFERENCE_COLOR = "#2D3142"  # Navy blue for reference signal

# Define font sizes with appropriate scaling
FONT_SIZE = 35
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)
LEGEND_SIZE = int(FONT_SIZE * 0.5)

# Larger font size for scientific notation
SCIENTIFIC_NOTATION_SIZE = int(FONT_SIZE * 0.5)

# Define paths using environment variables
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
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
# Function to apply scientific notation to y-axis with larger font size
def apply_scientific_notation(ax, fontsize=24):
    """Apply scientific notation formatting to the y-axis with custom font size"""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(
        (-2, 2)
    )  # Use scientific notation for numbers outside 0.01 to 100
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))

    # Set the font size for the tick labels (including scientific notation)
    ax.tick_params(axis="y", labelsize=fontsize)

    # Also set the offset text (the exponent part) font size
    ax.yaxis.offsetText.set_fontsize(fontsize)


# Define data range
START = 0
END = int(5 * FS)  # 5 seconds of data

# Choose channels to display
SELECTED_CHANNELS = [0, 1, 2, 3]
NUM_CHANNELS = len(SELECTED_CHANNELS)

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
        color=FILTERED_COLORS[i % len(FILTERED_COLORS)],
        linewidth=2,
        label=f"Channel {channel}",
    )

mpu.format_axis(
    ax_filtered,
    title="Bandpass + Notch Filtered Signal",
    xlabel=None,
    ylabel="Amplitude + Offset",
    xlim=(0, (END - START) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_filtered, SCIENTIFIC_NOTATION_SIZE)

# Plot 2: CMR Data (top right)
ax_cmr = fig.add_subplot(gs[0, 1])
for i, channel in enumerate(SELECTED_CHANNELS):
    # Add increasing offset for each channel
    offset = i * cmr_offset
    ax_cmr.plot(
        time_array,
        cmr_data[START:END, channel] + offset,
        color=CMR_COLORS[i % len(CMR_COLORS)],
        linewidth=2,
        label=f"Channel {channel}",
    )

mpu.format_axis(
    ax_cmr,
    title="Common Median Reference (CMR) Signal",
    xlabel=None,
    ylabel="Amplitude + Offset",
    xlim=(0, (END - START) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_cmr, SCIENTIFIC_NOTATION_SIZE)

# Plot 3: CMR Reference (bottom span)
ax_ref = fig.add_subplot(gs[1, :])
# Get the reference signal and ensure it's the right shape
reference_data = cmr_reference[START:END].compute().flatten()
ax_ref.plot(time_array, reference_data, color=REFERENCE_COLOR, linewidth=3)

mpu.format_axis(
    ax_ref,
    title="Common Median Reference Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, (END - START) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_ref, SCIENTIFIC_NOTATION_SIZE)

# Finalize the figure using mpu
mpu.finalize_figure(
    fig,
    title="Effect of Common Median Reference (CMR) on Neural Signals",
    title_y=0.98,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout to finalize positions before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

# Add panel labels using mpu.add_panel_label
mpu.add_panel_label(
    ax_filtered,
    "A",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_cmr,
    "B",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_ref,
    "C",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Figure output path
FIGURE_TITLE = "cmr_comparison_analysis"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

# Save figure using mpu
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print("CMR analysis complete.")

# %%
