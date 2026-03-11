"""
EMG and HD-EEG Time Series Visualization in Proper Perspective
Author: jpenalozaa
Real data visualization with 4 HD channels and 1 EMG channel
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data

# %%
# Set publication style
mpu.set_publication_style()

# Load environment variables
load_dotenv()
# %%
# Define paths and load data
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")
HD_STREAM_PATH = BASE_PATH.joinpath("HDEG.ant")

# Load streams
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()

stream_hd = StreamNode(str(HD_STREAM_PATH))
stream_hd.load_metadata()
stream_hd.load_data()

# Get sampling rate
FS = stream_emg.fs
# %%
# Create and apply filters for HD data
processor_hd = create_processing_node(stream_hd)

# Create filters
bandpass_filter_emg = create_bandpass_filter(10, 2000, fs=FS, order=5)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor_emg = FilterProcessor(
    filter_func=bandpass_filter_emg.get_filter_function(), overlap_samples=40
)
bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)

# Add processors
processor_hd.add_processor([notch_processor, bandpass_processor_hd], group="filters")

# Apply filters to HD data
filtered_hd = processor_hd.process(group=["filters"]).persist()

# %%
cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()

cmr_hd = cmr_processor.process(filtered_hd, fs=FS).persist()
# %%
whiten_hd = whiten_processor.process(cmr_hd, fs=FS).persist()
# %%
# Create and apply filters for EMG data
processor_emg = create_processing_node(stream_emg)
processor_emg.add_processor([notch_processor, bandpass_processor_emg], group="filters")
filtered_emg = processor_emg.process(group=["filters"]).persist()
# %%
# Extract data segments (2.5 seconds as in original)
NUM_SAMPLES = int(2.5 * FS)  # 2.5 seconds of data
START_SAMPLE = 0

# Get time slice
time_slice = slice(START_SAMPLE, START_SAMPLE + NUM_SAMPLES)

# Extract HD data (first 4 channels) and EMG data (first channel)
hd_channels = [2, 9, 15, 30]  # First 4 HD channels
hd_data = whiten_hd[time_slice, hd_channels].compute()
emg_data = filtered_emg[time_slice, 0].compute()

# Constants
NUM_CHANNELS = 5
SAMPLING_RATE = FS
TITLE = "EMG + HD-EEG"

# Combine data: 4 HD channels + 1 EMG channel
data = np.zeros((NUM_SAMPLES, NUM_CHANNELS))
data[:, :4] = hd_data  # HD channels 1-4
data[:, 4] = emg_data  # EMG channel 5

# Normalize each channel for better visualization
normalized_data = np.zeros_like(data, dtype=float)
for i in range(NUM_CHANNELS):
    channel_data = data[:, i]
    max_abs = np.max(np.abs(channel_data))
    if max_abs > 0:
        normalized_data[:, i] = channel_data / max_abs * 0.45  # Scale to avoid overlap
    else:
        normalized_data[:, i] = channel_data

# Create time array
time = np.arange(NUM_SAMPLES) / SAMPLING_RATE

# Create figure with custom background color
fig = plt.figure(figsize=(18, 7))

# Add the title at the top right
fig.text(1.0, 0.98, TITLE, ha="right", va="top", fontsize=12, fontstyle="italic")

# Create main plot area
ax = fig.add_subplot(111)
ax.set_title(
    "Multi-Modal Neural Recordings: HD-EEG and EMG in Perspective",
    fontsize=mpu.FONT_SIZES["title"],
    color="darkblue",
)

# Parameters for perspective
vertical_spacing = 1.0  # Vertical spacing between channels
skew_factor = 0.5  # Amount of horizontal skew for perspective

# Calculate the bounds for the perspective box
max_time = time[-1]
max_vert = vertical_spacing * NUM_CHANNELS  # Add an extra position for space
max_horiz_shift = skew_factor * max_vert

# Create vertical positions array for each channel - shifted up by one position
# Start at position 1 instead of 0 to leave space for the x-axis
y_positions = np.arange(1, NUM_CHANNELS + 1) * vertical_spacing

# Draw the perspective box outlines (the dashed lines defining the 3D space)
# Left diagonal
ax.plot([0, max_horiz_shift], [0, max_vert], "k--", linewidth=1, alpha=0.7)
# Right diagonal
ax.plot(
    [max_time, max_time + max_horiz_shift], [0, max_vert], "k--", linewidth=1, alpha=0.7
)
# Top horizontal
ax.plot(
    [max_horiz_shift, max_time + max_horiz_shift],
    [max_vert, max_vert],
    "k--",
    linewidth=1,
    alpha=0.7,
)
# Bottom horizontal (perspective-aligned x-axis)
ax.plot([0, max_time], [0, 0], "k-", linewidth=1)

# Define colors: HD channels in blue tones, EMG in distinctive color
channel_colors = [
    "#0173B2",  # HD Channel 1 - Blue
    "#5DADE2",  # HD Channel 2 - Light Blue
    "#3498DB",  # HD Channel 3 - Medium Blue
    "#1B4F72",  # HD Channel 4 - Dark Blue
    "#E74C3C",  # EMG Channel 5 - Red
]

# Define channel labels
channel_labels = [
    "HD Ch 1",
    "HD Ch 2",
    "HD Ch 3",
    "HD Ch 4",
    "EMG Ch 1",
]

# Plot each channel with shifted positions
for i in range(NUM_CHANNELS):
    # Calculate vertical position (starting from position 1, not 0)
    vert_pos = y_positions[i]

    # Calculate horizontal shift for perspective
    horiz_shift = (i + 1) * skew_factor  # Adjust skew to match vertical position

    # Shift time values for perspective
    shifted_time = time + horiz_shift

    # Shift data vertically
    shifted_data = normalized_data[:, i] + vert_pos

    # Plot time series with appropriate color
    linewidth = 2.0 if i == 4 else 1.5  # Make EMG line slightly thicker
    ax.plot(shifted_time, shifted_data, color=channel_colors[i], linewidth=linewidth)

    # Add channel label
    ax.text(
        shifted_time[0] - 0.05,
        vert_pos,
        channel_labels[i],
        ha="right",
        va="center",
        fontsize=mpu.FONT_SIZES["annotation"],
        fontweight="bold" if i == 4 else "normal",  # Bold for EMG
    )

    # Add horizontal reference line (very light)
    ax.axhline(
        y=vert_pos,
        xmin=horiz_shift / max(ax.get_xlim()) if max(ax.get_xlim()) > 0 else 0,
        xmax=1,
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )

# Configure axes
ax.set_xlim(-0.3, max_time + max_horiz_shift + 0.3)
ax.set_ylim(-0.6, max_vert + 0.6)

# Hide default axes
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.xaxis.set_visible(False)

# Add custom x-axis ticks and labels that follow the perspective
num_ticks = 6  # Number of tick marks you want
for i in range(num_ticks):
    tick_time = i * (max_time / (num_ticks - 1))

    # Draw tick mark
    ax.plot([tick_time, tick_time], [0, -0.1], "k-", linewidth=1)

    # Add tick label
    ax.text(
        tick_time,
        -0.2,
        f"{tick_time:.1f}",
        ha="center",
        va="top",
        fontsize=mpu.FONT_SIZES["tick_label"],
    )

    # Add perspective dash lines for each tick that extend up through the perspective box
    # Calculate the corresponding points at the top of the perspective
    tick_time_top = tick_time + max_horiz_shift

    # Draw the dashed line in perspective
    ax.plot([tick_time, tick_time_top], [0, max_vert], "k--", linewidth=0.7, alpha=0.3)

# Add time axis label
ax.text(
    max_time / 2,
    -0.5,
    "Time (s)",
    ha="center",
    va="top",
    fontsize=mpu.FONT_SIZES["axis_label"],
    fontweight="bold",
)

# Add data type legend
legend_elements = [
    plt.Line2D([0], [0], color="#0173B2", lw=2, label="HD-EEG Channels"),
    plt.Line2D([0], [0], color="#E74C3C", lw=2, label="EMG Channel"),
]
ax.legend(
    handles=legend_elements,
    loc="upper right",
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=mpu.FONT_SIZES["legend"],
)

# Add technical specifications text box
specs_text = f"Sampling Rate: {FS} Hz\nFilters: Notch (60Hz), Bandpass (10-2000Hz)\nDuration: {max_time:.1f} seconds"
ax.text(
    0.02,
    0.98,
    specs_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8),
)

# Finalize the figure
mpu.finalize_figure(fig)

# Save the figure
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath("emg_hd_perspective_visualization.png")
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

# Show the plot
plt.show()

# %%
a = plot_multi_channel_data(
    whiten_hd,
    time_window=[0, 1],
    color_mode="single",
    color="black",
    fs=FS,
    figsize=(15, 15),
)
# %%
