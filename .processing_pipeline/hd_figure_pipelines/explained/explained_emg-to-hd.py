"""
EMG to HD-EMG Visualization in 3D Perspective
Author: jpenalozaa
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)

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

# %%
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
# Create and apply filters
processor_hd = create_processing_node(stream_hd)

# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)
lowpass_filter = create_lowpass_filter(200, fs=FS, order=5)

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

# Add processors
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters
filtered_hd = processor_hd.process(group=["filters"]).persist()

# %%
# Helper functions for 3D perspective visualization


def create_perspective_shifts(num_channels, base_shift=0.5, perspective_factor=0.2):
    """
    Create perspective shifts to generate the 3D-like staggered effect

    Parameters:
        num_channels (int): Number of channels to visualize
        base_shift (float): Base vertical spacing between signals
        perspective_factor (float): Factor controlling horizontal offset for perspective

    Returns:
        tuple: (y_shifts, x_shifts) for vertical and horizontal positioning
    """
    y_shifts = np.arange(num_channels) * base_shift
    x_shifts = np.arange(num_channels) * perspective_factor

    return y_shifts, x_shifts


def draw_perspective_lines(
    ax, x_start, x_end, y_positions, line_style="k--", alpha=0.3
):
    """Draw perspective lines connecting points at same x-coordinate"""
    for x in [x_start, x_end]:
        for i in range(len(y_positions) - 1):
            ax.plot(
                [x, x], [y_positions[i], y_positions[i + 1]], line_style, alpha=alpha
            )


def create_arrow_with_text(
    ax, x, y, dx, dy, text, arrow_props=None, text_offset=(5, 5)
):
    """Create an arrow with text annotation"""
    if arrow_props is None:
        arrow_props = dict(arrowstyle="->", color="blue", lw=2)

    arrow = ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y), arrowprops=arrow_props)

    # Add text near the arrow
    ax.text(
        x + text_offset[0],
        y + text_offset[1],
        text,
        fontsize=14,
        color=arrow_props["color"],
    )

    return arrow


# %%
# Data preparation

# Constants in uppercase
START_TIME = 5  # seconds
DURATION = 1  # seconds
NUM_CHANNELS = 10
TITLE = "EMGtoHD"

START_SAMPLE = int(START_TIME * FS)
END_SAMPLE = START_SAMPLE + int(DURATION * FS)

# Select channels to display
SELECTED_CHANNELS = np.arange(NUM_CHANNELS)

# Extract data segment for visualization
hd_segment = filtered_hd[START_SAMPLE:END_SAMPLE, SELECTED_CHANNELS].compute()

# Create time array
time = np.arange(hd_segment.shape[0]) / FS

# Normalize each channel for better visualization
normalized_data = np.zeros_like(hd_segment, dtype=float)
for i in range(NUM_CHANNELS):
    channel_data = hd_segment[:, i]
    max_abs = np.max(np.abs(channel_data))
    if max_abs > 0:
        normalized_data[:, i] = channel_data / max_abs
    else:
        normalized_data[:, i] = channel_data

# %%
# Create Figure

# Set up figure
fig = plt.figure(figsize=(15, 10))
fig.text(1.0, 0.98, TITLE, ha="right", va="top", fontsize=12, fontstyle="italic")

# Create main plot area - only one section for the signal
ax1 = fig.add_subplot(111)

# Set title
ax1.set_title("EMG to HD-EMG Multichannel Recording", fontsize=16, color="darkblue")

# Generate perspective shifts
y_shifts, x_shifts = create_perspective_shifts(NUM_CHANNELS)

# Plot each channel with perspective shift
channel_names = []
for i in range(NUM_CHANNELS):
    channel_data = normalized_data[:, i]
    shifted_time = time + x_shifts[i]
    shifted_data = channel_data + y_shifts[i]

    # Plot the channel data
    ax1.plot(shifted_time, shifted_data, "b-", linewidth=1.5)

    # Add channel label
    channel_name = f"Ch {i + 1}"
    channel_names.append(channel_name)
    ax1.text(
        shifted_time[0] - 0.05,
        y_shifts[i],
        channel_name,
        ha="right",
        va="center",
        fontsize=10,
    )

# Draw perspective lines
draw_perspective_lines(ax1, time[0], time[-1], y_shifts)

# Add arrows and annotations for perspective
create_arrow_with_text(
    ax1,
    time[0],
    y_shifts[0] - 0.5,
    0.1,
    0,
    "Time (s)",
    arrow_props=dict(arrowstyle="->", color="blue", lw=2),
)

create_arrow_with_text(
    ax1,
    time[0] - 0.05,
    y_shifts[0],
    0,
    y_shifts[-1] - y_shifts[0],
    "Channels",
    arrow_props=dict(arrowstyle="->", color="green", lw=2),
)

# Add shaded region highlighting specific time period
HIGHLIGHT_START = 0.3  # seconds from start of segment
HIGHLIGHT_DURATION = 0.2  # seconds
highlight_start_idx = int(HIGHLIGHT_START * FS)
highlight_end_idx = highlight_start_idx + int(HIGHLIGHT_DURATION * FS)

# Draw highlight across all channels
for i in range(NUM_CHANNELS - 1):
    rect = Rectangle(
        (time[highlight_start_idx] + x_shifts[i], y_shifts[i]),
        time[highlight_end_idx] - time[highlight_start_idx],
        y_shifts[i + 1] - y_shifts[i],
        color="yellow",
        alpha=0.3,
        linewidth=0,
    )
    ax1.add_patch(rect)

# Draw connecting lines for the highlight
highlight_x1 = time[highlight_start_idx] + x_shifts[0]
highlight_x2 = time[highlight_end_idx] + x_shifts[0]
highlight_y1 = y_shifts[0]
highlight_y2 = y_shifts[-1]

ax1.plot(
    [highlight_x1, highlight_x1 + x_shifts[-1]],
    [highlight_y1, highlight_y2],
    "k--",
    alpha=0.5,
)
ax1.plot(
    [highlight_x2, highlight_x2 + x_shifts[-1]],
    [highlight_y1, highlight_y2],
    "k--",
    alpha=0.5,
)

# Format axes
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Channel Amplitude")
ax1.set_yticks([])  # Hide y-ticks since they're not meaningful with the shifts
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# %%
# Finalize the figure
plt.tight_layout()
mpu.finalize_figure(fig, title="EMG to HD-EMG Signal Analysis", title_fontsize=18)
# mpu.save_figure(fig, "emg_to_hd_perspective_visualization.png", dpi=600)
plt.show()
