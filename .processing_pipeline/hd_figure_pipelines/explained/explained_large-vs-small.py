"""
MYOMATRIX LARGE VS SMALL CONTACT Time Series Visualization in Proper Perspective
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
from matplotlib.ticker import MaxNLocator

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
    r"papers\2025_mp_emg diaphragm acquisition and processing\large-vs-small"
)
BASE_PATH_SMALL = BASE_PATH.joinpath(
    r"drv_00_baseline_25-02-26_9881-2_testSubject_topoMapping"
)
BASE_PATH_LARGE = BASE_PATH.joinpath(
    r"drv_01_baseline_24-11-01_3869-2_testSubject_DST+Contusion_large"
)
HD_SMALL_STREAM_PATH = BASE_PATH_SMALL.joinpath("HDEG.ant")
HD_LARGE_STREAM_PATH = BASE_PATH_LARGE.joinpath("EMGM.ant")  # old naming convention

# Load streams
stream_hd_small = StreamNode(str(HD_SMALL_STREAM_PATH))
stream_hd_small.load_metadata()
stream_hd_small.load_data()

stream_hd_large = StreamNode(str(HD_LARGE_STREAM_PATH))
stream_hd_large.load_metadata()
stream_hd_large.load_data()

# Get sampling rate
FS = stream_hd_small.fs
# %%
notch_filter = create_notch_filter(60, q=30, fs=FS)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=5)

# Create pre-processing functions
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)

cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()
# %%
# Create and apply filters for HD data
processor_hd_small = create_processing_node(stream_hd_small)

# Add processors
processor_hd_small.add_processor(
    [notch_processor, bandpass_processor_hd],
    group="filters",
)

processor_hd_large = create_processing_node(stream_hd_large)

# Add processors
processor_hd_large.add_processor(
    [notch_processor, bandpass_processor_hd],
    group="filters",
)
# %%
# Apply filters to HD data
filtered_hd_small = processor_hd_small.process(group=["filters"]).persist()
filtered_hd_large = processor_hd_large.process(group=["filters"]).persist()
# %%
cmr_hd_small = cmr_processor.process(filtered_hd_small, fs=FS).persist()
cmr_hd_large = cmr_processor.process(filtered_hd_large, fs=FS).persist()
# %%
whiten_hd_small = whiten_processor.process(cmr_hd_small, fs=FS).persist()
whiten_hd_large = whiten_processor.process(cmr_hd_large, fs=FS).persist()

# %%
# Define channels and data parameters
CHANNELS_SMALL = np.arange(0, 16, 2).tolist()
CHANNELS_LARGE = np.arange(8, 16, 1).tolist()

# Extract data segments
START_ = 1  # seconds
END_ = 5  # seconds
SAMPLING_RATE = FS
START_SAMPLE = int(START_ * FS)
END_SAMPLE = int(END_ * FS)

# Define zoom parameters (0.5 seconds zoom window)
START_ZOOM = int(2.5 * FS)  # Start zoom at 2.5 seconds
END_ZOOM = int(3.0 * FS)  # End zoom at 3.0 seconds (0.5 second window)

# Calculate time values for zoom highlighting
ZOOM_START_TIME = START_ZOOM / FS
ZOOM_END_TIME = END_ZOOM / FS
ZOOM_WIDTH = ZOOM_END_TIME - ZOOM_START_TIME

# Define font sizes with appropriate scaling
FONT_SIZE = 35
TITLE_SIZE = int(FONT_SIZE * 0.8)
SUBTITLE_SIZE = int(FONT_SIZE * 0.7)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)
LEGEND_SIZE = int(FONT_SIZE * 0.5)

# Get time slice
time_slice = slice(START_SAMPLE, END_SAMPLE)

TITLE = "Large contacts [100x200 um] vs Small Contacts [25 um^2]"

# Define colors for each dataset
LARGE_COLORS = [
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
]
SMALL_COLORS = [
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
    "#2D3142",
]

# Define highlight color for zoom boxes
HIGHLIGHT_COLOR = mpu.COLORS["red"]

# %%
# Create 2x2 comparison plot
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])

# Calculate time arrays
time_array = np.arange(END_SAMPLE - START_SAMPLE) / FS
zoom_time_array = np.arange(END_ZOOM - START_ZOOM) / FS

# Compute data to avoid lazy evaluation
large_data_computed = whiten_hd_large[START_SAMPLE:END_SAMPLE, :].compute()
small_data_computed = whiten_hd_small[START_SAMPLE:END_SAMPLE, :].compute()

# Calculate normalization and positioning parameters
y_spread = 1.0  # Vertical spacing between channels
norm_scale = 0.4  # Scale factor for normalized signals
y_offset = 0.0  # Baseline offset

# TOP LEFT: Large HD Data (Full View)
ax_large = fig.add_subplot(gs[0, 0])

# Calculate channel positions for large data (bottom to top)
large_channel_positions = []
large_channel_info = []

for idx, channel in enumerate(CHANNELS_LARGE):
    # Position channels from bottom to top
    channel_offset = y_offset + (len(CHANNELS_LARGE) - 1 - idx) * y_spread
    large_channel_positions.append(channel_offset)
    large_channel_info.append(f"Ch {channel}")

# Plot large HD data
for idx, channel in enumerate(CHANNELS_LARGE):
    # Get channel position
    channel_offset = large_channel_positions[idx]

    # Get data for this channel
    subset_data = large_data_computed[:, channel]

    # Normalize the data
    max_amplitude = np.max(np.abs(subset_data))
    if max_amplitude > 0:
        # Normalize and scale by y_spread * norm_scale
        norm_data = (
            subset_data / max_amplitude * (y_spread * norm_scale)
        ) + channel_offset
    else:
        # If flat signal, just offset it
        norm_data = np.zeros_like(subset_data) + channel_offset

    ax_large.plot(
        time_array,
        norm_data,
        color=LARGE_COLORS[idx % len(LARGE_COLORS)],
        linewidth=2,
        alpha=0.8,
        label=f"Channel {channel}",
    )

# Add highlight box for zoom region
y_min, y_max = ax_large.get_ylim()
height = y_max - y_min
large_rect = plt.Rectangle(
    (ZOOM_START_TIME - START_, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_large.add_patch(large_rect)

# Set up y-axis with channel labels and force integer ticks
ax_large.set_yticks(large_channel_positions)
ax_large.set_yticklabels(large_channel_info)
ax_large.set_ylim(
    y_offset - 0.5 * y_spread,
    y_offset + len(CHANNELS_LARGE) * y_spread - 0.5 * y_spread,
)

# Force y-axis to show only integer values
ax_large.yaxis.set_major_locator(MaxNLocator(integer=True))

mpu.format_axis(
    ax_large,
    title="Large Contacts [100x200 μm]",
    xlabel=None,
    ylabel="Channels",
    xlim=(0, (END_SAMPLE - START_SAMPLE) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# TOP RIGHT: Large HD Data (Zoom View)
ax_large_zoom = fig.add_subplot(gs[0, 1])

# Plot large HD data zoom
for idx, channel in enumerate(CHANNELS_LARGE):
    # Get channel position
    channel_offset = large_channel_positions[idx]

    # Get zoom data for this channel
    subset_data = whiten_hd_large[START_ZOOM:END_ZOOM, channel].compute()

    # Normalize the data
    max_amplitude = np.max(np.abs(subset_data))
    if max_amplitude > 0:
        # Normalize and scale by y_spread * norm_scale
        norm_data = (
            subset_data / max_amplitude * (y_spread * norm_scale)
        ) + channel_offset
    else:
        # If flat signal, just offset it
        norm_data = np.zeros_like(subset_data) + channel_offset

    ax_large_zoom.plot(
        zoom_time_array,
        norm_data,
        color=LARGE_COLORS[idx % len(LARGE_COLORS)],
        linewidth=2,
        alpha=0.8,
        label=f"Channel {channel}",
    )

# Set up y-axis with channel labels and force integer ticks
ax_large_zoom.set_yticks(large_channel_positions)
ax_large_zoom.set_yticklabels(large_channel_info)
ax_large_zoom.set_ylim(
    y_offset - 0.5 * y_spread,
    y_offset + len(CHANNELS_LARGE) * y_spread - 0.5 * y_spread,
)

# Force y-axis to show only integer values
ax_large_zoom.yaxis.set_major_locator(MaxNLocator(integer=True))

mpu.format_axis(
    ax_large_zoom,
    title="Large (Zoomed)",
    xlabel=None,
    ylabel="Channels",
    xlim=(0, ZOOM_WIDTH),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# BOTTOM LEFT: Small HD Data (Full View)
ax_small = fig.add_subplot(gs[1, 0])

# Calculate channel positions for small data (bottom to top)
small_channel_positions = []
small_channel_info = []

for idx, channel in enumerate(CHANNELS_SMALL):
    # Position channels from bottom to top
    channel_offset = y_offset + (len(CHANNELS_SMALL) - 1 - idx) * y_spread
    small_channel_positions.append(channel_offset)
    small_channel_info.append(f"Ch {channel}")

# Plot small HD data
for idx, channel in enumerate(CHANNELS_SMALL):
    # Get channel position
    channel_offset = small_channel_positions[idx]

    # Get data for this channel
    subset_data = small_data_computed[:, channel]

    # Normalize the data
    max_amplitude = np.max(np.abs(subset_data))
    if max_amplitude > 0:
        # Normalize and scale by y_spread * norm_scale
        norm_data = (
            subset_data / max_amplitude * (y_spread * norm_scale)
        ) + channel_offset
    else:
        # If flat signal, just offset it
        norm_data = np.zeros_like(subset_data) + channel_offset

    ax_small.plot(
        time_array,
        norm_data,
        color=SMALL_COLORS[idx % len(SMALL_COLORS)],
        linewidth=2,
        alpha=0.8,
        label=f"Channel {channel}",
    )

# Add highlight box for zoom region
y_min, y_max = ax_small.get_ylim()
height = y_max - y_min
small_rect = plt.Rectangle(
    (ZOOM_START_TIME - START_, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_small.add_patch(small_rect)

# Set up y-axis with channel labels and force integer ticks
ax_small.set_yticks(small_channel_positions)
ax_small.set_yticklabels(small_channel_info)
ax_small.set_ylim(
    y_offset - 0.5 * y_spread,
    y_offset + len(CHANNELS_SMALL) * y_spread - 0.5 * y_spread,
)

# Force y-axis to show only integer values
ax_small.yaxis.set_major_locator(MaxNLocator(integer=True))

mpu.format_axis(
    ax_small,
    title="Small Contacts [25 μm²]",
    xlabel="Time [s]",
    ylabel="Channels",
    xlim=(0, (END_SAMPLE - START_SAMPLE) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# BOTTOM RIGHT: Small HD Data (Zoom View)
ax_small_zoom = fig.add_subplot(gs[1, 1])

# Plot small HD data zoom
for idx, channel in enumerate(CHANNELS_SMALL):
    # Get channel position
    channel_offset = small_channel_positions[idx]

    # Get zoom data for this channel
    subset_data = whiten_hd_small[START_ZOOM:END_ZOOM, channel].compute()

    # Normalize the data
    max_amplitude = np.max(np.abs(subset_data))
    if max_amplitude > 0:
        # Normalize and scale by y_spread * norm_scale
        norm_data = (
            subset_data / max_amplitude * (y_spread * norm_scale)
        ) + channel_offset
    else:
        # If flat signal, just offset it
        norm_data = np.zeros_like(subset_data) + channel_offset

    ax_small_zoom.plot(
        zoom_time_array,
        norm_data,
        color=SMALL_COLORS[idx % len(SMALL_COLORS)],
        linewidth=2,
        alpha=0.8,
        label=f"Channel {channel}",
    )

# Set up y-axis with channel labels and force integer ticks
ax_small_zoom.set_yticks(small_channel_positions)
ax_small_zoom.set_yticklabels(small_channel_info)
ax_small_zoom.set_ylim(
    y_offset - 0.5 * y_spread,
    y_offset + len(CHANNELS_SMALL) * y_spread - 0.5 * y_spread,
)

# Force y-axis to show only integer values
ax_small_zoom.yaxis.set_major_locator(MaxNLocator(integer=True))

mpu.format_axis(
    ax_small_zoom,
    title="Small (Zoomed)",
    xlabel="Time [s]",
    ylabel="Channels",
    xlim=(0, ZOOM_WIDTH),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Finalize the figure
mpu.finalize_figure(
    fig,
    title=TITLE,
    title_y=0.95,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Add panel labels
mpu.add_panel_label(
    ax_large,
    "A",
    x_offset_factor=0.04,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_large_zoom,
    "B",
    x_offset_factor=0.04,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_small,
    "C",
    x_offset_factor=0.04,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_small_zoom,
    "D",
    x_offset_factor=0.04,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Figure output path
FIGURE_TITLE = "hd_comparison_2x2_zoom"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

# Save figure
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print("HD comparison analysis complete.")

# %%
