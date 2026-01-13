"""
Script to showcase CMR
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
Using mp_plotting_utils for standardized publication-quality visualization
Modified to use channel numbers on y-axis with scientific notation only on CMR reference
"""

# %%
import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec

# Force y-axis to show only integer values for panel A
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, ScalarFormatter

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

# Define paths using environment variables
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
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
# %%
raw_data = stream_hd.data.compute()[0 : int(FS * 120), :]
# %%

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)

# %%
bandpass_plot = bandpass_filter.plot_frequency_response()
notch_plot = notch_filter.plot_frequency_response()

# %%
bp_data = bandpass_filter.filter(raw_data, FS)
filtered_data = da.array(notch_filter.filter(bp_data, FS))

# %%
cmr_processor = create_cmr_processor_rs()
cmr_data = cmr_processor.process(filtered_data, FS).persist()
cmr_reference = cmr_processor.get_reference(filtered_data).persist()

# %%
a = plot_multi_channel_data(
    filtered_data, channels=[1, 2, 3, 4], fs=FS, time_window=[0, 5]
)

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator, ScalarFormatter

# Define custom color scheme
NAVY_BLUE = "#2D3142"
FILTERED_COLORS = [
    "#E0A500",
    "#4512FA",
    "#A43E2C",
    "#0B7A57",
]  # Different colors for each channel
CMR_COLORS = [
    "#E0A500",
    "#4512FA",
    "#A43E2C",
    "#0B7A57",
]  # Same colors for CMR
REFERENCE_COLOR = "#2D3142"  # Navy blue for reference signal

# Define font sizes with appropriate scaling
FONT_SIZE = 16
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.7)
TICK_SIZE = int(FONT_SIZE * 0.7)
LEGEND_SIZE = int(FONT_SIZE * 0.7)
SCIENTIFIC_NOTATION_SIZE = int(FONT_SIZE * 0.5)

# Define data range
START = 0
END = int(5 * FS)  # 5 seconds of data

# Choose channels to display
SELECTED_CHANNELS = [0, 1, 2, 3]
NUM_CHANNELS = len(SELECTED_CHANNELS)

# Create figure with nested GridSpec
fig = plt.figure(figsize=(7, 6))

# Outer GridSpec: 2 rows, 1 column
outer_gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.1)

# Inner GridSpec for top row (2 columns)
gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[0], wspace=0.1)

# Calculate time array
time_array = np.arange(END - START) / FS

# Compute data first to avoid lazy evaluation issues
filter_data_computed = filtered_data.compute()
cmr_data_computed = cmr_data.compute()

# Calculate normalization and positioning for each channel
y_spread = 1.0  # Vertical spacing between channels
norm_scale = 0.4  # Scale factor for normalized signals
y_offset = 0.0  # Baseline offset

# Calculate channel positions (bottom to top)
channel_positions = []
channel_info = []

for idx, channel in enumerate(SELECTED_CHANNELS):
    # Position channels from bottom to top (reverse order)
    channel_offset = y_offset + (len(SELECTED_CHANNELS) - 1 - idx) * y_spread
    channel_positions.append(channel_offset)
    channel_info.append(f"Ch {channel}")

# Plot 1: Filtered Data (top left)
ax_filtered = fig.add_subplot(gs_top[0, 0])
for idx, channel in enumerate(SELECTED_CHANNELS):
    # Get channel position
    channel_offset = channel_positions[idx]

    # Get data for this channel
    subset_data = filter_data_computed[START:END, channel]

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

    ax_filtered.plot(
        time_array,
        norm_data,
        color=FILTERED_COLORS[idx % len(FILTERED_COLORS)],
        linewidth=1.2,
        alpha=0.8,
    )

# Set up y-axis with channel labels
ax_filtered.set_yticks(channel_positions)
ax_filtered.set_yticklabels(channel_info)
ax_filtered.set_ylim(
    y_offset - 0.5 * y_spread,
    y_offset + len(SELECTED_CHANNELS) * y_spread - 0.5 * y_spread,
)

mpu.format_axis(
    ax_filtered,
    title="Pre-processed EMGdia",
    xlabel="",
    ylabel="EMGdia Channels",
    xlim=(0, (END - START) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Plot 2: CMR Data (top right)
ax_cmr = fig.add_subplot(gs_top[0, 1])
for idx, channel in enumerate(SELECTED_CHANNELS):
    # Get channel position
    channel_offset = channel_positions[idx]

    # Get data for this channel
    subset_data = cmr_data_computed[START:END, channel]

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

    ax_cmr.plot(
        time_array,
        norm_data,
        color=CMR_COLORS[idx % len(CMR_COLORS)],
        linewidth=1.2,
        alpha=0.8,
    )

# Set up y-axis with channel labels
ax_cmr.set_yticks(channel_positions)
ax_cmr.set_yticklabels(channel_info)
ax_cmr.set_ylim(
    y_offset - 0.5 * y_spread,
    y_offset + len(SELECTED_CHANNELS) * y_spread - 0.5 * y_spread,
)

mpu.format_axis(
    ax_cmr,
    title="CMR Filtered EMGdia",
    xlabel="",
    ylabel="",
    xlim=(0, (END - START) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Plot 3: CMR Reference Signal (bottom)
ax_ref = fig.add_subplot(outer_gs[1, 0])
# Get the reference signal and ensure it's the right shape
reference_data = cmr_reference[START:END].compute().flatten()
ax_ref.plot(time_array, reference_data, color=REFERENCE_COLOR, linewidth=1.2)

mpu.format_axis(
    ax_ref,
    title="Common Median Reference (CMR) Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, (END - START) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Apply scientific notation
formatter_ref = ScalarFormatter(useMathText=True)
formatter_ref.set_scientific(True)
formatter_ref.set_powerlimits((-2, 2))
ax_ref.yaxis.set_major_formatter(formatter_ref)
ax_ref.ticklabel_format(
    style="scientific", axis="y", scilimits=(0, 0), useMathText=True
)
ax_ref.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
ax_ref.tick_params(axis="y", labelsize=TICK_SIZE)

# Force y-axis to show only integer values for channel plots
ax_filtered.yaxis.set_major_locator(MaxNLocator(integer=True))
ax_cmr.yaxis.set_major_locator(MaxNLocator(integer=True))

# Hide x-tick labels for top row
for ax in [ax_filtered, ax_cmr]:
    ax.set_xticklabels([])

# Hide y-tick labels for right plot
ax_cmr.set_yticklabels([])

# Add panel labels using mpu.add_panel_label
mpu.add_panel_label(
    ax_filtered,
    "A",
    x_offset_factor=0.15,
    y_offset_factor=0.15,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_cmr,
    "B",
    x_offset_factor=0.1,
    y_offset_factor=0.15,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_ref,
    "C",
    x_offset_factor=0.15,
    y_offset_factor=0.1,
    fontsize=SUBTITLE_SIZE,
)

# Finalize the figure using mpu
mpu.finalize_figure(
    fig,
    title_y=0.98,
    left_margin=0.08,
    top_margin=0.05,
    title_fontsize=TITLE_SIZE,
)

label_x = -0.22  # Adjust this value to move labels left/right

ax_filtered.yaxis.set_label_coords(-0.13, 0.5)
all_axes = [ax_filtered, ax_cmr, ax_ref]

for ax in all_axes:
    # ax.yaxis.set_label_coords(label_x, 0.5)
    ax.tick_params(axis="both", pad=-3)


# Figure output path
FIGURE_TITLE = "fig05_cmr_comparison_analysis"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

# Save figure using mpu
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print("CMR analysis complete.")
# %%
