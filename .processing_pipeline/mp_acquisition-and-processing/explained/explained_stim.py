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
import mp_plotting_utils as mpu
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
    create_single_threshold_detector,
)

mpu.set_publication_style()
sns.set_theme(style="darkgrid")
dotenv.load_dotenv()
# %%
BASE_PATH = Path(os.getenv("DATA_DIR"))
DATA_PATH = BASE_PATH.joinpath(
    r"E:\jpenalozaa\papers\2025_mp_emg diaphragm acquisition and processing\drv_15-40-17_stim"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

EMG_PATH = DATA_PATH.joinpath(r"RawG.ant")
STIM_PATH = DATA_PATH.joinpath(r"MonA.ant")
# INSP_PATH = DATA_PATH.joinpath(r"insp.ant")
# %%
# Load EMG data
STREAM_EMG = StreamNode(str(EMG_PATH))
STREAM_EMG.load_metadata()
STREAM_EMG.load_data()
# Print STREAM_EMG summary
STREAM_EMG.summarize()


STREAM_STIM = StreamNode(str(STIM_PATH))
STREAM_STIM.load_metadata()
STREAM_STIM.load_data()
# Print STREAM_EMG summary
STREAM_STIM.summarize()


# %%
# Create and visualize filters before applying them
FS = STREAM_EMG.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
BANDPASS_FILTER = create_bandpass_filter(10, 4000, fs=FS, order=5)
NOTCH_FILTER = create_notch_filter(60, q=60, fs=FS)
LOWPASS_FILTER = create_lowpass_filter(50, FS)
# %%
# BANDPASS_PLOT = BANDPASS_FILTER.plot_frequency_response()
# NOTCH_PLOT = NOTCH_FILTER.plot_frequency_response()
# LOWPASS_PLOT = LOWPASS_FILTER.plot_frequency_response()
# %%
# Create processing node with filters
PROCESSOR_EMG = create_processing_node(STREAM_EMG)
# %%
# Create processors
NOTCH_PROCESSOR = FilterProcessor(
    filter_func=NOTCH_FILTER.get_filter_function(), overlap_samples=40
)
BANDPASS_PROCESSOR = FilterProcessor(
    filter_func=BANDPASS_FILTER.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
PROCESSOR_EMG.add_processor([NOTCH_PROCESSOR, BANDPASS_PROCESSOR], group="filters")

# Apply filters and plot results
FILTERED_EMG = PROCESSOR_EMG.process(group=["filters"]).persist()

# %%
FILTERED_DATA = da.concatenate([FILTERED_EMG[:, 1:2], STREAM_STIM.data[:, :1]], axis=1)

# %%

MULTICHANNEL_FIG = plot_multi_channel_data(FILTERED_DATA, fs=FS, time_window=[400, 440])

# %%

TKEO_PROCESSOR = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
TKEO_DATA = TKEO_PROCESSOR.process(FILTERED_EMG, fs=FS).persist()

# %%
TKEO_FIG = plot_multi_channel_data(TKEO_DATA, fs=FS, time_window=[100, 110])
# %%

DATA_ORGANIZED = da.concatenate(
    [
        STREAM_STIM.data[:, :1],
        TKEO_DATA[:, 1:2].compute(),
        FILTERED_EMG[:, 1:2],
    ],
    axis=1,
)
# %%
AGG_FIG = plot_multi_channel_data(DATA_ORGANIZED, fs=FS, time_window=[400, 800])
# %%

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


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


# Set the colorblind-friendly palette
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")

# Define font sizes with appropriate scaling
FONT_SIZE = 30
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.9)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.7)
TICK_SIZE = int(FONT_SIZE * 0.6)
LEGEND_SIZE = int(FONT_SIZE * 0.6)

# Larger font size for scientific notation
SCIENTIFIC_NOTATION_SIZE = int(FONT_SIZE * 0.5)

# Assuming we have these variables from the original code:
# STREAM_STIM.data, TKEO_DATA, FILTERED_EMG, FS

# Define data range for full view and zoomed view
FULL_DURATION = 100  # seconds
ZOOM_DURATION = 30  # seconds

START = int(400 * FS)
END = int(START + (FULL_DURATION * FS))

# The key change: calculate zoom start/end in relation to the time axis, not sample indices
ZOOM_START_SAMPLE = int(425 * FS)  # Start zoom at 440s in absolute time
ZOOM_END_SAMPLE = int(ZOOM_START_SAMPLE + (ZOOM_DURATION * FS))

# Convert to relative time for plotting - this is crucial
ZOOM_START_TIME = (ZOOM_START_SAMPLE - START) / FS  # Time relative to the plotted start
ZOOM_END_TIME = (ZOOM_END_SAMPLE - START) / FS
ZOOM_WIDTH = ZOOM_END_TIME - ZOOM_START_TIME

# Calculate time arrays
TIME_ARRAY = np.arange(END - START) / FS
ZOOM_TIME_ARRAY = np.arange(ZOOM_END_SAMPLE - ZOOM_START_SAMPLE) / FS

# Define data arrays for plotting
STIM_DATA_FULL = STREAM_STIM.data[START:END, 0]
TKEO_DATA_FULL = TKEO_DATA[START:END, 1]
EMG_DATA_FULL = FILTERED_EMG[START:END, 1]

STIM_DATA_ZOOM = STREAM_STIM.data[ZOOM_START_SAMPLE:ZOOM_END_SAMPLE, 0]
TKEO_DATA_ZOOM = TKEO_DATA[ZOOM_START_SAMPLE:ZOOM_END_SAMPLE, 1]
EMG_DATA_ZOOM = FILTERED_EMG[ZOOM_START_SAMPLE:ZOOM_END_SAMPLE, 1]

# Create figure with GridSpec for custom layout
# 6 rows, 2 columns with equal width
fig = plt.figure(figsize=(25, 12))
gs = GridSpec(6, 2, width_ratios=[1, 1])

# Dark grey with navy tint for main lines
DARK_GREY_NAVY = "#2D3142"

# Choose a distinct color for the highlight box
HIGHLIGHT_COLOR = PALETTE[3]  # Using a distinct color from the palette

# Left column plots (spans first column)

# Plot 1: Stimulation Data (top)
ax_stim = fig.add_subplot(gs[0:2, 0])
ax_stim.plot(TIME_ARRAY, STIM_DATA_FULL, color=DARK_GREY_NAVY, linewidth=2)

mpu.format_axis(
    ax_stim,
    title="Stimulation Signal",
    xlabel="",
    ylabel="Amplitude",
    xlim=(0, FULL_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_stim, SCIENTIFIC_NOTATION_SIZE)

# Add highlight box for zoomed region - now using correct time positioning
y_min, y_max = ax_stim.get_ylim()
height = y_max - y_min
stim_rect = Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_stim.add_patch(stim_rect)

# Plot 2: TKEO Data (middle)
ax_tkeo = fig.add_subplot(gs[2:4, 0])
ax_tkeo.plot(TIME_ARRAY, TKEO_DATA_FULL, color=PALETTE[1], linewidth=2)

mpu.format_axis(
    ax_tkeo,
    title="TKEO Envelope",
    xlabel="",
    ylabel="Amplitude",
    xlim=(0, FULL_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_tkeo, SCIENTIFIC_NOTATION_SIZE)

# Add highlight box for zoomed region
y_min, y_max = ax_tkeo.get_ylim()
height = y_max - y_min
tkeo_rect = Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_tkeo.add_patch(tkeo_rect)

# Plot 3: EMG Channel (bottom)
ax_emg = fig.add_subplot(gs[4:6, 0])
ax_emg.plot(TIME_ARRAY, EMG_DATA_FULL, color=PALETTE[2], linewidth=2)

mpu.format_axis(
    ax_emg,
    title="Filtered EMG Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, FULL_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_emg, SCIENTIFIC_NOTATION_SIZE)

# Add highlight box for zoomed region
y_min, y_max = ax_emg.get_ylim()
height = y_max - y_min
emg_rect = Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_emg.add_patch(emg_rect)

# Right column plots (spans second column)

# Plot 4: Stimulation Zoomed (top)
ax_stim_zoom = fig.add_subplot(gs[0:2, 1])
ax_stim_zoom.plot(ZOOM_TIME_ARRAY, STIM_DATA_ZOOM, color=DARK_GREY_NAVY, linewidth=2)

mpu.format_axis(
    ax_stim_zoom,
    title="Stimulation Signal (Zoomed)",
    xlabel="",
    ylabel="Amplitude",
    xlim=(0, ZOOM_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_stim_zoom, SCIENTIFIC_NOTATION_SIZE)

# Plot 5: TKEO Zoomed (middle)
ax_tkeo_zoom = fig.add_subplot(gs[2:4, 1])
ax_tkeo_zoom.plot(ZOOM_TIME_ARRAY, TKEO_DATA_ZOOM, color=PALETTE[1], linewidth=2)

mpu.format_axis(
    ax_tkeo_zoom,
    title="TKEO Envelope (Zoomed)",
    xlabel="",
    ylabel="Amplitude",
    xlim=(0, ZOOM_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_tkeo_zoom, SCIENTIFIC_NOTATION_SIZE)

# Plot 6: EMG Zoomed (bottom)
ax_emg_zoom = fig.add_subplot(gs[4:6, 1])
ax_emg_zoom.plot(ZOOM_TIME_ARRAY, EMG_DATA_ZOOM, color=PALETTE[2], linewidth=2)

mpu.format_axis(
    ax_emg_zoom,
    title="Filtered EMG Signal (Zoomed)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, ZOOM_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_emg_zoom, SCIENTIFIC_NOTATION_SIZE)

# Finalize the figure with mpu
mpu.finalize_figure(
    fig,
    # title="EMG Signal Processing with Stimulation",
    title_y=0.98,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout
plt.tight_layout(rect=[0, 0, 1, 0.95])


# Add panel labels
mpu.add_panel_label(
    ax_stim,
    "A",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Row 2 labels
mpu.add_panel_label(
    ax_stim_zoom,
    "B",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_tkeo,
    "C",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_tkeo_zoom,
    "D",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Row 3 labels
mpu.add_panel_label(
    ax_emg,
    "E",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_emg_zoom,
    "F",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)


# Figure output path
FIGURE_TITLE = "stim_explained"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")
# Save the figure
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

# %%
