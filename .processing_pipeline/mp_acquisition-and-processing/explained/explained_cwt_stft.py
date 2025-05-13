"""
Time-Frequency Analysis Visualization
Author: Jesus Penaloza (Enhanced with mp_plotting_utils for publication-quality plots)
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
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ssqueezepy import Wavelet, cwt, imshow, stft
from ssqueezepy.experimental import scale_to_freq

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data

# Set publication style
mpu.set_publication_style()
load_dotenv()

# %%
# CONSTANTS - Using capital letters for constants
# Color constants
HIGHLIGHT_COLOR = "#FFCC00"  # Bright mustard yellow
SIGNAL_COLOR = mpu.PRIMARY_COLOR  # Dark navy blue for time series
COLORMAP = "inferno"


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


# %%
# Load data - This would come from your actual data loading code
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")

# Load EMG data
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()
FS = stream_emg.fs  # Get sampling rate from the stream node

# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=60, fs=FS)

# Create processing node with filters
processor_emg = create_processing_node(stream_emg)
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters and get data
filter_data = processor_emg.process(group=["filters"]).persist()

# Extract data for time-frequency analysis
START = int(FS * 0)
END = int(FS * 30)
base_data = filter_data[START:END, :]

# %%
# Perform time-frequency analysis
x = base_data.compute()[:, 0]
wavelet = Wavelet()
cwt_data, scales = cwt(x, wavelet)
stft_data = stft(x)[::-1]
cwt_data = np.flipud(cwt_data)
cwt_frequencies = scale_to_freq(scales, wavelet, len(x), fs=FS)
stft_frequencies = np.linspace(1, 0, len(stft_data)) * FS / 2

# %%
# Signal processing parameters
CONTRAST_STFT = 0.7  # Lower values increase contrast
CONTRAST_CWT = 0.7  # Adjust based on your data

# Define data range
CHANNEL = 0
START_LONG = int(0 * FS)
END_LONG = int(5 * FS)

# Define zoom window (adjust as needed)
ZOOM_START = int(2 * FS)
ZOOM_END = int(3 * FS)  # 1 second window

# Get time values for the zoom highlight box
ZOOM_START_TIME = ZOOM_START / FS
ZOOM_END_TIME = ZOOM_END / FS
ZOOM_WIDTH = ZOOM_END_TIME - ZOOM_START_TIME

# Define frequency range for visualization
MAX_FREQ = 4000
MIN_FREQ = 20

# Get raw data
X = base_data.compute()[:, CHANNEL]
X_LONG = X[START_LONG:END_LONG]
X_ZOOM = X[ZOOM_START:ZOOM_END]

# Extract data for the long recording
STFT_CHANNEL_LONG = stft_data[:, START_LONG:END_LONG]
CWT_CHANNEL_LONG = np.flipud(cwt_data[:, START_LONG:END_LONG])

# Extract data for the zoomed section
STFT_CHANNEL_ZOOM = stft_data[:, ZOOM_START:ZOOM_END]
CWT_CHANNEL_ZOOM = np.flipud(cwt_data[:, ZOOM_START:ZOOM_END])

# Create time arrays
TIME_LONG = np.linspace(0, 5, END_LONG - START_LONG)  # 0 to 5 seconds
TIME_ZOOM = np.linspace(0, 1, ZOOM_END - ZOOM_START)  # Zoom window time

# Filter frequencies below MAX_FREQ Hz for all plots
STFT_MASK = stft_frequencies < MAX_FREQ
STFT_FREQS_FILTERED = stft_frequencies[STFT_MASK]
STFT_CHANNEL_LONG_FILTERED = STFT_CHANNEL_LONG[STFT_MASK, :]
STFT_CHANNEL_ZOOM_FILTERED = STFT_CHANNEL_ZOOM[STFT_MASK, :]

CWT_MASK = cwt_frequencies < MAX_FREQ
CWT_FREQS_FILTERED = cwt_frequencies[CWT_MASK]
CWT_CHANNEL_LONG_FILTERED = CWT_CHANNEL_LONG[CWT_MASK, :]
CWT_CHANNEL_ZOOM_FILTERED = CWT_CHANNEL_ZOOM[CWT_MASK, :]

# Calculate min/max values for each transform for normalization
STFT_MIN = np.percentile(np.abs(STFT_CHANNEL_LONG_FILTERED), 5)
STFT_MAX = np.percentile(np.abs(STFT_CHANNEL_LONG_FILTERED), 99)

CWT_MIN = np.percentile(np.abs(CWT_CHANNEL_LONG_FILTERED), 5)
CWT_MAX = np.percentile(np.abs(CWT_CHANNEL_LONG_FILTERED), 99)

# %%
# Create figure with GridSpec for custom layout
FIG = plt.figure(figsize=(20, 16))  # Increased height for 3 rows
GS = GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 1])  # 5 columns layout

# Plot 1: Raw Data (first row, spanning first 4 columns)
AX_RAW = FIG.add_subplot(GS[0, 0:4])
AX_RAW.plot(TIME_LONG, X_LONG, color=SIGNAL_COLOR, linewidth=1.5)
mpu.format_axis(
    AX_RAW,
    title="Filtered",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 5),
)

# Add highlight box for Raw data zoomed region
Y_MIN, Y_MAX = AX_RAW.get_ylim()
HEIGHT = Y_MAX - Y_MIN
RAW_RECT = Rectangle(
    (ZOOM_START_TIME, Y_MIN),
    ZOOM_WIDTH,
    HEIGHT,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.3,
)
AX_RAW.add_patch(RAW_RECT)

# Plot 2: STFT Long (second row, spanning first 4 columns)
AX_STFT = FIG.add_subplot(GS[1, 0:4])
STFT_MESH = AX_STFT.pcolormesh(
    TIME_LONG,
    STFT_FREQS_FILTERED,
    np.abs(STFT_CHANNEL_LONG_FILTERED),
    cmap=COLORMAP,
    shading="auto",
    norm=PowerNorm(gamma=CONTRAST_STFT, vmin=STFT_MIN, vmax=STFT_MAX),
)
mpu.format_axis(
    AX_STFT,
    title="STFT Scalogram",
    xlabel=None,
    ylabel="Frequency (Hz)",
    xlim=(0, 5),
    ylim=(MIN_FREQ, MAX_FREQ),
)

# Add highlight box for STFT zoomed region
Y_MIN, Y_MAX = AX_STFT.get_ylim()
HEIGHT = Y_MAX - Y_MIN
STFT_RECT = Rectangle(
    (ZOOM_START_TIME, Y_MIN),
    ZOOM_WIDTH,
    HEIGHT,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.3,
)
AX_STFT.add_patch(STFT_RECT)

# Plot 3: CWT Long (third row, spanning first 4 columns)
AX_CWT = FIG.add_subplot(GS[2, 0:4])
CWT_MESH = AX_CWT.pcolormesh(
    TIME_LONG,
    CWT_FREQS_FILTERED,
    np.abs(CWT_CHANNEL_LONG_FILTERED),
    cmap=COLORMAP,
    shading="auto",
    norm=PowerNorm(gamma=CONTRAST_CWT, vmin=CWT_MIN, vmax=CWT_MAX),
)
mpu.format_axis(
    AX_CWT,
    title="CWT Scalogram",
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    xlim=(0, 5),
    ylim=(MIN_FREQ, MAX_FREQ),
)

# Add highlight box for CWT zoomed region
Y_MIN, Y_MAX = AX_CWT.get_ylim()
HEIGHT = Y_MAX - Y_MIN
CWT_RECT = Rectangle(
    (ZOOM_START_TIME, Y_MIN),
    ZOOM_WIDTH,
    HEIGHT,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.3,
)
AX_CWT.add_patch(CWT_RECT)

# Plot 4: Raw Data Zoomed (first row, far right)
AX_RAW_ZOOM = FIG.add_subplot(GS[0, 4])
AX_RAW_ZOOM.plot(TIME_ZOOM, X_ZOOM, color=SIGNAL_COLOR, linewidth=1.5)
mpu.format_axis(
    AX_RAW_ZOOM,
    title="Filtered Signal (Zoomed)",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 1),
)

# Add empty colorbar-like space for raw zoom plot to make it match others
DIVIDER_RAW = make_axes_locatable(AX_RAW_ZOOM)
CAX_RAW = DIVIDER_RAW.append_axes("right", size="5%", pad=0.05)
CAX_RAW.axis("off")  # Hide the axis since we don't need a colorbar here

# Plot 5: STFT Zoomed (second row, far right)
AX_STFT_ZOOM = FIG.add_subplot(GS[1, 4])
STFT_ZOOM_MESH = AX_STFT_ZOOM.pcolormesh(
    TIME_ZOOM,
    STFT_FREQS_FILTERED,
    np.abs(STFT_CHANNEL_ZOOM_FILTERED),
    cmap=COLORMAP,
    shading="auto",
    norm=PowerNorm(gamma=CONTRAST_STFT, vmin=STFT_MIN, vmax=STFT_MAX),
)
mpu.format_axis(
    AX_STFT_ZOOM,
    title="STFT (Zoomed)",
    xlabel=None,
    ylabel="Frequency (Hz)",
    xlim=(0, 1),
    ylim=(MIN_FREQ, MAX_FREQ),
)

# Add STFT colorbar
DIVIDER_STFT = make_axes_locatable(AX_STFT_ZOOM)
CAX_STFT = DIVIDER_STFT.append_axes("right", size="5%", pad=0.05)
CBAR_STFT = plt.colorbar(STFT_ZOOM_MESH, cax=CAX_STFT)
CAX_STFT.tick_params(labelsize=mpu.FONT_SIZES["tick_label"])

# Plot 6: CWT Zoomed (third row, far right)
AX_CWT_ZOOM = FIG.add_subplot(GS[2, 4])
CWT_ZOOM_MESH = AX_CWT_ZOOM.pcolormesh(
    TIME_ZOOM,
    CWT_FREQS_FILTERED,
    np.abs(CWT_CHANNEL_ZOOM_FILTERED),
    cmap=COLORMAP,
    shading="auto",
    norm=PowerNorm(gamma=CONTRAST_CWT, vmin=CWT_MIN, vmax=CWT_MAX),
)
mpu.format_axis(
    AX_CWT_ZOOM,
    title="CWT (Zoomed)",
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    xlim=(0, 1),
    ylim=(MIN_FREQ, MAX_FREQ),
)

# Add CWT colorbar
DIVIDER_CWT = make_axes_locatable(AX_CWT_ZOOM)
CAX_CWT = DIVIDER_CWT.append_axes("right", size="5%", pad=0.05)
CBAR_CWT = plt.colorbar(CWT_ZOOM_MESH, cax=CAX_CWT)
CAX_CWT.tick_params(labelsize=mpu.FONT_SIZES["tick_label"])

# Finalize the figure
mpu.finalize_figure(
    FIG,
    title="Time-Frequency Analysis Comparison",
    title_y=0.98,
)

# Apply tight layout before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

# Add panel labels
add_panel_label(AX_RAW, "A", offset_factor=0.07)
add_panel_label(AX_STFT, "B", offset_factor=0.07)
add_panel_label(AX_CWT, "C", offset_factor=0.07)
add_panel_label(AX_RAW_ZOOM, "D", offset_factor=0.07)
add_panel_label(AX_STFT_ZOOM, "E", offset_factor=0.07)
add_panel_label(AX_CWT_ZOOM, "F", offset_factor=0.07)

# Save the figure using mpu
mpu.save_figure(FIG, "time_frequency_analysis.png", dpi=600)

plt.show()
# %%
