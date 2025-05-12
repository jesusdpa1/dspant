"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data

load_dotenv()
sns.set_theme(style="darkgrid")
# %%
data_dir = Path(os.getenv("DATA_DIR"))
base_path = data_dir.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
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

# View summary of the processing node
processor_emg.summarize()

# %%
# Apply filters and plot results
filter_data = processor_emg.process(group=["filters"]).persist()
# %%
multichannel_fig = plot_multi_channel_data(filter_data, fs=fs, time_window=[0, 10])

# %%
start = int(fs * 0)
end = int(fs * 30)
base_data = filter_data[start:end, :]
plt.plot(base_data)

# %%
from ssqueezepy import Wavelet, cwt, imshow, stft
from ssqueezepy.experimental import scale_to_freq

x = base_data.compute()[:, 0]
wavelet = Wavelet()
cwt_data, scales = cwt(x, wavelet)
stft_data = stft(x)[::-1]
cwt_data = np.flipud(cwt_data)
cwt_frequencies = scale_to_freq(scales, wavelet, len(x), fs=fs)
stft_frequencies = np.linspace(1, 0, len(stft_data)) * fs / 2

# %%
"""
Time-Frequency Analysis Visualization
Author: Jesus Penaloza (Enhanced for publication-quality plots)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# CONSTANTS - Using capital letters for constants
# Color constants
HIGHLIGHT_COLOR = "#FFCC00"  # Bright mustard yellow
SIGNAL_COLOR = "#2D3142"  # Dark navy blue for time series
COLORMAP = "inferno"

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")

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

# Signal processing parameters
CONTRAST_STFT = 0.7  # Lower values increase contrast
CONTRAST_CWT = 0.7  # Adjust based on your data

# Define data range
CHANNEL = 0
START_LONG = int(0 * fs)
END_LONG = int(5 * fs)

# Define zoom window (adjust as needed)
ZOOM_START = int(2 * fs)
ZOOM_END = int(3 * fs)  # 1 second window

# Get time values for the zoom highlight box
ZOOM_START_TIME = ZOOM_START / fs
ZOOM_END_TIME = ZOOM_END / fs
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

# Create figure with GridSpec for custom layout
FIG = plt.figure(figsize=(20, 16))  # Increased height for 3 rows
GS = GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 1])  # Removed the colorbar column

# Plot 1: Raw Data (first row, spanning first 4 columns)
AX_RAW = FIG.add_subplot(GS[0, 0:4])
AX_RAW.plot(TIME_LONG, X_LONG, color=SIGNAL_COLOR, linewidth=1.5)
AX_RAW.set_xlim(0, 5)
AX_RAW.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_RAW.tick_params(labelsize=TICK_SIZE)
AX_RAW.set_title("Filtered", fontsize=SUBTITLE_SIZE, weight="bold")

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
AX_STFT.set_ylim(MIN_FREQ, MAX_FREQ)
AX_STFT.set_xlim(0, 5)
AX_STFT.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_STFT.tick_params(labelsize=TICK_SIZE)
AX_STFT.set_title("STFT Scalogram", fontsize=SUBTITLE_SIZE, weight="bold")

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
AX_CWT.set_ylim(MIN_FREQ, MAX_FREQ)
AX_CWT.set_xlim(0, 5)
AX_CWT.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_CWT.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_CWT.tick_params(labelsize=TICK_SIZE)
AX_CWT.set_title("CWT Scalogram", fontsize=SUBTITLE_SIZE, weight="bold")

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
AX_RAW_ZOOM.set_xlim(0, 1)
AX_RAW_ZOOM.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_RAW_ZOOM.tick_params(labelsize=TICK_SIZE)
AX_RAW_ZOOM.set_title(
    "Filtered Signal (Zoomed)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
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
AX_STFT_ZOOM.set_ylim(MIN_FREQ, MAX_FREQ)
AX_STFT_ZOOM.set_xlim(0, 1)
AX_STFT_ZOOM.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_STFT_ZOOM.tick_params(labelsize=TICK_SIZE)
AX_STFT_ZOOM.set_title(
    "STFT (Zoomed)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)

# Add STFT colorbar
DIVIDER_STFT = make_axes_locatable(AX_STFT_ZOOM)
CAX_STFT = DIVIDER_STFT.append_axes("right", size="5%", pad=0.05)
CBAR_STFT = plt.colorbar(STFT_ZOOM_MESH, cax=CAX_STFT)
CBAR_STFT.ax.tick_params(labelsize=TICK_SIZE)

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
AX_CWT_ZOOM.set_ylim(MIN_FREQ, MAX_FREQ)
AX_CWT_ZOOM.set_xlim(0, 1)
AX_CWT_ZOOM.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_CWT_ZOOM.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
AX_CWT_ZOOM.tick_params(labelsize=TICK_SIZE)
AX_CWT_ZOOM.set_title(
    "CWT (Zoomed)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)

# Add CWT colorbar
DIVIDER_CWT = make_axes_locatable(AX_CWT_ZOOM)
CAX_CWT = DIVIDER_CWT.append_axes("right", size="5%", pad=0.05)
CBAR_CWT = plt.colorbar(CWT_ZOOM_MESH, cax=CAX_CWT)
CBAR_CWT.ax.tick_params(labelsize=TICK_SIZE)

# Add overall title
plt.suptitle(
    "Time-Frequency Analysis Comparison", fontsize=TITLE_SIZE, fontweight="bold", y=0.98
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

# Save the figure at 600 DPI
plt.savefig("time_frequency_analysis.png", dpi=600, bbox_inches="tight")

# Show the figure
plt.show()
