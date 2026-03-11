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
from matplotlib.ticker import ScalarFormatter
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
SIGNAL_COLOR = mpu.PRIMARY_COLOR  # Dark navy blue for time series
COLORMAP = "inferno"

# %%
# Load data - This would come from your actual data loading code
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
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

# Define frequency range for visualization
MAX_FREQ = 4000
MIN_FREQ = 20

# Get raw data
X = base_data.compute()[:, CHANNEL]
X_LONG = X[START_LONG:END_LONG]

# Extract data for the long recording
STFT_CHANNEL_LONG = stft_data[:, START_LONG:END_LONG]
CWT_CHANNEL_LONG = np.flipud(cwt_data[:, START_LONG:END_LONG])

# Create time arrays
TIME_LONG = np.linspace(0, 5, END_LONG - START_LONG)  # 0 to 5 seconds

# Filter frequencies below MAX_FREQ Hz for all plots
STFT_MASK = stft_frequencies < MAX_FREQ
STFT_FREQS_FILTERED = stft_frequencies[STFT_MASK]
STFT_CHANNEL_LONG_FILTERED = STFT_CHANNEL_LONG[STFT_MASK, :]

CWT_MASK = cwt_frequencies < MAX_FREQ
CWT_FREQS_FILTERED = cwt_frequencies[CWT_MASK]
CWT_CHANNEL_LONG_FILTERED = CWT_CHANNEL_LONG[CWT_MASK, :]

# Calculate min/max values for each transform for normalization
STFT_MIN = np.percentile(np.abs(STFT_CHANNEL_LONG_FILTERED), 5)
STFT_MAX = np.percentile(np.abs(STFT_CHANNEL_LONG_FILTERED), 99)

CWT_MIN = np.percentile(np.abs(CWT_CHANNEL_LONG_FILTERED), 5)
CWT_MAX = np.percentile(np.abs(CWT_CHANNEL_LONG_FILTERED), 99)

# Define font sizes with appropriate scaling

# %%
# Define font sizes with appropriate scaling (FROM IIR VS FIR PLOT)
FONT_SIZE = 16
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.7)
TICK_SIZE = int(FONT_SIZE * 0.65)

# Create figure with 3-row, 2-column layout (main plots + colorbar column)
FIG = plt.figure(figsize=(7, 5))
GS = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[20, 1])

# Plot 1: Raw Data (first row, first column)
AX_RAW = FIG.add_subplot(GS[0, 0])
AX_RAW.plot(TIME_LONG, X_LONG, color=SIGNAL_COLOR, linewidth=1.2)
mpu.format_axis(
    AX_RAW,
    title="Filtered EMGdia Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 5),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Format y-axis of raw data plot to scientific notation
formatter_raw = ScalarFormatter(useMathText=True)
formatter_raw.set_scientific(True)
formatter_raw.set_powerlimits((-2, 2))  # Forces scientific notation
AX_RAW.yaxis.set_major_formatter(formatter_raw)
AX_RAW.ticklabel_format(
    style="scientific", axis="y", scilimits=(0, 0), useMathText=True
)
AX_RAW.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

# Plot 2: STFT (second row, first column)
AX_STFT = FIG.add_subplot(GS[1, 0], sharex=AX_RAW)
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
    title="Short-Time Fourier Transform (STFT) Scalogram",
    xlabel=None,
    ylabel="Frequency (Hz)",
    xlim=(0, 5),
    ylim=(MIN_FREQ, MAX_FREQ),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Format y-axis of STFT plot to scientific notation
formatter_stft = ScalarFormatter(useMathText=True)
formatter_stft.set_scientific(True)
formatter_stft.set_powerlimits((-2, 2))  # Forces scientific notation
AX_STFT.yaxis.set_major_formatter(formatter_stft)
AX_STFT.ticklabel_format(
    style="scientific", axis="y", scilimits=(0, 0), useMathText=True
)
AX_STFT.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

# Add STFT colorbar in second column
CAX_STFT = FIG.add_subplot(GS[1, 1])
CBAR_STFT = plt.colorbar(STFT_MESH, cax=CAX_STFT)
CAX_STFT.tick_params(labelsize=TICK_SIZE)

# Format STFT colorbar to scientific notation
formatter_cbar_stft = ScalarFormatter(useMathText=True)
formatter_cbar_stft.set_scientific(True)
formatter_cbar_stft.set_powerlimits((-2, 2))
CAX_STFT.yaxis.set_major_formatter(formatter_cbar_stft)
CAX_STFT.ticklabel_format(
    style="scientific", axis="y", scilimits=(0, 0), useMathText=True
)
CAX_STFT.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

# Plot 3: CWT (third row, first column)
AX_CWT = FIG.add_subplot(GS[2, 0], sharex=AX_RAW)
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
    title="Continuous Wavelet Transform (CWT) Scalogram",
    xlabel="Time (s)",
    ylabel="Frequency (Hz)",
    xlim=(0, 5),
    ylim=(MIN_FREQ, MAX_FREQ),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Format y-axis of CWT plot to scientific notation
formatter_cwt = ScalarFormatter(useMathText=True)
formatter_cwt.set_scientific(True)
formatter_cwt.set_powerlimits((-2, 2))  # Forces scientific notation
AX_CWT.yaxis.set_major_formatter(formatter_cwt)
AX_CWT.ticklabel_format(
    style="scientific", axis="y", scilimits=(0, 0), useMathText=True
)
AX_CWT.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

# Add CWT colorbar in second column
CAX_CWT = FIG.add_subplot(GS[2, 1])
CBAR_CWT = plt.colorbar(CWT_MESH, cax=CAX_CWT)
CAX_CWT.tick_params(labelsize=TICK_SIZE)

# Format CWT colorbar to scientific notation
formatter_cbar_cwt = ScalarFormatter(useMathText=True)
formatter_cbar_cwt.set_scientific(True)
formatter_cbar_cwt.set_powerlimits((-2, 2))
CAX_CWT.yaxis.set_major_formatter(formatter_cbar_cwt)
CAX_CWT.ticklabel_format(
    style="scientific", axis="y", scilimits=(0, 0), useMathText=True
)
CAX_CWT.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

# Create axes list for consistent formatting
all_axes = [AX_RAW, AX_STFT, AX_CWT]

# Align all y-axis labels (FROM IIR VS FIR PLOT)
label_x = -0.05  # Adjusted for this layout
for ax in all_axes:
    ax.yaxis.set_label_coords(label_x, 0.5)

for ax in all_axes[:-1]:
    ax.xaxis.set_visible(False)

# Add panel labels using mpu.add_panel_label
mpu.add_panel_label(
    AX_RAW,
    "A",
    x_offset_factor=0.26,
    y_offset_factor=0.09,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    AX_STFT,
    "B",
    x_offset_factor=0.26,
    y_offset_factor=0.09,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    AX_CWT,
    "C",
    x_offset_factor=0.26,
    y_offset_factor=0.09,
    fontsize=SUBTITLE_SIZE,
)

# Finalize the figure
mpu.finalize_figure(
    FIG,
    title_y=0.96,
    left_margin=0.01,
    hspace=0.4,
    wspace=0.01,
    top_margin=0.1,
    title_fontsize=TITLE_SIZE,
)


for ax in all_axes:
    ax.tick_params(axis="both", pad=-5, labelsize=TICK_SIZE)

AX_RAW.yaxis.set_label_coords(-0.03, 0.5)
AX_STFT.yaxis.set_label_coords(-0.03, 0.5)
AX_CWT.yaxis.set_label_coords(-0.03, 0.5)
# %%
# Figure output path
FIGURE_TITLE = "fig06_time_frequency_analysis_comparison"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

# Save the figure using mpu
mpu.save_figure(FIG, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print("Time-frequency analysis complete.")

# %%
