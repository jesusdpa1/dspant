"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.vizualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

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

# Define data range
channel = 0
start_long = int(0 * fs)
end_long = int(5 * fs)

# Define zoom window (adjust as needed)
zoom_start = int(2 * fs)
zoom_end = int(3 * fs)  # 1 second window

# Get time values for the zoom highlight box
zoom_start_time = zoom_start / fs
zoom_end_time = zoom_end / fs
zoom_width = zoom_end_time - zoom_start_time

# Get raw data
x = base_data.compute()[:, 0]
x_long = x[start_long:end_long]
x_zoom = x[zoom_start:zoom_end]

# Extract data for the long recording
stft_channel_long = stft_data[:, start_long:end_long]
cwt_channel_long = np.flipud(cwt_data[:, start_long:end_long])

# Extract data for the zoomed section
stft_channel_zoom = stft_data[:, zoom_start:zoom_end]
cwt_channel_zoom = np.flipud(cwt_data[:, zoom_start:zoom_end])

# Create time arrays
time_long = np.linspace(0, 5, end_long - start_long)  # 0 to 5 seconds
time_zoom = np.linspace(0, 1, zoom_end - zoom_start)  # Zoom window time

# Filter frequencies below 4000 Hz for all plots
stft_mask = stft_frequencies < 4000
stft_freqs_filtered = stft_frequencies[stft_mask]
stft_channel_long_filtered = stft_channel_long[stft_mask, :]
stft_channel_zoom_filtered = stft_channel_zoom[stft_mask, :]

cwt_mask = cwt_frequencies < 4000
cwt_freqs_filtered = cwt_frequencies[cwt_mask]
cwt_channel_long_filtered = cwt_channel_long[cwt_mask, :]
cwt_channel_zoom_filtered = cwt_channel_zoom[cwt_mask, :]

# CONTRAST ENHANCEMENT: Apply power normalization to enhance contrast
stft_power = 0.7  # Lower values increase contrast
cwt_power = 0.7  # Adjust based on your data

# Create figure with GridSpec for custom layout
fig = plt.figure(figsize=(20, 16))  # Increased height for 3 rows
gs = GridSpec(3, 5, width_ratios=[1, 1, 1, 1, 1])  # Removed the colorbar column

# Mustard yellow highlighter color and dark navy blue for time series
highlight_color = "#FFCC00"  # Bright mustard yellow
dark_navy_blue = "#2D3142"  # Dark navy blue for time series
cmap = "inferno"

# Calculate min/max values for each transform for normalization
stft_min = np.percentile(np.abs(stft_channel_long_filtered), 5)
stft_max = np.percentile(np.abs(stft_channel_long_filtered), 99)

cwt_min = np.percentile(np.abs(cwt_channel_long_filtered), 5)
cwt_max = np.percentile(np.abs(cwt_channel_long_filtered), 99)

# Plot 1: Raw Data (first row, spanning first 4 columns)
ax_raw = fig.add_subplot(gs[0, 0:4])
ax_raw.plot(time_long, x_long, color=dark_navy_blue, linewidth=1.5)
ax_raw.set_xlim(0, 5)
ax_raw.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_raw.tick_params(labelsize=TICK_SIZE)
ax_raw.set_title("Filtered", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for Raw data zoomed region
y_min, y_max = ax_raw.get_ylim()
height = y_max - y_min
raw_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.3,
)
ax_raw.add_patch(raw_rect)

# Plot 2: STFT Long (second row, spanning first 4 columns)
ax_stft = fig.add_subplot(gs[1, 0:4])
stft_mesh = ax_stft.pcolormesh(
    time_long,
    stft_freqs_filtered,
    np.abs(stft_channel_long_filtered),
    cmap=cmap,
    shading="auto",
    norm=PowerNorm(gamma=stft_power, vmin=stft_min, vmax=stft_max),
)
ax_stft.set_ylim(20, 4000)
ax_stft.set_xlim(0, 5)
ax_stft.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_stft.tick_params(labelsize=TICK_SIZE)
ax_stft.set_title("STFT Scalogram", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for STFT zoomed region
y_min, y_max = ax_stft.get_ylim()
height = y_max - y_min
stft_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.3,
)
ax_stft.add_patch(stft_rect)

# Plot 3: CWT Long (third row, spanning first 4 columns)
ax_cwt = fig.add_subplot(gs[2, 0:4])
cwt_mesh = ax_cwt.pcolormesh(
    time_long,
    cwt_freqs_filtered,
    np.abs(cwt_channel_long_filtered),
    cmap=cmap,
    shading="auto",
    norm=PowerNorm(gamma=cwt_power, vmin=cwt_min, vmax=cwt_max),
)
ax_cwt.set_ylim(20, 4000)
ax_cwt.set_xlim(0, 5)
ax_cwt.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_cwt.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_cwt.tick_params(labelsize=TICK_SIZE)
ax_cwt.set_title("CWT Scalogram", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight box for CWT zoomed region
y_min, y_max = ax_cwt.get_ylim()
height = y_max - y_min
cwt_rect = Rectangle(
    (zoom_start_time, y_min),
    zoom_width,
    height,
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.3,
)
ax_cwt.add_patch(cwt_rect)

# Plot 4: Raw Data Zoomed (first row, far right)
ax_raw_zoom = fig.add_subplot(gs[0, 4])
ax_raw_zoom.plot(time_zoom, x_zoom, color=dark_navy_blue, linewidth=1.5)
ax_raw_zoom.set_xlim(0, 1)
ax_raw_zoom.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_raw_zoom.tick_params(labelsize=TICK_SIZE)
ax_raw_zoom.set_title(
    "Filtered Signal (Zoomed)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)

# Add empty colorbar-like space for raw zoom plot to make it match others
divider_raw = make_axes_locatable(ax_raw_zoom)
cax_raw = divider_raw.append_axes("right", size="5%", pad=0.05)
cax_raw.axis("off")  # Hide the axis since we don't need a colorbar here

# Plot 5: STFT Zoomed (second row, far right)
ax_stft_zoom = fig.add_subplot(gs[1, 4])
stft_zoom_mesh = ax_stft_zoom.pcolormesh(
    time_zoom,
    stft_freqs_filtered,
    np.abs(stft_channel_zoom_filtered),
    cmap=cmap,
    shading="auto",
    norm=PowerNorm(gamma=stft_power, vmin=stft_min, vmax=stft_max),
)
ax_stft_zoom.set_ylim(20, 4000)
ax_stft_zoom.set_xlim(0, 1)
ax_stft_zoom.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_stft_zoom.tick_params(labelsize=TICK_SIZE)
ax_stft_zoom.set_title(
    "STFT (Zoomed)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)

# Add STFT colorbar
divider_stft = make_axes_locatable(ax_stft_zoom)
cax_stft = divider_stft.append_axes("right", size="5%", pad=0.05)
cbar_stft = plt.colorbar(stft_zoom_mesh, cax=cax_stft)
cbar_stft.ax.tick_params(labelsize=TICK_SIZE)

# Plot 6: CWT Zoomed (third row, far right)
ax_cwt_zoom = fig.add_subplot(gs[2, 4])
cwt_zoom_mesh = ax_cwt_zoom.pcolormesh(
    time_zoom,
    cwt_freqs_filtered,
    np.abs(cwt_channel_zoom_filtered),
    cmap=cmap,
    shading="auto",
    norm=PowerNorm(gamma=cwt_power, vmin=cwt_min, vmax=cwt_max),
)
ax_cwt_zoom.set_ylim(20, 4000)
ax_cwt_zoom.set_xlim(0, 1)
ax_cwt_zoom.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_cwt_zoom.set_ylabel("Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_cwt_zoom.tick_params(labelsize=TICK_SIZE)
ax_cwt_zoom.set_title(
    "CWT (Zoomed)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)

# Add CWT colorbar
divider_cwt = make_axes_locatable(ax_cwt_zoom)
cax_cwt = divider_cwt.append_axes("right", size="5%", pad=0.05)
cbar_cwt = plt.colorbar(cwt_zoom_mesh, cax=cax_cwt)
cbar_cwt.ax.tick_params(labelsize=TICK_SIZE)

# Add overall title
plt.suptitle(
    "Time-Frequency Analysis Comparison", fontsize=TITLE_SIZE, fontweight="bold", y=0.98
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

plt.show()
# %%
