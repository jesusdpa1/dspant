"""
Testing zarr saved data
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import os
import time
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
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
from dspant.processors.filters.fir_filters import (
    create_moving_average,
    create_moving_rms,
)
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")

# %%

data_path = Path(os.getenv("DATA_DIR"))
base_path = data_path.joinpath(
    r"topoMapping/25-03-26_4902-1_testSubject_topoMapping/drv/zarr_test"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_stream_path = base_path.joinpath(r"data_RawG.zarr")

# %%
# Load EMG data
stream_emg = StreamNode(str(emg_stream_path), source="zarr")
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node
#%%
# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(20, fs)
# %%
bandpass_plot = bandpass_filter.plot_frequency_response()
notch_plot = notch_filter.plot_frequency_response()
lowpass_plot = lowpass_filter.plot_frequency_response()
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


lowpass_processor = FilterProcessor(
    filter_func=lowpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
# %%
# View summary of the processing node
processor_emg.summarize()

# %%
# Apply filters and plot results
filter_data = processor_emg.process(group=["filters"]).persist()
# %%
multichannel_fig = plot_multi_channel_data(filter_data, fs=fs, time_window=[0, 10])
square_processor = RectificationProcessor("square")
data_rectified = square_processor.process(filter_data)
# %%
start = int(fs * 5)
end = int(fs * 10)
base_raw_data = filter_data[start:end, :]
base_data = data_rectified[start:end, :]
plt.plot(base_data)
# %%
window_size = int(0.025 * fs)
moving_envelope_processor = create_moving_average(window_size=window_size, center=True)
rms_envelope_processor = create_moving_rms(window_size=window_size, center=True)

# %%
moving_envelope = moving_envelope_processor.process(base_data).compute()
rms_envelope = rms_envelope_processor.process(base_raw_data).compute()
tkeo_envelope = lowpass_processor.process(base_data, fs=fs).compute()

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

# Create a figure with a 2-row grid: top row for original signal, bottom row for envelope methods
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 3, height_ratios=[1, 1.5])

# Select one channel for visualization
channel_to_plot = 1

# Calculate time array for x-axis
# Calculate time array for x-axis
time_array = np.arange(len(tkeo_envelope[:, channel_to_plot])) / fs
len_to_plot = len(tkeo_envelope[:, channel_to_plot])
max_time = len_to_plot / fs

# Darker grey with navy tint
dark_grey_navy = "#2D3142"

# Plot the original signal spanning the full width (across all 4 columns)
ax_orig = fig.add_subplot(gs[0, :])
ax_orig.plot(
    time_array,
    base_data[:len_to_plot, channel_to_plot],
    color=dark_grey_navy,  # Using darker grey with navy tint
    linewidth=2,
)
ax_orig.set_xlim(0, max_time)
ax_orig.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_orig.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_orig.tick_params(labelsize=TICK_SIZE)
ax_orig.set_title("Original Squared Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 1: Moving Average Envelope
ax1 = fig.add_subplot(gs[1, 0])
# Plot the line
ax1.plot(
    time_array,
    moving_envelope[:len_to_plot, channel_to_plot],
    color=palette[0],
    linewidth=2,
)
# Fill the area under the curve
ax1.fill_between(
    time_array,
    moving_envelope[:len_to_plot, channel_to_plot],
    0,
    color=palette[0],
    alpha=0.3,
)
ax1.set_xlim(0, max_time)
ax1.set_ylim(bottom=0)  # Set bottom to 0 for envelope methods
ax1.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.tick_params(labelsize=TICK_SIZE)
ax1.set_title("Moving Average Envelope", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 2: RMS Envelope
ax2 = fig.add_subplot(gs[1, 1])
# Plot the line
ax2.plot(
    time_array,
    rms_envelope[:len_to_plot, channel_to_plot],
    color=palette[1],
    linewidth=2,
)
# Fill the area under the curve
ax2.fill_between(
    time_array,
    rms_envelope[:len_to_plot, channel_to_plot],
    0,
    color=palette[1],
    alpha=0.3,
)
ax2.set_xlim(0, max_time)
ax2.set_ylim(bottom=0)  # Set bottom to 0 for envelope methods
ax2.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.tick_params(labelsize=TICK_SIZE)
ax2.set_title("RMS Envelope", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 3: TKEO Envelope
ax3 = fig.add_subplot(gs[1, 2])
# Plot the line
ax3.plot(
    time_array,
    tkeo_envelope[:len_to_plot, channel_to_plot],
    color=palette[2],
    linewidth=2,
)
# Fill the area under the curve
ax3.fill_between(
    time_array,
    tkeo_envelope[:len_to_plot, channel_to_plot],
    0,
    color=palette[2],
    alpha=0.3,
)
ax3.set_xlim(0, max_time)
ax3.set_ylim(bottom=0)  # Set bottom to 0 for envelope methods
ax3.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax3.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax3.tick_params(labelsize=TICK_SIZE)
ax3.set_title("TKEO Envelope", fontsize=SUBTITLE_SIZE, weight="bold")

# # Plot 4: Empty placeholder for now (or you can add another envelope method if you have one)
# ax4 = fig.add_subplot(gs[1, 3])
# # Just set up the axes for consistency
# ax4.set_xlim(0, max_time)
# ax4.set_ylim(bottom=0)
# ax4.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
# ax4.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
# ax4.tick_params(labelsize=TICK_SIZE)
# ax4.set_title("Placeholder", fontsize=SUBTITLE_SIZE, weight="bold")
# ax4.text(
#     0.5,
#     0.5,
#     "Future Method",
#     horizontalalignment="center",
#     verticalalignment="center",
#     transform=ax4.transAxes,
#     fontsize=SUBTITLE_SIZE,
# )

# Add overall title
plt.suptitle(
    "EMG Envelope Detection Methods", fontsize=TITLE_SIZE, fontweight="bold", y=0.98
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

# # Add a caption/note if needed
# plt.figtext(
#     0.5,
#     0.01,
#     f"Channel {channel_to_plot} data, filtered with bandpass (10-2000 Hz) and notch (60 Hz) filters, window size: {window_size / fs * 1000:.0f}ms",
#     ha="center",
#     fontsize=CAPTION_SIZE,
#     fontstyle="italic",
# )

plt.show()

# %%
