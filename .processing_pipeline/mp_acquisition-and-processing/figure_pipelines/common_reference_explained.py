"""
Script to showcase CMR
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
from dspant.vizualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

hd_stream_path = base_path + r"/HDEG.ant"
# %%
# Load EMG data
stream_hd = StreamNode(hd_stream_path)
stream_hd.load_metadata()
stream_hd.load_data()
# Print stream_emg summary
stream_hd.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_hd.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=30, fs=fs)
# %%
bandpass_plot = bandpass_filter.plot_frequency_response()
notch_plot = notch_filter.plot_frequency_response()
# %%
# Create processing node with filters
processor_hd = create_processing_node(stream_hd)
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
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
# %%
# View summary of the processing node
processor_hd.summarize()

# %%
# Apply filters and plot results
filter_data = processor_hd.process(group=["filters"]).persist()

# %%
cmr_processor = create_cmr_processor_rs()
cmr_data = cmr_processor.process(filter_data, fs).persist()
cmr_reference = cmr_processor.get_reference(filter_data)

# %%
a = plot_multi_channel_data(
    filter_data, channels=[1, 2, 3, 4], fs=fs, time_window=[0, 5]
)
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

# Define data range
start = 0
end = int(5 * fs)  # 5 seconds of data

# Choose channels to display
selected_channels = [0, 1, 2, 3]
num_channels = len(selected_channels)

# Create figure with GridSpec for custom layout
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 2, height_ratios=[2, 1])

# Calculate time array
time_array = np.arange(end - start) / fs

# Darker grey with navy tint for reference line
dark_grey_navy = "#2D3142"

# Calculate vertical offsets for channels
# First, determine the approx amplitude range of signals
filtered_range = np.max(np.abs(filter_data[start:end, selected_channels])) * 2
cmr_range = np.max(np.abs(cmr_data[start:end, selected_channels])) * 2

# Calculate offset between channels (make it 1.5x the range for clear separation)
filtered_offset = filtered_range * 1.5
cmr_offset = cmr_range * 1.5

# Plot 1: Filtered Data (top left)
ax_filtered = fig.add_subplot(gs[0, 0])
for i, channel in enumerate(selected_channels):
    # Add increasing offset for each channel
    offset = i * filtered_offset
    ax_filtered.plot(
        time_array,
        filter_data[start:end, channel] + offset,
        color=palette[i],
        linewidth=1.5,
        label=f"Channel {channel}",
    )
ax_filtered.set_xlim(0, (end - start) / fs)
ax_filtered.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_filtered.tick_params(labelsize=TICK_SIZE)
ax_filtered.set_title(
    "Bandpass + Notch Filtered Signal", fontsize=SUBTITLE_SIZE, weight="bold"
)
ax_filtered.legend(fontsize=TICK_SIZE, loc="upper right")

# Plot 2: CMR Data (top right)
ax_cmr = fig.add_subplot(gs[0, 1])
for i, channel in enumerate(selected_channels):
    # Add increasing offset for each channel
    offset = i * cmr_offset
    ax_cmr.plot(
        time_array,
        cmr_data[start:end, channel] + offset,
        color=palette[i],
        linewidth=1.5,
        label=f"Channel {channel}",
    )
ax_cmr.set_xlim(0, (end - start) / fs)
ax_cmr.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_cmr.tick_params(labelsize=TICK_SIZE)
ax_cmr.set_title(
    "Common Median Reference (CMR) Signal", fontsize=SUBTITLE_SIZE, weight="bold"
)
ax_cmr.legend(fontsize=TICK_SIZE, loc="upper right")

# Plot 3: CMR Reference (bottom span)
ax_ref = fig.add_subplot(gs[1, :])
# Get the reference signal and ensure it's the right shape
reference_data = cmr_reference[start:end].compute().flatten()
ax_ref.plot(time_array, reference_data, color=dark_grey_navy, linewidth=2)
ax_ref.set_xlim(0, (end - start) / fs)
ax_ref.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_ref.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_ref.tick_params(labelsize=TICK_SIZE)
ax_ref.set_title(
    "Common Median Reference Signal", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Add overall title
plt.suptitle(
    "Effect of Common Median Reference (CMR) on Neural Signals",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    y=0.98,
)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

plt.show()

# %%
