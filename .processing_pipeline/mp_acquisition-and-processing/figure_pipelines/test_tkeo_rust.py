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

from dspant._rs import compute_tkeo
from dspant.emgproc.activity_detection import (
    EMGOnsetDetector,
    create_absolute_threshold_detector,
)
from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.basic import create_tkeo, create_tkeo_envelope

# Import our Rust-accelerated version for comparison
from dspant.processor.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)

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
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
# %%
fig_bp = bandpass_filter.plot_frequency_response(
    show_phase=True, cutoff_lines=True, freq_scale="log", y_min=-80
)
# plt.show()  # This displays and clears the current figure
# plt.savefig("bandpass_filter.png", dpi=300, bbox_inches='tight')
# %%
notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)
fig_notch = notch_filter.plot_frequency_response(
    title="60 Hz Notch Filter", cutoff_lines=True, freq_scale="log", y_min=-80
)
plt.show()  # This displays and clears the current figure
# plt.savefig("notch_filter.png", dpi=300, bbox_inches='tight')
# %%
# Create processing node with filters
processor_hd = create_processing_node(stream_emg)
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
# ======= ENVELOPE DETECTION TESTING =======
# Create and apply multiple envelope detection methods

# First, get the filtered data as our base
base_data = filter_data

# %%
# 4. Create TKEO envelope pipeline (Python version)
tkeo_pipeline = create_tkeo_envelope(
    method="modified", rectify=True, smooth=True, cutoff_freq=20, fs=fs
)

# Add to processing node
processor_tkeo = create_processing_node(stream_emg, name="TKEO Python")
# First add basic filtering
processor_tkeo.add_processor([notch_processor, bandpass_processor], group="preprocess")
# Add TKEO envelope processors
processor_tkeo.add_processor(tkeo_pipeline.processors["envelope"], group="envelope")
# View processor summary
processor_tkeo.summarize()

# %%
# Create Rust-accelerated TKEO envelope pipeline
tkeo_pipeline_rs = create_tkeo(method="modified")

# Add to processing node
processor_tkeo_rs = create_processing_node(stream_emg, name="TKEO Rust")
# First add basic filtering
processor_tkeo_rs.add_processor(
    [notch_processor, bandpass_processor], group="preprocess"
)
# Add Rust-accelerated TKEO envelope processors
processor_tkeo_rs.add_processor(tkeo_pipeline_rs, group="envelope")
# View processor summary
processor_tkeo_rs.summarize()

# %%
# Process data with both methods and measure timing
start_time = time.time()
tkeo_envelope = processor_tkeo.process(group=["preprocess", "envelope"]).compute()
python_time = time.time() - start_time
print(f"Python TKEO processing time: {python_time:.4f} seconds")

start_time = time.time()
tkeo_envelope_rs = processor_tkeo_rs.process(group=["preprocess", "envelope"]).compute()
rust_time = time.time() - start_time
print(f"Rust TKEO processing time: {rust_time:.4f} seconds")
print(f"Speedup: {python_time / rust_time:.2f}x")

# %%
# Verify that results are similar
difference = np.abs(tkeo_envelope - tkeo_envelope_rs)
mean_diff = np.mean(difference)
max_diff = np.max(difference)
print(f"Mean difference: {mean_diff:.8f}")
print(f"Max difference: {max_diff:.8f}")

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import signal

# Sample length and channel to plot (from original code)
sample_length = 50000  # 5 seconds of data at 1000 Hz
channel_to_plot = 0

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

# Create figure with GridSpec for better control over subplot spacing
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

# First subplot - Filtered EMG Signal
ax1 = fig.add_subplot(gs[0])
ax1.plot(base_data[:sample_length, channel_to_plot], color=palette[0], linewidth=1.5)
ax1.set_title("Filtered EMG Signal", fontsize=TITLE_SIZE, fontweight="bold")
ax1.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
ax1.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax1.grid(True, alpha=0.3)

# Second subplot - TKEO Envelope
ax2 = fig.add_subplot(gs[1])
ax2.plot(
    tkeo_envelope_rs[:sample_length, channel_to_plot], color=palette[3], linewidth=1.5
)
ax2.set_title("TKEO Envelope", fontsize=TITLE_SIZE, fontweight="bold")
ax2.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
ax2.set_xlabel("Samples", fontsize=AXIS_LABEL_SIZE)
ax2.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
ax2.grid(True, alpha=0.3)

# Add overall title
fig.suptitle("EMG Signal Analysis", fontsize=TITLE_SIZE + 2, fontweight="bold", y=0.98)

# Adjust layout to make room for titles
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Add a caption/note if needed
plt.figtext(
    0.5,
    0.01,
    "Note: Data sampled at 1000 Hz",
    ha="center",
    fontsize=CAPTION_SIZE,
    fontstyle="italic",
)

# Show the plot
plt.show()

# %%
# Test onset detection with both implementations

# Create onset detector
onset_detector = create_absolute_threshold_detector(
    threshold=0.2,
    min_duration=0.05,
)

# Detect onsets in Python envelope
python_onsets = onset_detector.process(tkeo_envelope, fs=fs)
python_onsets_df = onset_detector.to_dataframe(python_onsets)
print(f"Python TKEO detected {len(python_onsets_df)} onsets")

# Detect onsets in Rust envelope
rust_onsets = onset_detector.process(tkeo_envelope_rs, fs=fs)
rust_onsets_df = onset_detector.to_dataframe(rust_onsets)
print(f"Rust TKEO detected {len(rust_onsets_df)} onsets")

# %%
# Plot a section with detected onsets
plot_length = 10000  # 10 seconds
ch = 0  # Channel to plot

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.title("Python TKEO with Detected Onsets")
plt.plot(tkeo_envelope[:plot_length, ch])
# Plot onsets for this channel
channel_onsets = python_onsets_df.filter(pl.col("channel") == ch)
for onset_idx in channel_onsets["onset_idx"]:
    if onset_idx < plot_length:
        plt.axvline(x=onset_idx, color="r", linestyle="--", alpha=0.7)
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.title("Rust TKEO with Detected Onsets")
plt.plot(tkeo_envelope_rs[:plot_length, ch])
# Plot onsets for this channel
channel_onsets_rs = rust_onsets_df.filter(pl.col("channel") == ch)
for onset_idx in channel_onsets_rs["onset_idx"]:
    if onset_idx < plot_length:
        plt.axvline(x=onset_idx, color="r", linestyle="--", alpha=0.7)
plt.ylabel("Amplitude")
plt.xlabel("Samples")

plt.tight_layout()
plt.show()
