"""
Functions to extract MEPs with envelope detection - Complete test script
Author: Jesus Penaloza (Updated with envelope detection)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.basic import (
    TKEOProcessor,
    create_hilbert_envelope,
    create_rectify_smooth_envelope,
    create_rms_envelope,
    create_tkeo_envelope,
)
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)
from dspant.processor.spatial import create_cmr_processor, create_whitening_processor
from dspant.processor.spectral import LFCCProcessor, MFCCProcessor, SpectrogramProcessor

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
lowpass_filter = ButterFilter("lowpass", 100, order=4, fs=fs)
fig_lp = lowpass_filter.plot_frequency_response(
    show_phase=True, cutoff_lines=True, freq_scale="log", y_min=-80
)
# plt.savefig("lowpass_filter.png", dpi=300, bbox_inches='tight')
# %%

highpass_filter = ButterFilter("highpass", 1, order=4, fs=fs)
fig_hp = highpass_filter.plot_frequency_response(
    show_phase=True, cutoff_lines=True, freq_scale="log", y_min=-80
)
# plt.savefig("highpass_filter.png", dpi=300, bbox_inches='tight')

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
channel_id = 0
# Plot filtered data
plt.figure(figsize=(15, 6))
time_axis = np.arange(40000) / fs  # Create time axis in seconds

plt.subplot(211)
plt.plot(time_axis, stream_emg.data[0:40000, channel_id], "k", alpha=0.7, label="Raw")
plt.title("Raw Data")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(time_axis, filter_data[0:40000, channel_id], "b", label="Filtered")
plt.title("Filtered Data (Notch + Bandpass)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("filtered_data.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# ======= ENVELOPE DETECTION TESTING =======
# Create and apply multiple envelope detection methods

# First, get the filtered data as our base
base_data = filter_data

# Define the time window to plot
plot_start = 0
plot_end = 20000
plot_time = np.arange(plot_start, plot_end) / fs

# %%
# 1. Create Rectify+Smooth envelope pipeline
rectify_smooth_pipeline = create_rectify_smooth_envelope(
    cutoff_freq=20,  # 20 Hz lowpass for smoothing
    fs=fs,
    rect_method="abs",  # Full-wave rectification
)

# Add to processing node
processor_rect_smooth = create_processing_node(stream_emg, name="Rectify+Smooth")
# First add basic filtering
processor_rect_smooth.add_processor(
    [notch_processor, bandpass_processor], group="preprocess"
)
# Then add envelope processors
for proc in rectify_smooth_pipeline.get_group_processors("envelope"):
    processor_rect_smooth.add_processor(proc, group="envelope")

# Process data
rect_smooth_data = processor_rect_smooth.process(
    group=["preprocess", "envelope"]
).persist()

# %%
# 2. Create Hilbert envelope pipeline
hilbert_pipeline = create_hilbert_envelope()

# Add to processing node
processor_hilbert = create_processing_node(stream_emg, name="Hilbert")
# First add basic filtering
processor_hilbert.add_processor(
    [notch_processor, bandpass_processor], group="preprocess"
)
# Then add envelope processors
for proc in hilbert_pipeline.get_group_processors("envelope"):
    processor_hilbert.add_processor(proc, group="envelope")

# Process data
hilbert_data = processor_hilbert.process(group=["preprocess", "envelope"]).persist()

# %%
# 3. Create RMS envelope pipeline
rms_pipeline = create_rms_envelope(window_size=int(fs * 0.05), center=True)

# Add to processing node
processor_rms = create_processing_node(stream_emg, name="RMS")
# First add basic filtering
processor_rms.add_processor([notch_processor, bandpass_processor], group="preprocess")
# Then add envelope processors
for proc in rms_pipeline.get_group_processors("envelope"):
    processor_rms.add_processor(proc, group="envelope")

# Process data
rms_data = processor_rms.process(group=["preprocess", "envelope"]).persist()

# %%
# 4. Create TKEO envelope pipeline
tkeo_pipeline = create_tkeo_envelope(
    method="modified", rectify=True, smooth=True, cutoff_freq=20, fs=fs
)

# Add to processing node
processor_tkeo = create_processing_node(stream_emg, name="TKEO")
# First add basic filtering
processor_tkeo.add_processor([notch_processor, bandpass_processor], group="preprocess")
# Then add envelope processors
for proc in tkeo_pipeline.get_group_processors("envelope"):
    processor_tkeo.add_processor(proc, group="envelope")

# Process data
tkeo_data = processor_tkeo.process(group=["preprocess", "envelope"]).persist()

# %%
# Plot all envelope methods for comparison
plt.figure(figsize=(15, 10))

# Plot the filtered data first
plt.subplot(5, 1, 1)
plt.plot(plot_time, base_data[plot_start:plot_end, channel_id], "k", alpha=0.7)
plt.title("Filtered Data")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot the rectify+smooth envelope
plt.subplot(5, 1, 2)
plt.plot(plot_time, rect_smooth_data[plot_start:plot_end, channel_id], "r")
plt.title("Rectify + Smooth Envelope")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot the Hilbert envelope
plt.subplot(5, 1, 3)
plt.plot(plot_time, hilbert_data[plot_start:plot_end, channel_id], "g")
plt.title("Hilbert Envelope")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot the RMS envelope
plt.subplot(5, 1, 4)
plt.plot(plot_time, rms_data[plot_start:plot_end, channel_id], "b")
plt.title("RMS Envelope")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot the TKEO envelope
plt.subplot(5, 1, 5)
plt.plot(plot_time, tkeo_data[plot_start:plot_end, channel_id], "m")
plt.title("TKEO Envelope")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("envelope_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Plot all envelope methods overlaid for direct comparison
plt.figure(figsize=(15, 8))

# Plot the filtered data
plt.subplot(2, 1, 1)
plt.plot(
    plot_time,
    base_data[plot_start:plot_end, channel_id],
    "k",
    alpha=0.5,
    label="Filtered",
)
plt.title("Filtered Data")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

# Plot all envelopes overlaid
plt.subplot(2, 1, 2)
plt.plot(
    plot_time,
    rect_smooth_data[plot_start:plot_end, channel_id],
    "r",
    label="Rectify+Smooth",
)
plt.plot(plot_time, hilbert_data[plot_start:plot_end, channel_id], "g", label="Hilbert")
plt.plot(plot_time, rms_data[plot_start:plot_end, channel_id], "b", label="RMS")
plt.plot(plot_time, tkeo_data[plot_start:plot_end, channel_id], "m", label="TKEO")
plt.title("Envelope Methods Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("envelope_overlay.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Parameter comparison for TKEO method

# Create TKEO envelope with different parameters
tkeo_classic = create_tkeo_envelope(
    method="classic", rectify=True, smooth=True, cutoff_freq=20, fs=fs
)

tkeo_modified = create_tkeo_envelope(
    method="modified", rectify=True, smooth=True, cutoff_freq=20, fs=fs
)

tkeo_no_smoothing = create_tkeo_envelope(
    method="classic", rectify=True, smooth=False, fs=fs
)

# Add to processing nodes
processor_tkeo_classic = create_processing_node(stream_emg, name="TKEO-Classic")
processor_tkeo_modified = create_processing_node(stream_emg, name="TKEO-Modified")
processor_tkeo_no_smooth = create_processing_node(stream_emg, name="TKEO-NoSmooth")

# Add preprocessing
for processor in [
    processor_tkeo_classic,
    processor_tkeo_modified,
    processor_tkeo_no_smooth,
]:
    processor.add_processor([notch_processor, bandpass_processor], group="preprocess")

# Add envelope processors for each variant
for proc in tkeo_classic.get_group_processors("envelope"):
    processor_tkeo_classic.add_processor(proc, group="envelope")

for proc in tkeo_modified.get_group_processors("envelope"):
    processor_tkeo_modified.add_processor(proc, group="envelope")

for proc in tkeo_no_smoothing.get_group_processors("envelope"):
    processor_tkeo_no_smooth.add_processor(proc, group="envelope")

# Process data
tkeo_classic_data = processor_tkeo_classic.process(group=["preprocess", "envelope"])
tkeo_modified_data = processor_tkeo_modified.process(group=["preprocess", "envelope"])
tkeo_no_smooth_data = processor_tkeo_no_smooth.process(group=["preprocess", "envelope"])

# %%
# Plot TKEO parameter comparison
plt.figure(figsize=(15, 10))

# Plot the filtered data
plt.subplot(4, 1, 1)
plt.plot(plot_time, base_data[plot_start:plot_end, channel_id], "k", alpha=0.5)
plt.title("Filtered Data")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot TKEO classic
plt.subplot(4, 1, 2)
plt.plot(plot_time, tkeo_classic_data[plot_start:plot_end, channel_id], "b")
plt.title("TKEO Classic (3-point) with Smoothing")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot TKEO modified
plt.subplot(4, 1, 3)
plt.plot(plot_time, tkeo_modified_data[plot_start:plot_end, channel_id], "g")
plt.title("TKEO Modified (4-point) with Smoothing")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot TKEO without smoothing
plt.subplot(4, 1, 4)
plt.plot(plot_time, tkeo_no_smooth_data[plot_start:plot_end, channel_id], "r")
plt.title("TKEO Classic without Smoothing")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("tkeo_parameter_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Compare RMS window sizes

# Create different RMS window sizes for comparison
rms_small = create_rms_envelope(window_size=21, center=True)
rms_medium = create_rms_envelope(window_size=101, center=True)
rms_large = create_rms_envelope(window_size=501, center=True)

# Add to processing nodes
processor_rms_small = create_processing_node(stream_emg, name="RMS-Small")
processor_rms_medium = create_processing_node(stream_emg, name="RMS-Medium")
processor_rms_large = create_processing_node(stream_emg, name="RMS-Large")

# Add preprocessing
for processor in [processor_rms_small, processor_rms_medium, processor_rms_large]:
    processor.add_processor([notch_processor, bandpass_processor], group="preprocess")

# Add envelope processors for each variant
for proc in rms_small.get_group_processors("envelope"):
    processor_rms_small.add_processor(proc, group="envelope")

for proc in rms_medium.get_group_processors("envelope"):
    processor_rms_medium.add_processor(proc, group="envelope")

for proc in rms_large.get_group_processors("envelope"):
    processor_rms_large.add_processor(proc, group="envelope")

# Process data
rms_small_data = processor_rms_small.process(group=["preprocess", "envelope"])
rms_medium_data = processor_rms_medium.process(group=["preprocess", "envelope"])
rms_large_data = processor_rms_large.process(group=["preprocess", "envelope"])

# %%
# Plot RMS window size comparison
plt.figure(figsize=(15, 10))

# Plot the filtered data
plt.subplot(4, 1, 1)
plt.plot(plot_time, base_data[plot_start:plot_end, channel_id], "k", alpha=0.5)
plt.title("Filtered Data")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot RMS small window
plt.subplot(4, 1, 2)
plt.plot(plot_time, rms_small_data[plot_start:plot_end, channel_id], "b")
plt.title("RMS with Small Window (21 samples)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot RMS medium window
plt.subplot(4, 1, 3)
plt.plot(plot_time, rms_medium_data[plot_start:plot_end, channel_id], "g")
plt.title("RMS with Medium Window (101 samples)")
plt.ylabel("Amplitude")
plt.grid(True)

# Plot RMS large window
plt.subplot(4, 1, 4)
plt.plot(plot_time, rms_large_data[plot_start:plot_end, channel_id], "r")
plt.title("RMS with Large Window (501 samples)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.savefig("rms_window_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Create a 2x2 comparison of all envelope methods
plt.figure(figsize=(15, 10))

# 1. Rectify+Smooth
plt.subplot(2, 2, 1)
plt.plot(
    plot_time,
    base_data[plot_start:plot_end, channel_id],
    "k",
    alpha=0.3,
    label="Signal",
)
plt.plot(
    plot_time, rect_smooth_data[plot_start:plot_end, channel_id], "r", label="Envelope"
)
plt.title("Rectify + Smooth Envelope")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# 2. Hilbert
plt.subplot(2, 2, 2)
plt.plot(
    plot_time,
    base_data[plot_start:plot_end, channel_id],
    "k",
    alpha=0.3,
    label="Signal",
)
plt.plot(
    plot_time, hilbert_data[plot_start:plot_end, channel_id], "g", label="Envelope"
)
plt.title("Hilbert Transform Envelope")
plt.legend()
plt.grid(True)

# 3. RMS
plt.subplot(2, 2, 3)
plt.plot(
    plot_time,
    base_data[plot_start:plot_end, channel_id],
    "k",
    alpha=0.3,
    label="Signal",
)
plt.plot(
    plot_time, rms_medium_data[plot_start:plot_end, channel_id], "b", label="Envelope"
)
plt.title("RMS Envelope (101-pt window)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

# 4. TKEO
plt.subplot(2, 2, 4)
plt.plot(
    plot_time,
    base_data[plot_start:plot_end, channel_id],
    "k",
    alpha=0.3,
    label="Signal",
)
plt.plot(
    plot_time, tkeo_classic_data[plot_start:plot_end, channel_id], "m", label="Envelope"
)
plt.title("TKEO Envelope")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("envelope_methods_2x2.png", dpi=300, bbox_inches="tight")
plt.show()
