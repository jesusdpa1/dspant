"""
Script to showcase ECG removal by template subtraction
Author: Jesus Penaloza
Using mp_plotting_utils for standardized publication-quality visualization
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
from matplotlib.gridspec import GridSpec

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.pattern.detection.peak import create_positive_peak_detector
from dspant.pattern.detection.template import TemplateDetector
from dspant.pattern.subtraction.correlation import (
    CorrelationSubtractor,
    create_ecg_subtractor,
    subtract_templates,
)

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.extractors.template_extractor import (
    extract_template,
    extract_template_distributions,
)
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data

# Set publication style
load_dotenv()
# %%

DATA_DIR = Path(os.getenv("DATA_DIR"))

BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")
HD_STREAM_PATH = BASE_PATH.joinpath("HDEG.ant")

# %%
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()
# %%%
# Load EMG data
stream_hd = StreamNode(str(HD_STREAM_PATH))
stream_hd.load_metadata()
stream_hd.load_data()
# Print stream_emg summary
stream_hd.summarize()

# %%
# Create and visualize filters before applying them
FS = stream_hd.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)
# %%

notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%

# Create processing node with filters
processor_hd = create_processing_node(stream_hd)
processor_emg = create_processing_node(stream_emg)
# Add processors to the processing node
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# %%
# Apply filters and plot results
filter_hd = processor_hd.process(group=["filters"]).persist()
filter_emg = processor_emg.process(group=["filters"]).persist()

# %%
cmr_processor = create_cmr_processor_rs()
cmr_data = cmr_processor.process(filter_hd, FS).persist()
cmr_reference = cmr_processor.get_reference(filter_hd)

# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=20)
tkeo_data = tkeo_processor.process(filter_emg, fs=FS).persist()

# %%

a = plot_multi_channel_data(filter_hd, channels=[1, 2, 3, 4], fs=FS, time_window=[0, 5])

# %%

b = plot_multi_channel_data(filter_emg, fs=FS, time_window=[0, 5])
c = plot_multi_channel_data(tkeo_data, fs=FS, time_window=[0, 5])

# %%
# Step 1: Define time window and slice all relevant data
START_ = int(FS * 0)  # Starting from beginning of recording
END_ = int(FS * 5)  # 5 seconds of data
time_slice = slice(START_, END_)

# Slice the reference signal for peak detection
reference_data = cmr_reference[time_slice, :].persist()

# Create a peak detector optimized for R-peaks
r_peak_detector = create_positive_peak_detector(
    threshold=4.0,  # Adjust based on signal characteristics
    threshold_mode="mad",  # Robust to outliers
    refractory_period=0.1,  # 100ms minimum between peaks (600 BPM max)
)

# Detect R-peaks in the reference signal
peak_results = r_peak_detector.detect(reference_data, fs=FS)

# Extract peak indices - these are relative to the sliced data
r_peak_indices = peak_results["index"].compute()

print(f"Detected {len(r_peak_indices)} R-peaks in the reference signal.")

# Plot the reference signal with detected peaks
plt.figure(figsize=(15, 5))
plt.plot(reference_data, label="Reference Signal")
plt.plot(
    r_peak_indices,
    reference_data[r_peak_indices],
    "r*",
    markersize=10,
    label="Detected R-peaks",
)
plt.title("ECG Reference Signal with Detected R-peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# Step 2: Extract template using the correct template extractor
# Also slice the EMG data to match the reference data time window
emg_data = filter_emg[time_slice, :].compute()

# Define the window size for template extraction (in samples)
window_ms = 60  # Window size in milliseconds (+- 30 ms around peak)
waveform_processor = WaveformExtractor(filter_emg, FS)

# %%

ecg_waveforms = waveform_processor.extract_waveforms(
    r_peak_indices,
    100,
    100,
)

# %%
# Now using the correct template extractor function
ecg_template = extract_template(
    ecg_waveforms[0],
    axis=0,
)
template_ = plt.plot(ecg_template[:, 0])

# %%
# Plot the extracted template
plt.figure(figsize=(12, 6))
for ch in range(min(4, ecg_template.shape[1])):  # Plot up to 4 channels
    plt.subplot(2, 2, ch + 1)
    plt.plot(ecg_template[:, ch])
    plt.title(f"ECG Template - Channel {ch}")
    plt.xlabel("Samples")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
plt.tight_layout()
plt.show()
# %%
# Step 3: Use the template detector to identify all matches in the signal
# Create a template detector with appropriate parameters
template_detector = TemplateDetector(
    template=ecg_template[:, 0],  # Using the first channel of the template
    threshold=0.7,  # Start with 0.7 correlation threshold (adjust as needed)
    min_distance=0.25,  # Minimum 250ms between detections
    use_normalized_correlation=True,
    threshold_mode="absolute",
)

# Apply the detector to the filtered EMG data
detections = template_detector.detect(filter_emg[time_slice, 0:1], fs=FS)

# Compute the results
detection_results = detections.compute()

# Get indices and correlation scores
detected_indices = detection_results["index"]
correlation_scores = detection_results["correlation"]

print(f"Detected {len(detected_indices)} template matches")

# Plot the detection results
plt.figure(figsize=(15, 8))
plt.subplot(211)
plt.plot(filter_emg[time_slice, 0].compute(), label="EMG Signal")
plt.plot(
    r_peak_indices,
    filter_emg[time_slice, 0][r_peak_indices].compute(),
    "r*",
    markersize=10,
    label="Original R-peaks",
)
plt.title("Original Signal with R-peaks")
plt.legend()

plt.subplot(212)
plt.plot(filter_emg[time_slice, 0].compute(), label="EMG Signal")
plt.plot(
    detected_indices,
    filter_emg[time_slice, 0][detected_indices].compute(),
    "go",
    markersize=8,
    label="Template Matches",
)
for i, (idx, corr) in enumerate(zip(detected_indices, correlation_scores)):
    plt.text(
        idx,
        filter_emg[time_slice, 0][idx].compute() + 0.1,
        f"{corr:.2f}",
        fontsize=8,
        ha="center",
    )
plt.title("Template Detection Results")
plt.legend()

plt.tight_layout()
plt.show()

# Compare detection accuracy with original R-peaks
match_count = 0
max_distance = int(0.05 * FS)  # 50ms tolerance

for r_peak in r_peak_indices:
    for detected in detected_indices:
        if abs(r_peak - detected) <= max_distance:
            match_count += 1
            break

r_peak_recall = match_count / len(r_peak_indices) if len(r_peak_indices) > 0 else 0
precision = match_count / len(detected_indices) if len(detected_indices) > 0 else 0

print(f"Detection Performance:")
print(
    f"Recall: {r_peak_recall:.2f} ({match_count}/{len(r_peak_indices)} R-peaks detected)"
)
print(
    f"Precision: {precision:.2f} ({match_count}/{len(detected_indices)} detections are correct)"
)

# %%
# Step 3: Apply template subtraction using correlation.py
# Create a correlation-based ECG subtractor
# Extract a single channel template (1D array)

(ecg_template[:, 0],)

# Create a correlation-based ECG subtractor
ecg_subtractor = create_ecg_subtractor(
    ecg_template=ecg_template[:, 0],
    window_ms=window_ms,
    fs=FS,
)

# Apply ECG subtraction
cleaned_emg = ecg_subtractor.process(
    filter_emg[time_slice, 0],
    indices=r_peak_indices,
    fs=FS,
).compute()
# %%
plt.plot(filter_emg[time_slice, 0])
plt.plot(cleaned_emg)
# %%
