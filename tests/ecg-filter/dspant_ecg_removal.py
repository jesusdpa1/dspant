"""
Simplified script for ECG removal using template subtraction
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.signal import correlate

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.pattern.detection.peak import create_positive_peak_detector
from dspant.processors.extractors.template_extractor import extract_template
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs

# Load environment variables
load_dotenv()
# %%
# Define paths
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")
# HD_STREAM_PATH = BASE_PATH.joinpath("HDEG.ant")
# %%
# Load streams
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()

# stream_hd = StreamNode(str(HD_STREAM_PATH))
# stream_hd.load_metadata()
# stream_hd.load_data()

# Get sampling rate
FS = stream_emg.fs
# %%
# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)
lowpass_filter = create_lowpass_filter(200, fs=FS, order=5)

# Create filter processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
lowpass_processor = FilterProcessor(
    filter_func=lowpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Set up processing nodes
# processor_hd = create_processing_node(stream_hd)
processor_emg = create_processing_node(stream_emg)

# Add processors to the nodes
# processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# %%
# Apply filters
# filter_hd = processor_hd.process(group=["filters"]).persist()
filter_emg = processor_emg.process(group=["filters"]).persist()

# %%
START_ = int(FS * 0)  # Starting from beginning of recording
END_ = int(FS * 5)  # 5 seconds of data
time_slice = slice(START_, END_)
# Get common-mode reference
reference_ecg = lowpass_processor.process(filter_emg, FS)

plt.plot(reference_ecg[time_slice])

# %%

# Slice the reference signal for peak detection
reference_data = reference_ecg[time_slice, 0].persist()

# Create a peak detector optimized for R-peaks
r_peak_detector = create_positive_peak_detector(
    threshold=10.0,
    threshold_mode="mad",
    refractory_period=0.1,
)

# Detect R-peaks in the reference signal
peak_results = r_peak_detector.detect(reference_data, fs=FS)
r_peak_indices = peak_results["index"].compute()
print(f"Detected {len(r_peak_indices)} R-peaks in the reference signal.")

# Plot reference signal with detected peaks
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
# Simplified template extraction approach - mirroring the original working code
# Extract ECG templates
window_ms = 60  # Window size in milliseconds
window_samples = int((window_ms / 1000) * FS)
half_win = window_samples // 2
waveform_processor = WaveformExtractor(filter_emg[:, 0], FS)

# Extract waveforms
ecg_waveforms = waveform_processor.extract_waveforms(
    r_peak_indices,
    pre_samples=half_win,
    post_samples=half_win,
    time_unit='samples'
)

# Extract template
ecg_template = extract_template(ecg_waveforms[0], axis=0)
template = ecg_template[:, 0]  # Use first channel

# Plot the extracted template
plt.figure(figsize=(12, 4))
plt.plot(template)
plt.title(f"ECG Template (Using Extractors)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert to numpy arrays for processing
emg_data = filter_emg[time_slice, 0].compute()


# %%
# Template subtraction
def subtract_ecg_template(data, template, peak_indices, half_window):
    """
    Subtract ECG template from signal using correlation alignment.

    Args:
        data: EMG signal (numpy array)
        template: ECG template (numpy array)
        peak_indices: Indices of R-peaks
        half_window: Half window size in samples

    Returns:
        Cleaned signal
    """
    # Make a copy to avoid modifying input
    result = data.copy()

    # Process each peak
    for p in peak_indices:
        # Skip if peak is too close to edges
        if p < half_window or p >= len(result) - half_window:
            continue

        # Extract segment around peak
        segment = result[p - half_window : p + half_window]

        # Calculate correlation
        corr = correlate(segment, template, mode="valid")

        # Calculate shift
        shift = np.argmax(corr) - (len(corr) // 2)

        # Calculate start and end positions with shift
        start = p - half_window + shift
        end = start + len(template)

        # Safety check for boundaries
        if start >= 0 and end < len(result):
            # Re-extract segment at aligned position
            aligned_segment = result[start:end]

            # Calculate scaling factor
            scale = np.dot(aligned_segment, template) / np.dot(template, template)

            # Subtract scaled template
            result[start:end] -= scale * template

    return result


# Apply template subtraction
cleaned_emg = subtract_ecg_template(emg_data, template, r_peak_indices, half_win)

# Plot original and cleaned signal
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(emg_data, label="Original EMG Signal")
plt.plot(r_peak_indices, emg_data[r_peak_indices], "r*", markersize=8, label="R-peaks")
plt.title("Original EMG Signal with R-peaks")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(cleaned_emg, label="Cleaned EMG Signal")
plt.title("EMG Signal after ECG Template Subtraction")
plt.xlabel("Samples")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Zoom in on a section to better see the difference
zoom_start = 1000
zoom_end = 2000
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(emg_data[zoom_start:zoom_end], label="Original EMG Signal")
plt.title("Zoomed Original EMG Signal")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(cleaned_emg[zoom_start:zoom_end], label="Cleaned EMG Signal")
plt.title("Zoomed EMG Signal after ECG Template Subtraction")
plt.xlabel("Samples")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("ECG template subtraction completed.")

# %%
