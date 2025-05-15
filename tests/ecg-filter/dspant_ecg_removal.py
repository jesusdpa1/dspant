"""
Simplified script for ECG removal using template subtraction
"""

import os
from pathlib import Path

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
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs

# Load environment variables
load_dotenv()

# Define paths
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")
HD_STREAM_PATH = BASE_PATH.joinpath("HDEG.ant")

# Load streams
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()

stream_hd = StreamNode(str(HD_STREAM_PATH))
stream_hd.load_metadata()
stream_hd.load_data()

# Get sampling rate
FS = stream_hd.fs

# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)

# Create filter processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Set up processing nodes
processor_hd = create_processing_node(stream_hd)
processor_emg = create_processing_node(stream_emg)

# Add processors to the nodes
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters
filter_hd = processor_hd.process(group=["filters"]).persist()
filter_emg = processor_emg.process(group=["filters"]).persist()

# Get common-mode reference
cmr_processor = create_cmr_processor_rs()
cmr_data = cmr_processor.process(filter_hd, FS).persist()
cmr_reference = cmr_processor.get_reference(filter_hd)

# Define time window
START_ = int(FS * 0)  # Starting from beginning of recording
END_ = int(FS * 5)  # 5 seconds of data
time_slice = slice(START_, END_)

# Slice the reference signal for peak detection
reference_data = cmr_reference[time_slice, :].persist()

# Create a peak detector optimized for R-peaks
r_peak_detector = create_positive_peak_detector(
    threshold=4.0,
    threshold_mode="mad",
    refractory_period=0.1,
)

# Detect R-peaks in the reference signal
peak_results = r_peak_detector.detect(reference_data, fs=FS)
r_peak_indices = peak_results["index"].compute()
print(f"Detected {len(r_peak_indices)} R-peaks in the reference signal.")

# Extract ECG templates
window_ms = 60  # Window size in milliseconds (+- 30 ms around peak)
half_window = int((window_ms / 2000) * FS)  # Half window in samples
waveform_processor = WaveformExtractor(filter_emg, FS)

# Extract waveforms
ecg_waveforms = waveform_processor.extract_waveforms(
    r_peak_indices,
    pre_samples=100,
    post_samples=100,
)

# Extract template
ecg_template = extract_template(ecg_waveforms[0], axis=0)

# Convert to numpy arrays for processing
emg_data = filter_emg[time_slice, 0].compute()
template_data = ecg_template[:, 0]
window_samples = len(template_data)
half_win = window_samples // 2


# Define simple function for template subtraction using scipy correlate
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
        end = start + window_samples

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
cleaned_emg = subtract_ecg_template(emg_data, template_data, r_peak_indices, half_win)

print("ECG template subtraction completed.")
