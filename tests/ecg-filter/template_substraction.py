"""
Script to showcase CMR
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
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

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_highpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data

load_dotenv()
# %%

DATA_DIR = Path(os.getenv("DATA_DIR"))

BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"
EMG_STREAM_PATH = 
HD_STREAM_PATH = BASE_PATH.joinpath("HDEG.ant")

# %%
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
cmr_data = cmr_processor.process(filter_data, FS).persist()
cmr_reference = cmr_processor.get_reference(filter_data)
# %%
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from scipy import signal


def detect_peaks(
    data: Union[np.ndarray, da.Array],
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    width: Optional[float] = None,
    threshold: Optional[float] = None,
    return_properties: bool = False,
    normalize: Optional[str] = None,
    norm_window: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Detect peaks in a 1D signal, handling both NumPy and Dask arrays.

    This function provides a unified interface to scipy.signal.find_peaks,
    automatically handling both NumPy and Dask arrays.

    Args:
        data: Input signal (1D NumPy or Dask array)
        height: Minimum peak height
        distance: Minimum samples between peaks
        prominence: Minimum peak prominence
        width: Minimum peak width
        threshold: Minimum threshold of peaks (vertical distance to neighbors)
        return_properties: If True, return peak properties along with indices
        normalize: Normalization method to apply before peak detection:
            - 'minmax': Scale data to range [0, 1]
            - 'zscore': Apply z-score normalization
            - 'percent': Express values as percentage of max
            - 'robust': Apply robust scaling using median and IQR
            - 'local': Apply local normalization within windows
            - None: No normalization (default)
        norm_window: Window size for local normalization (if normalize='local')

    Returns:
        If return_properties is False: Array of peak indices
        If return_properties is True: Tuple of (indices, properties dictionary)
    """
    # Make a copy of data to avoid modifying the original
    if isinstance(data, np.ndarray):
        working_data = data.copy()
    else:
        working_data = data  # Dask arrays are immutable, so no need to copy

    # Apply normalization if requested
    if normalize is not None:
        working_data = _normalize_signal(
            working_data, method=normalize, window=norm_window
        )

    # Handle different array types
    if isinstance(working_data, np.ndarray):
        # For NumPy arrays, use scipy.signal.find_peaks directly
        return signal.find_peaks(
            working_data,
            height=height,
            distance=distance,
            prominence=prominence,
            width=width,
            threshold=threshold,
            rel_height=0.5,  # Default value from scipy
        )
    elif isinstance(working_data, da.Array):
        # For Dask arrays, we need to handle chunking

        # Ensure 1D array
        if working_data.ndim > 1:
            raise ValueError("Input must be a 1D array")

        # Determine chunk overlap based on parameters
        overlap = 0
        if distance is not None:
            overlap = max(overlap, int(distance * 2))
        if width is not None:
            overlap = max(overlap, int(width * 3))
        if prominence is not None:
            # Prominence calculation may need wider context
            overlap = max(overlap, 50)  # A reasonable default

        # Ensure some minimum overlap
        overlap = max(overlap, 20)

        # Define the function to apply to each chunk
        def _find_peaks_in_chunk(chunk, chunk_idx=None):
            # Get the global offset for this chunk
            if chunk_idx is None:
                offset = 0
            else:
                offset = sum(working_data.chunks[0][:chunk_idx])

            # Find peaks in this chunk
            peaks, properties = signal.find_peaks(
                chunk,
                height=height,
                distance=distance,
                prominence=prominence,
                width=width,
                threshold=threshold,
                rel_height=0.5,
            )

            # Adjust indices to global coordinates
            global_peaks = peaks + offset

            if return_properties:
                return global_peaks, properties
            else:
                return global_peaks

        # Use Dask's map_overlap to process chunks with proper overlap
        if return_properties:
            # For properties, we need to compute and then combine results
            chunks_result = []
            # Process each chunk separately to handle properties correctly
            for i in range(len(working_data.chunks[0])):
                chunk_start = sum(working_data.chunks[0][:i])
                chunk_end = chunk_start + working_data.chunks[0][i]

                # Add overlap
                overlap_start = max(0, chunk_start - overlap)
                overlap_end = min(working_data.shape[0], chunk_end + overlap)

                # Get the chunk with overlap
                chunk_with_overlap = working_data[overlap_start:overlap_end].compute()

                # Find peaks in this chunk
                local_peaks, local_props = _find_peaks_in_chunk(
                    chunk_with_overlap, chunk_idx=i
                )

                # Filter out peaks from overlap regions
                valid_mask = (local_peaks >= chunk_start) & (local_peaks < chunk_end)
                valid_peaks = local_peaks[valid_mask]

                # Filter properties accordingly
                valid_props = {
                    k: np.asarray(v)[valid_mask]
                    if isinstance(v, (list, np.ndarray))
                    else v
                    for k, v in local_props.items()
                }

                chunks_result.append((valid_peaks, valid_props))

            # Combine results from all chunks
            if not chunks_result:
                return np.array([], dtype=int), {}

            all_peaks = np.concatenate(
                [res[0] for res in chunks_result if len(res[0]) > 0]
            )

            # Handle case with no peaks
            if len(all_peaks) == 0:
                return np.array([], dtype=int), {}

            # Sort peaks by position
            sort_idx = np.argsort(all_peaks)
            all_peaks = all_peaks[sort_idx]

            # Combine properties
            all_props = {}
            for prop_name in chunks_result[0][1].keys():
                # Get property values from all chunks
                prop_values = []
                for i, (peaks, props) in enumerate(chunks_result):
                    if len(peaks) == 0:
                        continue  # Skip empty chunks

                    if isinstance(props[prop_name], (list, np.ndarray)):
                        prop_values.append(props[prop_name])
                    else:
                        # Handle scalar properties
                        prop_values.append(np.full(len(peaks), props[prop_name]))

                # Concatenate and sort by peak position
                if len(prop_values) > 0:
                    combined_prop = np.concatenate(prop_values)[sort_idx]
                    all_props[prop_name] = combined_prop

            return all_peaks, all_props
        else:
            # For indices only, we can use map_overlap more directly
            result = working_data.map_overlap(
                _find_peaks_in_chunk, depth=overlap, boundary="reflect", dtype=object
            )

            # Compute and combine results
            peaks_list = result.compute()
            if not peaks_list or all(len(p) == 0 for p in peaks_list):
                return np.array([], dtype=int)

            # Filter out empty arrays and concatenate
            non_empty_peaks = [p for p in peaks_list if len(p) > 0]
            if not non_empty_peaks:
                return np.array([], dtype=int)

            all_peaks = np.concatenate(non_empty_peaks)

            # Sort peaks by position and remove duplicates
            all_peaks = np.unique(all_peaks)

            return all_peaks
    else:
        raise TypeError("Input must be a NumPy or Dask array")


def _normalize_signal(
    data: Union[np.ndarray, da.Array],
    method: str = "minmax",
    window: Optional[int] = None,
) -> Union[np.ndarray, da.Array]:
    """
    Normalize signal data using various methods.

    Args:
        data: Input signal (NumPy or Dask array)
        method: Normalization method
            - 'minmax': Scale to range [0, 1]
            - 'zscore': Z-score normalization
            - 'percent': Express as percentage of max value
            - 'robust': Normalize using median and IQR
            - 'local': Apply local normalization in windows
        window: Window size for local normalization

    Returns:
        Normalized signal
    """
    if method == "minmax":
        # Min-max scaling to [0, 1]
        if isinstance(data, np.ndarray):
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max == data_min:
                return np.zeros_like(data)
            return (data - data_min) / (data_max - data_min)
        else:
            # For Dask array
            data_min = data.min().compute()
            data_max = data.max().compute()
            if data_max == data_min:
                return da.zeros_like(data)
            return (data - data_min) / (data_max - data_min)

    elif method == "zscore":
        # Z-score normalization: (x - mean) / std
        if isinstance(data, np.ndarray):
            data_mean = np.mean(data)
            data_std = np.std(data)
            if data_std == 0:
                return np.zeros_like(data)
            return (data - data_mean) / data_std
        else:
            # For Dask array
            data_mean = data.mean().compute()
            data_std = data.std().compute()
            if data_std == 0:
                return da.zeros_like(data)
            return (data - data_mean) / data_std

    elif method == "percent":
        # Express as percentage of max value
        if isinstance(data, np.ndarray):
            data_max = np.max(np.abs(data))
            if data_max == 0:
                return np.zeros_like(data)
            return data / data_max * 100
        else:
            # For Dask array
            data_max = data.max().compute()
            if data_max == 0:
                return da.zeros_like(data)
            return data / data_max * 100

    elif method == "robust":
        # Robust scaling using median and IQR
        if isinstance(data, np.ndarray):
            median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(data)
            return (data - median) / iqr
        else:
            # For Dask array
            # Note: Computing percentiles with Dask requires loading all data
            # This might not be efficient for very large arrays
            computed_data = data.compute()
            median = np.median(computed_data)
            q75, q25 = np.percentile(computed_data, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return da.zeros_like(data)
            return (data - median) / iqr

    elif method == "local":
        # Local normalization within windows
        if window is None:
            # Default window size - about 10% of data length
            if isinstance(data, np.ndarray):
                window = max(int(len(data) * 0.1), 10)
            else:
                window = max(int(data.shape[0] * 0.1), 10)

        # Ensure window is odd
        if window % 2 == 0:
            window += 1

        # Apply local normalization
        if isinstance(data, np.ndarray):
            # Create output array
            normalized = np.zeros_like(data)
            half_window = window // 2

            # Apply windowed normalization
            for i in range(len(data)):
                # Define window boundaries
                start = max(0, i - half_window)
                end = min(len(data), i + half_window + 1)

                # Get window data
                window_data = data[start:end]

                # Normalize using window statistics
                win_mean = np.mean(window_data)
                win_std = np.std(window_data)

                if win_std == 0:
                    normalized[i] = 0
                else:
                    normalized[i] = (data[i] - win_mean) / win_std

            return normalized
        else:
            # For Dask array, use map_overlap
            def _local_norm_chunk(chunk):
                # Create output array
                result = np.zeros_like(chunk)
                half_window = window // 2

                # Apply windowed normalization
                for i in range(len(chunk)):
                    # Define window boundaries
                    start = max(0, i - half_window)
                    end = min(len(chunk), i + half_window + 1)

                    # Get window data
                    window_data = chunk[start:end]

                    # Normalize using window statistics
                    win_mean = np.mean(window_data)
                    win_std = np.std(window_data)

                    if win_std == 0:
                        result[i] = 0
                    else:
                        result[i] = (chunk[i] - win_mean) / win_std

                return result

            # Apply with proper overlap
            return data.map_overlap(
                _local_norm_chunk, depth=window // 2, boundary="reflect"
            )
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# %%
# Option 1: Detect peaks on the filtered signal (recommended)
signal_ = cmr_reference
highpass_filter = create_highpass_filter(
    cutoff=100,
    fs=FS,
    order=5,
)
highpass_processor = FilterProcessor(
    filter_func=highpass_filter.get_filter_function(), overlap_samples=40
)

signal_filtered = highpass_processor.process(signal_, FS)


# Try median-based thresholding (robust to outliers)
def detect_spikes_median(signal_data, threshold_factor=5, min_distance=10):
    # Convert to numpy array if it's a dask array
    if isinstance(signal_data, da.Array):
        # Compute the dask array to convert to numpy
        signal_data = signal_data.compute()

    # Estimate noise level using median
    median = np.median(np.abs(signal_data)) / 0.6745  # Approximation of std
    threshold = threshold_factor * median

    # Find peaks above threshold
    peaks, properties = signal.find_peaks(
        np.abs(signal_data),  # Look for both positive and negative peaks
        height=threshold,
        distance=min_distance,
    )

    return peaks, threshold


# Apply median-based thresholding - get just one channel
signal_to_process = signal_filtered[:100000, 0]

# Convert to numpy if needed
if isinstance(signal_to_process, da.Array):
    signal_to_process = signal_to_process.compute()

# Now detect spikes
peaks, threshold = detect_spikes_median(
    signal_to_process,
    threshold_factor=4.5,  # Adjust this factor
    min_distance=int(FS * 0.001),  # Minimum 1ms between spikes
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(signal_to_process)
plt.plot(peaks, signal_to_process[peaks], "ro", label="Detected Spikes")
plt.axhline(threshold, color="g", linestyle="--", label="Threshold")
plt.axhline(-threshold, color="g", linestyle="--")
plt.legend()
plt.title("Median-based Spike Detection")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")


# %%
# Try median-based thresholding for positive peaks only
def detect_positive_spikes_median(signal_data, threshold_factor=5, min_distance=10):
    # Convert to numpy array if it's a dask array
    if isinstance(signal_data, da.Array):
        # Compute the dask array to convert to numpy
        signal_data = signal_data.compute()

    # Estimate noise level using median
    median = np.median(np.abs(signal_data)) / 0.6745  # Approximation of std
    threshold = threshold_factor * median

    # Find positive peaks above threshold (no abs() this time)
    peaks, properties = signal.find_peaks(
        signal_data,  # Use the raw signal, not the absolute value
        height=threshold,  # Only detect peaks above the positive threshold
        distance=min_distance,
    )

    return peaks, threshold


# Apply median-based thresholding for positive peaks
signal_to_process = signal_filtered[:100000, 0]

# Convert to numpy if needed
if isinstance(signal_to_process, da.Array):
    signal_to_process = signal_to_process.compute()

# Now detect positive spikes
peaks, threshold = detect_positive_spikes_median(
    signal_to_process,
    threshold_factor=4.5,  # Adjust this factor
    min_distance=int(FS * 0.01),  # Minimum 1ms between spikes
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(signal_to_process)
plt.plot(peaks, signal_to_process[peaks], "ro", label="Positive Spikes")
plt.axhline(threshold, color="g", linestyle="--", label="Threshold")
plt.legend()
plt.title("Positive Spike Detection")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
# %%


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def extract_spike_template_from_data(signal_data, initial_peaks=None, window_size=20):
    """
    Extract a spike template from the actual data.

    Args:
        signal_data: Input signal
        initial_peaks: Initial estimate of spike locations
        window_size: Window size for spike extraction

    Returns:
        Averaged spike template, spike waveforms array
    """
    if initial_peaks is None or len(initial_peaks) == 0:
        # If no initial peaks, use threshold crossing to find some
        threshold = np.mean(signal_data) + 3 * np.std(signal_data)
        initial_peaks = np.where(signal_data > threshold)[0]

    # Extract spike waveforms
    half_window = window_size // 2
    spike_waveforms = []

    for peak in initial_peaks:
        if peak >= half_window and peak < len(signal_data) - half_window:
            waveform = signal_data[peak - half_window : peak + half_window]
            spike_waveforms.append(waveform)

    if not spike_waveforms:
        raise ValueError("No valid spike waveforms found")

    # Stack and average
    spike_array = np.vstack(spike_waveforms)

    # Calculate average template
    template = np.mean(spike_array, axis=0)

    # Normalize
    template = template / np.sqrt(np.sum(template**2))

    return template, spike_array


def template_match_spike_detection(
    signal_data, template, threshold=0.5, min_distance=10, return_correlation=True
):
    """
    Detect spikes using template matching via convolution.

    Args:
        signal_data: Input signal (1D array)
        template: Spike template extracted from data
        threshold: Correlation threshold for detection
        min_distance: Minimum distance between detected spikes
        return_correlation: Whether to return correlation signal

    Returns:
        peaks: Indices of detected spikes
        properties: Dictionary of peak properties
        correlation: Correlation signal (if return_correlation=True)
    """
    # Convert to numpy array if it's a dask array
    if isinstance(signal_data, da.Array):
        signal_data = signal_data.compute()

    # Normalize signal for better correlation
    signal_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)

    # Cross-correlate signal with template
    correlation = signal.correlate(signal_norm, template, mode="same")
    correlation = correlation / np.max(np.abs(correlation))  # Normalize to [-1, 1]

    # Find peaks in correlation that exceed threshold
    peaks, properties = signal.find_peaks(
        correlation, height=threshold, distance=min_distance
    )

    if return_correlation:
        return peaks, properties, correlation
    else:
        return peaks, properties


def template_based_spike_detection(
    signal_data, window_size=None, threshold=0.5, min_distance=10
):
    """
    Complete template-based spike detection pipeline.

    Args:
        signal_data: Input signal
        window_size: Window size for template extraction
        threshold: Correlation threshold for detection
        min_distance: Minimum distance between spikes

    Returns:
        peaks: Detected spike indices
        template: Extracted spike template
        correlation: Correlation signal
    """
    # Convert to numpy if needed
    if isinstance(signal_data, da.Array):
        signal_data = signal_data.compute()

    # Normalize signal
    signal_norm = (signal_data - np.mean(signal_data)) / np.std(signal_data)

    # Set appropriate window size if not specified
    if window_size is None:
        if "FS" in globals():
            window_size = int(0.003 * FS)  # 3ms window
        else:
            window_size = 20  # Default if FS not available

    # Step 1: Find initial peaks for template extraction
    # Use a simple threshold approach to find candidate peaks
    threshold_value = np.mean(signal_norm) + 3 * np.std(signal_norm)
    initial_peaks, _ = signal.find_peaks(
        signal_norm, height=threshold_value, distance=min_distance
    )

    if len(initial_peaks) == 0:
        # Try with a lower threshold if no peaks found
        threshold_value = np.mean(signal_norm) + 2 * np.std(signal_norm)
        initial_peaks, _ = signal.find_peaks(
            signal_norm, height=threshold_value, distance=min_distance
        )

        if len(initial_peaks) == 0:
            raise ValueError("Could not find initial peaks for template extraction")

    # Step 2: Extract template from initial peaks
    template, spike_array = extract_spike_template_from_data(
        signal_norm, initial_peaks, window_size
    )

    # Step 3: Perform template matching with extracted template
    peaks, properties, correlation = template_match_spike_detection(
        signal_norm,
        template=template,
        threshold=threshold,
        min_distance=min_distance,
        return_correlation=True,
    )

    return peaks, template, correlation, spike_array


# Apply template-based spike detection
signal_to_process = signal_filtered[:100000, 0]

# Convert to numpy if needed
if isinstance(signal_to_process, da.Array):
    signal_to_process = signal_to_process.compute()

# Set appropriate template window size based on your data
template_window = int(0.002 * FS)  # 2ms window
min_spike_distance = int(FS * 0.001)  # 1ms minimum distance between spikes


# Detect spikes using template matching
peaks, template, correlation, spike_array = template_based_spike_detection(
    signal_to_process,
    window_size=template_window,
    threshold=0.45,  # Adjust this threshold (0-1)
    min_distance=min_spike_distance,
)

# Plot results
plt.figure(figsize=(12, 10))

# Plot the extracted template
plt.subplot(311)
plt.plot(np.arange(-template_window // 2, template_window // 2), template)
plt.title(f"Extracted Spike Template (n={len(spike_array)})")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

# Plot the signal with detected spikes
plt.subplot(312)
plt.plot(signal_to_process)
plt.plot(peaks, signal_to_process[peaks], "ro", label=f"Spikes ({len(peaks)})")
plt.legend()
plt.title("Template-Matched Spike Detection")
plt.ylabel("Amplitude")

# Plot the correlation signal
plt.subplot(313)
plt.plot(correlation)
plt.axhline(y=0.45, color="r", linestyle="--", label="Threshold (0.45)")
plt.title("Template Correlation")
plt.xlabel("Sample Index")
plt.ylabel("Correlation")
plt.legend()

plt.tight_layout()

# Additional visualization: show all extracted spikes
plt.figure(figsize=(10, 6))

# Plot individual spikes (up to 50)
max_to_plot = min(50, len(spike_array))
for i in range(max_to_plot):
    plt.plot(
        np.arange(-template_window // 2, template_window // 2),
        spike_array[i],
        "k-",
        alpha=0.2,
    )

# Plot mean spike
plt.plot(
    np.arange(-template_window // 2, template_window // 2),
    template,
    "r-",
    linewidth=2,
    label="Mean Template",
)

# Plot standard deviation
std_spike = np.std(spike_array, axis=0)
mean_spike = np.mean(spike_array, axis=0)  # Same as template but unscaled
plt.fill_between(
    np.arange(-template_window // 2, template_window // 2),
    mean_spike - std_spike,
    mean_spike + std_spike,
    color="r",
    alpha=0.2,
    label="±1 SD",
)

plt.axvline(x=0, color="b", linestyle="--", label="Peak Alignment")
plt.title(f"Aligned Spike Waveforms (n={len(spike_array)})")
plt.xlabel("Samples relative to peak")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)


# %%
