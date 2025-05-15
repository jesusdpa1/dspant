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
from dspant.pattern.subtraction.correlation import (
    create_correlation_subtractor,
    subtract_templates,
)
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
# %%
# Load streams
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()

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
processor_emg = create_processing_node(stream_emg)

# Add processors to the nodes
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# %%
# Apply filters
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
# Extract ECG templates
window_ms = 60  # Window size in milliseconds
window_samples = int((window_ms / 1000) * FS)
half_win = window_samples // 2
waveform_processor = WaveformExtractor(filter_emg[:, 0], FS)

# Extract waveforms
ecg_waveforms = waveform_processor.extract_waveforms(
    r_peak_indices, pre_samples=half_win, post_samples=half_win, time_unit="samples"
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
# Using the new subtraction function with the new interface
# Use the convenience function for direct template subtraction


# Alternatively, using the class-based approach
subtractor = create_correlation_subtractor()
cleaned_emg_alt = subtractor.process(
    data=filter_emg[time_slice, :1],
    template=ecg_template[:, :1],
    indices=r_peak_indices,
    fs=FS,
    mode=None,
).compute()


# %%
# For comparison, also use our manual implementation
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


# Apply manual template subtraction for comparison
cleaned_emg_manual = subtract_ecg_template(emg_data, template, r_peak_indices, half_win)

# %%
# Plot original and all cleaned signals for comparison
plt.figure(figsize=(15, 15))
plt.subplot(3, 1, 1)
plt.plot(emg_data, label="Original EMG Signal")
plt.plot(r_peak_indices, emg_data[r_peak_indices], "r*", markersize=8, label="R-peaks")
plt.title("Original EMG Signal with R-peaks")
plt.legend()
plt.grid(True)


plt.subplot(3, 1, 2)
plt.plot(cleaned_emg_alt[:, 0], label="Cleaned EMG (create_correlation_subtractor)")
plt.title("EMG Signal after ECG Template Subtraction (class-based approach)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(cleaned_emg_manual, label="Cleaned EMG (manual implementation)")
plt.title("EMG Signal after ECG Template Subtraction (manual)")
plt.xlabel("Samples")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
"""
Cross-correlation based template subtraction.

This module implements template subtraction using correlation for optimal
alignment and scaling. It's particularly effective for removing artifacts
like ECG from EMG or other electrophysiological signals.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import dask.array as da
import numpy as np
from scipy.signal import correlate

from dspant.core.internals import public_api
from dspant.pattern.subtraction.base import BaseSubtractor


@public_api
class CorrelationSubtractor(BaseSubtractor):
    """
    Template subtraction using cross-correlation for alignment.

    This subtractor aligns templates with signal segments using cross-correlation,
    scales them optimally, and subtracts them from the original signal.
    Useful for removing artifacts like ECG from EMG or EEG signals.
    """

    def __init__(self):
        super().__init__()

    def process(
        self,
        data: da.Array,
        template: Union[np.ndarray, da.Array],
        fs: Optional[float] = None,
        mode: Optional[Literal["global", None]] = None,
        indices: Optional[Union[np.ndarray, List[int]]] = None,
        half_window: Optional[int] = None,
        **kwargs,
    ) -> da.Array:
        """
        Subtract templates from data at specified indices.

        Args:
            data: Dask array of shape (samples, channels).
            template: Template array of shape (samples, channels) or (samples,).
            fs: Sampling frequency in Hz (optional).
            mode: 'global' applies the first channel of template to all channels;
                  None uses per-channel subtraction.
            indices: List or array of sample indices where templates are subtracted.
            half_window: Half-window size in samples (defaults to half template length).
            **kwargs: Reserved for future options.

        Returns:
            Data array with template artifacts subtracted.
        """
        if indices is None:
            raise ValueError("Indices must be provided for template subtraction.")

        # Ensure template is contiguous numpy array
        if isinstance(template, da.Array):
            template = np.ascontiguousarray(template.compute())
        else:
            template = np.ascontiguousarray(template)

        if template.ndim == 1:
            template = template[:, None]

        # Validate dimensions
        if mode is None and template.shape[1] != data.shape[1]:
            raise ValueError(
                f"Template channels ({template.shape[1]}) must match data channels ({data.shape[1]}) "
                f"when mode is None. Use mode='global' for single-channel templates."
            )

        # Set up window parameters
        template_len = template.shape[0]
        half_window = half_window or template_len // 2
        window_size = 2 * template_len

        # Use a large overlap to ensure we don't miss any indices
        # This is crucial - we need to have plenty of room around chunk boundaries
        self._overlap_samples = window_size * 3

        # Ensure indices are sorted contiguous array for efficiency
        indices = np.sort(np.asarray(indices))

        # Adjust indices for sliced arrays
        data_offset = 0
        if hasattr(data, "_key"):
            try:
                key_info = data._key
                if isinstance(key_info, tuple) and len(key_info) >= 2:
                    slice_info = key_info[1]
                    if isinstance(slice_info, tuple) and len(slice_info) > 0:
                        if isinstance(slice_info[0], slice):
                            start = slice_info[0].start
                            if start is not None:
                                data_offset = start
                                # Check if indices need adjustment
                                if np.any(indices >= data.shape[0] + start):
                                    indices = indices - start
            except Exception as e:
                print(f"Warning: Error adjusting indices: {e}")

        def subtract_chunk(chunk, block_info=None):
            """Process a single chunk with proper boundary handling"""
            # Ensure chunk is 2D contiguous
            if chunk.ndim == 1:
                chunk = np.ascontiguousarray(chunk[:, None])
            else:
                chunk = np.ascontiguousarray(chunk)

            # Get chunk position
            chunk_start = 0
            if block_info and 0 in block_info:
                chunk_start = block_info[0]["array-location"][0][0]

            # Use a large margin to catch indices that might affect this chunk
            margin = window_size * 2
            chunk_end = chunk_start + chunk.shape[0]

            # Find indices that might influence this chunk
            # Use a very wide range to be safe
            mask = (indices >= chunk_start - margin) & (indices < chunk_end + margin)
            relevant_indices = indices[mask]

            # If no relevant indices, return chunk unchanged
            if len(relevant_indices) == 0:
                return chunk

            # Create output array
            result = chunk.copy()

            # Create array of local indices within this chunk
            local_indices = relevant_indices - chunk_start

            # Process each channel
            if mode == "global":
                # Global mode: use same template channel for all data channels
                template_ch = np.ascontiguousarray(template[:, 0])
                for ch in range(result.shape[1]):
                    channel_data = np.ascontiguousarray(result[:, ch])
                    result[:, ch] = self._subtract_from_channel(
                        channel_data, template_ch, local_indices, half_window
                    )
            else:
                # Per-channel mode: use matching template channel
                for ch in range(result.shape[1]):
                    # Get template channel (use last one if out of bounds)
                    t_ch = min(ch, template.shape[1] - 1)
                    template_ch = np.ascontiguousarray(template[:, t_ch])

                    # Process channel
                    channel_data = np.ascontiguousarray(result[:, ch])
                    result[:, ch] = self._subtract_from_channel(
                        channel_data, template_ch, local_indices, half_window
                    )

            return result

        # Process data with overlap
        result = data.map_overlap(
            subtract_chunk,
            depth={-2: self._overlap_samples},  # Keep this format
            boundary="reflect",
            dtype=data.dtype,
            block_info=True,
        )

        # Save stats
        self._subtraction_stats.update(
            {
                "num_indices": len(indices),
                "window_size": window_size,
                "template_length": template_len,
                "fs": fs,
                "mode": mode,
                "data_offset": data_offset,
            }
        )

        return result

    def _subtract_from_channel(
        self,
        data: np.ndarray,  # Changed from 'signal' to 'data'
        template: np.ndarray,
        indices: np.ndarray,
        half_window: int,
    ) -> np.ndarray:
        """Subtract template from a single channel at specified indices"""
        # Make a copy and ensure it's contiguous
        result = np.ascontiguousarray(data.copy())  # Changed from 'signal' to 'data'
        n_samples = len(result)

        # Process each index
        for idx in indices:
            # Skip if out of range or too close to edge
            if idx < half_window or idx >= n_samples - half_window:
                continue

            # Extract segment for correlation
            segment = result[idx - half_window : idx + half_window]

            # Skip if segment size is wrong
            if len(segment) != 2 * half_window:
                continue

            # Calculate correlation with full mode for better alignment
            # Cross-correlation finds how much to shift the template to align with signal
            corr = correlate(segment, template, mode="full")

            # Find optimal lag (shift) to align template with signal
            # The -1 term adjusts for zero-indexing
            lag = np.argmax(corr) - (len(template) - 1)

            # Calculate start and end positions with shift
            start = idx - half_window + lag
            end = start + len(template)

            # Skip if adjustment puts us out of bounds
            if start < 0 or end > n_samples:
                continue

            # Extract the aligned segment
            aligned = result[start:end]

            # Skip if lengths don't match
            if len(aligned) != len(template):
                continue

            # Calculate optimal scaling factor to match template amplitude
            energy = np.dot(template, template)
            if energy > 1e-10:  # Avoid division by zero
                scale = np.dot(aligned, template) / energy

                # Subtract scaled template
                result[start:end] -= scale * template

        return result

    @property
    def summary(self) -> Dict[str, Any]:
        return super().summary


@public_api
def subtract_templates(
    data: Union[np.ndarray, da.Array],
    template: np.ndarray,
    indices: Union[np.ndarray, List[int]],
    half_window: Optional[int] = None,
    mode: Optional[Literal["global", None]] = None,
) -> Union[np.ndarray, da.Array]:
    """
    Convenience wrapper for one-off template subtraction.

    Args:
        data: Input signal array (samples × channels).
        template: Template to subtract (samples × channels or 1D).
        indices: Where to subtract template.
        half_window: Half window around each index (optional).
        mode: 'global' for shared template; None for per-channel.

    Returns:
        Signal with artifacts removed.
    """
    subtractor = CorrelationSubtractor()
    return subtractor.process(
        data=data,
        template=template,
        indices=indices,
        half_window=half_window,
        mode=mode,
    )


@public_api
def create_correlation_subtractor() -> CorrelationSubtractor:
    """
    Create a correlation-based template subtractor.

    Returns:
        Configured CorrelationSubtractor
    """
    return CorrelationSubtractor()


#%%
subtractor = create_correlation_subtractor()
cleaned_emg_alt = subtractor.process(
    data=filter_emg[time_slice, :1],
    template=ecg_template[:, :1],
    indices=r_peak_indices,
    fs=FS,
    mode=None,
).compute()

# %%
plt.plot(filter_emg[time_slice, :1])
plt.plot(cleaned_emg_alt)

# %%
