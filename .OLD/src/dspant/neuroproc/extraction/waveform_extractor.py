"""
Functions for extracting and aligning spike waveforms using Dask arrays with Numba acceleration.

This module provides functionality to:
1. Extract spike waveforms from raw data based on detected spike indices
2. Align spikes to their peak for better comparison
3. Efficiently process out-of-memory data with Dask
4. Accelerate computations with Numba when appropriate

The implementation uses Numba for speeding up operations on extracted segments
while maintaining Dask for handling large datasets efficiently.
"""

from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit
from scipy import interpolate


@jit(nopython=True, cache=True)
def _align_waveform(
    waveform: np.ndarray,
    pre_samples: int,
    align_window: int,
    max_jitter: int,
    is_negative: bool,
) -> Tuple[np.ndarray, int]:
    """
    Align waveform to its peak with Numba acceleration.

    Args:
        waveform: Waveform to align
        pre_samples: Number of samples before the peak
        align_window: Window around detected peak to search for alignment
        max_jitter: Maximum allowed jitter for alignment
        is_negative: Whether to align to negative or positive peak

    Returns:
        Tuple of (aligned_waveform, shift)
    """
    waveform_length = waveform.shape[0]
    n_channels = waveform.shape[1]

    # Search for peak within alignment window
    search_start = max(0, pre_samples - align_window)
    search_end = min(waveform_length, pre_samples + align_window + 1)

    # Use first channel for alignment
    search_window = waveform[search_start:search_end, 0]

    # Find position of peak in search window
    if is_negative:
        # For negative spikes, find minimum
        peak_idx = np.argmin(search_window) + search_start
    else:
        # For positive spikes, find maximum
        peak_idx = np.argmax(search_window) + search_start

    # Calculate shift required
    shift = pre_samples - peak_idx

    # Apply shift if within jitter limits
    if abs(shift) <= max_jitter:
        # Create aligned waveform
        aligned = np.zeros_like(waveform)

        # Apply integer shift
        if shift > 0:
            # Shift right
            if shift < waveform_length:
                aligned[shift:, :] = waveform[:-shift, :]
        else:
            # Shift left
            shift_abs = -shift
            if shift_abs < waveform_length:
                aligned[:-shift_abs, :] = waveform[shift_abs:, :]

        return aligned, shift

    # If shift is too large, return original waveform
    return waveform.copy(), 0


@jit(nopython=True, cache=True)  # Removed parallel=True
def _align_multiple_waveforms(
    waveforms: np.ndarray,
    pre_samples: int,
    align_window: int,
    max_jitter: int,
    polarities: np.ndarray,
) -> np.ndarray:
    """
    Align multiple waveforms to their peaks with Numba acceleration.

    Args:
        waveforms: Array of waveforms (spikes, samples, channels)
        pre_samples: Number of samples before the peak
        align_window: Window around detected peak to search for alignment
        max_jitter: Maximum allowed jitter for alignment
        polarities: Array indicating whether each spike is negative (True) or positive (False)

    Returns:
        Array of aligned waveforms
    """
    n_spikes, n_samples, n_channels = waveforms.shape
    aligned_waveforms = np.zeros_like(waveforms)

    for i in range(n_spikes):  # Removed prange
        is_negative = polarities[i]
        aligned_waveforms[i], _ = _align_waveform(
            waveforms[i], pre_samples, align_window, max_jitter, is_negative
        )

    return aligned_waveforms


def extract_spike_waveforms(
    data: da.Array,
    spike_indices: Union[np.ndarray, pl.DataFrame],
    pre_samples: int = 10,
    post_samples: int = 40,
    align_to_min: bool = True,
    align_window: int = 5,
    align_interp: bool = False,
    max_jitter: int = 3,
    use_numba: bool = True,
    compute_result: bool = False,
) -> Tuple[da.Array, np.ndarray, Dict]:
    """
    Extract spike waveforms from Dask array and align them.

    Args:
        data: Raw data Dask array (samples x channels)
        spike_indices: Spike detection results (DataFrame or structured array)
        pre_samples: Number of samples to take before the spike peak
        post_samples: Number of samples to take after the spike peak
        align_to_min: Whether to align spikes to their minimum (negative peak)
        align_window: Window around detected peak to search for alignment
        align_interp: Whether to use cubic interpolation for sub-sample alignment
        max_jitter: Maximum allowed jitter for alignment
        use_numba: Whether to use Numba acceleration for alignment (when computing)
        compute_result: Whether to immediately compute Dask results into NumPy arrays

    Returns:
        Tuple of (waveforms, spike_times, metadata)
        - waveforms: Dask array of aligned spike waveforms (spikes x samples)
        - spike_times: Array of spike times in samples
        - metadata: Dictionary with additional information
    """
    # Convert spike_indices to numpy arrays if it's a DataFrame
    if isinstance(spike_indices, pl.DataFrame):
        spike_times = spike_indices["index"].to_numpy()
        channels = (
            spike_indices["channel"].to_numpy()
            if "channel" in spike_indices.columns
            else np.zeros(len(spike_times), dtype=np.int32)
        )
        amplitudes = (
            spike_indices["amplitude"].to_numpy()
            if "amplitude" in spike_indices.columns
            else np.zeros(len(spike_times), dtype=np.float32)
        )
    else:
        # Assume structured numpy array or plain array of indices
        try:
            spike_times = spike_indices["index"]
            channels = (
                spike_indices["channel"]
                if "channel" in spike_indices.dtype.names
                else np.zeros(len(spike_times), dtype=np.int32)
            )
            amplitudes = (
                spike_indices["amplitude"]
                if "amplitude" in spike_indices.dtype.names
                else np.zeros(len(spike_times), dtype=np.float32)
            )
        except (TypeError, IndexError, KeyError):
            # Assume a plain array of indices
            spike_times = np.asarray(spike_indices)
            channels = np.zeros(len(spike_times), dtype=np.int32)
            amplitudes = np.zeros(len(spike_times), dtype=np.float32)

    # Get number of spikes and data shape
    n_spikes = len(spike_times)
    if n_spikes == 0:
        # Return empty result
        empty_result = da.empty(
            (0, pre_samples + post_samples + 1, data.shape[1] if data.ndim > 1 else 1),
            chunks=(
                1,
                pre_samples + post_samples + 1,
                data.shape[1] if data.ndim > 1 else 1,
            ),
            dtype=data.dtype,
        )
        if compute_result:
            empty_result = empty_result.compute()
        return (empty_result, np.array([]), {"n_spikes": 0, "channels": np.array([])})

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    waveform_length = pre_samples + post_samples + 1

    # Filter spike times to prevent errors
    valid_mask = (spike_times >= pre_samples) & (spike_times < n_samples - post_samples)
    if not np.all(valid_mask):
        spike_times = spike_times[valid_mask]
        channels = channels[valid_mask]
        amplitudes = amplitudes[valid_mask]
        print(f"Filtered {np.sum(~valid_mask)} invalid spike times")

    # Extract waveforms from Dask array
    # The choice to use Numba for alignment is handled inside the function
    return _extract_dask_waveforms(
        data=data,
        spike_times=spike_times,
        channels=channels,
        amplitudes=amplitudes,
        pre_samples=pre_samples,
        post_samples=post_samples,
        align_to_min=align_to_min,
        align_window=align_window,
        align_interp=align_interp,
        max_jitter=max_jitter,
        use_numba=use_numba,
        compute_result=compute_result,
    )


def _extract_dask_waveforms(
    data: da.Array,
    spike_times: np.ndarray,
    channels: np.ndarray,
    amplitudes: np.ndarray,
    pre_samples: int,
    post_samples: int,
    align_to_min: bool,
    align_window: int,
    align_interp: bool,
    max_jitter: int,
    use_numba: bool = True,
    compute_result: bool = False,
) -> Tuple[Union[da.Array, np.ndarray], np.ndarray, Dict]:
    """
    Extract spike waveforms from a Dask array.

    Args:
        data: Raw data Dask array (samples x channels)
        spike_times: Array of spike times in samples
        channels: Array of channel indices
        amplitudes: Array of spike amplitudes
        pre_samples: Number of samples to take before the spike peak
        post_samples: Number of samples to take after the spike peak
        align_to_min: Whether to align spikes to their minimum (negative peak)
        align_window: Window around detected peak to search for alignment
        align_interp: Whether to use cubic interpolation for sub-sample alignment
        max_jitter: Maximum allowed jitter for alignment
        use_numba: Whether to use Numba acceleration for alignment (when computing)
        compute_result: Whether to immediately compute Dask results into NumPy arrays

    Returns:
        Tuple of (waveforms, spike_times, metadata)
    """
    # Using sequential extraction to avoid parallel execution issues
    # Prepare lists to store waveform data
    waveform_chunks = []
    valid_indices = []
    waveform_polarities = []  # Track whether each waveform is negative or positive

    # Extract waveforms
    for i, (spike_time, channel, amplitude) in enumerate(
        zip(spike_times, channels, amplitudes)
    ):
        try:
            # Extract waveform
            waveform_chunk = data[
                spike_time - pre_samples : spike_time + post_samples + 1, :
            ]

            # For alignment, we'll need the polarity
            is_negative = amplitude < 0 or (amplitude == 0 and align_to_min)
            waveform_polarities.append(is_negative)

            # Store waveform and index
            waveform_chunks.append(waveform_chunk)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error extracting spike at time {spike_time}: {str(e)}")
            continue

    # Handle case of no valid waveforms
    if not waveform_chunks:
        empty_result = da.empty(
            (0, pre_samples + post_samples + 1, data.shape[1]),
            chunks=(1, pre_samples + post_samples + 1, data.shape[1]),
            dtype=data.dtype,
        )
        if compute_result:
            empty_result = empty_result.compute()
        return (
            empty_result,
            np.array([]),
            {"n_spikes": 0, "channels": np.array([]), "amplitudes": np.array([])},
        )

    # Stack waveforms into a single Dask array
    waveforms = da.stack(waveform_chunks)

    # Filter spike times and channels
    valid_indices = np.array(valid_indices)
    spike_times = spike_times[valid_indices]
    channels = channels[valid_indices]
    amplitudes = amplitudes[valid_indices]

    # Create metadata dictionary
    metadata = {
        "n_spikes": len(valid_indices),
        "channels": channels,
        "amplitudes": amplitudes,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "aligned": align_to_min,
    }

    # Compute and align if requested
    if compute_result:
        # First compute to convert to NumPy
        waveforms_np = waveforms.compute()

        # Then apply alignment if requested
        if align_to_min:
            if use_numba:
                # Efficient alignment with Numba
                waveforms_np = _align_multiple_waveforms(
                    waveforms_np,
                    pre_samples,
                    align_window,
                    max_jitter,
                    np.array(waveform_polarities, dtype=np.bool_),
                )
            else:
                # Plain NumPy alignment
                aligned_waveforms = np.zeros_like(waveforms_np)
                for i, is_negative in enumerate(waveform_polarities):
                    # Search for peak within alignment window
                    search_start = max(0, pre_samples - align_window)
                    search_end = min(
                        waveforms_np.shape[1], pre_samples + align_window + 1
                    )

                    # Align based on polarity
                    if is_negative:
                        peak_idx = (
                            np.argmin(waveforms_np[i, search_start:search_end, 0])
                            + search_start
                        )
                    else:
                        peak_idx = (
                            np.argmax(waveforms_np[i, search_start:search_end, 0])
                            + search_start
                        )

                    # Calculate shift
                    shift = pre_samples - peak_idx

                    # Apply shift if within limits
                    if abs(shift) <= max_jitter:
                        if shift > 0:
                            # Shift right
                            aligned_waveforms[i, shift:, :] = waveforms_np[
                                i, :-shift, :
                            ]
                        elif shift < 0:
                            # Shift left
                            aligned_waveforms[i, :shift, :] = waveforms_np[
                                i, -shift:, :
                            ]
                        else:
                            # No shift needed
                            aligned_waveforms[i] = waveforms_np[i]
                    else:
                        # Shift too large, keep original
                        aligned_waveforms[i] = waveforms_np[i]

                waveforms_np = aligned_waveforms

            return waveforms_np, spike_times, metadata

    # For Dask array output (not computed), return as is
    # Alignment will be applied when computed or via align_waveforms function
    return waveforms, spike_times, metadata


def align_waveforms(
    waveforms: Union[da.Array, np.ndarray],
    align_to_min: bool = True,
    align_window: int = 5,
    max_jitter: int = 3,
    use_numba: bool = True,
    compute_result: bool = False,
) -> Union[da.Array, np.ndarray]:
    """
    Align spike waveforms to their peak.
    Supports both Dask arrays and NumPy arrays.

    Args:
        waveforms: Array of spike waveforms (spikes x samples x channels)
        align_to_min: Whether to align to minimum (negative peak)
        align_window: Window size to search for peak
        max_jitter: Maximum allowed jitter in samples
        use_numba: Whether to use Numba acceleration (when using NumPy arrays)
        compute_result: Whether to immediately compute Dask results into NumPy arrays

    Returns:
        Array of aligned waveforms
    """
    # Handle NumPy array input
    if isinstance(waveforms, np.ndarray):
        n_spikes, n_samples, n_channels = waveforms.shape

        if n_spikes == 0:
            return waveforms

        # Center sample index
        pre_samples = n_samples // 2

        # Determine polarities for each waveform
        polarities = np.array(
            [np.min(waveforms[i, :, 0]) < 0 for i in range(n_spikes)], dtype=np.bool_
        )

        # Use Numba acceleration if requested
        if use_numba:
            return _align_multiple_waveforms(
                waveforms, pre_samples, align_window, max_jitter, polarities
            )
        else:
            # Plain NumPy alignment
            aligned_waveforms = np.zeros_like(waveforms)
            for i, is_negative in enumerate(polarities):
                # Search for peak within alignment window
                search_start = max(0, pre_samples - align_window)
                search_end = min(n_samples, pre_samples + align_window + 1)

                # Find peak position
                if is_negative:
                    peak_idx = (
                        np.argmin(waveforms[i, search_start:search_end, 0])
                        + search_start
                    )
                else:
                    peak_idx = (
                        np.argmax(waveforms[i, search_start:search_end, 0])
                        + search_start
                    )

                # Calculate shift
                shift = pre_samples - peak_idx

                # Apply shift if within limits
                if abs(shift) <= max_jitter:
                    if shift > 0:
                        # Shift right
                        aligned_waveforms[i, shift:, :] = waveforms[i, :-shift, :]
                    elif shift < 0:
                        # Shift left
                        aligned_waveforms[i, :shift, :] = waveforms[i, -shift:, :]
                    else:
                        # No shift needed
                        aligned_waveforms[i] = waveforms[i]
                else:
                    # Shift too large, keep original
                    aligned_waveforms[i] = waveforms[i]

            return aligned_waveforms

    # Handle Dask array input
    n_spikes, n_samples, n_channels = waveforms.shape

    if n_spikes == 0:
        if compute_result:
            return waveforms.compute()
        return waveforms

    # Center sample index
    pre_samples = n_samples // 2

    # If we're computing the result, convert to NumPy and use Numba acceleration
    if compute_result:
        waveforms_np = waveforms.compute()
        polarities = np.array(
            [np.min(waveforms_np[i, :, 0]) < 0 for i in range(n_spikes)], dtype=np.bool_
        )

        if use_numba:
            return _align_multiple_waveforms(
                waveforms_np, pre_samples, align_window, max_jitter, polarities
            )
        else:
            # Use the NumPy implementation above
            return align_waveforms(
                waveforms_np, align_to_min, align_window, max_jitter, use_numba=False
            )

    # For Dask without computing, we need to process each waveform separately
    aligned_chunks = []

    # Process each waveform
    for i in range(n_spikes):
        # Compute the first channel for determining polarity
        first_channel = waveforms[i, :, 0].compute()
        is_negative = np.min(first_channel) < 0

        # Define search window
        search_start = max(0, pre_samples - align_window)
        search_end = min(n_samples, pre_samples + align_window + 1)

        # Compute search window for peak finding
        search_window = waveforms[i, search_start:search_end, 0].compute()

        # Find peak position
        if is_negative:
            peak_idx = np.argmin(search_window) + search_start
        else:
            peak_idx = np.argmax(search_window) + search_start

        # Calculate shift
        shift = pre_samples - peak_idx

        # Apply shift if within jitter limits
        if abs(shift) <= max_jitter:
            if shift != 0:
                # Create shifted version
                if shift > 0:
                    # Shift right
                    aligned_chunk = da.pad(
                        waveforms[i, :-shift]
                        if shift < n_samples
                        else waveforms[i, 0:0],
                        ((shift, 0), (0, 0)),
                        mode="constant",
                    )
                else:
                    # Shift left
                    aligned_chunk = da.pad(
                        waveforms[i, -shift:]
                        if -shift < n_samples
                        else waveforms[i, 0:0],
                        ((0, -shift), (0, 0)),
                        mode="constant",
                    )
            else:
                # No shift needed
                aligned_chunk = waveforms[i]
        else:
            # Shift too large, keep original
            aligned_chunk = waveforms[i]

        aligned_chunks.append(aligned_chunk)

    # Stack aligned waveforms into a Dask array
    aligned_waveforms = da.stack(aligned_chunks)

    return aligned_waveforms


def sequential_extract_waveforms(
    data: da.Array,
    spike_indices: Union[np.ndarray, pl.DataFrame],
    pre_samples: int = 10,
    post_samples: int = 40,
    align_to_min: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract spike waveforms sequentially to avoid Dask parallelism issues.

    Args:
        data: Raw data Dask array (samples x channels)
        spike_indices: Spike detection results (DataFrame or structured array)
        pre_samples: Number of samples to take before the spike peak
        post_samples: Number of samples to take after the spike peak
        align_to_min: Whether to align spikes to their minimum (negative peak)

    Returns:
        Tuple of (waveforms, spike_times, metadata)
    """
    # Convert to numpy for indexing
    if isinstance(spike_indices, pl.DataFrame):
        spike_times = spike_indices["index"].to_numpy()
    else:
        # Assume it's already a numpy array
        spike_times = np.asarray(spike_indices)

    # Storage for results
    waveforms_list = []
    valid_indices = []

    # Extract each waveform sequentially
    for i, spike_time in enumerate(spike_times):
        if i % 1000 == 0:
            print(f"Processing spike {i}/{len(spike_times)}")

        # Check bounds
        if spike_time < pre_samples or spike_time >= data.shape[0] - post_samples:
            continue

        try:
            # Extract without stacking to avoid the Dask parallel execution issue
            start_idx = spike_time - pre_samples
            end_idx = spike_time + post_samples + 1

            # Get the window and compute immediately - this is the key difference
            window = data[start_idx:end_idx].compute()

            # Verify shape
            if window.shape[0] == pre_samples + post_samples + 1:
                waveforms_list.append(window)
                valid_indices.append(i)
        except Exception as e:
            print(f"Error at index {spike_time}: {str(e)}")
            continue

    # Stack results
    if waveforms_list:
        # Stack in memory since we've already computed each window
        stacked_waveforms = np.stack(waveforms_list)
        valid_spike_times = spike_times[valid_indices]

        # Create metadata
        metadata = {
            "n_spikes": len(valid_spike_times),
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "aligned": align_to_min,
        }

        return stacked_waveforms, valid_spike_times, metadata
    else:
        # Handle empty result
        return np.array([]), np.array([]), {"n_spikes": 0}
