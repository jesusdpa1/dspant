"""
Functions for extracting and aligning spike waveforms.

This module provides functionality to:
1. Extract spike waveforms from raw data based on detected spike indices
2. Align spikes to their peak for better comparison
3. Perform various alignments and normalization operations
"""

import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numba import jit, prange
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


@jit(nopython=True, parallel=True, cache=True)
def _extract_and_align_waveforms_numba(
    data: np.ndarray,
    spike_times: np.ndarray,
    pre_samples: int,
    post_samples: int,
    align_to_min: bool,
    align_window: int,
    max_jitter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and align spike waveforms with Numba acceleration.

    Args:
        data: Raw data array (samples x channels)
        spike_times: Spike times in samples
        pre_samples: Number of samples to take before the spike peak
        post_samples: Number of samples to take after the spike peak
        align_to_min: Whether to align spikes to negative peak
        align_window: Window around detected peak to search for alignment
        max_jitter: Maximum allowed jitter for alignment

    Returns:
        Tuple of (waveforms, valid_indices)
    """
    n_spikes = len(spike_times)
    n_samples, n_channels = data.shape
    waveform_length = pre_samples + post_samples + 1

    # Pre-allocate arrays
    waveforms = np.zeros((n_spikes, waveform_length, n_channels))
    valid_indices = np.ones(n_spikes, dtype=np.bool_)

    # Extract and align waveforms
    for i in prange(n_spikes):
        spike_time = spike_times[i]

        # Check if waveform is within valid range
        if spike_time < pre_samples or spike_time >= n_samples - post_samples:
            valid_indices[i] = False
            continue

        # Extract waveform
        window_start = spike_time - pre_samples
        window_end = spike_time + post_samples + 1
        waveform = data[window_start:window_end, :]

        # Align waveform if requested
        if align_to_min and waveform.shape[0] == waveform_length:
            aligned_waveform, _ = _align_waveform(
                waveform,
                pre_samples,
                align_window,
                max_jitter,
                is_negative=(data[spike_time, 0] < 0),
            )
            waveforms[i] = aligned_waveform
        else:
            # Use as is
            waveforms[i] = waveform

    return waveforms, valid_indices


def extract_spike_waveforms(
    data: np.ndarray,
    spike_indices: Union[np.ndarray, pl.DataFrame],
    pre_samples: int = 10,
    post_samples: int = 40,
    align_to_min: bool = True,
    align_window: int = 5,
    align_interp: bool = False,
    max_jitter: int = 3,
    max_workers: Optional[int] = None,
    use_numba: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract spike waveforms from raw data and align them.

    Args:
        data: Raw data array (samples x channels)
        spike_indices: Spike detection results (DataFrame or structured array)
        pre_samples: Number of samples to take before the spike peak
        post_samples: Number of samples to take after the spike peak
        align_to_min: Whether to align spikes to their minimum (negative peak)
        align_window: Window around detected peak to search for alignment
        align_interp: Whether to use cubic interpolation for sub-sample alignment
        max_jitter: Maximum allowed jitter for alignment
        max_workers: Maximum number of worker threads for parallel processing
        use_numba: Whether to use Numba acceleration

    Returns:
        Tuple of (waveforms, spike_times, metadata)
        - waveforms: Array of aligned spike waveforms (spikes x samples)
        - spike_times: Array of spike times in samples
        - metadata: Dictionary with additional information
    """
    # Convert spike_indices to numpy arrays if it's a DataFrame
    if isinstance(spike_indices, pl.DataFrame):
        spike_times = spike_indices["index"].to_numpy()
        channels = spike_indices["channel"].to_numpy()
        amplitudes = spike_indices["amplitude"].to_numpy()
    else:
        # Assume structured numpy array
        spike_times = spike_indices["index"]
        channels = spike_indices["channel"]
        amplitudes = spike_indices["amplitude"]

    # Get number of spikes and data shape
    n_spikes = len(spike_times)
    if n_spikes == 0:
        return (np.array([]), np.array([]), {"n_spikes": 0, "channels": np.array([])})

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    waveform_length = pre_samples + post_samples + 1

    # Use Numba acceleration if available and requested
    if use_numba:
        # Extract waveforms with Numba
        waveforms, valid_indices = _extract_and_align_waveforms_numba(
            data,
            spike_times,
            pre_samples,
            post_samples,
            align_to_min,
            align_window,
            max_jitter,
        )

        # Filter out invalid waveforms
        waveforms = waveforms[valid_indices]
        spike_times = spike_times[valid_indices]
        channels = channels[valid_indices]
        amplitudes = amplitudes[valid_indices]
    else:
        # Use non-Numba implementation (with interpolation option)
        # Pre-allocate array for waveforms
        waveforms = np.zeros((n_spikes, waveform_length, n_channels))
        valid_indices = np.ones(n_spikes, dtype=bool)

        # Function to extract a single waveform
        def extract_waveform(i):
            spike_time = spike_times[i]
            channel = min(channels[i], n_channels - 1)  # Ensure channel is valid

            # Check if waveform is within valid range
            if spike_time < pre_samples or spike_time >= n_samples - post_samples:
                return i, None, False

            # Extract waveform
            waveform = data[spike_time - pre_samples : spike_time + post_samples + 1]

            # Align waveform if requested
            if align_to_min:
                # Search for minimum within alignment window
                search_start = max(0, pre_samples - align_window)
                search_end = min(waveform_length, pre_samples + align_window + 1)
                search_window = waveform[search_start:search_end, channel]

                if len(search_window) > 0:
                    # Find position of minimum in search window
                    if amplitudes[i] < 0:
                        # For negative spikes, find minimum
                        min_idx = np.argmin(search_window) + search_start
                    else:
                        # For positive spikes, find maximum
                        min_idx = np.argmax(search_window) + search_start

                    shift = pre_samples - min_idx

                    # Apply shift if within jitter limits
                    if abs(shift) <= max_jitter:
                        if shift != 0:
                            if align_interp and abs(shift) < waveform_length - 2:
                                # Use cubic interpolation for sub-sample alignment
                                x = np.arange(waveform_length)
                                new_waveform = np.zeros_like(waveform)

                                for ch in range(n_channels):
                                    f = interpolate.interp1d(
                                        x,
                                        waveform[:, ch],
                                        kind="cubic",
                                        bounds_error=False,
                                        fill_value=0,
                                    )
                                    new_x = x - shift
                                    new_waveform[:, ch] = f(new_x)

                                waveform = new_waveform
                            else:
                                # Apply integer shift
                                temp = np.zeros_like(waveform)
                                if shift > 0:
                                    temp[shift:] = waveform[:-shift]
                                else:
                                    temp[:shift] = waveform[-shift:]
                                waveform = temp

            return i, waveform, True

        # Extract waveforms
        if max_workers is not None and n_spikes > 10:
            # Process in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(extract_waveform, i) for i in range(n_spikes)
                ]
                for future in concurrent.futures.as_completed(futures):
                    i, waveform, is_valid = future.result()
                    if is_valid:
                        waveforms[i] = waveform
                    else:
                        valid_indices[i] = False
        else:
            # Process sequentially
            for i in range(n_spikes):
                i, waveform, is_valid = extract_waveform(i)
                if is_valid:
                    waveforms[i] = waveform
                else:
                    valid_indices[i] = False

        # Filter out invalid waveforms
        waveforms = waveforms[valid_indices]
        spike_times = spike_times[valid_indices]
        channels = channels[valid_indices]
        amplitudes = amplitudes[valid_indices]

    # Create metadata dictionary
    metadata = {
        "n_spikes": len(waveforms),
        "channels": channels,
        "amplitudes": amplitudes,
        "pre_samples": pre_samples,
        "post_samples": post_samples,
        "aligned": align_to_min,
    }

    return waveforms, spike_times, metadata


def align_waveforms(
    waveforms: np.ndarray,
    align_to_min: bool = True,
    align_window: int = 5,
    max_jitter: int = 3,
    use_numba: bool = True,
) -> np.ndarray:
    """
    Align spike waveforms to their peak.

    Args:
        waveforms: Array of spike waveforms (spikes x samples x channels)
        align_to_min: Whether to align to minimum (negative peak)
        align_window: Window size to search for peak
        max_jitter: Maximum allowed jitter in samples
        use_numba: Whether to use Numba acceleration

    Returns:
        Array of aligned waveforms
    """
    n_spikes, n_samples, n_channels = waveforms.shape
    aligned_waveforms = np.zeros_like(waveforms)

    if n_spikes == 0:
        return waveforms

    # Center sample index
    pre_samples = n_samples // 2

    # Use Numba acceleration if available and requested
    if use_numba:
        for i in range(n_spikes):
            is_negative = np.min(waveforms[i, :, 0]) < 0
            aligned_waveforms[i], _ = _align_waveform(
                waveforms[i], pre_samples, align_window, max_jitter, is_negative
            )
    else:
        for i in range(n_spikes):
            # Determine polarity based on first channel
            is_negative = np.min(waveforms[i, :, 0]) < 0

            # Search for peak within alignment window
            search_start = max(0, pre_samples - align_window)
            search_end = min(n_samples, pre_samples + align_window + 1)

            if is_negative:
                # Find negative peak
                peak_idx = (
                    np.argmin(waveforms[i, search_start:search_end, 0]) + search_start
                )
            else:
                # Find positive peak
                peak_idx = (
                    np.argmax(waveforms[i, search_start:search_end, 0]) + search_start
                )

            # Calculate shift
            shift = pre_samples - peak_idx

            # Apply shift if within jitter limits
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
