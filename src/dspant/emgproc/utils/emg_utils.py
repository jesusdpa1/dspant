"""
Utility functions for EMG signal processing.

This module provides various utility functions for EMG signal analysis,
including template extraction, amplitude estimation, and signal quality assessment.
These functions are optimized with numba for performance on large datasets.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def _extract_template_numba(waveforms: np.ndarray, align_index: int) -> np.ndarray:
    """
    Compute average template from multiple waveforms with numba acceleration.

    Parameters
    ----------
    waveforms : np.ndarray
        Array of waveforms (n_waveforms, n_samples)
    align_index : int
        Index to align the waveforms on (e.g., peak location)

    Returns
    -------
    template : np.ndarray
        Average template waveform
    """
    n_waveforms, n_samples = waveforms.shape
    template = np.zeros(n_samples, dtype=np.float32)

    # Compute the mean across waveforms
    for i in range(n_samples):
        sum_val = 0.0
        for j in range(n_waveforms):
            sum_val += waveforms[j, i]
        template[i] = sum_val / n_waveforms

    return template


@jit(nopython=True, parallel=True, cache=True)
def _extract_aligned_waveforms_numba(
    data: np.ndarray,
    indices: np.ndarray,
    window_pre: int,
    window_post: int,
    align_to_max: bool = False,
    align_to_min: bool = True,
    max_jitter: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract aligned waveforms around specified indices with numba acceleration.

    Parameters
    ----------
    data : np.ndarray
        Input data array (samples × channels)
    indices : np.ndarray
        Array of indices to extract waveforms from
    window_pre : int
        Number of samples before each index to include
    window_post : int
        Number of samples after each index to include
    align_to_max : bool
        Whether to align waveforms to their maximum value
    align_to_min : bool
        Whether to align waveforms to their minimum value
    max_jitter : int
        Maximum allowed jitter for alignment

    Returns
    -------
    waveforms : np.ndarray
        Array of aligned waveforms (n_waveforms, window_pre+window_post+1, n_channels)
    aligned_indices : np.ndarray
        Array of aligned indices after jitter correction
    """
    n_samples, n_channels = data.shape
    window_size = window_pre + window_post + 1

    # Count valid indices
    n_valid = 0
    for idx in indices:
        if window_pre <= idx < n_samples - window_post:
            n_valid += 1

    # Pre-allocate arrays
    waveforms = np.zeros((n_valid, window_size, n_channels), dtype=np.float32)
    aligned_indices = np.zeros(n_valid, dtype=np.int64)

    # Extract waveforms
    i_valid = 0
    for i in range(len(indices)):
        idx = indices[i]

        # Skip if outside valid range
        if idx < window_pre or idx >= n_samples - window_post:
            continue

        # Extract raw waveform
        start = idx - window_pre
        end = idx + window_post + 1

        for c in range(n_channels):
            waveform = data[start:end, c]

            # Align if requested
            align_shift = 0

            if align_to_max or align_to_min:
                # Find alignment point
                if align_to_max and align_to_min:
                    # Use extremum with largest magnitude
                    max_idx = np.argmax(waveform)
                    min_idx = np.argmin(waveform)

                    if abs(waveform[max_idx]) >= abs(waveform[min_idx]):
                        align_point = max_idx
                    else:
                        align_point = min_idx
                elif align_to_max:
                    align_point = np.argmax(waveform)
                else:  # align_to_min
                    align_point = np.argmin(waveform)

                # Calculate shift (how far from center the alignment point is)
                align_shift = window_pre - align_point

                # Limit jitter
                if abs(align_shift) > max_jitter:
                    align_shift = 0

            # Apply alignment shift if non-zero
            if align_shift != 0:
                shifted_waveform = np.zeros_like(waveform)

                if align_shift > 0:
                    # Shift right
                    shifted_waveform[align_shift:] = waveform[:-align_shift]
                else:
                    # Shift left (align_shift is negative)
                    shifted_waveform[:align_shift] = waveform[-align_shift:]

                waveforms[i_valid, :, c] = shifted_waveform
            else:
                waveforms[i_valid, :, c] = waveform

        # Store aligned index
        aligned_indices[i_valid] = idx + align_shift
        i_valid += 1

    return waveforms, aligned_indices


def extract_emg_templates(
    data: np.ndarray,
    onsets: Dict[str, List[np.ndarray]],
    window_pre: int = 50,
    window_post: int = 150,
    align_to_max: bool = False,
    align_to_min: bool = True,
    max_jitter: int = 2,
    max_onsets_per_template: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Extract average EMG templates for each onset group.

    Parameters
    ----------
    data : np.ndarray
        Input data array (samples × channels)
    onsets : Dict[str, List[np.ndarray]]
        Dictionary mapping onset IDs to lists of onset times in samples
    window_pre : int, default: 50
        Number of samples before onset to include
    window_post : int, default: 150
        Number of samples after onset to include
    align_to_max : bool, default: False
        Whether to align waveforms to their maximum value
    align_to_min : bool, default: True
        Whether to align waveforms to their minimum value (EMG often has negative peaks)
    max_jitter : int, default: 2
        Maximum allowed jitter for alignment
    max_onsets_per_template : int, default: 100
        Maximum number of onsets to use for computing a template

    Returns
    -------
    templates : Dict[str, np.ndarray]
        Dictionary mapping onset IDs to average templates
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    templates = {}

    for onset_id, onset_times_list in onsets.items():
        all_waveforms = []

        for segment_onsets in onset_times_list:
            if len(segment_onsets) == 0:
                continue

            # Limit number of onsets if needed
            if len(segment_onsets) > max_onsets_per_template:
                segment_onsets = np.random.choice(
                    segment_onsets, max_onsets_per_template, replace=False
                )

            # Extract waveforms
            waveforms, _ = _extract_aligned_waveforms_numba(
                data,
                segment_onsets,
                window_pre,
                window_post,
                align_to_max,
                align_to_min,
                max_jitter,
            )

            if len(waveforms) > 0:
                all_waveforms.append(waveforms)

        if all_waveforms:
            # Concatenate waveforms from all segments
            combined_waveforms = np.vstack(all_waveforms)

            # Compute template for each channel
            template = np.zeros(
                (window_pre + window_post + 1, n_channels), dtype=np.float32
            )

            for c in range(n_channels):
                template[:, c] = _extract_template_numba(
                    combined_waveforms[:, :, c], window_pre
                )

            templates[onset_id] = template
        else:
            # Create empty template if no waveforms found
            templates[onset_id] = np.zeros(
                (window_pre + window_post + 1, n_channels), dtype=np.float32
            )

    return templates


def compute_emg_amplitudes(
    data: np.ndarray,
    onsets: Dict[str, List[np.ndarray]],
    offsets: Optional[Dict[str, List[np.ndarray]]] = None,
    rms_window: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute EMG burst amplitudes for each onset.

    Parameters
    ----------
    data : np.ndarray
        Input data array (samples × channels)
    onsets : Dict[str, List[np.ndarray]]
        Dictionary mapping onset IDs to lists of onset times in samples
    offsets : Dict[str, List[np.ndarray]], optional
        Dictionary mapping onset IDs to lists of offset times in samples.
        If None, a fixed window after onset is used.
    rms_window : int, default: 10
        Window size in samples for RMS calculation

    Returns
    -------
    amplitudes : Dict[str, np.ndarray]
        Dictionary mapping onset IDs to arrays of burst amplitudes
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    amplitudes = {}

    for onset_id, onset_times_list in onsets.items():
        all_amplitudes = []

        # Get corresponding offsets if available
        if offsets is not None and onset_id in offsets:
            offset_times_list = offsets[onset_id]
        else:
            offset_times_list = None

        for segment_idx, segment_onsets in enumerate(onset_times_list):
            if len(segment_onsets) == 0:
                continue

            # Get offsets for this segment if available
            if offset_times_list is not None and segment_idx < len(offset_times_list):
                segment_offsets = offset_times_list[segment_idx]

                # Match onsets with offsets
                if len(segment_offsets) == len(segment_onsets):
                    segment_durations = segment_offsets - segment_onsets
                else:
                    # Use default duration if mismatch
                    segment_durations = np.full_like(
                        segment_onsets, 100
                    )  # 100 samples default
            else:
                # Use default duration
                segment_durations = np.full_like(segment_onsets, 100)

            # Calculate amplitude for each onset
            segment_amplitudes = np.zeros(
                (len(segment_onsets), n_channels), dtype=np.float32
            )

            for i, (onset, duration) in enumerate(
                zip(segment_onsets, segment_durations)
            ):
                if onset >= 0 and onset + duration < n_samples:
                    # Extract burst
                    burst = data[onset : onset + int(duration)]

                    # Calculate RMS amplitude
                    for c in range(n_channels):
                        # Use sliding window RMS and take maximum
                        max_rms = 0

                        for j in range(0, len(burst) - rms_window + 1):
                            window = burst[j : j + rms_window, c]
                            rms = np.sqrt(np.mean(window**2))
                            max_rms = max(max_rms, rms)

                        segment_amplitudes[i, c] = max_rms

            all_amplitudes.append(segment_amplitudes)

        if all_amplitudes:
            amplitudes[onset_id] = np.vstack(all_amplitudes)
        else:
            amplitudes[onset_id] = np.array([])

    return amplitudes


@jit(nopython=True, cache=True)
def _compute_emg_snr_numba(
    signal: np.ndarray,
    onsets: np.ndarray,
    durations: np.ndarray,
    rest_window: int = 100,
) -> Tuple[float, float, float]:
    """
    Compute EMG signal-to-noise ratio with numba acceleration.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array (1D)
    onsets : np.ndarray
        Onset indices
    durations : np.ndarray
        Burst durations in samples
    rest_window : int, default: 100
        Window size to capture rest periods before onsets

    Returns
    -------
    snr : float
        Signal-to-noise ratio
    signal_rms : float
        RMS of active periods
    noise_rms : float
        RMS of rest periods
    """
    n_samples = len(signal)

    # Calculate noise from rest periods
    noise_segments = []
    for onset in onsets:
        # Use period before onset as rest
        start = max(0, onset - rest_window)
        if start < onset:
            noise_segments.append(signal[start:onset])

    # Calculate signal from active periods
    signal_segments = []
    for onset, duration in zip(onsets, durations):
        end = min(n_samples, onset + int(duration))
        if onset < end:
            signal_segments.append(signal[onset:end])

    # Calculate RMS values
    noise_rms = 0.0
    total_noise_samples = 0
    for segment in noise_segments:
        for sample in segment:
            noise_rms += sample**2
            total_noise_samples += 1

    signal_rms = 0.0
    total_signal_samples = 0
    for segment in signal_segments:
        for sample in segment:
            signal_rms += sample**2
            total_signal_samples += 1

    # Avoid division by zero
    if total_noise_samples > 0:
        noise_rms = np.sqrt(noise_rms / total_noise_samples)
    else:
        noise_rms = 1e-10

    if total_signal_samples > 0:
        signal_rms = np.sqrt(signal_rms / total_signal_samples)
    else:
        signal_rms = 0.0

    # Calculate SNR
    snr = 0.0
    if noise_rms > 0:
        snr = 20 * np.log10(signal_rms / noise_rms)

    return snr, signal_rms, noise_rms


def calculate_emg_snr(
    data: np.ndarray,
    onsets: Dict[str, List[np.ndarray]],
    offsets: Optional[Dict[str, List[np.ndarray]]] = None,
    rest_window: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate EMG signal-to-noise ratio for each channel and onset group.

    Parameters
    ----------
    data : np.ndarray
        Input data array (samples × channels)
    onsets : Dict[str, List[np.ndarray]]
        Dictionary mapping onset IDs to lists of onset times in samples
    offsets : Dict[str, List[np.ndarray]], optional
        Dictionary mapping onset IDs to lists of offset times in samples
        If None, a fixed duration is used for active periods
    rest_window : int, default: 100
        Window size in samples to capture rest periods before onsets

    Returns
    -------
    snr_values : Dict[str, Dict[str, float]]
        Dictionary mapping onset IDs to dictionaries of:
        - snr: signal-to-noise ratio in dB
        - signal_rms: RMS of signal during active periods
        - noise_rms: RMS of signal during rest periods
        for each channel
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape
    snr_values = {}

    for onset_id, onset_times_list in onsets.items():
        # Collect all onsets and durations
        all_onsets = []
        all_durations = []

        # Get corresponding offsets if available
        if offsets is not None and onset_id in offsets:
            offset_times_list = offsets[onset_id]
        else:
            offset_times_list = None

        for segment_idx, segment_onsets in enumerate(onset_times_list):
            if len(segment_onsets) == 0:
                continue

            # Get offsets for this segment if available
            if offset_times_list is not None and segment_idx < len(offset_times_list):
                segment_offsets = offset_times_list[segment_idx]

                # Match onsets with offsets
                if len(segment_offsets) == len(segment_onsets):
                    segment_durations = segment_offsets - segment_onsets
                else:
                    # Use default duration if mismatch
                    segment_durations = np.full_like(
                        segment_onsets, 100
                    )  # 100 samples default
            else:
                # Use default duration
                segment_durations = np.full_like(segment_onsets, 100)

            all_onsets.extend(segment_onsets)
            all_durations.extend(segment_durations)

        # Skip if no onsets
        if not all_onsets:
            snr_values[onset_id] = {
                f"channel_{c}": {"snr": 0, "signal_rms": 0, "noise_rms": 0}
                for c in range(n_channels)
            }
            continue

        # Convert to numpy arrays
        all_onsets = np.array(all_onsets)
        all_durations = np.array(all_durations)

        # Calculate SNR for each channel
        channel_snrs = {}
        for c in range(n_channels):
            snr, signal_rms, noise_rms = _compute_emg_snr_numba(
                data[:, c], all_onsets, all_durations, rest_window
            )

            channel_snrs[f"channel_{c}"] = {
                "snr": snr,
                "signal_rms": signal_rms,
                "noise_rms": noise_rms,
            }

        snr_values[onset_id] = channel_snrs

    return snr_values


@jit(nopython=True, parallel=True, cache=True)
def _calculate_onset_consistency_numba(
    templates: np.ndarray, align_index: int
) -> np.ndarray:
    """
    Calculate consistency between templates with numba acceleration.

    Parameters
    ----------
    templates : np.ndarray
        Array of templates (n_templates, n_samples)
    align_index : int
        Index to align the templates on

    Returns
    -------
    consistency_matrix : np.ndarray
        Matrix of consistency scores between templates
    """
    n_templates, n_samples = templates.shape
    consistency_matrix = np.zeros((n_templates, n_templates), dtype=np.float32)

    # Calculate cross-correlation between all template pairs
    for i in prange(n_templates):
        template_i = templates[i]

        for j in range(i, n_templates):
            template_j = templates[j]

            # Calculate correlation
            mean_i = 0.0
            mean_j = 0.0

            for k in range(n_samples):
                mean_i += template_i[k]
                mean_j += template_j[k]

            mean_i /= n_samples
            mean_j /= n_samples

            numerator = 0.0
            denom_i = 0.0
            denom_j = 0.0

            for k in range(n_samples):
                diff_i = template_i[k] - mean_i
                diff_j = template_j[k] - mean_j

                numerator += diff_i * diff_j
                denom_i += diff_i * diff_i
                denom_j += diff_j * diff_j

            # Pearson correlation
            correlation = 0.0
            if denom_i > 0 and denom_j > 0:
                correlation = numerator / np.sqrt(denom_i * denom_j)

            # Store correlation
            consistency_matrix[i, j] = correlation
            consistency_matrix[j, i] = correlation

    return consistency_matrix


def calculate_onset_consistency(
    templates: Dict[str, np.ndarray], window_pre: int = 50
) -> Dict[str, Dict[str, float]]:
    """
    Calculate consistency between onset templates.

    Parameters
    ----------
    templates : Dict[str, np.ndarray]
        Dictionary mapping onset IDs to templates
    window_pre : int, default: 50
        Number of samples before onset (for alignment)

    Returns
    -------
    consistency_scores : Dict[str, Dict[str, float]]
        Dictionary mapping onset IDs to dictionaries of consistency scores
        with other onsets
    """
    onset_ids = list(templates.keys())
    n_onsets = len(onset_ids)

    consistency_scores = {onset_id: {} for onset_id in onset_ids}

    # Process each channel separately
    n_channels = templates[onset_ids[0]].shape[1] if n_onsets > 0 else 0

    for c in range(n_channels):
        # Extract templates for this channel
        channel_templates = np.array(
            [templates[onset_id][:, c] for onset_id in onset_ids]
        )

        if len(channel_templates) > 1:
            # Calculate consistency matrix
            consistency_matrix = _calculate_onset_consistency_numba(
                channel_templates, window_pre
            )

            # Store results
            for i, onset_i in enumerate(onset_ids):
                for j, onset_j in enumerate(onset_ids):
                    if i != j:  # Skip self-comparison
                        key = f"{onset_j}_ch{c}"
                        consistency_scores[onset_i][key] = consistency_matrix[i, j]

    return consistency_scores
