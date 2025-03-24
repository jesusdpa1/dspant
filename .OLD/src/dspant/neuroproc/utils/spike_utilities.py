"""
Utility functions for working with spike data.

This module provides utilities for extracting, processing, and manipulating
spike data for use with dspant's neural processing modules. It includes functions
for converting between different spike data formats and extracting spike-related information.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


def extract_spike_times_from_binary(
    spike_data: np.ndarray,
    threshold: float,
    peak_sign: str = "neg",
    dead_time_samples: int = 5,
    assign_to_units: bool = False,
    n_units: int = 1,
) -> Union[np.ndarray, Dict[str, List[np.ndarray]]]:
    """
    Extract spike times from binary data using a threshold crossing method.

    Parameters
    ----------
    spike_data : np.ndarray
        Input data array (samples × channels)
    threshold : float
        Threshold for spike detection
    peak_sign : str, default: "neg"
        Sign of spikes to detect. Can be "neg", "pos", or "both".
    dead_time_samples : int, default: 5
        Minimum number of samples between consecutive spikes
    assign_to_units : bool, default: False
        Whether to assign spikes to units. If True, spikes are randomly
        assigned to n_units based on channel.
    n_units : int, default: 1
        Number of units to create if assign_to_units is True

    Returns
    -------
    spike_times : np.ndarray or Dict[str, List[np.ndarray]]
        If assign_to_units is False, a numpy array of spike times in samples.
        If assign_to_units is True, a dictionary mapping unit IDs to lists of
        spike times in samples.
    """
    # Handle multiple channels
    n_samples, n_channels = (
        spike_data.shape if spike_data.ndim > 1 else (len(spike_data), 1)
    )
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(-1, 1)

    # Initialize list to store spike times for each channel
    all_spike_times = []

    for ch in range(n_channels):
        channel_data = spike_data[:, ch]

        # Apply threshold
        if peak_sign == "neg":
            threshold_crossings = np.where(channel_data < -abs(threshold))[0]
        elif peak_sign == "pos":
            threshold_crossings = np.where(channel_data > abs(threshold))[0]
        else:  # "both"
            threshold_crossings = np.where(abs(channel_data) > abs(threshold))[0]

        # Apply dead time
        if len(threshold_crossings) > 0:
            spike_times = [threshold_crossings[0]]

            for i in range(1, len(threshold_crossings)):
                if threshold_crossings[i] - spike_times[-1] >= dead_time_samples:
                    spike_times.append(threshold_crossings[i])

            all_spike_times.extend([(t, ch) for t in spike_times])

    # Sort by time
    all_spike_times.sort(key=lambda x: x[0])

    if not assign_to_units:
        # Return just the spike times
        return np.array([t for t, _ in all_spike_times])
    else:
        # Assign spikes to units based on channels
        unit_spike_times = {f"unit_{i}": [np.array([])] for i in range(n_units)}

        # Distribute spikes among units based on channel
        for t, ch in all_spike_times:
            unit_id = f"unit_{ch % n_units}"
            unit_spike_times[unit_id][0] = np.append(unit_spike_times[unit_id][0], t)

        return unit_spike_times


def extract_spike_amplitudes(
    spike_data: np.ndarray,
    spike_times: Dict[str, List[np.ndarray]],
    window_samples: int = 11,
    peak_sign: str = "neg",
) -> Dict[str, np.ndarray]:
    """
    Extract spike amplitudes for detected spikes.

    Parameters
    ----------
    spike_data : np.ndarray
        Input data array (samples × channels)
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times in samples
    window_samples : int, default: 11
        Window size in samples around each spike to extract
    peak_sign : str, default: "neg"
        Sign of spike peaks to extract. Can be "neg", "pos", or "both".

    Returns
    -------
    spike_amplitudes : Dict[str, np.ndarray]
        Dictionary mapping unit IDs to arrays of spike amplitudes
    """
    # Handle multiple channels
    n_samples, n_channels = (
        spike_data.shape if spike_data.ndim > 1 else (len(spike_data), 1)
    )
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(-1, 1)

    # Initialize dictionary to store spike amplitudes
    spike_amplitudes = {}

    for unit_id, unit_spike_times_list in spike_times.items():
        all_amplitudes = []

        for segment_spike_times in unit_spike_times_list:
            if len(segment_spike_times) == 0:
                continue

            # Determine which channel to use for this unit (simple mapping based on unit ID)
            try:
                channel = int(unit_id.split("_")[-1]) % n_channels
            except (ValueError, IndexError):
                channel = 0

            segment_amplitudes = []

            for spike_time in segment_spike_times:
                # Ensure spike time is within data bounds
                if (
                    spike_time < window_samples // 2
                    or spike_time >= n_samples - window_samples // 2
                ):
                    continue

                # Extract spike waveform
                start = spike_time - window_samples // 2
                end = spike_time + window_samples // 2 + 1
                waveform = spike_data[start:end, channel]

                # Extract amplitude based on peak sign
                if peak_sign == "neg":
                    amplitude = np.min(waveform)
                elif peak_sign == "pos":
                    amplitude = np.max(waveform)
                else:  # "both"
                    amp_pos = np.max(waveform)
                    amp_neg = np.min(waveform)
                    amplitude = amp_pos if abs(amp_pos) > abs(amp_neg) else amp_neg

                segment_amplitudes.append(amplitude)

            all_amplitudes.extend(segment_amplitudes)

        spike_amplitudes[unit_id] = np.array(all_amplitudes)

    return spike_amplitudes


def extract_templates(
    spike_data: np.ndarray,
    spike_times: Dict[str, List[np.ndarray]],
    window_samples: int = 21,
    max_spikes_per_unit: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Extract average templates for each unit.

    Parameters
    ----------
    spike_data : np.ndarray
        Input data array (samples × channels)
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times in samples
    window_samples : int, default: 21
        Window size in samples around each spike to extract
    max_spikes_per_unit : int, default: 1000
        Maximum number of spikes to use for computing templates

    Returns
    -------
    templates : Dict[str, np.ndarray]
        Dictionary mapping unit IDs to average templates
    """
    # Handle multiple channels
    n_samples, n_channels = (
        spike_data.shape if spike_data.ndim > 1 else (len(spike_data), 1)
    )
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(-1, 1)

    # Initialize dictionary to store templates
    templates = {}

    for unit_id, unit_spike_times_list in spike_times.items():
        all_waveforms = []

        for segment_spike_times in unit_spike_times_list:
            if len(segment_spike_times) == 0:
                continue

            # Determine which channel to use for this unit (simple mapping based on unit ID)
            try:
                channel = int(unit_id.split("_")[-1]) % n_channels
            except (ValueError, IndexError):
                channel = 0

            # Randomly select spikes if there are too many
            if len(segment_spike_times) > max_spikes_per_unit:
                segment_spike_times = np.random.choice(
                    segment_spike_times, max_spikes_per_unit, replace=False
                )

            for spike_time in segment_spike_times:
                # Ensure spike time is within data bounds
                if (
                    spike_time < window_samples // 2
                    or spike_time >= n_samples - window_samples // 2
                ):
                    continue

                # Extract spike waveform
                start = spike_time - window_samples // 2
                end = spike_time + window_samples // 2 + 1
                waveform = spike_data[start:end, channel]

                all_waveforms.append(waveform)

        if all_waveforms:
            # Compute average template
            templates[unit_id] = np.mean(np.vstack(all_waveforms), axis=0)
        else:
            templates[unit_id] = np.zeros(window_samples)

    return templates


def estimate_noise_levels(
    spike_data: np.ndarray,
    spike_times: Dict[str, List[np.ndarray]],
    window_samples: int = 21,
) -> Dict[str, float]:
    """
    Estimate noise levels for each unit.

    Parameters
    ----------
    spike_data : np.ndarray
        Input data array (samples × channels)
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times in samples
    window_samples : int, default: 21
        Window size in samples to exclude around spikes for noise estimation

    Returns
    -------
    noise_levels : Dict[str, float]
        Dictionary mapping unit IDs to noise standard deviations
    """
    # Handle multiple channels
    n_samples, n_channels = (
        spike_data.shape if spike_data.ndim > 1 else (len(spike_data), 1)
    )
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(-1, 1)

    # Initialize dictionary to store noise levels
    noise_levels = {}

    # Create a mask for all spikes
    all_spikes_mask = np.ones(n_samples, dtype=bool)
    for unit_spike_times_list in spike_times.values():
        for segment_spike_times in unit_spike_times_list:
            for spike_time in segment_spike_times:
                if 0 <= spike_time < n_samples:
                    start = max(0, spike_time - window_samples // 2)
                    end = min(n_samples, spike_time + window_samples // 2 + 1)
                    all_spikes_mask[start:end] = False

    for unit_id in spike_times.keys():
        # Determine which channel to use for this unit (simple mapping based on unit ID)
        try:
            channel = int(unit_id.split("_")[-1]) % n_channels
        except (ValueError, IndexError):
            channel = 0

        # Extract noise segments (data outside of spikes)
        noise_data = spike_data[all_spikes_mask, channel]

        if len(noise_data) > 0:
            # Compute noise standard deviation
            noise_levels[unit_id] = np.std(noise_data)
        else:
            # If no noise segments found, use a default approach
            noise_levels[unit_id] = np.std(spike_data[:, channel]) / 2  # Heuristic

    return noise_levels


def compute_spike_positions(
    spike_times: Dict[str, List[np.ndarray]],
    position_data: Optional[np.ndarray] = None,
    channel_positions: Optional[np.ndarray] = None,
    units_to_channels: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute spike positions for quality metrics.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times in samples
    position_data : np.ndarray, optional
        Array of positions over time with shape (n_samples, 3)
        If provided, each spike is assigned the position at its time.
    channel_positions : np.ndarray, optional
        Array of channel positions with shape (n_channels, 3)
        If provided with units_to_channels, each spike is assigned its channel's position.
    units_to_channels : Dict[str, int], optional
        Dictionary mapping unit IDs to channel indices
        Required if using channel_positions.

    Returns
    -------
    spike_positions : Dict[str, np.ndarray]
        Dictionary mapping unit IDs to arrays of spike positions with shape (n_spikes, 3)
    """
    spike_positions = {}

    if position_data is not None:
        # Use position data over time
        n_samples = position_data.shape[0]

        for unit_id, unit_spike_times_list in spike_times.items():
            all_positions = []

            for segment_spike_times in unit_spike_times_list:
                for spike_time in segment_spike_times:
                    if 0 <= spike_time < n_samples:
                        all_positions.append(position_data[spike_time])

            if all_positions:
                spike_positions[unit_id] = np.vstack(all_positions)
            else:
                spike_positions[unit_id] = np.zeros((0, 3))

    elif channel_positions is not None and units_to_channels is not None:
        # Use static channel positions
        for unit_id, unit_spike_times_list in spike_times.items():
            if unit_id not in units_to_channels:
                spike_positions[unit_id] = np.zeros((0, 3))
                continue

            channel_idx = units_to_channels[unit_id]

            # Count total spikes for this unit
            total_spikes = sum(len(spikes) for spikes in unit_spike_times_list)

            # Assign the same channel position to all spikes
            if total_spikes > 0:
                spike_positions[unit_id] = np.tile(
                    channel_positions[channel_idx], (total_spikes, 1)
                )
            else:
                spike_positions[unit_id] = np.zeros((0, 3))

    else:
        # Create random positions as placeholder
        for unit_id, unit_spike_times_list in spike_times.items():
            total_spikes = sum(len(spikes) for spikes in unit_spike_times_list)

            if total_spikes > 0:
                # Use unit_id to seed random generator for reproducibility
                try:
                    seed = int(unit_id.split("_")[-1])
                except (ValueError, IndexError):
                    seed = hash(unit_id) % 2**32

                np.random.seed(seed)
                spike_positions[unit_id] = np.random.randn(total_spikes, 3)
            else:
                spike_positions[unit_id] = np.zeros((0, 3))

    return spike_positions


def prepare_spike_data_for_metrics(
    spike_data: np.ndarray,
    threshold: Optional[float] = None,
    spike_times: Optional[Dict[str, List[np.ndarray]]] = None,
    sampling_frequency: float = 30000.0,
    peak_sign: str = "neg",
    window_samples: int = 21,
    dead_time_samples: int = 5,
    n_units: int = 1,
    compute_templates: bool = True,
    compute_amplitudes: bool = True,
    compute_noise: bool = True,
    compute_positions: bool = False,
    channel_positions: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Prepare spike data for quality metrics computation.

    Parameters
    ----------
    spike_data : np.ndarray
        Input data array (samples × channels)
    threshold : float, optional
        Threshold for spike detection. Required if spike_times is None.
    spike_times : Dict[str, List[np.ndarray]], optional
        Dictionary mapping unit IDs to lists of spike times in samples.
        If None, spikes are detected using the threshold.
    sampling_frequency : float, default: 30000.0
        Sampling frequency in Hz
    peak_sign : str, default: "neg"
        Sign of spikes to detect. Can be "neg", "pos", or "both".
    window_samples : int, default: 21
        Window size in samples for extracting spike waveforms
    dead_time_samples : int, default: 5
        Minimum number of samples between consecutive spikes
    n_units : int, default: 1
        Number of units to create if spike_times is None
    compute_templates : bool, default: True
        Whether to compute templates
    compute_amplitudes : bool, default: True
        Whether to compute spike amplitudes
    compute_noise : bool, default: True
        Whether to compute noise levels
    compute_positions : bool, default: False
        Whether to compute spike positions
    channel_positions : np.ndarray, optional
        Array of channel positions with shape (n_channels, 3)
        Required if compute_positions is True

    Returns
    -------
    spike_data : Dict[str, Any]
        Dictionary containing prepared spike data:
        - spike_times: Dict[str, List[np.ndarray]]
        - sampling_frequency: float
        - total_duration: float
        - templates (optional): Dict[str, np.ndarray]
        - spike_amplitudes (optional): Dict[str, np.ndarray]
        - noise_levels (optional): Dict[str, float]
        - spike_positions (optional): Dict[str, np.ndarray]
    """
    # Handle multiple channels
    n_samples, n_channels = (
        spike_data.shape if spike_data.ndim > 1 else (len(spike_data), 1)
    )
    if spike_data.ndim == 1:
        spike_data = spike_data.reshape(-1, 1)

    # Detect spikes if not provided
    if spike_times is None:
        if threshold is None:
            raise ValueError("Either spike_times or threshold must be provided")

        spike_times = extract_spike_times_from_binary(
            spike_data,
            threshold,
            peak_sign,
            dead_time_samples,
            assign_to_units=True,
            n_units=n_units,
        )

    # Calculate total duration
    total_duration = n_samples / sampling_frequency

    # Initialize output dictionary
    output = {
        "spike_times": spike_times,
        "sampling_frequency": sampling_frequency,
        "total_duration": total_duration,
    }

    # Compute templates if requested
    if compute_templates:
        output["templates"] = extract_templates(spike_data, spike_times, window_samples)

    # Compute spike amplitudes if requested
    if compute_amplitudes:
        output["spike_amplitudes"] = extract_spike_amplitudes(
            spike_data, spike_times, window_samples, peak_sign
        )

    # Compute noise levels if requested
    if compute_noise:
        output["noise_levels"] = estimate_noise_levels(
            spike_data, spike_times, window_samples
        )

    # Compute spike positions if requested
    if compute_positions:
        # Create units_to_channels mapping
        units_to_channels = {}
        for unit_id in spike_times.keys():
            try:
                units_to_channels[unit_id] = int(unit_id.split("_")[-1]) % n_channels
            except (ValueError, IndexError):
                units_to_channels[unit_id] = 0

        output["spike_positions"] = compute_spike_positions(
            spike_times, None, channel_positions, units_to_channels
        )

    return output
