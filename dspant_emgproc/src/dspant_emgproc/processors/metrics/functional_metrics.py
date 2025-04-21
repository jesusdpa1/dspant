"""
Functional metrics for EMG signal analysis.

This module provides functions to compute various functional metrics from EMG signals,
including activation timing, force estimation, fatigue indicators, and muscle coordination
measures. These metrics help quantify muscle function during motor tasks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit, prange

from ...engine.base import BaseProcessor


@jit(nopython=True, cache=True)
def _compute_activation_timing_numba(
    onsets: np.ndarray,
    offsets: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute activation timing metrics with Numba acceleration.

    Parameters
    ----------
    onsets : np.ndarray
        Array of onset indices
    offsets : np.ndarray
        Array of offset indices
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    durations : np.ndarray
        Array of activation durations in seconds
    onset_times : np.ndarray
        Array of onset times in seconds
    offset_times : np.ndarray
        Array of offset times in seconds
    """
    n_events = len(onsets)
    durations = np.zeros(n_events, dtype=np.float32)
    onset_times = np.zeros(n_events, dtype=np.float32)
    offset_times = np.zeros(n_events, dtype=np.float32)

    for i in range(n_events):
        # Convert indices to time in seconds
        onset_times[i] = onsets[i] / fs
        offset_times[i] = offsets[i] / fs

        # Calculate duration
        durations[i] = (offsets[i] - onsets[i]) / fs

    return durations, onset_times, offset_times


@jit(nopython=True, cache=True)
def _compute_burst_metrics_numba(
    signal: np.ndarray,
    onsets: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute EMG burst metrics with Numba acceleration.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector
    onsets : np.ndarray
        Array of onset indices
    offsets : np.ndarray
        Array of offset indices

    Returns
    -------
    mean_amplitudes : np.ndarray
        Mean amplitude during each burst
    peak_amplitudes : np.ndarray
        Peak amplitude during each burst
    areas : np.ndarray
        Area under the curve for each burst
    rms_values : np.ndarray
        RMS value for each burst
    """
    n_events = len(onsets)
    mean_amplitudes = np.zeros(n_events, dtype=np.float32)
    peak_amplitudes = np.zeros(n_events, dtype=np.float32)
    areas = np.zeros(n_events, dtype=np.float32)
    rms_values = np.zeros(n_events, dtype=np.float32)

    for i in range(n_events):
        onset = onsets[i]
        offset = offsets[i]

        # Skip invalid indices
        if onset >= offset or onset < 0 or offset >= len(signal):
            continue

        # Extract burst
        burst = signal[onset:offset]

        # Skip empty bursts
        if len(burst) == 0:
            continue

        # Compute metrics
        mean_amplitudes[i] = np.mean(np.abs(burst))
        peak_amplitudes[i] = np.max(np.abs(burst))
        areas[i] = np.sum(np.abs(burst))
        rms_values[i] = np.sqrt(np.mean(burst * burst))

    return mean_amplitudes, peak_amplitudes, areas, rms_values


@jit(nopython=True, cache=True)
def _compute_rate_coding_numba(
    onsets: np.ndarray,
    fs: float,
    window_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rate coding metrics (firing rate) with Numba acceleration.

    Parameters
    ----------
    onsets : np.ndarray
        Array of onset indices
    fs : float
        Sampling frequency in Hz
    window_size : float
        Window size in seconds for rate computation

    Returns
    -------
    rates : np.ndarray
        Array of instantaneous firing rates in Hz
    times : np.ndarray
        Array of time points for rates in seconds
    """
    if len(onsets) < 2:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)

    # Convert onsets to times in seconds
    onset_times = onsets / fs

    # Calculate time windows
    min_time = onset_times[0]
    max_time = onset_times[-1]
    n_windows = max(1, int((max_time - min_time) / window_size))

    rates = np.zeros(n_windows, dtype=np.float32)
    times = np.zeros(n_windows, dtype=np.float32)

    for i in range(n_windows):
        start_time = min_time + i * window_size
        end_time = start_time + window_size

        # Count onsets in this window
        count = 0
        for t in onset_times:
            if start_time <= t < end_time:
                count += 1

        # Calculate rate
        rates[i] = count / window_size
        times[i] = start_time + window_size / 2  # Center of window

    return rates, times


@jit(nopython=True, cache=True)
def _compute_co_activation_numba(
    onsets1: np.ndarray,
    offsets1: np.ndarray,
    onsets2: np.ndarray,
    offsets2: np.ndarray,
    total_samples: int,
) -> float:
    """
    Compute co-activation index between two muscles with Numba acceleration.

    Parameters
    ----------
    onsets1 : np.ndarray
        Onset indices for first muscle
    offsets1 : np.ndarray
        Offset indices for first muscle
    onsets2 : np.ndarray
        Onset indices for second muscle
    offsets2 : np.ndarray
        Offset indices for second muscle
    total_samples : int
        Total number of samples in the recording

    Returns
    -------
    co_activation_index : float
        Co-activation index (0-1) representing the proportion of
        time both muscles are simultaneously active
    """
    # Create activation masks
    mask1 = np.zeros(total_samples, dtype=np.int8)
    mask2 = np.zeros(total_samples, dtype=np.int8)

    # Fill masks for muscle 1
    for i in range(len(onsets1)):
        onset = onsets1[i]
        offset = offsets1[i]

        if onset < offset and onset >= 0 and offset < total_samples:
            mask1[onset:offset] = 1

    # Fill masks for muscle 2
    for i in range(len(onsets2)):
        onset = onsets2[i]
        offset = offsets2[i]

        if onset < offset and onset >= 0 and offset < total_samples:
            mask2[onset:offset] = 1

    # Calculate co-activation
    co_activation = np.sum(mask1 & mask2)
    m1_activation = np.sum(mask1)
    m2_activation = np.sum(mask2)

    # Calculate index
    total_activation = m1_activation + m2_activation - co_activation

    if total_activation > 0:
        co_activation_index = co_activation / total_activation
    else:
        co_activation_index = 0.0

    return co_activation_index


class EMGFunctionalMetricsProcessor(BaseProcessor):
    """
    Processor for computing functional metrics from EMG signals.

    This processor calculates various functional metrics from EMG signals,
    including activation timing, burst characteristics, and rate coding measures.
    """

    def __init__(
        self,
        metrics: List[str] = ["timing", "burst", "rate"],
        rate_window_size: float = 1.0,  # window size in seconds for rate calculation
    ):
        """
        Initialize the EMG functional metrics processor.

        Parameters
        ----------
        metrics : List[str]
            List of metrics to compute
        rate_window_size : float
            Window size in seconds for rate calculation
        """
        self.metrics = metrics
        self.rate_window_size = rate_window_size
        self._overlap_samples = 0

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> Dict:
        """
        Compute functional metrics for EMG signals.

        Parameters
        ----------
        data : da.Array
            EMG signal data (samples × channels)
        fs : float
            Sampling frequency (Hz)
        **kwargs : dict
            Additional keyword arguments:
            - onsets: Dict[str, List[np.ndarray]]
                Dictionary of onset indices
            - offsets: Dict[str, List[np.ndarray]]
                Dictionary of offset indices

        Returns
        -------
        metrics : dict
            Dictionary of computed metrics
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Convert dask array to numpy for processing
        data_np = data.compute()

        # Ensure 2D array
        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)

        n_samples, n_channels = data_np.shape

        # Get onsets and offsets
        onsets = kwargs.get("onsets", None)
        offsets = kwargs.get("offsets", None)

        if onsets is None:
            raise ValueError("Onset indices must be provided")

        if offsets is None and "timing" in self.metrics:
            raise ValueError("Offset indices must be provided for timing metrics")

        # Initialize results
        results = {}

        # Get all unique muscle/channel identifiers
        all_ids = list(onsets.keys())

        for muscle_id in all_ids:
            muscle_results = {}

            # Flatten onsets and offsets for this muscle
            all_onsets = []
            all_offsets = []

            for segment_idx, segment_onsets in enumerate(onsets[muscle_id]):
                if len(segment_onsets) == 0:
                    continue

                # Get corresponding offsets
                if (
                    offsets is not None
                    and muscle_id in offsets
                    and segment_idx < len(offsets[muscle_id])
                ):
                    segment_offsets = offsets[muscle_id][segment_idx]

                    # Ensure matching lengths
                    if len(segment_offsets) == len(segment_onsets):
                        all_onsets.extend(segment_onsets)
                        all_offsets.extend(segment_offsets)
                    else:
                        # Use fixed duration if lengths don't match
                        fixed_duration = int(0.5 * fs)  # 500 ms default
                        all_onsets.extend(segment_onsets)
                        all_offsets.extend(segment_onsets + fixed_duration)
                else:
                    # Use fixed duration if offsets not provided
                    fixed_duration = int(0.5 * fs)  # 500 ms default
                    all_onsets.extend(segment_onsets)
                    all_offsets.extend(segment_onsets + fixed_duration)

            # Skip if no events
            if not all_onsets:
                results[muscle_id] = {"error": "No valid onsets found"}
                continue

            # Convert to numpy arrays
            all_onsets = np.array(all_onsets)
            all_offsets = np.array(all_offsets)

            # Sort by onset time
            if len(all_onsets) > 0:
                sort_idx = np.argsort(all_onsets)
                all_onsets = all_onsets[sort_idx]
                all_offsets = all_offsets[sort_idx]

            # Compute timing metrics
            if "timing" in self.metrics:
                durations, onset_times, offset_times = _compute_activation_timing_numba(
                    all_onsets, all_offsets, fs
                )

                muscle_results["durations"] = durations
                muscle_results["onset_times"] = onset_times
                muscle_results["offset_times"] = offset_times

                # Summary statistics
                muscle_results["mean_duration"] = (
                    np.mean(durations) if len(durations) > 0 else 0
                )
                muscle_results["std_duration"] = (
                    np.std(durations) if len(durations) > 0 else 0
                )
                muscle_results["total_active_time"] = np.sum(durations)
                muscle_results["activation_count"] = len(durations)

                if len(onset_times) > 1:
                    # Calculate inter-onset intervals
                    ioi = np.diff(onset_times)
                    muscle_results["mean_ioi"] = np.mean(ioi)
                    muscle_results["std_ioi"] = np.std(ioi)
                    muscle_results["min_ioi"] = np.min(ioi)
                    muscle_results["max_ioi"] = np.max(ioi)
                else:
                    muscle_results["mean_ioi"] = 0
                    muscle_results["std_ioi"] = 0
                    muscle_results["min_ioi"] = 0
                    muscle_results["max_ioi"] = 0

            # Determine which channel to use for this muscle
            # Simple mapping based on muscle_id
            try:
                channel = int(muscle_id.split("_")[-1]) % n_channels
            except (ValueError, IndexError):
                channel = 0

            channel_data = data_np[:, channel]

            # Compute burst metrics
            if "burst" in self.metrics:
                mean_amps, peak_amps, areas, rms_values = _compute_burst_metrics_numba(
                    channel_data, all_onsets, all_offsets
                )

                muscle_results["mean_amplitudes"] = mean_amps
                muscle_results["peak_amplitudes"] = peak_amps
                muscle_results["burst_areas"] = areas
                muscle_results["burst_rms"] = rms_values

                # Summary statistics
                muscle_results["mean_amplitude"] = (
                    np.mean(mean_amps) if len(mean_amps) > 0 else 0
                )
                muscle_results["mean_peak"] = (
                    np.mean(peak_amps) if len(peak_amps) > 0 else 0
                )
                muscle_results["max_peak"] = (
                    np.max(peak_amps) if len(peak_amps) > 0 else 0
                )
                muscle_results["mean_area"] = np.mean(areas) if len(areas) > 0 else 0
                muscle_results["mean_rms"] = (
                    np.mean(rms_values) if len(rms_values) > 0 else 0
                )

            # Compute rate coding metrics
            if "rate" in self.metrics:
                rates, times = _compute_rate_coding_numba(
                    all_onsets, fs, self.rate_window_size
                )

                muscle_results["firing_rates"] = rates
                muscle_results["rate_times"] = times

                # Summary statistics
                muscle_results["mean_rate"] = np.mean(rates) if len(rates) > 0 else 0
                muscle_results["peak_rate"] = np.max(rates) if len(rates) > 0 else 0
                muscle_results["std_rate"] = np.std(rates) if len(rates) > 0 else 0

            results[muscle_id] = muscle_results

        # Compute co-activation metrics if multiple muscles provided
        if len(all_ids) > 1 and "coactivation" in self.metrics:
            coactivation_results = {}

            for i, muscle1 in enumerate(all_ids):
                for j, muscle2 in enumerate(all_ids):
                    if i >= j:  # Skip self-comparisons and duplicates
                        continue

                    # Get onsets and offsets for both muscles
                    onsets1 = (
                        np.concatenate([arr for arr in onsets[muscle1] if len(arr) > 0])
                        if onsets[muscle1]
                        else np.array([])
                    )
                    onsets2 = (
                        np.concatenate([arr for arr in onsets[muscle2] if len(arr) > 0])
                        if onsets[muscle2]
                        else np.array([])
                    )

                    if (
                        offsets is not None
                        and muscle1 in offsets
                        and muscle2 in offsets
                    ):
                        offsets1 = (
                            np.concatenate(
                                [arr for arr in offsets[muscle1] if len(arr) > 0]
                            )
                            if offsets[muscle1]
                            else np.array([])
                        )
                        offsets2 = (
                            np.concatenate(
                                [arr for arr in offsets[muscle2] if len(arr) > 0]
                            )
                            if offsets[muscle2]
                            else np.array([])
                        )
                    else:
                        # Use fixed duration if offsets not provided
                        fixed_duration = int(0.5 * fs)  # 500 ms default
                        offsets1 = (
                            onsets1 + fixed_duration
                            if len(onsets1) > 0
                            else np.array([])
                        )
                        offsets2 = (
                            onsets2 + fixed_duration
                            if len(onsets2) > 0
                            else np.array([])
                        )

                    # Skip if not enough data
                    if len(onsets1) == 0 or len(onsets2) == 0:
                        continue

                    # Compute co-activation index
                    coactivation_index = _compute_co_activation_numba(
                        onsets1, offsets1, onsets2, offsets2, n_samples
                    )

                    pair_key = f"{muscle1}_{muscle2}"
                    coactivation_results[pair_key] = coactivation_index

            if coactivation_results:
                results["coactivation"] = coactivation_results

        return results

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap processing."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "metrics": self.metrics,
                "rate_window_size": self.rate_window_size,
            }
        )
        return base_summary


# Factory functions for creating processors with common configurations


def create_basic_functional_processor() -> EMGFunctionalMetricsProcessor:
    """
    Create a basic EMG functional metrics processor.

    Returns
    -------
    processor : EMGFunctionalMetricsProcessor
        Configured processor for basic functional metrics
    """
    return EMGFunctionalMetricsProcessor(
        metrics=["timing", "burst"],
    )


def create_comprehensive_functional_processor() -> EMGFunctionalMetricsProcessor:
    """
    Create a comprehensive EMG functional metrics processor.

    Returns
    -------
    processor : EMGFunctionalMetricsProcessor
        Configured processor for comprehensive functional metrics
    """
    return EMGFunctionalMetricsProcessor(
        metrics=["timing", "burst", "rate", "coactivation"],
        rate_window_size=0.5,  # 500 ms window for higher time resolution
    )


def compute_activation_timing(
    onsets: Dict[str, List[np.ndarray]],
    offsets: Dict[str, List[np.ndarray]],
    fs: float,
) -> Dict[str, Dict[str, float]]:
    """
    Compute activation timing metrics from onset/offset data.

    Parameters
    ----------
    onsets : Dict[str, List[np.ndarray]]
        Dictionary mapping muscle IDs to lists of onset times in samples
    offsets : Dict[str, List[np.ndarray]]
        Dictionary mapping muscle IDs to lists of offset times in samples
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    timing_metrics : Dict[str, Dict[str, float]]
        Dictionary mapping muscle IDs to timing metrics
    """
    # Create and use the processor for calculation
    processor = EMGFunctionalMetricsProcessor(metrics=["timing"])

    # Process with minimal data (1 sample, 1 channel)
    dummy_data = da.zeros((1, 1))
    results = processor.process(dummy_data, fs=fs, onsets=onsets, offsets=offsets)

    # Extract just the timing metrics
    timing_metrics = {}
    for muscle_id, metrics in results.items():
        if "activation_count" in metrics:  # Check if it's valid timing metrics
            timing_metrics[muscle_id] = {
                "mean_duration": metrics["mean_duration"],
                "total_active_time": metrics["total_active_time"],
                "activation_count": metrics["activation_count"],
                "mean_ioi": metrics["mean_ioi"],
            }

    return timing_metrics


def compute_burst_metrics(
    data: np.ndarray,
    onsets: Dict[str, List[np.ndarray]],
    offsets: Dict[str, List[np.ndarray]],
    fs: float,
) -> Dict[str, Dict[str, float]]:
    """
    Compute EMG burst metrics from signal and onset/offset data.

    Parameters
    ----------
    data : np.ndarray
        Input data array (samples × channels)
    onsets : Dict[str, List[np.ndarray]]
        Dictionary mapping muscle IDs to lists of onset times in samples
    offsets : Dict[str, List[np.ndarray]]
        Dictionary mapping muscle IDs to lists of offset times in samples
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    burst_metrics : Dict[str, Dict[str, float]]
        Dictionary mapping muscle IDs to burst metrics
    """
    # Create and use the processor for calculation
    processor = EMGFunctionalMetricsProcessor(metrics=["burst"])

    results = processor.process(data, fs=fs, onsets=onsets, offsets=offsets)

    # Extract just the burst metrics
    burst_metrics = {}
    for muscle_id, metrics in results.items():
        if "mean_amplitude" in metrics:  # Check if it's valid burst metrics
            burst_metrics[muscle_id] = {
                "mean_amplitude": metrics["mean_amplitude"],
                "mean_peak": metrics["mean_peak"],
                "max_peak": metrics["max_peak"],
                "mean_area": metrics["mean_area"],
                "mean_rms": metrics["mean_rms"],
            }

    return burst_metrics


def compute_co_activation(
    onsets: Dict[str, List[np.ndarray]],
    offsets: Optional[Dict[str, List[np.ndarray]]] = None,
    total_samples: int = 0,
    fs: float = 1000.0,
) -> Dict[str, float]:
    """
    Compute co-activation indices between multiple muscles.

    Parameters
    ----------
    onsets : Dict[str, List[np.ndarray]]
        Dictionary mapping muscle IDs to lists of onset times in samples
    offsets : Dict[str, List[np.ndarray]], optional
        Dictionary mapping muscle IDs to lists of offset times in samples
        If None, a fixed duration of 500 ms is used
    total_samples : int
        Total number of samples in the recording
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    co_activation : Dict[str, float]
        Dictionary mapping muscle pairs to co-activation indices
    """
    # If total_samples is not provided, estimate from onsets
    if total_samples <= 0:
        max_onset = 0
        for muscle_id, onset_list in onsets.items():
            for segment_onsets in onset_list:
                if len(segment_onsets) > 0:
                    max_onset = max(max_onset, np.max(segment_onsets))

        # Add a buffer to max onset
        total_samples = max_onset + int(fs * 1.0)  # Add 1 second buffer

    # Create and use the processor for calculation
    processor = EMGFunctionalMetricsProcessor(metrics=["coactivation"])

    # Process with minimal data but correct length
    dummy_data = da.zeros((total_samples, 1))
    results = processor.process(dummy_data, fs=fs, onsets=onsets, offsets=offsets)

    # Extract co-activation metrics
    if "coactivation" in results:
        return results["coactivation"]
    else:
        return {}
