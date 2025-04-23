"""
Sample Entropy based EMG onset detection.

This module implements an EMG onset detection technique based on Sample Entropy (SampEn),
which measures signal complexity or irregularity. Muscle activation typically causes
a decrease in entropy, making this approach effective for detecting activation onsets
even in noisy conditions.
"""

import concurrent.futures
from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit, prange

from dspant.engine.base import BaseProcessor


@jit(nopython=True, cache=True)
def _sample_entropy_numba(
    data: np.ndarray, m: int = 2, r: float = 0.2, normalize: bool = True
) -> float:
    """
    Calculate sample entropy of a time series using Numba acceleration.

    Sample entropy measures the irregularity and complexity of a time series.
    Lower values indicate more regularity in the signal.

    Args:
        data: Input time series (1D array)
        m: Pattern length (embedding dimension)
        r: Similarity threshold (typically 0.1 to 0.25 times signal std)
        normalize: Whether to normalize the signal before calculation

    Returns:
        Sample entropy value
    """
    # Ensure input is 1D
    x = data.flatten()
    n = len(x)

    if n < m + 2:
        return np.nan  # Not enough data points

    # Normalize if requested
    if normalize:
        # Subtract mean and divide by std
        x = (x - np.mean(x)) / (np.std(x) + 1e-10)

    # If not normalized, calculate r as a proportion of std
    if not normalize:
        r = r * np.std(x)

    # Initialize template and forward template counts
    A = 0.0  # Count for m+1
    B = 0.0  # Count for m

    # For each template (excluding last m+1 points)
    for i in range(n - m):
        # Calculate number of matches for m and m+1
        matches_m = 0
        matches_m_plus_1 = 0

        # Compare with other templates (no self-matching)
        for j in range(n - m):
            if i == j:
                continue  # Skip self-comparison

            # Check for match of length m
            match_m = True
            for k in range(m):
                if abs(x[i + k] - x[j + k]) > r:
                    match_m = False
                    break

            # If we have an m-match, increment count
            if match_m:
                matches_m += 1

                # Check for match of length m+1 if possible
                if i < n - m - 1 and j < n - m - 1:
                    if abs(x[i + m] - x[j + m]) <= r:
                        matches_m_plus_1 += 1

        # Add matches to running totals
        B += matches_m
        A += matches_m_plus_1

    # Calculate sample entropy
    if B == 0 or A == 0:
        return np.inf  # Not enough matches found
    else:
        return -np.log(A / B)


@jit(nopython=True, cache=True)
def _windowed_entropy_numba(
    data: np.ndarray,
    window_size: int,
    step_size: int = 1,
    m: int = 2,
    r: float = 0.2,
    normalize: bool = True,
) -> np.ndarray:
    """
    Calculate sample entropy in sliding windows across the signal.

    Args:
        data: Input signal (1D array)
        window_size: Size of the sliding window
        step_size: Number of samples to advance the window in each step
        m: Pattern length for sample entropy
        r: Similarity threshold for sample entropy
        normalize: Whether to normalize each window before calculating entropy

    Returns:
        Array of entropy values for each window position
    """
    # Ensure data is flattened
    x = data.flatten()
    n = len(x)

    # Calculate number of windows
    n_windows = max(0, (n - window_size) // step_size + 1)

    # Initialize results array
    entropy_values = np.ones(n_windows, dtype=np.float32) * np.nan

    # Process each window
    for i in range(n_windows):
        start_idx = i * step_size
        window = x[start_idx : start_idx + window_size]

        # Skip windows that are too short
        if len(window) >= m + 2:
            entropy_values[i] = _sample_entropy_numba(window, m, r, normalize)

    return entropy_values


@jit(nopython=True, parallel=True, cache=True)
def _detect_entropy_changes_numba(
    entropy_values: np.ndarray,
    signal: np.ndarray,
    window_positions: np.ndarray,
    threshold: float,
    min_duration_samples: int,
    min_change_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect significant changes in entropy values that indicate EMG onsets.

    Args:
        entropy_values: Array of entropy values from sliding windows
        signal: Original signal
        window_positions: Array of indices for the center of each window
        threshold: Threshold for significant entropy change (negative for activation)
        min_duration_samples: Minimum duration of activation
        min_change_samples: Minimum number of samples with entropy below threshold

    Returns:
        Tuple of arrays (onset_indices, offset_indices, amplitudes, durations)
    """
    n = len(entropy_values)

    # Pre-allocate result arrays with generous size
    max_events = n // min_duration_samples + 1
    onset_indices = np.zeros(max_events, dtype=np.int64)
    offset_indices = np.zeros(max_events, dtype=np.int64)
    amplitudes = np.zeros(max_events, dtype=np.float32)
    durations = np.zeros(max_events, dtype=np.int32)

    # Set threshold direction (decreasing entropy for activation)
    below_threshold = entropy_values < threshold

    # Find continuous segments below threshold
    in_segment = False
    segment_start = 0
    count_below = 0
    event_count = 0

    for i in range(n):
        if below_threshold[i]:
            count_below += 1
            if not in_segment:
                in_segment = True
                segment_start = i
        else:
            if in_segment:
                # Check if the segment is long enough
                if count_below >= min_change_samples:
                    segment_end = i - 1

                    # Get corresponding indices in the original signal
                    onset_idx = window_positions[segment_start]
                    offset_idx = window_positions[segment_end]

                    # Calculate duration in samples
                    duration = offset_idx - onset_idx + 1

                    # Only keep segments meeting minimum duration
                    if duration >= min_duration_samples:
                        # Calculate amplitude (maximum in segment)
                        segment_signal = signal[onset_idx : offset_idx + 1]
                        amplitude = np.max(segment_signal)

                        # Store the event
                        onset_indices[event_count] = onset_idx
                        offset_indices[event_count] = offset_idx
                        amplitudes[event_count] = amplitude
                        durations[event_count] = duration
                        event_count += 1

                # Reset segment tracking
                in_segment = False
                count_below = 0

    # Check for a segment that extends to the end of the data
    if in_segment and count_below >= min_change_samples:
        segment_end = n - 1
        onset_idx = window_positions[segment_start]
        offset_idx = window_positions[segment_end]
        duration = offset_idx - onset_idx + 1

        if duration >= min_duration_samples:
            segment_signal = signal[onset_idx : offset_idx + 1]
            amplitude = np.max(segment_signal)

            onset_indices[event_count] = onset_idx
            offset_indices[event_count] = offset_idx
            amplitudes[event_count] = amplitude
            durations[event_count] = duration
            event_count += 1

    # Return only the valid events
    return (
        onset_indices[:event_count],
        offset_indices[:event_count],
        amplitudes[:event_count],
        durations[:event_count],
    )


class SampleEntropyOnsetDetector(BaseProcessor):
    """
    EMG onset detection processor using Sample Entropy.

    This processor calculates Sample Entropy in a sliding window and detects
    significant decreases in entropy, which correspond to muscle activation.
    It's particularly effective for signals with varying noise levels or non-stationary
    characteristics where simple thresholding may fail.
    """

    def __init__(
        self,
        window_size: float = 0.1,  # seconds
        step_size: float = 0.02,  # seconds
        threshold_method: Literal["adaptive", "absolute", "percentile"] = "adaptive",
        threshold_value: float = 1.5,  # std devs below mean for adaptive
        entropy_m: int = 2,  # embedding dimension
        entropy_r: float = 0.2,  # similarity threshold
        min_duration: float = 0.05,  # seconds
        baseline_window: Optional[Tuple[float, float]] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the Sample Entropy onset detector.

        Args:
            window_size: Size of sliding window for entropy calculation (seconds)
            step_size: Step size for sliding window (seconds)
            threshold_method: Method for determining activation threshold
                "adaptive": Threshold as number of std deviations below mean
                "absolute": Fixed threshold value
                "percentile": Threshold at specified percentile of entropy values
            threshold_value: Threshold parameter value
            entropy_m: Pattern length (embedding dimension) for SampEn
            entropy_r: Similarity threshold (typically 0.1-0.25 times signal std)
            min_duration: Minimum duration for a valid activation (seconds)
            baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
            max_workers: Maximum number of worker threads to use
        """
        self.window_size = window_size
        self.step_size = step_size
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.entropy_m = entropy_m
        self.entropy_r = entropy_r
        self.min_duration = min_duration
        self.baseline_window = baseline_window
        self.max_workers = max_workers

        # Use overlap to ensure no events are missed at chunk boundaries
        self._overlap_samples = None  # Will be set based on window size

        # Storage for computed threshold
        self.entropy_threshold = None

        # Define dtype for output
        self._dtype = np.dtype(
            [
                ("onset_idx", np.int64),
                ("offset_idx", np.int64),
                ("channel", np.int32),
                ("amplitude", np.float32),
                ("duration", np.float32),
                (
                    "entropy_drop",
                    np.float32,
                ),  # Additional information about entropy change
            ]
        )

    def compute_entropy_threshold(self, entropy_values: np.ndarray) -> float:
        """
        Compute activation threshold based on entropy values.

        Args:
            entropy_values: Array of entropy values

        Returns:
            Threshold value for activation detection
        """
        if self.threshold_method == "absolute":
            return self.threshold_value
        elif self.threshold_method == "adaptive":
            # Calculate threshold as mean - threshold_value * std
            mean_entropy = np.nanmean(entropy_values)
            std_entropy = np.nanstd(entropy_values)
            return mean_entropy - self.threshold_value * std_entropy
        elif self.threshold_method == "percentile":
            # Calculate threshold as a percentile of the entropy values
            return np.nanpercentile(entropy_values, self.threshold_value)
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Detect EMG onset events using Sample Entropy.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with onset events (can be converted to DataFrame)
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for onset detection")

        # Convert parameters to samples
        window_samples = int(self.window_size * fs)
        step_samples = max(1, int(self.step_size * fs))
        min_duration_samples = int(self.min_duration * fs)
        min_change_samples = max(
            3, window_samples // 10
        )  # Minimum number of windows with change

        # Set overlap samples based on window size and step size
        self._overlap_samples = window_samples * 2

        # Pre-compute entropy threshold if requested
        if kwargs.get("precompute_threshold", False):
            # This could be expensive for large datasets
            sample_data = data.compute()
            if sample_data.ndim == 1:
                sample_data = sample_data.reshape(-1, 1)

            # Calculate entropy for the first channel
            entropy_values = _windowed_entropy_numba(
                sample_data[:, 0],
                window_samples,
                step_samples,
                self.entropy_m,
                self.entropy_r,
            )

            self.entropy_threshold = self.compute_entropy_threshold(entropy_values)

        # Prepare baseline indices if specified
        baseline_start = 0
        baseline_end = None
        if self.baseline_window is not None:
            baseline_start = int(self.baseline_window[0] * fs)
            baseline_end = int(self.baseline_window[1] * fs)

        def detect_onsets_chunk(chunk: np.ndarray, block_info=None) -> np.ndarray:
            """Process a chunk of data to detect onsets using Sample Entropy"""
            # Get chunk offset from block_info
            chunk_offset = 0
            if block_info and len(block_info) > 0:
                chunk_offset = block_info[0]["array-location"][0][0]

            # Ensure the input is a contiguous array
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case by adding extra dimension if needed
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]

            # Create result array for all channels
            all_results = []

            # Helper function to process a single channel
            def process_channel(channel_ind):
                # Get the channel data
                channel_data = chunk[:, channel_ind]

                # Generate window center positions
                window_positions = np.arange(
                    window_samples // 2,
                    len(channel_data) - window_samples // 2,
                    step_samples,
                )

                # Skip if there's not enough data
                if len(window_positions) < 5:  # Need at least a few windows
                    return np.array([], dtype=self._dtype)

                # Calculate entropy in sliding windows
                entropy_values = _windowed_entropy_numba(
                    channel_data,
                    window_samples,
                    step_samples,
                    self.entropy_m,
                    self.entropy_r,
                )

                # Compute entropy threshold if not already computed
                threshold = self.entropy_threshold
                if threshold is None:
                    if baseline_end is not None and baseline_end > baseline_start:
                        # Use baseline window for threshold calculation
                        baseline_data = chunk[baseline_start:baseline_end, channel_ind]
                        baseline_entropy = _windowed_entropy_numba(
                            baseline_data,
                            window_samples,
                            step_samples,
                            self.entropy_m,
                            self.entropy_r,
                        )
                        threshold = self.compute_entropy_threshold(baseline_entropy)
                    else:
                        # Use current chunk for threshold calculation
                        threshold = self.compute_entropy_threshold(entropy_values)

                # Detect entropy changes indicating onsets
                onset_indices, offset_indices, amplitudes, durations = (
                    _detect_entropy_changes_numba(
                        entropy_values,
                        channel_data,
                        window_positions,
                        threshold,
                        min_duration_samples,
                        min_change_samples,
                    )
                )

                # Skip if no events detected
                if len(onset_indices) == 0:
                    return np.array([], dtype=self._dtype)

                # Create structured array for results
                result = np.zeros(len(onset_indices), dtype=self._dtype)
                result["onset_idx"] = onset_indices + chunk_offset
                result["offset_idx"] = offset_indices + chunk_offset
                result["channel"] = channel_ind
                result["amplitude"] = amplitudes
                result["duration"] = durations / fs  # Convert to seconds

                # Calculate entropy drop for each event
                for i, (onset_idx, offset_idx) in enumerate(
                    zip(onset_indices, offset_indices)
                ):
                    # Find corresponding indices in entropy array
                    onset_window_idx = np.abs(window_positions - onset_idx).argmin()
                    offset_window_idx = np.abs(window_positions - offset_idx).argmin()

                    # Ensure valid indices
                    if onset_window_idx < len(
                        entropy_values
                    ) and offset_window_idx < len(entropy_values):
                        # Calculate entropy change during activation
                        pre_entropy = np.nanmean(
                            entropy_values[
                                max(0, onset_window_idx - 3) : onset_window_idx + 1
                            ]
                        )
                        event_entropy = np.nanmean(
                            entropy_values[onset_window_idx : offset_window_idx + 1]
                        )
                        entropy_drop = pre_entropy - event_entropy
                        result["entropy_drop"][i] = entropy_drop

                return result

            # Use ThreadPoolExecutor if we have multiple channels and max_workers is specified
            if chunk.shape[1] > 1 and self.max_workers is not None:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    # Process each channel in parallel
                    futures = [
                        executor.submit(process_channel, i)
                        for i in range(chunk.shape[1])
                    ]

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        channel_result = future.result()
                        if len(channel_result) > 0:
                            all_results.append(channel_result)
            else:
                # Process channels sequentially
                for channel_ind in range(chunk.shape[1]):
                    channel_result = process_channel(channel_ind)
                    if len(channel_result) > 0:
                        all_results.append(channel_result)

            # Combine all channel results
            if not all_results:
                return np.array([], dtype=self._dtype)

            return np.concatenate(all_results)

        # Ensure input is 2D
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Use map_overlap with explicit boundary handling
        result = data.map_overlap(
            detect_onsets_chunk,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=self._dtype,
            meta=np.array([], dtype=self._dtype),
            drop_axis=None,
        )

        return result

    def to_dataframe(self, events_array: Union[np.ndarray, da.Array]) -> pl.DataFrame:
        """
        Convert events array to a Polars DataFrame.

        Args:
            events_array: Array of detected events

        Returns:
            Polars DataFrame with onset events
        """
        # Convert dask array to numpy if needed
        if isinstance(events_array, da.Array):
            events_array = events_array.compute()

        # Convert numpy structured array to Polars DataFrame
        return pl.from_numpy(events_array)

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        if self._overlap_samples is None:
            return 100  # Default value if not yet set
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "window_size": self.window_size,
                "step_size": self.step_size,
                "threshold_method": self.threshold_method,
                "threshold_value": self.threshold_value,
                "entropy_m": self.entropy_m,
                "entropy_r": self.entropy_r,
                "min_duration": self.min_duration,
                "accelerated": True,
                "parallel": self.max_workers is not None,
            }
        )
        return base_summary


# Factory functions for common configurations
def create_entropy_onset_detector(
    window_size: float = 0.1,
    step_size: float = 0.02,
    threshold_value: float = 1.5,
    threshold_method: str = "adaptive",
    min_duration: float = 0.05,
    max_workers: Optional[int] = None,
) -> SampleEntropyOnsetDetector:
    """
    Create a Sample Entropy EMG onset detector with standard settings.

    Args:
        window_size: Size of sliding window (seconds)
        step_size: Step size for sliding windows (seconds)
        threshold_value: Threshold value (meaning depends on threshold_method)
        threshold_method: Method for threshold calculation
        min_duration: Minimum activation duration (seconds)
        max_workers: Maximum number of worker threads

    Returns:
        Configured SampleEntropyOnsetDetector
    """
    return SampleEntropyOnsetDetector(
        window_size=window_size,
        step_size=step_size,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        min_duration=min_duration,
        max_workers=max_workers,
    )


def create_adaptive_entropy_detector(
    sensitivity: float = 1.5,
    window_size: float = 0.1,
    min_duration: float = 0.03,
    max_workers: Optional[int] = None,
) -> SampleEntropyOnsetDetector:
    """
    Create a Sample Entropy detector with adaptive thresholding.

    This configuration adjusts to varying noise levels automatically by using
    threshold_value as the number of standard deviations below the mean entropy.

    Args:
        sensitivity: Number of standard deviations below mean for threshold
        window_size: Size of sliding window (seconds)
        min_duration: Minimum activation duration (seconds)
        max_workers: Maximum number of worker threads

    Returns:
        Configured SampleEntropyOnsetDetector with adaptive thresholds
    """
    return SampleEntropyOnsetDetector(
        window_size=window_size,
        step_size=window_size / 5,  # Overlap windows for smoother results
        threshold_method="adaptive",
        threshold_value=sensitivity,
        entropy_m=2,
        entropy_r=0.2,
        min_duration=min_duration,
        max_workers=max_workers,
    )


def create_percentile_entropy_detector(
    percentile: float = 15.0,
    window_size: float = 0.08,
    min_duration: float = 0.04,
    max_workers: Optional[int] = None,
) -> SampleEntropyOnsetDetector:
    """
    Create a Sample Entropy detector with percentile-based thresholding.

    This configuration uses a percentile of the entropy distribution as threshold,
    making it robust against outliers and non-normal entropy distributions.

    Args:
        percentile: Percentile value for threshold (lower percentile = higher sensitivity)
        window_size: Size of sliding window (seconds)
        min_duration: Minimum activation duration (seconds)
        max_workers: Maximum number of worker threads

    Returns:
        Configured SampleEntropyOnsetDetector with percentile thresholds
    """
    return SampleEntropyOnsetDetector(
        window_size=window_size,
        step_size=window_size / 5,
        threshold_method="percentile",
        threshold_value=percentile,
        entropy_m=2,
        entropy_r=0.15,
        min_duration=min_duration,
        max_workers=max_workers,
    )


def create_high_resolution_entropy_detector(
    window_size: float = 0.05,
    step_size: float = 0.01,
    sensitivity: float = 1.75,
    max_workers: Optional[int] = None,
) -> SampleEntropyOnsetDetector:
    """
    Create a Sample Entropy detector optimized for high temporal resolution.

    This configuration uses smaller windows and step size for finer temporal detail,
    at the cost of increased computational demands.

    Args:
        window_size: Size of sliding window (seconds)
        step_size: Step size for sliding windows (seconds)
        sensitivity: Number of standard deviations below mean for threshold
        max_workers: Maximum number of worker threads

    Returns:
        Configured SampleEntropyOnsetDetector with high temporal resolution
    """
    return SampleEntropyOnsetDetector(
        window_size=window_size,
        step_size=step_size,
        threshold_method="adaptive",
        threshold_value=sensitivity,
        entropy_m=2,
        entropy_r=0.2,
        min_duration=0.03,
        max_workers=max_workers,
    )
