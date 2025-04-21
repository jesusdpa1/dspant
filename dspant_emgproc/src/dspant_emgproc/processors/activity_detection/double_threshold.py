"""
Double-threshold EMG onset detection algorithm.

This module implements an advanced EMG onset detection technique that uses two thresholds
to reduce false positives. The signal must first cross a higher threshold and then
remain above a lower threshold for a specified duration to be considered an activation.
"""

import concurrent.futures
from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit, prange

from dspant.engine.base import BaseProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_thresholds_numba(
    data: np.ndarray, method: str, high_value: float, low_value: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute high and low activation thresholds using Numba parallel acceleration.

    Args:
        data: Input data array
        method: Threshold method ("absolute", "std", "rms", "percent_max")
        high_value: High threshold value
        low_value: Low threshold value

    Returns:
        Tuple of arrays (high_thresholds, low_thresholds) for each channel
    """
    n_channels = data.shape[1]
    high_thresholds = np.zeros(n_channels, dtype=np.float32)
    low_thresholds = np.zeros(n_channels, dtype=np.float32)

    # Parallel processing across channels
    for chan in prange(n_channels):
        channel_data = data[:, chan]

        if method == "absolute":
            high_thresholds[chan] = high_value
            low_thresholds[chan] = low_value
        elif method == "std":
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            high_thresholds[chan] = mean + high_value * std
            low_thresholds[chan] = mean + low_value * std
        elif method == "rms":
            rms = np.sqrt(np.mean(channel_data * channel_data))
            high_thresholds[chan] = high_value * rms
            low_thresholds[chan] = low_value * rms
        elif method == "percent_max":
            max_val = np.max(channel_data)
            high_thresholds[chan] = high_value * max_val / 100.0
            low_thresholds[chan] = low_value * max_val / 100.0

    return high_thresholds, low_thresholds


class DoubleThresholdOnsetDetector(BaseProcessor):
    """
    Double-threshold EMG onset detection processor with Numba acceleration.

    This class implements the double-threshold approach for robust EMG onset detection.
    The signal must first cross a high threshold and then remain above a low threshold
    for a specified period to be considered a valid activation. This approach reduces
    false positives compared to single threshold methods.
    """

    def __init__(
        self,
        threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
        high_threshold_value: float = 3.0,
        low_threshold_value: float = 1.0,
        min_duration: float = 0.01,  # seconds
        max_onset_delay: float = 0.1,  # seconds to search backward for true onset
        baseline_window: Optional[Tuple[float, float]] = None,
        max_workers: Optional[int] = None,
        debug_info: bool = False,
    ):
        """
        Initialize the double-threshold EMG onset detector.

        Args:
            threshold_method: Method for determining the activation threshold
                "absolute": Fixed threshold values
                "std": Multiples of standard deviation above mean
                "rms": Multiples of RMS value
                "percent_max": Percentages of maximum value
            high_threshold_value: High threshold parameter value (for initial detection)
            low_threshold_value: Low threshold parameter value (for confirmation)
            min_duration: Minimum duration for a valid activation (seconds)
            max_onset_delay: Maximum time to search backward for true onset (seconds)
            baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
            max_workers: Maximum number of worker threads for parallel processing
            debug_info: Whether to include debugging information in the results
        """
        self.threshold_method = threshold_method
        self.high_threshold_value = high_threshold_value
        self.low_threshold_value = low_threshold_value
        self.min_duration = min_duration
        self.max_onset_delay = max_onset_delay
        self.baseline_window = baseline_window
        self.max_workers = max_workers
        self.debug_info = debug_info

        # Use small overlap to ensure no events are missed at chunk boundaries
        self._overlap_samples = 20

        # Store computed thresholds
        self.high_thresholds = None
        self.low_thresholds = None

        # Define dtype for output
        dtype_list = [
            ("onset_idx", np.int64),
            ("offset_idx", np.int64),
            ("channel", np.int32),
            ("amplitude", np.float32),
            ("duration", np.float32),
        ]

        # Add debug fields if requested
        if debug_info:
            dtype_list.extend(
                [
                    ("true_onset_idx", np.int64),  # Initial threshold crossing
                    ("high_threshold", np.float32),
                    ("low_threshold", np.float32),
                    ("confirmation_time", np.float32),
                ]
            )

        self._dtype = np.dtype(dtype_list)

    def compute_thresholds(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute high and low thresholds based on the specified method.

        Args:
            data: Input data array

        Returns:
            Tuple of arrays (high_thresholds, low_thresholds) for each channel
        """
        return _compute_thresholds_numba(
            data,
            self.threshold_method,
            self.high_threshold_value,
            self.low_threshold_value,
        )

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Detect EMG onset events using the double-threshold technique.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments
                precompute_thresholds: Whether to precompute thresholds from the entire dataset

        Returns:
            Dask array with onset events (can be converted to DataFrame)
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for onset detection")

        # Convert durations to samples
        min_samples = int(self.min_duration * fs)
        max_onset_delay_samples = int(self.max_onset_delay * fs)

        # Pre-compute the thresholds for the entire dataset if possible
        if self.threshold_method == "absolute":
            self.high_thresholds = np.array(
                [self.high_threshold_value], dtype=np.float32
            )
            self.low_thresholds = np.array([self.low_threshold_value], dtype=np.float32)
        elif kwargs.get("precompute_thresholds", False):
            # Optionally precompute thresholds from the entire dataset
            sample_data = data.compute()
            if sample_data.ndim == 1:
                sample_data = sample_data.reshape(-1, 1)
            self.high_thresholds, self.low_thresholds = self.compute_thresholds(
                sample_data
            )

        # Prepare baseline indices if specified
        baseline_start = 0
        baseline_end = None
        if self.baseline_window is not None:
            baseline_start = int(self.baseline_window[0] * fs)
            baseline_end = int(self.baseline_window[1] * fs)

        def detect_onsets_chunk(chunk: np.ndarray, block_info=None) -> np.ndarray:
            """Process a chunk of data to detect onsets using double-threshold method"""
            # Get chunk offset from block_info
            chunk_offset = 0
            if block_info and len(block_info) > 0:
                chunk_offset = block_info[0]["array-location"][0][0]

            # Ensure the input is a contiguous array with correct memory layout
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case by adding extra dimension if needed
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]

            # Compute thresholds if not already computed
            if (
                self.high_thresholds is None
                or self.low_thresholds is None
                or len(self.high_thresholds) != chunk.shape[1]
            ):
                if baseline_end is not None:
                    baseline_data = chunk[baseline_start:baseline_end]
                    if baseline_data.size > 0:
                        self.high_thresholds, self.low_thresholds = (
                            self.compute_thresholds(baseline_data)
                        )
                    else:
                        self.high_thresholds, self.low_thresholds = (
                            self.compute_thresholds(chunk)
                        )
                else:
                    self.high_thresholds, self.low_thresholds = self.compute_thresholds(
                        chunk
                    )

            # Handle threshold shape correction
            if len(self.high_thresholds) == 1 and chunk.shape[1] > 1:
                # Expand thresholds to match number of channels
                self.high_thresholds = np.repeat(self.high_thresholds, chunk.shape[1])
                self.low_thresholds = np.repeat(self.low_thresholds, chunk.shape[1])

            # Detect threshold crossings
            above_high_threshold = chunk > self.high_thresholds[None, :]
            above_low_threshold = chunk > self.low_thresholds[None, :]

            # Create result array for all channels
            all_results = []

            # Helper function to process a single channel
            def process_channel(channel_ind):
                # Get high threshold crossings for this channel
                high_transitions = np.diff(
                    above_high_threshold[:, channel_ind].astype(int)
                )

                # Find rising edges (1) for high threshold
                high_onset_indices = np.where(high_transitions == 1)[0]

                # Add one to onset indices (diff reduces length by 1)
                high_onset_indices = high_onset_indices + 1

                # Skip if no high threshold crossings
                if len(high_onset_indices) == 0:
                    return np.array([], dtype=self._dtype)

                # Pre-allocate arrays for results
                max_events = len(high_onset_indices)
                result_onsets = np.zeros(max_events, dtype=np.int64)
                result_offsets = np.zeros(max_events, dtype=np.int64)
                result_amplitudes = np.zeros(max_events, dtype=np.float32)
                result_durations = np.zeros(max_events, dtype=np.float32)

                # Debug info if requested
                if self.debug_info:
                    true_onsets = np.zeros(max_events, dtype=np.int64)
                    high_threshold_values = np.zeros(max_events, dtype=np.float32)
                    low_threshold_values = np.zeros(max_events, dtype=np.float32)
                    confirmation_times = np.zeros(max_events, dtype=np.float32)

                # Counter for actual valid events
                event_count = 0

                # Process each potential onset
                for idx, high_onset_idx in enumerate(high_onset_indices):
                    # Skip if too close to the end
                    if high_onset_idx + min_samples >= len(chunk):
                        continue

                    # This is the initial high threshold crossing point
                    true_onset_idx = high_onset_idx

                    # Check if the signal stays above the low threshold for the minimum duration
                    is_valid = True
                    low_threshold = self.low_thresholds[channel_ind]

                    # Count consecutive samples above low threshold
                    consecutive_count = 0
                    offset_idx = high_onset_idx

                    for i in range(
                        high_onset_idx,
                        min(len(chunk), high_onset_idx + min_samples * 2),
                    ):
                        if chunk[i, channel_ind] > low_threshold:
                            consecutive_count += 1
                            offset_idx = i
                        else:
                            # Break on first sample below low threshold
                            break

                    # Check if minimum duration was met
                    if consecutive_count < min_samples:
                        is_valid = False

                    if is_valid:
                        # Search backward to find the "true" onset (first time signal crossed low threshold)
                        search_start = max(0, high_onset_idx - max_onset_delay_samples)
                        onset_idx = high_onset_idx

                        for i in range(high_onset_idx - 1, search_start - 1, -1):
                            if (
                                chunk[i, channel_ind]
                                <= self.low_thresholds[channel_ind]
                            ):
                                # Found the first point below threshold
                                onset_idx = i + 1
                                break
                            else:
                                # Keep searching backward
                                onset_idx = i

                        # Calculate duration
                        duration_samples = offset_idx - onset_idx + 1
                        duration_sec = duration_samples / fs

                        # Skip if shorter than minimum duration (unlikely but possible)
                        if duration_samples < min_samples:
                            continue

                        # Calculate amplitude (max value during activation)
                        segment = chunk[onset_idx : offset_idx + 1, channel_ind]
                        amplitude = np.max(segment)

                        # Store in pre-allocated arrays
                        result_onsets[event_count] = onset_idx + chunk_offset
                        result_offsets[event_count] = offset_idx + chunk_offset
                        result_amplitudes[event_count] = amplitude
                        result_durations[event_count] = duration_sec

                        # Store debug info if requested
                        if self.debug_info:
                            true_onsets[event_count] = true_onset_idx + chunk_offset
                            high_threshold_values[event_count] = self.high_thresholds[
                                channel_ind
                            ]
                            low_threshold_values[event_count] = self.low_thresholds[
                                channel_ind
                            ]
                            confirmation_times[event_count] = consecutive_count / fs

                        event_count += 1

                # Create structured array from filled portion only
                if event_count > 0:
                    result = np.zeros(event_count, dtype=self._dtype)
                    result["onset_idx"] = result_onsets[:event_count]
                    result["offset_idx"] = result_offsets[:event_count]
                    result["channel"] = channel_ind
                    result["amplitude"] = result_amplitudes[:event_count]
                    result["duration"] = result_durations[:event_count]

                    if self.debug_info:
                        result["true_onset_idx"] = true_onsets[:event_count]
                        result["high_threshold"] = high_threshold_values[:event_count]
                        result["low_threshold"] = low_threshold_values[:event_count]
                        result["confirmation_time"] = confirmation_times[:event_count]

                    return result
                else:
                    return np.array([], dtype=self._dtype)

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

        # Ensure input is 2D using a safer method
        if data.ndim == 1:
            # Use indexing syntax instead of reshape
            data = data[:, np.newaxis]

        # Use map_overlap with explicit boundary handling and drop_axis parameter
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
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "threshold_method": self.threshold_method,
                "high_threshold_value": self.high_threshold_value,
                "low_threshold_value": self.low_threshold_value,
                "min_duration": self.min_duration,
                "max_onset_delay": self.max_onset_delay,
                "accelerated": True,
                "parallel": self.max_workers is not None,
                "debug_info": self.debug_info,
            }
        )
        return base_summary


# Factory functions for common configurations
def create_double_threshold_detector(
    high_threshold: float = 3.0,
    low_threshold: float = 1.0,
    threshold_method: Literal["std", "rms", "absolute", "percent_max"] = "std",
    min_duration: float = 0.02,
    max_onset_delay: float = 0.1,
    max_workers: Optional[int] = None,
) -> DoubleThresholdOnsetDetector:
    """
    Create a double-threshold EMG onset detector with custom thresholds.

    Args:
        high_threshold: High threshold value for initial detection
        low_threshold: Low threshold value for confirmation
        threshold_method: Method for determining thresholds
        min_duration: Minimum duration above low threshold for valid activation (seconds)
        max_onset_delay: Maximum time to search backward for true onset (seconds)
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        Configured DoubleThresholdOnsetDetector
    """
    return DoubleThresholdOnsetDetector(
        threshold_method=threshold_method,
        high_threshold_value=high_threshold,
        low_threshold_value=low_threshold,
        min_duration=min_duration,
        max_onset_delay=max_onset_delay,
        max_workers=max_workers,
    )


def create_std_double_threshold_detector(
    high_std: float = 3.0,
    low_std: float = 1.0,
    min_duration: float = 0.02,
    baseline_window: Optional[Tuple[float, float]] = None,
    max_workers: Optional[int] = None,
) -> DoubleThresholdOnsetDetector:
    """
    Create a double-threshold EMG onset detector using standard deviation thresholds.

    Args:
        high_std: High threshold as a multiple of standard deviation
        low_std: Low threshold as a multiple of standard deviation
        min_duration: Minimum duration above low threshold for valid activation (seconds)
        baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        Configured DoubleThresholdOnsetDetector using standard deviation thresholds
    """
    return DoubleThresholdOnsetDetector(
        threshold_method="std",
        high_threshold_value=high_std,
        low_threshold_value=low_std,
        min_duration=min_duration,
        baseline_window=baseline_window,
        max_workers=max_workers,
    )


def create_rms_double_threshold_detector(
    high_factor: float = 3.0,
    low_factor: float = 1.2,
    min_duration: float = 0.02,
    max_workers: Optional[int] = None,
) -> DoubleThresholdOnsetDetector:
    """
    Create a double-threshold EMG onset detector using RMS-based thresholds.

    Args:
        high_factor: High threshold as a multiple of RMS
        low_factor: Low threshold as a multiple of RMS
        min_duration: Minimum duration above low threshold for valid activation (seconds)
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        Configured DoubleThresholdOnsetDetector using RMS-based thresholds
    """
    return DoubleThresholdOnsetDetector(
        threshold_method="rms",
        high_threshold_value=high_factor,
        low_threshold_value=low_factor,
        min_duration=min_duration,
        max_workers=max_workers,
    )


def create_robust_double_threshold_detector(
    high_threshold: float = 3.0,
    low_threshold: float = 1.0,
    min_duration: float = 0.03,
    max_onset_delay: float = 0.2,
    threshold_method: str = "rms",
    debug_info: bool = True,
    max_workers: Optional[int] = None,
) -> DoubleThresholdOnsetDetector:
    """
    Create a noise-robust double-threshold EMG onset detector with conservative settings.

    This configuration uses longer minimum durations and more separation between thresholds
    to maximize robustness against false positives in noisy environments.

    Args:
        high_threshold: High threshold value
        low_threshold: Low threshold value
        min_duration: Minimum duration above low threshold (longer for robustness)
        max_onset_delay: Maximum time to search backward for true onset (seconds)
        threshold_method: Method for determining thresholds
        debug_info: Whether to include debugging information in results
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        Configured DoubleThresholdOnsetDetector with robust settings
    """
    return DoubleThresholdOnsetDetector(
        threshold_method=threshold_method,
        high_threshold_value=high_threshold,
        low_threshold_value=low_threshold,
        min_duration=min_duration,
        max_onset_delay=max_onset_delay,
        debug_info=debug_info,
        max_workers=max_workers,
    )
