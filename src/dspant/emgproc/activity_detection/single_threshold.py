"""
EMG onset detection algorithms for identifying muscle activation.

This module provides methods to detect the onset of muscle activity in EMG signals
using various methods including threshold-based detection.
"""

import concurrent.futures
from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit, prange

from ...engine.base import BaseProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_thresholds_numba(
    data: np.ndarray, method: str, value: float
) -> np.ndarray:
    """
    Compute activation thresholds using Numba parallel acceleration.

    Args:
        data: Input data array
        method: Threshold method
        value: Threshold value

    Returns:
        Array of threshold values for each channel
    """
    n_channels = data.shape[1]
    thresholds = np.zeros(n_channels, dtype=np.float32)

    # Parallel processing across channels
    for chan in prange(n_channels):
        channel_data = data[:, chan]

        if method == "absolute":
            thresholds[chan] = value
        elif method == "std":
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            thresholds[chan] = mean + value * std
        elif method == "rms":
            rms = np.sqrt(np.mean(channel_data * channel_data))
            thresholds[chan] = value * rms
        elif method == "percent_max":
            max_val = np.max(channel_data)
            thresholds[chan] = value * max_val / 100.0

    return thresholds


class EMGOnsetDetector(BaseProcessor):
    """
    EMG onset detection processor implementation with thread pool and Numba acceleration.

    Detects onset of muscle activation in EMG signals using threshold crossing.
    """

    def __init__(
        self,
        threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
        threshold_value: float = 3.0,
        min_duration: float = 0.01,  # seconds
        baseline_window: Optional[Tuple[float, float]] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the EMG onset detector.

        Args:
            threshold_method: Method for determining the activation threshold
                "absolute": Fixed threshold value
                "std": Multiple of standard deviation above mean
                "rms": Multiple of RMS value
                "percent_max": Percentage of maximum value
            threshold_value: Threshold parameter value
            min_duration: Minimum duration for a valid activation (seconds)
            baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
            max_workers: Maximum number of worker threads to use
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_duration = min_duration
        self.baseline_window = baseline_window
        self.max_workers = max_workers

        # Use small overlap to ensure no events are missed at chunk boundaries
        self._overlap_samples = 10

        # Store computed thresholds
        self.thresholds = None

        # Define dtype for output
        self._dtype = np.dtype(
            [
                ("onset_idx", np.int64),
                ("offset_idx", np.int64),
                ("channel", np.int32),
                ("amplitude", np.float32),
                ("duration", np.float32),
            ]
        )

    def compute_thresholds(self, data: np.ndarray) -> np.ndarray:
        """
        Compute activation thresholds based on the specified method.

        Args:
            data: Input data array

        Returns:
            Array of threshold values for each channel
        """
        # Use Numba accelerated function
        return _compute_thresholds_numba(
            data, self.threshold_method, self.threshold_value
        )

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Detect EMG onset events in the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with onset events (can be converted to DataFrame)
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for onset detection")

        # Convert minimum duration to samples
        min_samples = int(self.min_duration * fs)

        # Pre-compute the thresholds for the entire dataset if possible
        # This can help ensure consistent threshold application across chunks
        if self.threshold_method == "absolute":
            self.thresholds = np.array([self.threshold_value], dtype=np.float32)
        elif kwargs.get("precompute_thresholds", False):
            # Optionally precompute thresholds from the entire dataset
            # This could be expensive for large datasets
            sample_data = data.compute()
            if sample_data.ndim == 1:
                sample_data = sample_data.reshape(-1, 1)
            self.thresholds = self.compute_thresholds(sample_data)

        # Prepare baseline indices if specified
        baseline_start = 0
        baseline_end = None
        if self.baseline_window is not None:
            baseline_start = int(self.baseline_window[0] * fs)
            baseline_end = int(self.baseline_window[1] * fs)

        def detect_onsets_chunk(chunk: np.ndarray, block_info=None) -> np.ndarray:
            """Process a chunk of data to detect onsets"""
            # Get chunk offset from block_info
            chunk_offset = 0
            print(block_info)
            if block_info and len(block_info) > 0:
                chunk_offset = block_info[0]["array-location"][0][0]
                print(chunk_offset)
            # Debug: print chunk info to see which chunks are being processed
            # print(f"Processing chunk with offset {chunk_offset}, shape {chunk.shape}")

            # Ensure the input is a contiguous array and has correct memory layout
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case by adding extra dimension if needed
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]  # Use np.newaxis instead of reshape

            # Compute thresholds if not already computed
            if self.thresholds is None or len(self.thresholds) != chunk.shape[1]:
                if baseline_end is not None:
                    baseline_data = chunk[baseline_start:baseline_end]
                    if baseline_data.size > 0:
                        self.thresholds = self.compute_thresholds(baseline_data)
                    else:
                        self.thresholds = self.compute_thresholds(chunk)
                else:
                    self.thresholds = self.compute_thresholds(chunk)

            # Handle threshold shape correction
            if len(self.thresholds) == 1 and chunk.shape[1] > 1:
                # Expand threshold to match number of channels
                self.thresholds = np.repeat(self.thresholds, chunk.shape[1])

            # Detect threshold crossings
            above_threshold = chunk > self.thresholds[None, :]

            # Create result array for all channels
            all_results = []

            # Helper function to process a single channel
            def process_channel(channel_ind):
                # Get transitions for this channel
                channel_transitions = np.diff(
                    above_threshold[:, channel_ind].astype(int)
                )

                # Find rising edges (1) and falling edges (-1)
                onset_indices = np.where(channel_transitions == 1)[0]
                offset_indices = np.where(channel_transitions == -1)[0]

                # Add one to onset indices (diff reduces length by 1)
                onset_indices = onset_indices + 1
                offset_indices = offset_indices + 1

                # Estimate max possible events (can't be more than number of onsets)
                max_events = len(onset_indices)

                # Skip further processing if no onsets
                if max_events == 0:
                    return np.array([], dtype=self._dtype)

                # Pre-allocate arrays for results
                result_onsets = np.zeros(max_events, dtype=np.int64)
                result_offsets = np.zeros(max_events, dtype=np.int64)
                result_amplitudes = np.zeros(max_events, dtype=np.float32)
                result_durations = np.zeros(max_events, dtype=np.float32)

                # Counter for actual valid events
                event_count = 0

                # Process valid onset-offset pairs
                for onset_idx in onset_indices:
                    # Find the next offset after this onset
                    valid_offsets = offset_indices[offset_indices > onset_idx]
                    if len(valid_offsets) == 0:
                        continue

                    offset_idx = valid_offsets[0]

                    # Calculate duration
                    duration_samples = offset_idx - onset_idx

                    # Skip if shorter than minimum duration
                    if duration_samples < min_samples:
                        continue

                    # Calculate duration in seconds
                    duration_sec = duration_samples / fs

                    # Calculate amplitude
                    segment = chunk[onset_idx : offset_idx + 1, channel_ind]
                    amplitude = np.max(segment)

                    # Store in pre-allocated arrays
                    result_onsets[event_count] = onset_idx + chunk_offset
                    result_offsets[event_count] = offset_idx + chunk_offset
                    result_amplitudes[event_count] = amplitude
                    result_durations[event_count] = duration_sec
                    event_count += 1

                # Create structured array from filled portion only
                if event_count > 0:
                    result = np.zeros(event_count, dtype=self._dtype)
                    result["onset_idx"] = result_onsets[:event_count]
                    result["offset_idx"] = result_offsets[:event_count]
                    result["channel"] = channel_ind
                    result["amplitude"] = result_amplitudes[:event_count]
                    result["duration"] = result_durations[:event_count]
                    return result
                else:
                    return np.array([], dtype=self._dtype)

            # Use ThreadPoolExecutor if we have multiple channels and max_workers is specified
            print(chunk.shape)
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
                "threshold_value": self.threshold_value,
                "min_duration": self.min_duration,
                "accelerated": True,
                "parallel": True,
            }
        )
        return base_summary


# Factory functions for common configurations
def create_std_onset_detector(
    threshold_std: float = 3.0,
    min_duration: float = 0.01,
    baseline_window: Optional[Tuple[float, float]] = None,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using standard deviation thresholding.
    """
    return EMGOnsetDetector(
        threshold_method="std",
        threshold_value=threshold_std,
        min_duration=min_duration,
        baseline_window=baseline_window,
        max_workers=max_workers,
    )


def create_rms_onset_detector(
    threshold_factor: float = 1.5,
    min_duration: float = 0.01,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using RMS thresholding.
    """
    return EMGOnsetDetector(
        threshold_method="rms",
        threshold_value=threshold_factor,
        min_duration=min_duration,
        max_workers=max_workers,
    )


def create_absolute_threshold_detector(
    threshold: float,
    min_duration: float = 0.01,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using absolute thresholding.
    """
    return EMGOnsetDetector(
        threshold_method="absolute",
        threshold_value=threshold,
        min_duration=min_duration,
        max_workers=max_workers,
    )
