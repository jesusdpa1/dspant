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

        # Store original data for amplitude computation
        self._original_data = None

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

    def process(
        self, data: da.Array, fs: Optional[float] = None, **kwargs
    ) -> Union[da.Array, np.ndarray]:
        """
        Detect EMG onset events in the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments
                return_mask_only: If True, return only the binary mask

        Returns:
            Binary mask (if return_mask_only=True) or array of onset events
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for onset detection")

        # Store original data for amplitude calculation
        self._original_data = data

        # Convert minimum duration to samples
        min_samples = int(self.min_duration * fs)

        # Pre-compute the thresholds for the entire dataset if possible
        if self.threshold_method == "absolute":
            self.thresholds = np.array([self.threshold_value], dtype=np.float32)
        elif kwargs.get("precompute_thresholds", False):
            # Optionally precompute thresholds from the entire dataset
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

        def detect_activation_mask(chunk: np.ndarray, block_info=None) -> np.ndarray:
            """Process a chunk of data to create an activation mask"""
            # Get chunk offset from block_info
            chunk_offset = 0
            if block_info and len(block_info) > 0:
                chunk_offset = block_info[0]["array-location"][0][0]

            # Ensure the input is a contiguous array
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]

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

            # Create a binary mask
            mask = np.zeros_like(chunk, dtype=np.int8)

            # Apply threshold and identify regions above threshold
            above_threshold = chunk > self.thresholds[None, :]

            # For each channel, find continuous segments above threshold
            for channel in range(chunk.shape[1]):
                # Get transitions
                transitions = np.diff(
                    above_threshold[:, channel].astype(int), prepend=0
                )

                # Find rising edges (1) and falling edges (-1)
                onsets = np.where(transitions == 1)[0]
                offsets = np.where(transitions == -1)[0]

                # Process pairs
                for i, onset in enumerate(onsets):
                    # Find corresponding offset
                    valid_offsets = offsets[offsets > onset]
                    if len(valid_offsets) == 0:
                        offset = len(chunk)
                    else:
                        offset = valid_offsets[0]

                    # Check minimum duration
                    if offset - onset >= min_samples:
                        # Mark this region in the mask
                        mask[onset:offset, channel] = 1

            return mask

        # Create a binary activation mask
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Use map_overlap to create a binary mask
        activation_mask = data.map_overlap(
            detect_activation_mask,
            depth={0: self._overlap_samples},
            boundary="reflect",
            dtype=np.int8,
        )

        # Store this mask for later use
        self._activation_mask = activation_mask

        # If only mask requested, return it
        if kwargs.get("return_mask_only", False):
            return activation_mask

        # Post-process the mask to extract events
        # This is more efficiently done after computing the full mask
        events = self._extract_events_from_mask(activation_mask, data, fs)

        return events

    def _extract_events_from_mask(
        self, mask: da.Array, data: da.Array, fs: float
    ) -> np.ndarray:
        """
        Extract onset events from a binary activation mask.

        Args:
            mask: Binary activation mask
            data: Original data array
            fs: Sampling frequency

        Returns:
            Array of onset events
        """
        # Compute the mask and original data to get numpy arrays
        mask_array = mask.compute()
        data_array = data.compute()

        # Extract events from the mask
        events_list = []

        # Process each channel
        for channel in range(mask_array.shape[1]):
            channel_mask = mask_array[:, channel]

            # Find transitions
            transitions = np.diff(channel_mask, prepend=0)

            # Find rising and falling edges
            onsets = np.where(transitions == 1)[0]
            offsets = np.where(transitions == -1)[0]

            # If no falling edge after last onset, use the end of the data
            if len(onsets) > len(offsets):
                offsets = np.append(offsets, len(channel_mask))

            # Create events for each onset-offset pair
            for onset, offset in zip(onsets, offsets):
                # Calculate duration
                duration_samples = offset - onset
                duration_sec = duration_samples / fs

                # Calculate amplitude from original data
                segment = data_array[onset:offset, channel]
                if len(segment) > 0:
                    amplitude = np.max(segment)
                else:
                    amplitude = 0.0

                # Create event
                event = (onset, offset, channel, amplitude, duration_sec)
                events_list.append(event)

        # Create structured array
        if events_list:
            events_array = np.array(events_list, dtype=self._dtype)
        else:
            events_array = np.array([], dtype=self._dtype)

        return events_array

    def create_binary_mask(self, data: da.Array, fs: float) -> da.Array:
        """
        Create a binary mask showing activation periods.

        Args:
            data: Input data array
            fs: Sampling frequency

        Returns:
            Binary mask array (1 where active, 0 elsewhere)
        """
        # Process the data but only return the mask
        return self.process(data, fs, return_mask_only=True)

    def to_dataframe(self, events_array: Union[np.ndarray, da.Array]) -> pl.DataFrame:
        """
        Convert events array to a Polars DataFrame.

        Args:
            events_array: Array of detected events

        Returns:
            Polars DataFrame with onset events
        """
        # Handle empty arrays
        if isinstance(events_array, da.Array) and events_array.size == 0:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                {
                    "onset_idx": [],
                    "offset_idx": [],
                    "channel": [],
                    "amplitude": [],
                    "duration": [],
                }
            )

        # Convert dask array to numpy if needed
        if isinstance(events_array, da.Array):
            try:
                events_array = events_array.compute()
            except Exception as e:
                print(f"Error computing events array: {e}")
                # Return empty DataFrame with correct schema
                return pl.DataFrame(
                    {
                        "onset_idx": [],
                        "offset_idx": [],
                        "channel": [],
                        "amplitude": [],
                        "duration": [],
                    }
                )

        # Handle empty numpy array case
        if len(events_array) == 0:
            return pl.DataFrame(
                {
                    "onset_idx": [],
                    "offset_idx": [],
                    "channel": [],
                    "amplitude": [],
                    "duration": [],
                }
            )

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

    Args:
        threshold_std: Threshold as multiple of standard deviation above mean
        min_duration: Minimum duration for a valid activation (seconds)
        baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
        max_workers: Maximum number of worker threads to use

    Returns:
        Configured EMGOnsetDetector for standard deviation-based detection
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
    baseline_window: Optional[Tuple[float, float]] = None,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using RMS thresholding.

    Args:
        threshold_factor: Multiplier for RMS value to determine threshold
        min_duration: Minimum duration for a valid activation (seconds)
        baseline_window: Optional window for baseline calculation
        max_workers: Maximum number of worker threads to use

    Returns:
        Configured EMGOnsetDetector for RMS-based detection
    """
    return EMGOnsetDetector(
        threshold_method="rms",
        threshold_value=threshold_factor,
        min_duration=min_duration,
        baseline_window=baseline_window,
        max_workers=max_workers,
    )


def create_absolute_threshold_detector(
    threshold: float,
    min_duration: float = 0.01,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using absolute thresholding.

    Args:
        threshold: Fixed threshold value for activation detection
        min_duration: Minimum duration for a valid activation (seconds)
        max_workers: Maximum number of worker threads to use

    Returns:
        Configured EMGOnsetDetector for absolute threshold-based detection
    """
    return EMGOnsetDetector(
        threshold_method="absolute",
        threshold_value=threshold,
        min_duration=min_duration,
        max_workers=max_workers,
    )


def create_percent_max_detector(
    percent: float = 10.0,  # 10% of maximum by default
    min_duration: float = 0.01,
    baseline_window: Optional[Tuple[float, float]] = None,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using percentage of maximum value thresholding.

    Args:
        percent: Percentage of maximum value to use as threshold
        min_duration: Minimum duration for a valid activation (seconds)
        baseline_window: Optional window for baseline calculation
        max_workers: Maximum number of worker threads to use

    Returns:
        Configured EMGOnsetDetector for percentage of maximum-based detection
    """
    return EMGOnsetDetector(
        threshold_method="percent_max",
        threshold_value=percent,
        min_duration=min_duration,
        baseline_window=baseline_window,
        max_workers=max_workers,
    )


def create_normalized_onset_detector(
    threshold: float = 0.5,
    threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "absolute",
    min_duration: float = 0.01,
    max_workers: Optional[int] = None,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector optimized for normalized data.

    This factory function creates a detector with parameters suited for data
    that has already been normalized to a standard range (e.g., 0-1).

    Args:
        threshold: Threshold value (appropriate for normalized data)
        threshold_method: Method for determining threshold (typically 'absolute' for normalized data)
        min_duration: Minimum duration for a valid activation (seconds)
        max_workers: Maximum number of worker threads to use

    Returns:
        Configured EMGOnsetDetector for normalized data
    """
    return EMGOnsetDetector(
        threshold_method=threshold_method,
        threshold_value=threshold,
        min_duration=min_duration,
        max_workers=max_workers,
    )
