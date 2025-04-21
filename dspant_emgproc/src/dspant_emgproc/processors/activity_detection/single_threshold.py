"""
EMG onset detection algorithms for identifying muscle activation.

This module provides methods to detect the onset of muscle activity in EMG signals
using threshold-based detection.
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from dspant.engine.base import BaseProcessor

from ...core.internals import public_api


@public_api
class EMGOnsetDetector(BaseProcessor):
    """
    EMG onset detection processor implementation.

    Detects onset of muscle activation in EMG signals using threshold crossing.
    """

    def __init__(
        self,
        threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
        threshold_value: float = 3.0,
        min_duration: float = 0.01,  # seconds
        baseline_window: Optional[Tuple[float, float]] = None,
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
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_duration = min_duration
        self.baseline_window = baseline_window

        # Use overlap based on minimum duration to avoid missing events at chunk boundaries
        self._overlap_samples = 20  # Somewhat larger than needed for safety

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

    def compute_thresholds(self, data: np.ndarray) -> float:
        """
        Compute activation thresholds based on the specified method.

        Args:
            data: Input data array

        Returns:
            Threshold value
        """
        # Flatten data for single channel processing
        data_flat = data.flatten()

        if self.threshold_method == "absolute":
            return self.threshold_value
        elif self.threshold_method == "std":
            mean = np.mean(data_flat)
            std = np.std(data_flat)
            return mean + self.threshold_value * std
        elif self.threshold_method == "rms":
            rms = np.sqrt(np.mean(data_flat * data_flat))
            return self.threshold_value * rms
        elif self.threshold_method == "percent_max":
            max_val = np.max(data_flat)
            return self.threshold_value * max_val / 100.0
        else:
            return self.threshold_value

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

        # Pre-compute threshold - use the provided absolute value for simplicity
        if (
            isinstance(self.threshold_value, (int, float))
            and self.threshold_method == "absolute"
        ):
            threshold = self.threshold_value
        else:
            # Sample a small portion for threshold calculation
            sample_size = min(int(1e6), data.shape[0])  # Use at most 1M samples
            sample_data = data[:sample_size].compute()
            threshold = self.compute_thresholds(sample_data)

        def detect_onsets_chunk(chunk: np.ndarray) -> np.ndarray:
            """Process a chunk of data to detect onsets"""
            # No need to access block_info directly, Dask handles the offsets with map_overlap

            # Handle different input shapes by flattening if needed
            if chunk.ndim > 1:
                # For now, just use the first channel if there are multiple
                chunk_data = chunk[:, 0] if chunk.shape[1] > 0 else chunk.flatten()
            else:
                chunk_data = chunk

            # Detect threshold crossings
            above_threshold = chunk_data > threshold

            # Find transitions (diff == 1 for rising edge, diff == -1 for falling edge)
            transitions = np.diff(above_threshold.astype(int))

            # Find rising and falling edges
            onset_indices = np.where(transitions == 1)[0] + 1  # +1 to correct for diff
            offset_indices = (
                np.where(transitions == -1)[0] + 1
            )  # +1 to correct for diff

            # If no transitions, return empty result
            if len(onset_indices) == 0 or len(offset_indices) == 0:
                return np.array([], dtype=self._dtype)

            # Process each onset
            results = []

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

                # Calculate peak amplitude during the event
                segment = chunk_data[onset_idx : offset_idx + 1]
                amplitude = np.max(segment)

                # Create result record
                results.append(
                    (
                        onset_idx,  # Onset index
                        offset_idx,  # Offset index
                        0,  # Channel number (always 0 for single channel)
                        amplitude,  # Peak amplitude
                        duration_sec,  # Duration in seconds
                    )
                )

            # Convert results to structured array
            return (
                np.array(results, dtype=self._dtype)
                if results
                else np.array([], dtype=self._dtype)
            )

        # Make empty array with the correct dtype for meta
        empty = np.array([], dtype=self._dtype)

        # Prepare data shape
        # Handle single channel case properly
        if data.ndim == 1:
            # Keep it as 1D
            proc_data = data
        else:
            # Take first channel if multivariate
            proc_data = data[:, 0]

        # Set overlap depth to handle events that span chunk boundaries
        overlap = max(self._overlap_samples, min_samples)

        # Use map_blocks with drop_axis to ensure output has correct shape
        result = proc_data.map_overlap(
            detect_onsets_chunk,
            depth=overlap,
            boundary="reflect",
            meta=empty,
            drop_axis=0,  # Drop time axis
            dtype=self._dtype,
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
        if len(events_array) > 0:
            return pl.from_numpy(events_array)
        else:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    "onset_idx": pl.Int64,
                    "offset_idx": pl.Int64,
                    "channel": pl.Int32,
                    "amplitude": pl.Float32,
                    "duration": pl.Float32,
                }
            )

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
            }
        )
        return base_summary


@public_api
def create_absolute_threshold_detector(
    threshold: float,
    min_duration: float = 0.01,
) -> EMGOnsetDetector:
    """
    Create an EMG onset detector using absolute thresholding.

    Args:
        threshold: Absolute threshold value
        min_duration: Minimum duration for valid activation in seconds

    Returns:
        Configured EMGOnsetDetector
    """
    return EMGOnsetDetector(
        threshold_method="absolute",
        threshold_value=threshold,
        min_duration=min_duration,
    )
