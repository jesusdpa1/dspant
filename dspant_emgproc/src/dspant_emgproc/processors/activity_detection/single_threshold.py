"""
EMG onset detection algorithms for identifying muscle activation using zero-crossing.

This module provides methods to detect the onset of muscle activity in EMG signals
using a zero-crossing approach for threshold detection.
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
    EMG onset detection processor implementation using zero-crossing.

    Detects onset of muscle activation in EMG signals by monitoring when the
    difference between signal and threshold crosses zero.
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
            # Handle different input shapes by flattening if needed
            if chunk.ndim > 1:
                # For now, just use the first channel if there are multiple
                chunk_data = chunk[:, 0] if chunk.shape[1] > 0 else chunk.flatten()
            else:
                chunk_data = chunk

            # Calculate difference signal (signal - threshold)
            signs = chunk_data >= threshold

            # Find all zero crossings (from negative to positive for onset, positive to negative for offset)
            # Use sign function instead of just testing < 0 for more reliability with floating point
            # signs = np.sign(diff_signal)

            # Get indices where the sign changes
            # Rising edge: -1 to 1 or -1 to 0 to 1
            # Falling edge: 1 to -1 or 1 to 0 to -1
            zero_crossings = (
                np.where(np.diff(signs) != 0)[0] + 1
            )  # +1 to correct for diff

            # Classify as onset (rising) or offset (falling)
            if len(zero_crossings) < 2:  # Need at least one onset and one offset
                return np.array([], dtype=self._dtype)

            # Determine direction of each crossing
            crossing_directions = []
            for i, zc in enumerate(zero_crossings):
                if zc > 0 and zc < len(signs) - 1:  # Ensure we're not at the edge
                    # If the sign after the crossing is positive, it's a rising edge (onset)
                    if signs[zc] > 0:
                        crossing_directions.append(1)  # onset
                    else:
                        crossing_directions.append(-1)  # offset
                elif i > 0:  # If at edge, assume opposite of previous
                    crossing_directions.append(-crossing_directions[-1])
                else:
                    # First crossing and at edge, make an educated guess
                    crossing_directions.append(1 if chunk_data[zc] > threshold else -1)

            # Group into onset-offset pairs
            onset_indices = []
            offset_indices = []

            current_onset = None
            for i, (zc, direction) in enumerate(
                zip(zero_crossings, crossing_directions)
            ):
                if direction > 0 and current_onset is None:  # onset
                    current_onset = zc
                elif direction < 0 and current_onset is not None:  # offset
                    onset_indices.append(current_onset)
                    offset_indices.append(zc)
                    current_onset = None

            # Process each onset-offset pair
            results = []
            for onset_idx, offset_idx in zip(onset_indices, offset_indices):
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
                "detection_method": "zero-crossing",
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
