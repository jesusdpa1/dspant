"""
EMG Onset Detection Module: Double Threshold Implementation

This module provides a robust method for detecting muscle activation
events in EMG signals using a double-threshold approach.
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor


@public_api
class DoubleThresholdDetector(BaseProcessor):
    """
    EMG onset detection processor implementation using double-threshold.

    Detects onset of muscle activation in EMG signals by requiring the signal to cross
    two different thresholds: a higher primary threshold and a lower secondary threshold.
    The dual threshold approach increases detection robustness by reducing false positives.
    """

    def __init__(
        self,
        threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
        primary_threshold_value: float = 3.0,
        secondary_threshold_value: float = 1.5,  # Lower threshold value
        min_contraction_duration: float = 0.01,  # seconds
        refractory_period: float = 0.1,  # seconds
        secondary_points_required: int = 3,  # Minimum consecutive points above secondary threshold
        baseline_window: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the EMG double threshold detector.

        Args:
            threshold_method: Method for determining the activation threshold
                "absolute": Fixed threshold values
                "std": Multiple of standard deviation above mean
                "rms": Multiple of RMS value
                "percent_max": Percentage of maximum value
            primary_threshold_value: Higher threshold parameter value
            secondary_threshold_value: Lower threshold parameter value
            min_contraction_duration: Minimum duration for a valid contraction (seconds)
            refractory_period: Minimum time between detections (seconds)
            secondary_points_required: Number of consecutive points required above secondary threshold
            baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
        """
        self.threshold_method = threshold_method
        self.primary_threshold_value = primary_threshold_value
        self.secondary_threshold_value = secondary_threshold_value
        self.min_contraction_duration = min_contraction_duration
        self.refractory_period = refractory_period
        self.secondary_points_required = secondary_points_required
        self.baseline_window = baseline_window

        # Dynamically adjust overlap to ensure sufficient context for detection
        self._overlap_samples = 0

        # Store computed thresholds per channel
        self.primary_thresholds = None
        self.secondary_thresholds = None

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

    def compute_thresholds(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Compute primary and secondary activation thresholds based on the specified method.

        Args:
            data: Input data array

        Returns:
            Tuple of (primary_threshold, secondary_threshold)
        """
        # Flatten data for single channel processing
        data_flat = data.flatten()

        if self.threshold_method == "absolute":
            primary = self.primary_threshold_value
            secondary = self.secondary_threshold_value
        elif self.threshold_method == "std":
            mean = np.mean(data_flat)
            std = np.std(data_flat)
            primary = mean + self.primary_threshold_value * std
            secondary = mean + self.secondary_threshold_value * std
        elif self.threshold_method == "rms":
            rms = np.sqrt(np.mean(data_flat * data_flat))
            primary = self.primary_threshold_value * rms
            secondary = self.secondary_threshold_value * rms
        elif self.threshold_method == "percent_max":
            max_val = np.max(data_flat)
            primary = self.primary_threshold_value * max_val / 100.0
            secondary = self.secondary_threshold_value * max_val / 100.0
        else:
            primary = self.primary_threshold_value
            secondary = self.secondary_threshold_value

        return primary, secondary

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Detect EMG onset events in the input data using double threshold.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with onset events (can be converted to DataFrame)
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for onset detection")

        # Convert time-based parameters to sample counts
        min_samples = max(int(self.min_contraction_duration * fs), 1)
        refractory_samples = max(int(self.refractory_period * fs), 1)

        # Dynamically calculate overlap to ensure sufficient context
        max_overlap = min(
            data.shape[0] // 10,  # No more than 10% of total data
            max(
                refractory_samples * 2,  # Ensure enough context for refractory period
                min_samples * 2,  # Ensure enough context for minimum duration
                int(0.5 * fs),  # Additional buffer (half a second)
            ),
        )
        self._overlap_samples = max_overlap

        def detect_events_in_chunk(
            chunk: np.ndarray, chunk_loc: Optional[Tuple[int, int]] = None
        ) -> np.ndarray:
            """
            Detect events within a chunk, with optional global context.

            Args:
                chunk: Input data chunk
                chunk_loc: Optional tuple of (start_idx, end_idx) for global context

            Returns:
                Array of detected events
            """
            n_samples, n_channels = chunk.shape

            # Adjust global index tracking
            global_start = chunk_loc[0] if chunk_loc else 0

            # Pre-allocate a list to store detected events
            events_list = []

            # Ensure thresholds are computed per channel
            if self.primary_thresholds is None:
                self.primary_thresholds = {}
                self.secondary_thresholds = {}

            # Process each channel separately
            for channel in range(n_channels):
                channel_data = chunk[:, channel]

                # Calculate threshold based on selected method
                if self.baseline_window is not None:
                    # Use specified baseline window if provided
                    start_idx = max(0, int(self.baseline_window[0] * fs))
                    end_idx = min(n_samples, int(self.baseline_window[1] * fs))
                    baseline_data = channel_data[start_idx:end_idx]
                else:
                    # Use entire signal for baseline
                    baseline_data = channel_data

                # Compute thresholds for this channel
                primary_threshold, secondary_threshold = self.compute_thresholds(
                    baseline_data
                )

                # Store thresholds for monitoring/debugging
                self.primary_thresholds[channel] = primary_threshold
                self.secondary_thresholds[channel] = secondary_threshold

                # Find raw threshold crossings
                primary_mask = channel_data > primary_threshold
                secondary_mask = channel_data > secondary_threshold

                # Find primary threshold crossings
                primary_transitions = np.diff(primary_mask.astype(int), prepend=0)
                primary_onsets = np.where(primary_transitions == 1)[0]
                primary_offsets = np.where(primary_transitions == -1)[0]

                # Handle edge case: signal ends while still above threshold
                if len(primary_onsets) > len(primary_offsets):
                    primary_offsets = np.append(primary_offsets, n_samples - 1)

                # Apply refractory period to primary onsets
                if len(primary_onsets) > 0:
                    # First onset is always kept
                    filtered_primary_onsets = [primary_onsets[0]]

                    # Filter subsequent onsets based on refractory period
                    for i in range(1, len(primary_onsets)):
                        if (
                            primary_onsets[i] - filtered_primary_onsets[-1]
                            >= refractory_samples
                        ):
                            filtered_primary_onsets.append(primary_onsets[i])

                    primary_onsets = np.array(filtered_primary_onsets)
                else:
                    primary_onsets = np.array([])

                # Apply refractory period to primary offsets
                if len(primary_offsets) > 0:
                    # First offset is always kept
                    filtered_primary_offsets = [primary_offsets[0]]

                    # Filter subsequent offsets based on refractory period
                    for i in range(1, len(primary_offsets)):
                        if (
                            primary_offsets[i] - filtered_primary_offsets[-1]
                            >= refractory_samples
                        ):
                            filtered_primary_offsets.append(primary_offsets[i])

                    primary_offsets = np.array(filtered_primary_offsets)
                else:
                    primary_offsets = np.array([])

                # Validate events using secondary threshold
                valid_events = []

                # Skip if we don't have both onsets and offsets
                if len(primary_onsets) == 0 or len(primary_offsets) == 0:
                    continue

                # Match each onset with the next offset
                for onset_idx in primary_onsets:
                    # Find the first offset after this onset
                    valid_offsets = primary_offsets[primary_offsets > onset_idx]

                    if len(valid_offsets) == 0:
                        # No valid offset found for this onset
                        continue

                    offset_idx = valid_offsets[0]

                    # Check minimum contraction duration
                    duration_samples = offset_idx - onset_idx
                    if duration_samples < min_samples:
                        continue

                    # Extract segment for secondary threshold analysis
                    segment = secondary_mask[onset_idx:offset_idx]

                    # Count consecutive points above secondary threshold
                    consecutive_count = 0
                    max_consecutive = 0

                    for point in segment:
                        if point:
                            consecutive_count += 1
                            max_consecutive = max(max_consecutive, consecutive_count)
                        else:
                            consecutive_count = 0

                    # Validate using secondary threshold criteria
                    if max_consecutive >= self.secondary_points_required:
                        # Calculate duration
                        duration_seconds = duration_samples / fs

                        # Extract signal during contraction for amplitude
                        contraction_segment = channel_data[onset_idx:offset_idx]
                        peak_amplitude = np.max(contraction_segment)

                        # Adjust onset and offset indices to global context
                        global_onset_idx = global_start + onset_idx
                        global_offset_idx = global_start + offset_idx

                        # Add event to list
                        valid_events.append(
                            (
                                global_onset_idx,
                                global_offset_idx,
                                channel,
                                peak_amplitude,
                                duration_seconds,
                            )
                        )

                # Add validated events to the main list
                events_list.extend(valid_events)

            # Convert events list to structured array
            if events_list:
                return np.array(events_list, dtype=self._dtype)
            else:
                # Return empty array with correct dtype if no events
                return np.array([], dtype=self._dtype)

        # Apply detection function to dask array with improved overlap handling
        events_da = data.map_overlap(
            detect_events_in_chunk,
            depth={-2: self._overlap_samples},  # Explicit overlap depth
            boundary="none",  # No zero padding at boundaries
            dtype=self._dtype,
            drop_axis=(
                0,
                1,
            ),  # Input has shape [samples, channels], output is 1D array of events
            new_axis=0,  # Output will be 1D array of event records
        )

        return events_da

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
                "primary_threshold_value": self.primary_threshold_value,
                "secondary_threshold_value": self.secondary_threshold_value,
                "min_contraction_duration": self.min_contraction_duration,
                "refractory_period": self.refractory_period,
                "secondary_points_required": self.secondary_points_required,
                "detection_method": "double-threshold",
                "primary_thresholds": self.primary_thresholds,
                "secondary_thresholds": self.secondary_thresholds,
            }
        )
        return base_summary


@public_api
def create_double_threshold_detector(
    primary_threshold: float,
    secondary_threshold: float,
    threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
    secondary_points_required: int = 3,
    min_contraction_duration: float = 0.01,
    refractory_period: float = 0.1,
) -> DoubleThresholdDetector:
    """
    Create an EMG onset detector using double thresholding.

    Args:
        threshold_method: Method for determining the activation threshold
            "absolute": Fixed threshold values
            "std": Multiple of standard deviation above mean
            "rms": Multiple of RMS value
            "percent_max": Percentage of maximum value
        primary_threshold: Higher absolute threshold value
        secondary_threshold: Lower absolute threshold value
        secondary_points_required: Minimum consecutive points required above secondary threshold
        min_contraction_duration: Minimum duration for valid contraction in seconds
        refractory_period: Minimum time between consecutive events in seconds

    Returns:
        Configured EMGDoubleThresholdDetector
    """
    return DoubleThresholdDetector(
        threshold_method=threshold_method,
        primary_threshold_value=primary_threshold,
        secondary_threshold_value=secondary_threshold,
        secondary_points_required=secondary_points_required,
        min_contraction_duration=min_contraction_duration,
        refractory_period=refractory_period,
    )
