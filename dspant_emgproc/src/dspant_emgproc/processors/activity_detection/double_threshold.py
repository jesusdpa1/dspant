"""
EMG onset detection algorithms for identifying muscle activation using double-threshold.

This module provides methods to detect the onset of muscle activity in EMG signals
using a double-threshold approach for more robust detection.
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from dspant.engine.base import BaseProcessor

from ...core.internals import public_api


@public_api
class EMGDoubleThresholdDetector(BaseProcessor):
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
        min_event_spacing: float = 0.1,  # seconds
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
            min_event_spacing: Minimum time between consecutive events (onset-to-onset or offset-to-offset) in seconds
            secondary_points_required: Number of consecutive points required above secondary threshold
            baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
        """
        self.threshold_method = threshold_method
        self.primary_threshold_value = primary_threshold_value
        self.secondary_threshold_value = secondary_threshold_value
        self.min_contraction_duration = min_contraction_duration
        self.min_event_spacing = min_event_spacing
        self.secondary_points_required = secondary_points_required
        self.baseline_window = baseline_window

        # Use overlap based on minimum duration to avoid missing events at chunk boundaries
        self._overlap_samples = 20  # Somewhat larger than needed for safety

        # Store computed thresholds
        self.primary_threshold = None
        self.secondary_threshold = None

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

        # Convert minimum durations to samples
        min_contraction_samples = int(self.min_contraction_duration * fs)
        min_event_spacing_samples = int(self.min_event_spacing * fs)

        # Pre-compute thresholds
        if (
            isinstance(self.primary_threshold_value, (int, float))
            and isinstance(self.secondary_threshold_value, (int, float))
            and self.threshold_method == "absolute"
        ):
            primary_threshold = self.primary_threshold_value
            secondary_threshold = self.secondary_threshold_value
        else:
            # Sample a small portion for threshold calculation
            sample_size = min(int(1e6), data.shape[0])  # Use at most 1M samples
            sample_data = data[:sample_size].compute()
            primary_threshold, secondary_threshold = self.compute_thresholds(
                sample_data
            )

        # Store computed thresholds for reference
        self.primary_threshold = primary_threshold
        self.secondary_threshold = secondary_threshold

        def detect_onsets_chunk(chunk: np.ndarray) -> np.ndarray:
            """Process a chunk of data to detect onsets using a double threshold approach"""
            # Handle different input shapes by flattening if needed
            if chunk.ndim > 1:
                # For now, just use the first channel if there are multiple
                chunk_data = chunk[:, 0] if chunk.shape[1] > 0 else chunk.flatten()
            else:
                chunk_data = chunk

            # Initialize masks for primary and secondary thresholds
            primary_mask = chunk_data >= primary_threshold
            secondary_mask = chunk_data >= secondary_threshold

            # Find all potential onset candidates (where signal crosses primary threshold)
            primary_crossings = np.where(np.diff(primary_mask))[0] + 1

            # If no primary crossings, return empty result
            if len(primary_crossings) < 2:  # Need at least one onset and one offset
                return np.array([], dtype=self._dtype)

            # Determine direction of each primary crossing (onset or offset)
            crossing_directions = []
            for i, pc in enumerate(primary_crossings):
                if pc > 0 and pc < len(primary_mask) - 1:
                    if primary_mask[pc]:
                        crossing_directions.append(1)  # onset
                    else:
                        crossing_directions.append(-1)  # offset
                elif i > 0:
                    crossing_directions.append(-crossing_directions[-1])
                else:
                    crossing_directions.append(
                        1 if chunk_data[pc] > primary_threshold else -1
                    )

            # Extract all primary onsets and offsets
            primary_onsets = []
            primary_offsets = []

            for i, (pc, direction) in enumerate(
                zip(primary_crossings, crossing_directions)
            ):
                if direction > 0:  # onset
                    primary_onsets.append(pc)
                elif direction < 0:  # offset
                    primary_offsets.append(pc)

            # If no onsets or offsets, return empty
            if not primary_onsets or not primary_offsets:
                return np.array([], dtype=self._dtype)

            # Validate each primary onset using the secondary threshold criteria
            valid_onsets = []

            for onset in primary_onsets:
                # Find potential offset after this onset
                potential_offsets = [off for off in primary_offsets if off > onset]
                if not potential_offsets:
                    continue

                offset = potential_offsets[0]

                # Double threshold validation: Check if at least N consecutive points
                # are above the secondary threshold in the segment
                segment = secondary_mask[onset:offset]

                # Use a sliding window approach to check for consecutive points
                consecutive_count = 0
                max_consecutive = 0

                for point in segment:
                    if point:
                        consecutive_count += 1
                        max_consecutive = max(max_consecutive, consecutive_count)
                    else:
                        consecutive_count = 0

                # If we have enough consecutive points above secondary threshold
                if max_consecutive >= self.secondary_points_required:
                    valid_onsets.append((onset, offset))

            # Filter for minimum contraction duration and event spacing
            filtered_pairs = []

            if valid_onsets:
                # Add first valid pair
                filtered_pairs.append(valid_onsets[0])

                # Filter subsequent pairs based on spacing
                for pair in valid_onsets[1:]:
                    onset, offset = pair
                    prev_onset, prev_offset = filtered_pairs[-1]

                    # Check spacing between onsets
                    if onset - prev_onset >= min_event_spacing_samples:
                        # Check minimum contraction duration
                        if offset - onset >= min_contraction_samples:
                            filtered_pairs.append(pair)

            # Build final results
            results = []

            for onset_idx, offset_idx in filtered_pairs:
                # Calculate duration in seconds
                duration_sec = (offset_idx - onset_idx) / fs

                # Calculate amplitude
                segment = chunk_data[onset_idx : offset_idx + 1]
                amplitude = float(np.max(segment))

                # Add to results
                results.append(
                    (
                        onset_idx,
                        offset_idx,
                        0,  # channel
                        amplitude,
                        duration_sec,
                    )
                )

            # Convert results to structured array
            return np.array(results, dtype=self._dtype)

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
        overlap = max(
            self._overlap_samples,
            min_contraction_samples,
            min_event_spacing_samples,
            self.secondary_points_required
            * 2,  # Additional overlap for secondary threshold checking
        )

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
                "primary_threshold_value": self.primary_threshold_value,
                "secondary_threshold_value": self.secondary_threshold_value,
                "min_contraction_duration": self.min_contraction_duration,
                "min_event_spacing": self.min_event_spacing,
                "secondary_points_required": self.secondary_points_required,
                "detection_method": "double-threshold",
                "primary_threshold": self.primary_threshold,
                "secondary_threshold": self.secondary_threshold,
            }
        )
        return base_summary


@public_api
def create_double_threshold_detector(
    primary_threshold: float,
    secondary_threshold: float,
    secondary_points_required: int = 3,
    min_contraction_duration: float = 0.01,
    min_event_spacing: float = 0.1,
) -> EMGDoubleThresholdDetector:
    """
    Create an EMG onset detector using double thresholding.

    Args:
        primary_threshold: Higher absolute threshold value
        secondary_threshold: Lower absolute threshold value
        secondary_points_required: Minimum consecutive points required above secondary threshold
        min_contraction_duration: Minimum duration for valid contraction in seconds
        min_event_spacing: Minimum time between consecutive events in seconds

    Returns:
        Configured EMGDoubleThresholdDetector
    """
    return EMGDoubleThresholdDetector(
        threshold_method="absolute",
        primary_threshold_value=primary_threshold,
        secondary_threshold_value=secondary_threshold,
        secondary_points_required=secondary_points_required,
        min_contraction_duration=min_contraction_duration,
        min_event_spacing=min_event_spacing,
    )
