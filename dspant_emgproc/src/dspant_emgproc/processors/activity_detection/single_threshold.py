from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor


@public_api
class SingleThresholdDetector(BaseProcessor):
    """
    EMG onset detection processor implementation using single-threshold.

    Detects onset of muscle activation in EMG signals by identifying where the signal
    crosses a specific threshold. Incorporates a refractory period to prevent multiple
    detections of the same muscle activation event.
    """

    def __init__(
        self,
        threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
        threshold_value: float = 3.0,
        min_contraction_duration: float = 0.01,  # seconds
        refractory_period: float = 0.1,  # seconds
        baseline_window: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the EMG single threshold detector.

        Args:
            threshold_method: Method for determining the activation threshold
                "absolute": Fixed threshold value
                "std": Multiple of standard deviation above mean
                "rms": Multiple of RMS value
                "percent_max": Percentage of maximum value
            threshold_value: Threshold parameter value
            min_contraction_duration: Minimum duration for a valid contraction (seconds)
            refractory_period: Minimum time after a detection before another can be registered (seconds)
            baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_contraction_duration = min_contraction_duration
        self.refractory_period = refractory_period
        self.baseline_window = baseline_window

        # Dynamically adjust overlap to ensure sufficient context for detection
        self._overlap_samples = 0

        # Store computed threshold per channel
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

    def compute_threshold(self, data: np.ndarray) -> float:
        """
        Compute activation threshold based on the specified method.

        Args:
            data: Input data array

        Returns:
            Computed threshold value
        """
        # Flatten data for single channel processing
        data_flat = data.flatten()

        if self.threshold_method == "absolute":
            threshold = self.threshold_value
        elif self.threshold_method == "std":
            mean = np.mean(data_flat)
            std = np.std(data_flat)
            threshold = mean + self.threshold_value * std
        elif self.threshold_method == "rms":
            rms = np.sqrt(np.mean(data_flat * data_flat))
            threshold = self.threshold_value * rms
        elif self.threshold_method == "percent_max":
            max_val = np.max(data_flat)
            threshold = self.threshold_value * max_val / 100.0
        else:
            threshold = self.threshold_value

        return threshold

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process the input EMG data to detect muscle activation events.

        Args:
            data: Input EMG data array [samples, channels]
            fs: Sampling frequency in Hz (required)
            **kwargs: Additional keyword arguments

        Returns:
            Detected events array with fields:
            - onset_idx: Index of onset detection
            - offset_idx: Index of offset detection
            - channel: Channel number where event was detected
            - amplitude: Peak amplitude during the contraction
            - duration: Duration of the contraction in seconds
        """
        if fs is None:
            raise ValueError(
                "Sampling frequency (fs) is required for EMG threshold detection"
            )

        # Convert time-based parameters to sample counts
        min_samples = max(int(self.min_contraction_duration * fs), 1)
        refractory_samples = max(int(self.refractory_period * fs), 1)

        # Dynamically calculate overlap to ensure sufficient context
        # Use the minimum of actual data length and a reasonable overlap
        max_overlap = min(
            data.shape[0] // 10,  # No more than 10% of total data
            max(
                refractory_samples * 2,  # Ensure enough context for refractory period
                min_samples * 2,  # Ensure enough context for minimum duration
                int(0.5 * fs),  # Additional buffer (half a second)
            ),
        )
        self._overlap_samples = max_overlap

        # Define the detection function to apply to each chunk
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

                # Calculate threshold using the compute_threshold method
                threshold = self.compute_threshold(baseline_data)

                # Store threshold for monitoring/debugging
                if self.thresholds is None:
                    self.thresholds = {}
                self.thresholds[channel] = threshold

                # Find all threshold crossings (binary mask of activations)
                above_threshold = channel_data > threshold

                # Find changes from below to above threshold (potential onsets)
                # Use rolling comparison to find transitions
                transitions = np.diff(above_threshold.astype(int), prepend=0)

                # Get all potential onsets and offsets
                raw_onsets = np.where(transitions == 1)[0]
                raw_offsets = np.where(transitions == -1)[0]

                # Handle edge case: signal ends while still above threshold
                if len(raw_onsets) > len(raw_offsets):
                    # Add an offset at the end of the signal
                    raw_offsets = np.append(raw_offsets, n_samples - 1)

                # Apply refractory period to onsets
                if len(raw_onsets) > 0:
                    # First onset is always kept
                    filtered_onsets = [raw_onsets[0]]

                    # Filter subsequent onsets based on refractory period
                    for i in range(1, len(raw_onsets)):
                        if raw_onsets[i] - filtered_onsets[-1] >= refractory_samples:
                            filtered_onsets.append(raw_onsets[i])

                    onsets = np.array(filtered_onsets)
                else:
                    onsets = np.array([])

                # Apply refractory period to offsets
                if len(raw_offsets) > 0:
                    # First offset is always kept
                    filtered_offsets = [raw_offsets[0]]

                    # Filter subsequent offsets based on refractory period
                    for i in range(1, len(raw_offsets)):
                        if raw_offsets[i] - filtered_offsets[-1] >= refractory_samples:
                            filtered_offsets.append(raw_offsets[i])

                    offsets = np.array(filtered_offsets)
                else:
                    offsets = np.array([])

                # Match onsets and offsets and apply further validation
                valid_events = []

                # Skip if we don't have both onsets and offsets
                if len(onsets) == 0 or len(offsets) == 0:
                    continue

                # Match each onset with the next offset
                for onset_idx in onsets:
                    # Find the first offset after this onset
                    valid_offsets = offsets[offsets > onset_idx]

                    if len(valid_offsets) == 0:
                        # No valid offset found for this onset
                        continue

                    offset_idx = valid_offsets[0]

                    # Check minimum contraction duration
                    duration_samples = offset_idx - onset_idx
                    if duration_samples < min_samples:
                        continue

                    # Calculate event properties
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
                "threshold_value": self.threshold_value,
                "min_contraction_duration": self.min_contraction_duration,
                "refractory_period": self.refractory_period,
                "detection_method": "single-threshold",
                "thresholds": self.thresholds,
            }
        )
        return base_summary


@public_api
def create_single_threshold_detector(
    threshold_method: Literal["absolute", "std", "rms", "percent_max"] = "std",
    threshold_value: float = 3.0,
    min_contraction_duration: float = 0.01,  # seconds
    refractory_period: float = 0.1,  # seconds
    baseline_window: Optional[Tuple[float, float]] = None,
) -> SingleThresholdDetector:
    """
    Create an EMG single threshold detector processor.

    Args:
        threshold_method: Method for determining the activation threshold
            "absolute": Fixed threshold value
            "std": Multiple of standard deviation above mean
            "rms": Multiple of RMS value
            "percent_max": Percentage of maximum value
        threshold_value: Threshold parameter value
        min_contraction_duration: Minimum duration for valid contraction (seconds)
        refractory_period: Minimum time between detections (seconds)
        baseline_window: Optional window for baseline calculation (start_time, end_time) in seconds

    Returns:
        Configured EMGSingleThresholdDetector instance
    """
    return SingleThresholdDetector(
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        min_contraction_duration=min_contraction_duration,
        refractory_period=refractory_period,
        baseline_window=baseline_window,
    )
