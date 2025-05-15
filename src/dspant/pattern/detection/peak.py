"""
Peak detection algorithms for neural spike detection.

This module implements various peak detection methods for identifying
neural spikes and events in extracellular recordings.


missing

Prominence calculation

Width calculation

Plateau detection
"""

from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit

from dspant.processors.quality_metrics import NoiseEstimationProcessor
from dspant_neuroproc.processors.detection.base import ThresholdDetector


@jit(nopython=True, cache=True)
def _detect_peaks_numba(
    data: np.ndarray,
    positive_threshold: float,
    negative_threshold: float,
    detect_positive: bool,
    detect_negative: bool,
    min_distance_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect peaks in signal using sequential processing.

    Args:
        data: Input data array of shape (samples, channels)
        positive_threshold: Positive threshold value (scalar)
        negative_threshold: Negative threshold value (scalar)
        detect_positive: Whether to detect positive peaks
        detect_negative: Whether to detect negative peaks
        min_distance_samples: Minimum distance between peaks in samples

    Returns:
        Tuple of (peak_indices, peak_amplitudes, peak_channels)
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape

    # Initialize output lists with a maximum possible size
    max_peaks_per_channel = n_samples // min_distance_samples + 1
    peak_indices = np.zeros(max_peaks_per_channel * n_channels, dtype=np.int64)
    peak_amplitudes = np.zeros(max_peaks_per_channel * n_channels, dtype=np.float32)
    peak_channels = np.zeros(max_peaks_per_channel * n_channels, dtype=np.int32)

    # Tracking number of peaks found
    peak_count = 0

    # Process each channel sequentially
    for channel in range(n_channels):
        # Get channel data and thresholds
        channel_data = data[:, channel]
        pos_threshold = positive_threshold
        neg_threshold = negative_threshold

        # Keep track of last peak index for this channel
        last_peak_index = -min_distance_samples

        # Detect peaks
        for i in range(1, n_samples - 1):
            is_peak = False

            # Check if current point is higher than neighbors
            higher_than_neighbors = (
                channel_data[i] > channel_data[i - 1]
                and channel_data[i] > channel_data[i + 1]
            )

            if (
                detect_positive
                and higher_than_neighbors
                and channel_data[i] > pos_threshold
            ):
                is_peak = True

            # Check if current point is lower than neighbors (negative peak)
            lower_than_neighbors = (
                channel_data[i] < channel_data[i - 1]
                and channel_data[i] < channel_data[i + 1]
            )

            if (
                detect_negative
                and lower_than_neighbors
                and channel_data[i] < neg_threshold
            ):
                is_peak = True

            # If peak detected, check distance from previous peak
            if is_peak and (i - last_peak_index) >= min_distance_samples:
                # Ensure we don't exceed preallocated array
                if peak_count < len(peak_indices):
                    peak_indices[peak_count] = i
                    peak_amplitudes[peak_count] = channel_data[i]
                    peak_channels[peak_count] = channel
                    last_peak_index = i
                    peak_count += 1

    # Trim to actual number of peaks found
    return (
        peak_indices[:peak_count],
        peak_amplitudes[:peak_count],
        peak_channels[:peak_count],
    )


class PeakDetector(ThresholdDetector):
    """
    Peak detector for neural spikes based on threshold crossing and local extrema.

    This detector identifies peaks by first applying thresholds and then
    finding local maxima/minima. It supports configurable polarity, refractory
    periods, and noise-based adaptive thresholds.
    """

    def __init__(
        self,
        threshold: float = 4.0,
        threshold_mode: Literal["absolute", "std", "mad", "rms"] = "mad",
        polarity: Literal["positive", "negative", "both"] = "negative",
        refractory_period: float = 0.001,  # 1ms default refractory period
        align_to_peak: bool = True,  # Align detection to peak rather than threshold crossing
        noise_estimation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the peak detector.

        Args:
            threshold: Threshold value (interpretation depends on threshold_mode)
            threshold_mode: How to interpret the threshold
                "absolute": Use raw value
                "std": Multiple of signal standard deviation
                "mad": Multiple of median absolute deviation (more robust)
                "rms": Multiple of root mean square
            polarity: Which peaks to detect
                "positive": Only detect positive peaks
                "negative": Only detect negative peaks (typical for extracellular spikes)
                "both": Detect both positive and negative peaks
            refractory_period: Minimum time between spikes in seconds
            align_to_peak: Whether to align detection to peak rather than threshold crossing
            noise_estimation_kwargs: Parameters for noise estimation if threshold_mode is "mad"
        """
        super().__init__(
            threshold=threshold,
            threshold_mode=threshold_mode,
            polarity=polarity,
            refractory_period=refractory_period,
        )

        self.align_to_peak = align_to_peak

        # Set up noise estimation if needed
        self.noise_estimation_kwargs = noise_estimation_kwargs or {}

        if self.threshold_mode == "mad":
            self.noise_estimator = NoiseEstimationProcessor(
                method="mad", **self.noise_estimation_kwargs
            )
        else:
            self.noise_estimator = None

        # Overlap samples will be calculated dynamically based on fs and refractory_period
        self._overlap_samples = None

        # Initialize detection flags
        self.detect_positive = polarity in ["positive", "both"]
        self.detect_negative = polarity in ["negative", "both"]

    def detect(self, data: da.Array, fs: float, **kwargs) -> da.Array:
        """
        Detect peaks in the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency in Hz
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with detection results
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Calculate minimum distance in samples
        min_distance_samples = max(int(self.refractory_period * fs), 1)

        # Set overlap samples based on refractory period and sampling rate
        # Using 3x refractory period ensures we don't miss peaks at chunk boundaries
        self._overlap_samples = max(min_distance_samples * 3, 10)

        # Calculate adaptive thresholds if using MAD-based threshold
        if self.threshold_mode == "mad" and self.noise_estimator is not None:
            noise_levels = self.noise_estimator.process(data, fs=fs)
            noise_levels = noise_levels.compute()
            mean_noise = float(np.mean(noise_levels))
            pos_threshold = self.threshold * mean_noise
            neg_threshold = -self.threshold * mean_noise
        else:
            # Use compute_threshold from base class for other threshold types
            thresholds = self.compute_threshold(data.compute(), fs)
            pos_threshold = float(np.mean(thresholds["positive"]))
            neg_threshold = float(np.mean(thresholds["negative"]))

        # Define struct dtype for consistent output
        output_dtype = np.dtype(
            [
                ("index", np.int64),
                ("amplitude", np.float32),
                ("channel", np.int32),
            ]
        )

        # For small data, just compute directly without chunks
        if data.shape[1] < self._overlap_samples:
            # Process the whole array at once for small data
            computed_data = data.compute()
            peak_indices, peak_amplitudes, peak_channels = _detect_peaks_numba(
                computed_data,
                pos_threshold,
                neg_threshold,
                self.detect_positive,
                self.detect_negative,
                min_distance_samples,
            )

            # Create empty result with correct structure when no peaks are found
            if len(peak_indices) == 0:
                empty_result = np.array([], dtype=output_dtype)
                return da.from_array(empty_result, chunks=1)

            # Create structured array for output
            result = np.zeros(len(peak_indices), dtype=output_dtype)
            result["index"] = peak_indices
            result["amplitude"] = peak_amplitudes
            result["channel"] = peak_channels

            # Convert to dask array for consistent return type
            return da.from_array(result, chunks=1)

        # For larger data, use map_overlap with chunks
        # Define the chunk processing function
        def detect_peaks_chunk(chunk: np.ndarray) -> np.ndarray:
            # Ensure chunk is contiguous
            chunk = np.ascontiguousarray(chunk)

            # Use numba-accelerated implementation
            peak_indices, peak_amplitudes, peak_channels = _detect_peaks_numba(
                chunk,
                pos_threshold,
                neg_threshold,
                self.detect_positive,
                self.detect_negative,
                min_distance_samples,
            )

            # Create empty result with correct structure when no peaks are found
            if len(peak_indices) == 0:
                return np.array([], dtype=output_dtype)

            # Create structured array for output
            result = np.zeros(len(peak_indices), dtype=output_dtype)
            result["index"] = peak_indices
            result["amplitude"] = peak_amplitudes
            result["channel"] = peak_channels

            return result

        # Create an empty array with the right structure for meta
        empty_meta = np.array([], dtype=output_dtype)

        # Calculate safe overlap depth - cannot be larger than the smallest dimension
        safe_overlap = min(self._overlap_samples, data.shape[0] // 2, data.shape[1])

        # Apply detection to chunks with overlap
        result = data.map_overlap(
            detect_peaks_chunk,
            depth=safe_overlap,
            boundary="reflect",
            dtype=output_dtype,
            drop_axis=0,  # Drop time dimension
            meta=empty_meta,
        )

        # Update detection stats
        self._detection_stats = {
            "fs": fs,
            "positive_threshold": pos_threshold,
            "negative_threshold": neg_threshold,
            "min_distance_samples": min_distance_samples,
            "overlap_samples": safe_overlap,
        }

        return result

    def to_dataframe(
        self, events_array: Union[np.ndarray, da.Array], fs: float
    ) -> pl.DataFrame:
        """
        Convert events array to a Polars DataFrame.

        Args:
            events_array: Array of detected events
            fs: Sampling frequency in Hz

        Returns:
            Polars DataFrame with onset events
        """
        # Convert dask array to numpy if needed
        if isinstance(events_array, da.Array):
            try:
                events_array = events_array.compute()
            except Exception as e:
                print(f"Error computing events: {e}")
                # Return empty DataFrame if computation fails
                return pl.DataFrame(
                    schema={
                        "index": pl.Int64,
                        "amplitude": pl.Float32,
                        "channel": pl.Int32,
                        "time_sec": pl.Float64,
                    }
                )

        # Return empty DataFrame if no events found
        if len(events_array) == 0:
            return pl.DataFrame(
                schema={
                    "index": pl.Int64,
                    "amplitude": pl.Float32,
                    "channel": pl.Int32,
                    "time_sec": pl.Float64,
                }
            )

        # Convert numpy structured array to Polars DataFrame
        df = pl.from_numpy(events_array)

        # Add time in seconds column
        df = df.with_columns((pl.col("index") / fs).alias("time_sec"))

        return df

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        return self._overlap_samples or 10  # Default to 10 if not yet calculated

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the peak detector configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "align_to_peak": self.align_to_peak,
                "detect_positive": self.detect_positive,
                "detect_negative": self.detect_negative,
                "overlap_samples": self._overlap_samples,
            }
        )
        return base_summary


# Factory functions for common configurations


def create_threshold_detector(
    threshold: float = 4.0,
    threshold_mode: str = "mad",
    polarity: str = "negative",
    refractory_period: float = 0.001,
) -> PeakDetector:
    """
    Create a simple threshold-based peak detector.

    Args:
        threshold: Threshold value
        threshold_mode: Threshold interpretation mode
        polarity: Which polarity to detect
        refractory_period: Minimum time between detections

    Returns:
        Configured PeakDetector instance
    """
    return PeakDetector(
        threshold=threshold,
        threshold_mode=threshold_mode,
        polarity=polarity,
        refractory_period=refractory_period,
        align_to_peak=False,  # Just threshold crossing, not peak alignment
    )


def create_negative_peak_detector(
    threshold: float = 4.0,
    refractory_period: float = 0.001,
    threshold_mode: Literal["absolute", "std", "mad", "rms"] = "mad",
) -> PeakDetector:
    """
    Create a detector optimized for negative spikes (common in extracellular recordings).

    Args:
        threshold: MAD threshold multiplier
        refractory_period: Minimum time between spikes
        threshold_mode: How to interpret the threshold
            "absolute": Use raw value
            "std": Multiple of signal standard deviation
            "mad": Multiple of median absolute deviation (more robust)
            "rms": Multiple of root mean square

    Returns:
        Configured PeakDetector instance
    """
    return PeakDetector(
        threshold=threshold,
        threshold_mode=threshold_mode,
        polarity="negative",
        refractory_period=refractory_period,
        align_to_peak=True,
        noise_estimation_kwargs={"relative_start": 0.0, "relative_stop": 0.2},
    )


def create_positive_peak_detector(
    threshold: float = 4.0,
    refractory_period: float = 0.001,
    threshold_mode: Literal["absolute", "std", "mad", "rms"] = "mad",
) -> PeakDetector:
    """
    Create a detector optimized for positive spikes.

    Args:
        threshold: MAD threshold multiplier
        refractory_period: Minimum time between spikes
        threshold_mode: How to interpret the threshold
            "absolute": Use raw value
            "std": Multiple of signal standard deviation
            "mad": Multiple of median absolute deviation (more robust)
            "rms": Multiple of root mean square

    Returns:
        Configured PeakDetector instance
    """
    return PeakDetector(
        threshold=threshold,
        threshold_mode=threshold_mode,
        polarity="positive",
        refractory_period=refractory_period,
        align_to_peak=True,
        noise_estimation_kwargs={"relative_start": 0.0, "relative_stop": 0.2},
    )


def create_bipolar_peak_detector(
    threshold: float = 4.0,
    refractory_period: float = 0.001,
    threshold_mode: Literal["absolute", "std", "mad", "rms"] = "mad",
) -> PeakDetector:
    """
    Create a detector that finds both positive and negative peaks.

    Args:
        threshold: MAD threshold multiplier
        refractory_period: Minimum time between spikes
        threshold_mode: How to interpret the threshold
            "absolute": Use raw value
            "std": Multiple of signal standard deviation
            "mad": Multiple of median absolute deviation (more robust)
            "rms": Multiple of root mean square

    Returns:
        Configured PeakDetector instance
    """
    return PeakDetector(
        threshold=threshold,
        threshold_mode=threshold_mode,
        polarity="both",
        refractory_period=refractory_period,
        align_to_peak=True,
        noise_estimation_kwargs={"relative_start": 0.0, "relative_stop": 0.2},
    )
