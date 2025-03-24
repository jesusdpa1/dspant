"""
Base classes and interfaces for spike detection algorithms.

This module provides abstract base classes and shared functionality
for different spike detection strategies in neural data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from ...engine.base import BaseProcessor


class BaseDetector(BaseProcessor):
    """
    Abstract base class for all spike/event detectors.

    This class defines the interface that all detector implementations
    should follow, with methods for detecting spikes/events and
    calculating detection statistics.
    """

    def __init__(self):
        self._detection_stats = {}
        self._overlap_samples = 0  # Subclasses should set appropriate overlap

    @abstractmethod
    def detect(self, data: da.Array, fs: float, **kwargs) -> da.Array:
        """
        Core detection algorithm to be implemented by subclasses.

        Args:
            data: Input signal data
            fs: Sampling frequency in Hz
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dask array with detected events that can be converted to a Polars DataFrame
        """
        pass

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to detect spikes/events.

        This implements the BaseProcessor interface, delegating to the
        detect() method for the actual detection algorithm.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with detection results
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for spike detection")

        return self.detect(data, fs, **kwargs)

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
    def detection_stats(self) -> Dict[str, Any]:
        """Get statistics from the last detection run"""
        return self._detection_stats

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of detector configuration"""
        base_summary = super().summary
        # Subclasses should extend this with detector-specific information
        return base_summary


class ThresholdDetector(BaseDetector):
    """
    Base class for threshold-based detectors.

    This class provides common functionality for detectors that use
    amplitude thresholds to identify events.
    """

    def __init__(
        self,
        threshold: float,
        threshold_mode: str = "absolute",
        polarity: str = "both",
        refractory_period: float = 0.001,  # 1ms default refractory period
    ):
        """
        Initialize the threshold detector.

        Args:
            threshold: Threshold value (interpretation depends on threshold_mode)
            threshold_mode: How to interpret the threshold
                "absolute": Use raw value
                "std": Multiple of signal standard deviation
                "mad": Multiple of median absolute deviation
                "rms": Multiple of root mean square
            polarity: Which threshold crossings to detect
                "positive": Only detect positive-going threshold crossings
                "negative": Only detect negative-going threshold crossings
                "both": Detect both positive and negative threshold crossings
            refractory_period: Minimum time between spikes in seconds
        """
        super().__init__()
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.polarity = polarity
        self.refractory_period = refractory_period

        # For tracking multiple thresholds (as in double threshold detection)
        self._thresholds = {
            "positive": None,
            "negative": None,
        }

        # Validation
        if self.threshold_mode not in ["absolute", "std", "mad", "rms"]:
            raise ValueError(
                f"Invalid threshold_mode: {self.threshold_mode}. "
                "Must be one of: 'absolute', 'std', 'mad', 'rms'"
            )

        if self.polarity not in ["positive", "negative", "both"]:
            raise ValueError(
                f"Invalid polarity: {self.polarity}. "
                "Must be one of: 'positive', 'negative', 'both'"
            )

    def compute_threshold(
        self, data: np.ndarray, fs: float, **kwargs
    ) -> Dict[str, float]:
        """
        Compute detection thresholds based on signal properties.

        Args:
            data: Input signal data
            fs: Sampling frequency in Hz
            **kwargs: Additional parameters

        Returns:
            Dictionary with positive and/or negative thresholds
        """
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_channels = data.shape[1]
        result = {"positive": [], "negative": []}

        for chan in range(n_channels):
            chan_data = data[:, chan]

            if self.threshold_mode == "absolute":
                pos_thresh = self.threshold
                neg_thresh = -self.threshold

            elif self.threshold_mode == "std":
                std_val = np.std(chan_data)
                mean_val = np.mean(chan_data)
                pos_thresh = mean_val + self.threshold * std_val
                neg_thresh = mean_val - self.threshold * std_val

            elif self.threshold_mode == "mad":
                median_val = np.median(chan_data)
                mad_val = np.median(np.abs(chan_data - median_val))
                # Scale MAD to approximate standard deviation
                mad_val = mad_val / 0.6744897501960817
                pos_thresh = median_val + self.threshold * mad_val
                neg_thresh = median_val - self.threshold * mad_val

            elif self.threshold_mode == "rms":
                rms_val = np.sqrt(np.mean(np.square(chan_data)))
                pos_thresh = self.threshold * rms_val
                neg_thresh = -self.threshold * rms_val

            result["positive"].append(pos_thresh)
            result["negative"].append(neg_thresh)

        # Convert to numpy arrays
        result["positive"] = np.array(result["positive"])
        result["negative"] = np.array(result["negative"])

        return result

    def apply_refractory_period(self, indices: np.ndarray, fs: float) -> np.ndarray:
        """
        Filter spike indices to respect the refractory period.

        Args:
            indices: Array of spike indices
            fs: Sampling frequency in Hz

        Returns:
            Filtered array of spike indices
        """
        if len(indices) <= 1:
            return indices

        # Sort indices (they should already be sorted, but just to be sure)
        sorted_indices = np.sort(indices)

        # Convert refractory period to samples
        refractory_samples = int(self.refractory_period * fs)

        # Initialize with the first spike
        filtered_indices = [sorted_indices[0]]

        # Filter subsequent spikes
        for idx in sorted_indices[1:]:
            if idx - filtered_indices[-1] >= refractory_samples:
                filtered_indices.append(idx)

        return np.array(filtered_indices)

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of threshold detector configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "threshold": self.threshold,
                "threshold_mode": self.threshold_mode,
                "polarity": self.polarity,
                "refractory_period": self.refractory_period,
            }
        )
        return base_summary


class DoubleThresholdDetector(ThresholdDetector):
    """
    Base class for double-threshold detectors.

    This class extends ThresholdDetector by implementing a double-threshold
    approach, where a second threshold must be crossed within a specified
    time window for an event to be detected.
    """

    def __init__(
        self,
        threshold1: float,
        threshold2: float,
        threshold_mode: str = "absolute",
        polarity: str = "both",
        refractory_period: float = 0.001,
        window_size: float = 0.001,  # 1ms default window
    ):
        """
        Initialize the double threshold detector.

        Args:
            threshold1: First (triggering) threshold value
            threshold2: Second (confirmation) threshold value
            threshold_mode: How to interpret the thresholds
            polarity: Which threshold crossings to detect
            refractory_period: Minimum time between spikes in seconds
            window_size: Time window in seconds for the second threshold crossing
        """
        super().__init__(
            threshold=threshold1,
            threshold_mode=threshold_mode,
            polarity=polarity,
            refractory_period=refractory_period,
        )
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.window_size = window_size

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of double threshold detector configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "threshold1": self.threshold1,
                "threshold2": self.threshold2,
                "window_size": self.window_size,
            }
        )
        return base_summary
