"""
Bayesian changepoint detection algorithms for EMG activity detection.

This module provides methods to detect the onset of muscle activity in EMG signals
using Bayesian changepoint detection, which is particularly effective for detecting
subtle changes in signal characteristics.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit, prange
from scipy import stats

from dspant.engine.base import BaseProcessor


@jit(nopython=True, cache=True)
def _bayesian_offline_changepoint_detection(
    data: np.ndarray,
    hazard_function: float,
    observation_likelihood: str = "normal_meanvar",
    max_segments: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Bayesian offline changepoint detection with Numba acceleration.

    This implements the algorithm from "Bayesian Online Changepoint Detection"
    by Adams & MacKay (2007) but adapted for offline use.

    Args:
        data: 1D input signal
        hazard_function: Constant hazard/probability of changepoint (1/expected_run_length)
        observation_likelihood: Statistical model for the data ("normal_meanvar" or "normal_mean")
        max_segments: Maximum number of segments to detect

    Returns:
        Tuple of (probabilities, log_messages)
    """
    # Handle empty data
    if len(data) == 0:
        return np.array([]), np.array([])

    # Length of data
    n_samples = len(data)

    # Default max_segments
    if max_segments is None:
        max_segments = n_samples

    # Initialize matrices for probabilities and messages
    probs = np.zeros((n_samples, n_samples))
    log_msgs = np.zeros((n_samples, n_samples))

    # Initialize message using noninformative prior
    log_msgs[0, 0] = 0
    probs[0, 0] = 1

    # For statistical calculations
    t_values = np.arange(1, n_samples)

    # For all data points
    for t in range(1, n_samples):
        # Compute growth probabilities
        for i in range(t):
            if observation_likelihood == "normal_meanvar":
                # Calculate sufficient statistics
                mean_t = np.sum(data[i : t + 1]) / (t - i + 1)
                var_t = np.sum((data[i : t + 1] - mean_t) ** 2) / (t - i + 1)
                if var_t < 1e-8:  # Avoid numerical issues
                    var_t = 1e-8

                # Calculate log-likelihood for normal distribution with mean and variance estimation
                log_likelihood = (
                    -0.5 * (t - i + 1) * (np.log(2 * np.pi) + np.log(var_t))
                    - 0.5 * np.sum((data[i : t + 1] - mean_t) ** 2) / var_t
                )

            elif observation_likelihood == "normal_mean":
                # Fixed variance model - simpler calculation
                mean_t = np.sum(data[i : t + 1]) / (t - i + 1)
                var_t = 1.0  # Fixed variance

                # Calculate log-likelihood for normal distribution with mean estimation only
                log_likelihood = -0.5 * (t - i + 1) * np.log(2 * np.pi) - 0.5 * np.sum(
                    (data[i : t + 1] - mean_t) ** 2
                )
            else:
                # Default to normal_meanvar
                mean_t = np.sum(data[i : t + 1]) / (t - i + 1)
                var_t = np.sum((data[i : t + 1] - mean_t) ** 2) / (t - i + 1)
                if var_t < 1e-8:
                    var_t = 1e-8

                log_likelihood = (
                    -0.5 * (t - i + 1) * (np.log(2 * np.pi) + np.log(var_t))
                    - 0.5 * np.sum((data[i : t + 1] - mean_t) ** 2) / var_t
                )

            # Calculate growth probability
            log_msgs[t, t - i] = (
                log_likelihood + np.log(1 - hazard_function) + log_msgs[i, 0]
            )

        # Compute changepoint probabilities
        for i in range(t):
            log_msgs[t, 0] = np.logaddexp(
                log_msgs[t, 0], log_msgs[i, 0] + np.log(hazard_function)
            )

        # Normalize
        log_norm = np.logaddexp(log_msgs[t, 0], np.sum(log_msgs[t, 1 : t + 1]))
        log_msgs[t, : t + 1] = log_msgs[t, : t + 1] - log_norm

        # Convert log messages to probabilities
        probs[t, : t + 1] = np.exp(log_msgs[t, : t + 1])

    return probs, log_msgs


@jit(nopython=True, cache=True)
def _detect_changepoints_from_probabilities(
    probs: np.ndarray,
    threshold: float = 0.3,
    min_distance: int = 10,
) -> np.ndarray:
    """
    Detect changepoints from the Bayesian changepoint probabilities.

    Args:
        probs: Probability matrix from Bayesian changepoint detection
        threshold: Probability threshold for declaring a changepoint
        min_distance: Minimum distance between changepoints

    Returns:
        Array of changepoint indices
    """
    n_samples = probs.shape[0]

    # Calculate the run length probabilities
    run_length_probs = np.zeros(n_samples)
    for t in range(n_samples):
        run_length_probs[t] = probs[t, 0]  # Probability of a changepoint at t

    # Find peaks in the changepoint probability
    changepoints = []
    for t in range(1, n_samples - 1):
        if (
            run_length_probs[t] > threshold
            and run_length_probs[t] > run_length_probs[t - 1]
            and run_length_probs[t] >= run_length_probs[t + 1]
        ):
            # Check if this changepoint is far enough from the previous one
            if not changepoints or t - changepoints[-1] >= min_distance:
                changepoints.append(t)

    return np.array(changepoints)


class BayesianChangepointDetector(BaseProcessor):
    """
    EMG activity detection processor using Bayesian changepoint detection.

    This processor detects changes in EMG signal characteristics using Bayesian
    changepoint detection methods, which can identify subtle changes in signal
    properties that might be missed by threshold-based approaches.
    """

    def __init__(
        self,
        hazard_prob: float = 0.01,
        observation_model: Literal["normal_meanvar", "normal_mean"] = "normal_meanvar",
        prob_threshold: float = 0.3,
        min_distance_s: float = 0.1,  # seconds
        min_duration: float = 0.01,  # seconds
        max_segments: Optional[int] = None,
        post_process: bool = True,
    ):
        """
        Initialize the Bayesian changepoint detector.

        Args:
            hazard_prob: Hazard function value (probability of a changepoint at each point)
            observation_model: Statistical model for the observations
                "normal_meanvar": Normal distribution with unknown mean and variance
                "normal_mean": Normal distribution with unknown mean, fixed variance
            prob_threshold: Probability threshold for declaring a changepoint
            min_distance_s: Minimum distance between changepoints in seconds
            min_duration: Minimum duration for a valid activation in seconds
            max_segments: Maximum number of segments to detect
            post_process: Whether to apply post-processing to refine changepoints
        """
        self.hazard_prob = hazard_prob
        self.observation_model = observation_model
        self.prob_threshold = prob_threshold
        self.min_distance_s = min_distance_s
        self.min_duration = min_duration
        self.max_segments = max_segments
        self.post_process = post_process

        # Set large overlap to ensure accurate detection at chunk boundaries
        self._overlap_samples = 200  # This will be adjusted based on sampling frequency

        # Define dtype for output
        self._dtype = np.dtype(
            [
                ("onset_idx", np.int64),
                ("offset_idx", np.int64),
                ("channel", np.int32),
                ("probability", np.float32),
                ("duration", np.float32),
            ]
        )

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Detect EMG activity using Bayesian changepoint detection.

        Args:
            data: Input dask array
            fs: Sampling frequency (required)
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with detected activity events
        """
        if fs is None:
            raise ValueError(
                "Sampling frequency (fs) is required for changepoint detection"
            )

        # Convert time parameters to samples
        min_distance_samples = int(self.min_distance_s * fs)
        min_duration_samples = int(self.min_duration * fs)

        # Adjust overlap based on sampling frequency
        self._overlap_samples = max(
            int(0.5 * fs), 200
        )  # At least 0.5 seconds or 200 samples

        def detect_changepoints_chunk(chunk: np.ndarray, block_info=None) -> np.ndarray:
            """Process a chunk of data to detect changepoints"""
            # Get chunk offset from block_info
            chunk_offset = 0
            if block_info and len(block_info) > 0:
                chunk_offset = block_info[0]["array-location"][0][0]

            # Ensure the input is a contiguous array
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case by adding extra dimension if needed
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]

            # Create result array for all channels
            all_results = []

            # Process each channel
            for channel_ind in range(chunk.shape[1]):
                channel_data = chunk[:, channel_ind]

                # Run Bayesian changepoint detection
                probs, _ = _bayesian_offline_changepoint_detection(
                    channel_data,
                    self.hazard_prob,
                    self.observation_model,
                    self.max_segments,
                )

                # Detect changepoints
                changepoints = _detect_changepoints_from_probabilities(
                    probs, self.prob_threshold, min_distance_samples
                )

                # Skip if no changepoints
                if len(changepoints) < 2:
                    continue

                # Group changepoints into onset-offset pairs
                # For EMG, we typically expect a pair of changepoints for each activity
                # (onset followed by offset)
                for i in range(0, len(changepoints) - 1, 2):
                    onset_idx = changepoints[i]
                    offset_idx = (
                        changepoints[i + 1]
                        if i + 1 < len(changepoints)
                        else len(channel_data) - 1
                    )

                    # Calculate duration in samples
                    duration_samples = offset_idx - onset_idx

                    # Skip if duration is too short
                    if duration_samples < min_duration_samples:
                        continue

                    # Calculate duration in seconds and probability
                    duration_sec = duration_samples / fs
                    probability = probs[onset_idx, 0]  # Probability at onset point

                    # Create result for this event
                    result = np.zeros(1, dtype=self._dtype)
                    result["onset_idx"] = onset_idx + chunk_offset
                    result["offset_idx"] = offset_idx + chunk_offset
                    result["channel"] = channel_ind
                    result["probability"] = probability
                    result["duration"] = duration_sec

                    all_results.append(result)

            # Combine all results
            if not all_results:
                return np.array([], dtype=self._dtype)

            return np.concatenate(all_results)

        # Ensure input is 2D
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Use map_overlap with explicit boundary handling
        result = data.map_overlap(
            detect_changepoints_chunk,
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
            Polars DataFrame with events
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
                "hazard_prob": self.hazard_prob,
                "observation_model": self.observation_model,
                "prob_threshold": self.prob_threshold,
                "min_distance_s": self.min_distance_s,
                "min_duration": self.min_duration,
                "accelerated": True,
            }
        )
        return base_summary


# Factory functions for common configurations


def create_sensitive_changepoint_detector(
    min_duration: float = 0.01,
    min_distance_s: float = 0.1,
) -> BayesianChangepointDetector:
    """
    Create a Bayesian changepoint detector that is sensitive to subtle EMG activity.

    This configuration uses a lower hazard probability and threshold,
    making it more sensitive to small changes in the signal.

    Args:
        min_duration: Minimum duration for a valid activation in seconds
        min_distance_s: Minimum distance between changepoints in seconds

    Returns:
        Configured BayesianChangepointDetector
    """
    return BayesianChangepointDetector(
        hazard_prob=0.005,  # Lower hazard probability
        observation_model="normal_meanvar",
        prob_threshold=0.2,  # Lower threshold for changepoint detection
        min_distance_s=min_distance_s,
        min_duration=min_duration,
        post_process=True,
    )


def create_robust_changepoint_detector(
    min_duration: float = 0.03,
    min_distance_s: float = 0.2,
) -> BayesianChangepointDetector:
    """
    Create a Bayesian changepoint detector that is robust to noise.

    This configuration uses a higher hazard probability and threshold,
    making it less likely to detect false positives in noisy signals.

    Args:
        min_duration: Minimum duration for a valid activation in seconds
        min_distance_s: Minimum distance between changepoints in seconds

    Returns:
        Configured BayesianChangepointDetector
    """
    return BayesianChangepointDetector(
        hazard_prob=0.02,  # Higher hazard probability
        observation_model="normal_meanvar",
        prob_threshold=0.4,  # Higher threshold for changepoint detection
        min_distance_s=min_distance_s,
        min_duration=min_duration,
        post_process=True,
    )


def create_realtime_changepoint_detector(
    min_duration: float = 0.02,
) -> BayesianChangepointDetector:
    """
    Create a Bayesian changepoint detector optimized for near-real-time processing.

    This configuration uses parameters that balance sensitivity with computational efficiency.

    Args:
        min_duration: Minimum duration for a valid activation in seconds

    Returns:
        Configured BayesianChangepointDetector
    """
    return BayesianChangepointDetector(
        hazard_prob=0.01,
        observation_model="normal_mean",  # Simpler model for faster computation
        prob_threshold=0.3,
        min_distance_s=0.1,
        min_duration=min_duration,
        post_process=False,  # Skip post-processing for speed
    )
