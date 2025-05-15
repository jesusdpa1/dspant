"""
Template-based detection using convolution.

This module provides functionality for detecting signal patterns by matching
a template against a signal using convolution. It is useful for identifying
specific waveforms like QRS complexes in ECG or action potentials in neural data.
"""

from typing import Optional, Tuple

import dask.array as da
import numpy as np
from numba import jit, prange

from dspant.pattern.detection.base import BaseDetector

try:
    from dspant._rs import apply_template_matching

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@jit(nopython=True, cache=True)
def _normalize_template(template: np.ndarray) -> np.ndarray:
    """
    Normalize template for correlation.

    Args:
        template: Template to normalize

    Returns:
        Normalized template
    """
    # Handle multi-channel templates
    if template.ndim == 1:
        template = template.reshape(-1, 1)

    n_samples, n_channels = template.shape
    normalized = np.zeros_like(template)

    # Normalize each channel
    for ch in range(n_channels):
        channel_data = template[:, ch]
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)

        # Add small constant to avoid division by zero
        normalized[:, ch] = (channel_data - mean_val) / (std_val + 1e-10)

    return normalized


@jit(nopython=True, cache=True)
def _sliding_normalize(
    data: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rolling mean and standard deviation.

    Args:
        data: Data to normalize
        window_size: Size of sliding window

    Returns:
        Tuple of (rolling_mean, rolling_std)
    """
    n_samples = len(data)
    rolling_mean = np.zeros_like(data)
    rolling_std = np.zeros_like(data)

    # Calculate initial window
    half_win = window_size // 2

    # Calculate rolling statistics with center alignment
    for i in range(n_samples - window_size + 1):
        window = data[i : i + window_size]
        center_idx = i + half_win
        if center_idx < n_samples:
            rolling_mean[center_idx] = np.mean(window)
            rolling_std[center_idx] = np.std(window)

    # Fill edges with nearest valid values
    for i in range(half_win):
        rolling_mean[i] = rolling_mean[half_win]
        rolling_std[i] = rolling_std[half_win]

    for i in range(n_samples - half_win, n_samples):
        rolling_mean[i] = rolling_mean[n_samples - half_win - 1]
        rolling_std[i] = rolling_std[n_samples - half_win - 1]

    return rolling_mean, rolling_std


@jit(nopython=True, cache=True)
def _normalized_correlation_single_channel(
    data: np.ndarray, template: np.ndarray, window_size: int
) -> np.ndarray:
    """
    Calculate normalized correlation for a single channel.

    Args:
        data: 1D data array
        template: 1D template array
        window_size: Size of template

    Returns:
        Correlation values
    """
    n_samples = len(data)
    result = np.zeros(n_samples)

    # Get mean and std of template
    template_mean = np.mean(template)
    template_std = np.std(template)
    if template_std == 0:
        template_std = 1

    # Normalize template
    norm_template = (template - template_mean) / template_std

    # Get rolling mean and std for data
    data_mean, data_std = _sliding_normalize(data, window_size)

    # Add small constant to avoid division by zero
    data_std += 1e-10

    # Pre-normalize data
    norm_data = (data - data_mean) / data_std

    # Direct correlation using sliding dot product
    # This is faster than scipy.signal.correlate for this specific case
    for i in range(n_samples - window_size + 1):
        # Calculate dot product manually
        dot_product = 0.0
        for j in range(window_size):
            dot_product += norm_data[i + j] * norm_template[j]

        # Normalize by window size
        result[i + window_size // 2] = dot_product / window_size

    return result


@jit(nopython=True, parallel=True, cache=True)
def _normalized_correlation_multi_channel(
    data: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """
    Calculate normalized correlation for multi-channel data.

    Args:
        data: 2D data array (samples x channels)
        template: 2D template array (samples x channels)

    Returns:
        Correlation values averaged across channels
    """
    n_samples, n_channels = data.shape
    template_samples = template.shape[0]

    # Initialize correlation matrix
    correlations = np.zeros((n_samples, n_channels))

    # Process each channel in parallel
    for ch in prange(n_channels):
        # Get channel data
        data_ch = data[:, ch]
        template_ch = template[:, min(ch, template.shape[1] - 1)]

        # Calculate normalized correlation
        correlations[:, ch] = _normalized_correlation_single_channel(
            data_ch, template_ch, template_samples
        )

    # Average across channels
    result = np.mean(correlations, axis=1)

    return result


@jit(nopython=True, cache=True)
def _find_correlation_peaks(
    correlation: np.ndarray, threshold: float, min_distance: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in correlation signal.

    Args:
        correlation: Correlation values
        threshold: Threshold for detection
        min_distance: Minimum distance between peaks

    Returns:
        Tuple of (peak_indices, peak_values)
    """
    n_samples = len(correlation)
    # Pre-allocate with maximum possible number of peaks
    max_peaks = n_samples // min_distance + 1
    peak_indices = np.zeros(max_peaks, dtype=np.int64)
    peak_values = np.zeros(max_peaks, dtype=np.float64)

    # Initial peak count
    peak_count = 0

    # Find peaks by scanning the correlation signal
    for i in range(1, n_samples - 1):
        # Check if higher than neighbors
        is_peak = (
            correlation[i] > correlation[i - 1]
            and correlation[i] > correlation[i + 1]
            and correlation[i] > threshold
        )

        if is_peak:
            # Check minimum distance from previous peak
            is_valid = True
            for j in range(peak_count):
                if abs(i - peak_indices[j]) < min_distance:
                    # If we find a higher peak within min_distance, replace it
                    if correlation[i] > peak_values[j]:
                        peak_indices[j] = i
                        peak_values[j] = correlation[i]
                    is_valid = False
                    break

            # Add new peak if it's valid
            if is_valid:
                if peak_count < max_peaks:
                    peak_indices[peak_count] = i
                    peak_values[peak_count] = correlation[i]
                    peak_count += 1

    # Return actual peaks found
    return peak_indices[:peak_count], peak_values[:peak_count]


@jit(nopython=True, cache=True)
def _detect_template_matches_numba(
    data: np.ndarray,
    template: np.ndarray,
    threshold: float,
    min_distance: int,
    use_normalized_correlation: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect template matches using Numba-accelerated implementation.

    Args:
        data: Input data (samples x channels)
        template: Template to match
        threshold: Correlation threshold
        min_distance: Minimum distance between detections
        use_normalized_correlation: Whether to use normalized correlation

    Returns:
        Tuple of (match_indices, match_correlations, match_channels)
    """
    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Ensure template is 2D
    if template.ndim == 1:
        template = template.reshape(-1, 1)

    n_samples, n_channels = data.shape

    # Calculate correlation
    if use_normalized_correlation:
        correlation = _normalized_correlation_multi_channel(data, template)
    else:
        # Simple non-normalized correlation - placeholder for simplicity
        # In practice, this would be more complex
        correlation = np.zeros(n_samples)
        for ch in range(min(n_channels, template.shape[1])):
            for i in range(n_samples - template.shape[0] + 1):
                dot_product = 0.0
                for j in range(template.shape[0]):
                    dot_product += data[i + j, ch] * template[j, ch]
                correlation[i + template.shape[0] // 2] += dot_product
        correlation /= n_channels * template.shape[0]

    # Find peaks
    match_indices, match_correlations = _find_correlation_peaks(
        correlation, threshold, min_distance
    )

    # Set channel to 0 for all matches (multi-channel matching is averaged)
    match_channels = np.zeros_like(match_indices)

    return match_indices, match_correlations, match_channels


class TemplateDetector(BaseDetector):
    """
    Detect patterns in signals by template matching using convolution.

    This detector uses normalized cross-correlation to identify where
    a template pattern appears in a signal, regardless of amplitude scaling.
    """

    def __init__(
        self,
        template: Optional[np.ndarray] = None,
        threshold: float = 0.6,
        min_distance: float = 0.25,  # Seconds
        use_normalized_correlation: bool = True,
        use_rust: bool = True,
        threshold_mode: str = "absolute",
    ):
        """
        Initialize the template detector.

        Args:
            template: Template pattern to detect (if None, must be set later)
            threshold: Correlation threshold for detection
            min_distance: Minimum distance between detections in seconds
            use_normalized_correlation: Whether to use amplitude-invariant matching
            use_rust: Whether to use Rust acceleration if available
            threshold_mode: How to interpret the threshold
                "absolute": Raw correlation value (0-1 for normalized)
                "relative": Multiple of signal standard deviation
        """
        super().__init__()
        self.template = template
        self.threshold = threshold
        self.min_distance_sec = min_distance
        self.use_normalized_correlation = use_normalized_correlation
        self.use_rust = use_rust and _HAS_RUST
        self.threshold_mode = threshold_mode

        # Will be set during processing
        self._min_distance_samples = None
        self._overlap_samples = None
        self._fs = None

        # Initialize template properties
        if template is not None:
            self._set_template_properties()

    def _set_template_properties(self):
        """Set internal properties based on template"""
        if self.template is None:
            return

        # Determine template dimensions
        if self.template.ndim == 1:
            self._template_samples = len(self.template)
            self._template_channels = 1
            # Reshape to 2D for consistent handling
            self.template = self.template.reshape(-1, 1)
        else:
            self._template_samples, self._template_channels = self.template.shape

        # Normalize template for consistent correlation
        if self.use_normalized_correlation:
            # Use Numba-accelerated function
            self._normalized_template = _normalize_template(self.template)
        else:
            self._normalized_template = self.template

    def set_template(self, template: np.ndarray):
        """
        Set or update the template.

        Args:
            template: New template to use for detection
        """
        self.template = template
        self._set_template_properties()

    def detect(self, data: da.Array, fs: float, **kwargs) -> da.Array:
        """
        Detect template matches in the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency in Hz
            **kwargs: Additional keyword arguments
                template: Optional template override for this detection
                threshold: Optional threshold override
                min_distance: Optional distance override

        Returns:
            Dask array with detection results
        """
        # Check for template override
        if "template" in kwargs:
            self.set_template(kwargs["template"])

        # Ensure we have a template
        if self.template is None:
            raise ValueError("No template provided for detection")

        # Update parameters if provided
        threshold = kwargs.get("threshold", self.threshold)
        min_distance = kwargs.get("min_distance", self.min_distance_sec)

        # Calculate distance in samples
        self._min_distance_samples = max(int(min_distance * fs), 1)
        self._fs = fs

        # Set overlap samples based on template size and minimum distance
        self._overlap_samples = max(self._template_samples, self._min_distance_samples)

        # Define struct dtype for consistent output
        output_dtype = np.dtype(
            [
                ("index", np.int64),
                ("correlation", np.float32),
                ("channel", np.int32),
            ]
        )

        # Process data
        if self.use_rust and _HAS_RUST:
            # Use Rust implementation if available
            # This would be implemented in the Rust extension
            # Placeholder for now
            raise NotImplementedError("Rust implementation not yet available")
        else:
            # Python implementation with Numba acceleration
            # Define processing function for chunks
            def detect_templates_in_chunk(chunk, chunk_offset=0):
                """Process a single chunk of data"""
                # Ensure contiguous array for Numba
                chunk = np.ascontiguousarray(chunk)

                # Calculate adaptive threshold if needed
                if self.threshold_mode == "relative":
                    # We'll need to compute this before detection
                    computed_threshold = np.std(chunk) * threshold
                else:
                    computed_threshold = threshold

                # Use Numba-accelerated function
                match_indices, match_correlations, match_channels = (
                    _detect_template_matches_numba(
                        chunk,
                        self._normalized_template,
                        computed_threshold,
                        self._min_distance_samples,
                        self.use_normalized_correlation,
                    )
                )

                # Apply offset to indices
                match_indices = match_indices + chunk_offset

                # Create structured array for output
                result = np.zeros(len(match_indices), dtype=output_dtype)
                result["index"] = match_indices
                result["correlation"] = match_correlations
                result["channel"] = match_channels

                return result

            # Create empty array with right structure for meta
            empty_meta = np.array([], dtype=output_dtype)

            # Process with overlap
            result = data.map_overlap(
                detect_templates_in_chunk,
                depth=self._overlap_samples,
                boundary="reflect",
                dtype=output_dtype,
                drop_axis=0,  # Drop time dimension
                meta=empty_meta,
            )

            # Store detection stats
            self._detection_stats = {
                "fs": fs,
                "threshold": threshold,
                "min_distance_samples": self._min_distance_samples,
                "overlap_samples": self._overlap_samples,
                "template_samples": self._template_samples,
                "template_channels": self._template_channels,
            }

            return result
