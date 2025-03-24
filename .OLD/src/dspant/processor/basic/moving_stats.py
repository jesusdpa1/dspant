import concurrent.futures
from typing import Any, Dict, Literal, Optional, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from ...engine.base import BaseProcessor


class MovingAverageProcessor(BaseProcessor):
    """
    Moving average processor implementation with Numba acceleration and parallelization.

    Applies a moving average (simple, weighted, or exponential) to signals.
    """

    def __init__(
        self,
        window_size: int = 11,
        method: Literal["simple", "weighted", "exponential"] = "simple",
        weights: Optional[np.ndarray] = None,
        alpha: float = 0.3,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the moving average processor.

        Args:
            window_size: Size of the moving window for simple and weighted methods
            method: Moving average method to use
                "simple": Equal weight to all samples in window
                "weighted": Customizable or triangular weighting
                "exponential": Exponential weighting (uses alpha, not window_size)
            weights: Optional custom weights for weighted moving average
                     If None and method is "weighted", uses triangular weights
            alpha: Smoothing factor for exponential moving average (0 < alpha < 1)
                   Higher values give more weight to recent observations
            max_workers: Maximum number of worker threads for parallel processing
                         If None, uses the default ThreadPoolExecutor behavior
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        self.max_workers = max_workers

        # Validate window size
        if self.window_size < 1:
            raise ValueError("Window size must be at least 1")

        # Set up weights for weighted moving average
        if self.method == "weighted":
            if weights is not None:
                # Use provided weights
                self.weights = np.array(weights, dtype=np.float32)
                # Normalize weights to sum to 1
                self.weights = self.weights / np.sum(self.weights)
                # Update window size to match weights length
                self.window_size = len(self.weights)
            else:
                # Default to triangular weights
                self.weights = np.linspace(1, self.window_size, self.window_size)
                # Normalize weights to sum to 1
                self.weights = self.weights / np.sum(self.weights)
        else:
            self.weights = None

        # Validate alpha for exponential moving average
        if self.method == "exponential" and (self.alpha <= 0 or self.alpha >= 1):
            raise ValueError("Alpha must be between 0 and 1 exclusive")

        # Set overlap for map_overlap (need full window size - 1 samples)
        self._overlap_samples = window_size - 1 if self.method != "exponential" else 0

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply moving average to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments

        Returns:
            Data array with moving average applied
        """
        # Set max_workers from kwargs if provided
        max_workers = kwargs.get("max_workers", self.max_workers)
        use_parallel = (
            kwargs.get("parallel", True) and data.ndim > 1 and data.shape[1] > 1
        )

        if self.method == "simple":
            return data.map_overlap(
                _apply_simple_moving_average_parallel
                if use_parallel
                else _apply_simple_moving_average,
                depth=(self._overlap_samples, 0),
                boundary="reflect",
                window_size=self.window_size,
                max_workers=max_workers,
                dtype=data.dtype,
            )
        elif self.method == "weighted":
            return data.map_overlap(
                _apply_weighted_moving_average_parallel
                if use_parallel
                else _apply_weighted_moving_average,
                depth=(self._overlap_samples, 0),
                boundary="reflect",
                weights=self.weights,
                max_workers=max_workers,
                dtype=data.dtype,
            )
        elif self.method == "exponential":
            return data.map_blocks(
                _apply_exponential_moving_average_parallel
                if use_parallel
                else _apply_exponential_moving_average,
                alpha=self.alpha,
                max_workers=max_workers,
                dtype=data.dtype,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

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
                "method": self.method,
                "window_size": self.window_size
                if self.method != "exponential"
                else "N/A",
                "alpha": self.alpha if self.method == "exponential" else "N/A",
                "accelerated": True,
                "parallel": self.max_workers is not None,
            }
        )
        return base_summary


class MovingRMSProcessor(BaseProcessor):
    """
    Moving Root Mean Square (RMS) processor implementation with Numba acceleration.

    Calculates the RMS value over a sliding window.
    """

    def __init__(
        self,
        window_size: int = 11,
        center: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the moving RMS processor.

        Args:
            window_size: Size of the moving window
            center: If True, window is centered on each point
                   If False, window includes only past samples
            max_workers: Maximum number of worker threads for parallel processing
                         If None, uses the default ThreadPoolExecutor behavior
        """
        self.window_size = window_size
        self.center = center
        self.max_workers = max_workers

        # Validate window size
        if self.window_size < 1:
            raise ValueError("Window size must be at least 1")

        # Set overlap for map_overlap
        self._overlap_samples = window_size - 1

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply moving RMS to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments

        Returns:
            Data array with moving RMS applied
        """
        # Set max_workers from kwargs if provided
        max_workers = kwargs.get("max_workers", self.max_workers)
        use_parallel = (
            kwargs.get("parallel", True) and data.ndim > 1 and data.shape[1] > 1
        )

        return data.map_overlap(
            _apply_moving_rms_parallel if use_parallel else _apply_moving_rms,
            depth=(self._overlap_samples, 0),
            boundary="reflect",
            window_size=self.window_size,
            center=self.center,
            max_workers=max_workers,
            dtype=data.dtype,
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
                "window_size": self.window_size,
                "center": self.center,
                "accelerated": True,
                "parallel": self.max_workers is not None,
            }
        )
        return base_summary


# Optimized version using numba's prange for parallelism within a chunk
@jit(nopython=True, parallel=True, cache=True)
def _apply_simple_moving_average_parallel(
    data: np.ndarray, window_size: int, max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Apply simple moving average with equal weights using parallel processing.

    Args:
        data: Input array
        window_size: Size of moving window
        max_workers: Maximum number of workers (unused in numba version but kept for API compatibility)

    Returns:
        Smoothed array
    """
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        for i in range(n_samples):
            # Determine window boundaries
            window_start = max(0, i - (window_size // 2))
            window_end = min(n_samples, i + (window_size // 2) + 1)

            # Calculate mean for this window
            result[i] = np.mean(data[window_start:window_end])

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        # Parallel processing across channels
        for c in prange(n_channels):
            for i in range(n_samples):
                # Determine window boundaries
                window_start = max(0, i - (window_size // 2))
                window_end = min(n_samples, i + (window_size // 2) + 1)

                # Calculate mean for this window and channel
                result[i, c] = np.mean(data[window_start:window_end, c])

    return result


# Keep the original version for smaller data or when parallel processing is not needed
@jit(nopython=True, cache=True)
def _apply_simple_moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply simple moving average with equal weights.

    Args:
        data: Input array
        window_size: Size of moving window

    Returns:
        Smoothed array
    """
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        for i in range(n_samples):
            # Determine window boundaries
            window_start = max(0, i - (window_size // 2))
            window_end = min(n_samples, i + (window_size // 2) + 1)

            # Calculate mean for this window
            result[i] = np.mean(data[window_start:window_end])

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        for c in range(n_channels):
            for i in range(n_samples):
                # Determine window boundaries
                window_start = max(0, i - (window_size // 2))
                window_end = min(n_samples, i + (window_size // 2) + 1)

                # Calculate mean for this window and channel
                result[i, c] = np.mean(data[window_start:window_end, c])

    return result


# Parallel version for weighted moving average
@jit(nopython=True, parallel=True, cache=True)
def _apply_weighted_moving_average_parallel(
    data: np.ndarray, weights: np.ndarray, max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Apply weighted moving average with parallel processing.

    Args:
        data: Input array
        weights: Weight array
        max_workers: Maximum number of workers (unused in numba version but kept for API compatibility)

    Returns:
        Smoothed array
    """
    window_size = len(weights)
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        for i in range(n_samples):
            # Determine window boundaries
            half_window = window_size // 2
            window_start = max(0, i - half_window)
            window_end = min(n_samples, i + half_window + 1)

            # Get actual window size which may be smaller at boundaries
            actual_window = data[window_start:window_end]

            # Get matching weights
            if i < half_window:
                actual_weights = weights[-(window_end - window_start) :]
            elif i >= n_samples - half_window:
                actual_weights = weights[: (window_end - window_start)]
            else:
                actual_weights = weights

            # Normalize weights to sum to 1
            norm_weights = actual_weights / np.sum(actual_weights)

            # Calculate weighted average
            result[i] = np.sum(actual_window * norm_weights)

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        # Parallel processing across channels
        for c in prange(n_channels):
            for i in range(n_samples):
                # Determine window boundaries
                half_window = window_size // 2
                window_start = max(0, i - half_window)
                window_end = min(n_samples, i + half_window + 1)

                # Get actual window size which may be smaller at boundaries
                actual_window = data[window_start:window_end, c]

                # Get matching weights
                if i < half_window:
                    actual_weights = weights[-(window_end - window_start) :]
                elif i >= n_samples - half_window:
                    actual_weights = weights[: (window_end - window_start)]
                else:
                    actual_weights = weights

                # Normalize weights to sum to 1
                norm_weights = actual_weights / np.sum(actual_weights)

                # Calculate weighted average
                result[i, c] = np.sum(actual_window * norm_weights)

    return result


# Keep the original version
@jit(nopython=True, cache=True)
def _apply_weighted_moving_average(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Apply weighted moving average.

    Args:
        data: Input array
        weights: Weight array

    Returns:
        Smoothed array
    """
    window_size = len(weights)
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        for i in range(n_samples):
            # Determine window boundaries
            half_window = window_size // 2
            window_start = max(0, i - half_window)
            window_end = min(n_samples, i + half_window + 1)

            # Get actual window size which may be smaller at boundaries
            actual_window = data[window_start:window_end]

            # Get matching weights
            if i < half_window:
                actual_weights = weights[-(window_end - window_start) :]
            elif i >= n_samples - half_window:
                actual_weights = weights[: (window_end - window_start)]
            else:
                actual_weights = weights

            # Normalize weights to sum to 1
            norm_weights = actual_weights / np.sum(actual_weights)

            # Calculate weighted average
            result[i] = np.sum(actual_window * norm_weights)

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        for c in range(n_channels):
            for i in range(n_samples):
                # Determine window boundaries
                half_window = window_size // 2
                window_start = max(0, i - half_window)
                window_end = min(n_samples, i + half_window + 1)

                # Get actual window size which may be smaller at boundaries
                actual_window = data[window_start:window_end, c]

                # Get matching weights
                if i < half_window:
                    actual_weights = weights[-(window_end - window_start) :]
                elif i >= n_samples - half_window:
                    actual_weights = weights[: (window_end - window_start)]
                else:
                    actual_weights = weights

                # Normalize weights to sum to 1
                norm_weights = actual_weights / np.sum(actual_weights)

                # Calculate weighted average
                result[i, c] = np.sum(actual_window * norm_weights)

    return result


# Parallel version for exponential moving average
@jit(nopython=True, parallel=True, cache=True)
def _apply_exponential_moving_average_parallel(
    data: np.ndarray, alpha: float, max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Apply exponential moving average with parallel processing.

    Formula: y[t] = α * x[t] + (1-α) * y[t-1]

    Args:
        data: Input array
        alpha: Smoothing factor (0 < alpha < 1)
        max_workers: Maximum number of workers (unused in numba version but kept for API compatibility)

    Returns:
        Smoothed array
    """
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        # Initialize with first value
        result[0] = data[0]

        # Apply EMA formula
        for i in range(1, n_samples):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        # Initialize with first values
        result[0, :] = data[0, :]

        # Parallel processing across channels
        for c in prange(n_channels):
            for i in range(1, n_samples):
                result[i, c] = alpha * data[i, c] + (1 - alpha) * result[i - 1, c]

    return result


# Keep the original version
@jit(nopython=True, cache=True)
def _apply_exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential moving average.

    Formula: y[t] = α * x[t] + (1-α) * y[t-1]

    Args:
        data: Input array
        alpha: Smoothing factor (0 < alpha < 1)

    Returns:
        Smoothed array
    """
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        # Initialize with first value
        result[0] = data[0]

        # Apply EMA formula
        for i in range(1, n_samples):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        # Initialize with first values
        result[0, :] = data[0, :]

        # Apply EMA formula to each channel
        for c in range(n_channels):
            for i in range(1, n_samples):
                result[i, c] = alpha * data[i, c] + (1 - alpha) * result[i - 1, c]

    return result


# Parallel version for moving RMS
@jit(nopython=True, parallel=True, cache=True)
def _apply_moving_rms_parallel(
    data: np.ndarray, window_size: int, center: bool, max_workers: Optional[int] = None
) -> np.ndarray:
    """
    Apply moving RMS calculation with parallel processing.

    Args:
        data: Input array
        window_size: Size of moving window
        center: Whether to center the window
        max_workers: Maximum number of workers (unused in numba version but kept for API compatibility)

    Returns:
        RMS array
    """
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        for i in range(n_samples):
            if center:
                # Centered window
                half_window = window_size // 2
                window_start = max(0, i - half_window)
                window_end = min(n_samples, i + half_window + 1)
            else:
                # Past-only window
                window_start = max(0, i - window_size + 1)
                window_end = i + 1

            # Calculate RMS for this window
            window_data = data[window_start:window_end]
            result[i] = np.sqrt(np.mean(window_data**2))

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        # Parallel processing across channels
        for c in prange(n_channels):
            for i in range(n_samples):
                if center:
                    # Centered window
                    half_window = window_size // 2
                    window_start = max(0, i - half_window)
                    window_end = min(n_samples, i + half_window + 1)
                else:
                    # Past-only window
                    window_start = max(0, i - window_size + 1)
                    window_end = i + 1

                # Calculate RMS for this window and channel
                window_data = data[window_start:window_end, c]
                result[i, c] = np.sqrt(np.mean(window_data**2))

    return result


# Keep the original version
@jit(nopython=True, cache=True)
def _apply_moving_rms(data: np.ndarray, window_size: int, center: bool) -> np.ndarray:
    """
    Apply moving RMS calculation.

    Args:
        data: Input array
        window_size: Size of moving window
        center: Whether to center the window

    Returns:
        RMS array
    """
    result = np.empty_like(data)

    if data.ndim == 1:
        # Single channel case
        n_samples = data.shape[0]

        for i in range(n_samples):
            if center:
                # Centered window
                half_window = window_size // 2
                window_start = max(0, i - half_window)
                window_end = min(n_samples, i + half_window + 1)
            else:
                # Past-only window
                window_start = max(0, i - window_size + 1)
                window_end = i + 1

            # Calculate RMS for this window
            window_data = data[window_start:window_end]
            result[i] = np.sqrt(np.mean(window_data**2))

    else:
        # Multi-channel case
        n_samples = data.shape[0]
        n_channels = data.shape[1]

        for c in range(n_channels):
            for i in range(n_samples):
                if center:
                    # Centered window
                    half_window = window_size // 2
                    window_start = max(0, i - half_window)
                    window_end = min(n_samples, i + half_window + 1)
                else:
                    # Past-only window
                    window_start = max(0, i - window_size + 1)
                    window_end = i + 1

                # Calculate RMS for this window and channel
                window_data = data[window_start:window_end, c]
                result[i, c] = np.sqrt(np.mean(window_data**2))

    return result


# Additional version using ThreadPoolExecutor directly for cases where Numba's prange isn't optimal
def _apply_multichannel_processing(
    data: np.ndarray, channel_function, max_workers: Optional[int] = None, **kwargs
):
    """
    Apply a function to each channel in parallel using ThreadPoolExecutor.

    Args:
        data: Input array with shape (samples, channels)
        channel_function: Function to apply to each channel
        max_workers: Maximum number of worker threads
        **kwargs: Additional arguments to pass to channel_function

    Returns:
        Processed array
    """
    # Only use ThreadPoolExecutor for multi-channel data
    if data.ndim == 1 or data.shape[1] == 1:
        if data.ndim == 1:
            return channel_function(data, **kwargs)
        else:
            return channel_function(data[:, 0], **kwargs).reshape(-1, 1)

    n_channels = data.shape[1]
    result = np.empty_like(data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Submit jobs for each channel
        for c in range(n_channels):
            futures.append(executor.submit(channel_function, data[:, c], **kwargs))

        # Get results as they complete
        for c, future in enumerate(futures):
            result[:, c] = future.result()

    return result
