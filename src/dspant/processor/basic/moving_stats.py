from typing import Any, Dict, Literal, Optional, Union

import dask.array as da
import numpy as np
from numba import jit

from ...engine.base import BaseProcessor


class MovingAverageProcessor(BaseProcessor):
    """
    Moving average processor implementation with Numba acceleration.

    Applies a moving average (simple, weighted, or exponential) to signals.
    """

    def __init__(
        self,
        window_size: int = 11,
        method: Literal["simple", "weighted", "exponential"] = "simple",
        weights: Optional[np.ndarray] = None,
        alpha: float = 0.3,
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
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha

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
        if self.method == "simple":
            return data.map_overlap(
                _apply_simple_moving_average,
                depth=(self._overlap_samples, 0),
                boundary="reflect",
                window_size=self.window_size,
                dtype=data.dtype,
            )
        elif self.method == "weighted":
            return data.map_overlap(
                _apply_weighted_moving_average,
                depth=(self._overlap_samples, 0),
                boundary="reflect",
                weights=self.weights,
                dtype=data.dtype,
            )
        elif self.method == "exponential":
            return data.map_blocks(
                _apply_exponential_moving_average, alpha=self.alpha, dtype=data.dtype
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
            }
        )
        return base_summary


class MovingRMSProcessor(BaseProcessor):
    """
    Moving Root Mean Square (RMS) processor implementation with Numba acceleration.

    Calculates the RMS value over a sliding window.
    """

    def __init__(self, window_size: int = 11, center: bool = True):
        """
        Initialize the moving RMS processor.

        Args:
            window_size: Size of the moving window
            center: If True, window is centered on each point
                   If False, window includes only past samples
        """
        self.window_size = window_size
        self.center = center

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
        return data.map_overlap(
            _apply_moving_rms,
            depth=(self._overlap_samples, 0),
            boundary="reflect",
            window_size=self.window_size,
            center=self.center,
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
            }
        )
        return base_summary


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
            # If at the start, use weights from the end of the window
            # If at the end, use weights from the start of the window
            if i < half_window:
                actual_weights = weights[-(window_end - window_start) :]
            elif i >= n_samples - half_window:
                actual_weights = weights[: window_end - window_start]
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
                    actual_weights = weights[: window_end - window_start]
                else:
                    actual_weights = weights

                # Normalize weights to sum to 1
                norm_weights = actual_weights / np.sum(actual_weights)

                # Calculate weighted average
                result[i, c] = np.sum(actual_window * norm_weights)

    return result


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
