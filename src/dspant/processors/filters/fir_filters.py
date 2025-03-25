"""
FIR filter implementations with Rust acceleration.

This module provides high-performance implementations of various FIR filters
(Moving Average, Weighted Moving Average, RMS, Sinc, etc.) with Rust acceleration
for multi-channel data.
"""

from typing import Any, Dict, Optional

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor, ProcessingFunction

try:
    from dspant._rs import (
        apply_exponential_moving_average,
        apply_fir_filter,
        apply_median_filter,
        apply_moving_average,
        apply_moving_rms,
        apply_savgol_filter,
        apply_sinc_filter,
        apply_weighted_moving_average,
        generate_sinc_window,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )
    # Fallback imports would go here if needed


class FIRFilterProcessor(BaseProcessor):
    """
    FIR filter processor implementation with Rust acceleration.

    Applies various FIR filtering methods to signals with optional
    parallel processing and multi-channel support.
    """

    def __init__(
        self,
        filter_type: str = "moving_average",
        window_size: int = 11,
        weights: Optional[np.ndarray] = None,
        center: bool = True,
        poly_order: Optional[int] = None,
        method: str = "simple",
        alpha: float = 0.3,
        filter_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the FIR filter processor.

        Args:
            filter_type: Type of FIR filter ('moving_average', 'weighted',
                         'rms', 'sinc', 'median', 'savgol', 'exponential')
            window_size: Size of the moving window
            weights: Optional custom weights for weighted moving average
            center: Whether to center the window
            poly_order: Polynomial order for Savitzky-Golay filter
            method: Additional method specification (if applicable)
            alpha: Smoothing factor for exponential moving average
            filter_kwargs: Additional filter-specific keyword arguments
        """
        self.filter_type = filter_type
        self.window_size = window_size
        self.weights = weights
        self.center = center
        self.poly_order = poly_order
        self.method = method
        self.alpha = alpha
        self.filter_kwargs = filter_kwargs or {}

        # Validate parameters based on filter type
        self._validate_parameters()

        # Set overlap based on window size
        self._overlap_samples = (
            self.window_size - 1 if self.filter_type != "exponential" else 0
        )

    def _validate_parameters(self):
        """Validate filter parameters based on filter type."""
        if self.filter_type not in [
            "moving_average",
            "weighted",
            "rms",
            "sinc",
            "median",
            "savgol",
            "exponential",
            "fir",
        ]:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")

        if (
            self.filter_type
            in ["moving_average", "weighted", "rms", "median", "savgol"]
            and self.window_size < 1
        ):
            raise ValueError("Window size must be at least 1")

        if self.filter_type == "savgol":
            if self.poly_order is None:
                raise ValueError(
                    "Polynomial order must be specified for Savitzky-Golay filter"
                )
            if self.window_size < self.poly_order + 1:
                raise ValueError("Window size must be greater than polynomial order")

        if self.filter_type == "exponential":
            if not 0 < self.alpha < 1:
                raise ValueError("Alpha must be between 0 and 1 (exclusive)")

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply the specified FIR filter to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments
                use_rust: Whether to use Rust implementation
                parallel: Whether to use parallel processing

        Returns:
            Filtered data array
        """
        # Use kwargs or default to True
        use_rust = kwargs.get("use_rust", _HAS_RUST)

        # Ensure data is 2D
        needs_reshape = False
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            needs_reshape = True

        # Select appropriate filter function
        if not use_rust:
            # Fallback to Python implementation (to be implemented if needed)
            raise NotImplementedError("Python fallback not yet implemented")

        # Define a function to apply the specific filter
        def apply_filter(chunk):
            # Choose filter based on type
            if self.filter_type == "moving_average":
                return apply_moving_average(chunk, self.window_size, self.center)
            elif self.filter_type == "weighted":
                if self.weights is None:
                    raise ValueError(
                        "Weights must be provided for weighted moving average"
                    )
                return apply_weighted_moving_average(chunk, self.weights, self.center)
            elif self.filter_type == "rms":
                return apply_moving_rms(chunk, self.window_size, self.center)
            elif self.filter_type == "sinc":
                # Additional parameters for sinc filter
                cutoff_freq = self.filter_kwargs.get("cutoff_freq")
                window_type = self.filter_kwargs.get("window_type", "hann")
                fs = self.filter_kwargs.get("fs", 1.0)

                if cutoff_freq is None:
                    raise ValueError(
                        "Cutoff frequency must be specified for sinc filter"
                    )

                return apply_sinc_filter(
                    chunk, cutoff_freq, self.window_size, window_type, fs, self.center
                )
            elif self.filter_type == "median":
                return apply_median_filter(chunk, self.window_size, self.center)
            elif self.filter_type == "savgol":
                if self.poly_order is None:
                    raise ValueError(
                        "Polynomial order must be specified for Savitzky-Golay filter"
                    )

                return apply_savgol_filter(
                    chunk, self.window_size, self.poly_order, self.center
                )
            elif self.filter_type == "exponential":
                return apply_exponential_moving_average(chunk, self.alpha)
            elif self.filter_type == "fir":
                if self.weights is None:
                    raise ValueError("Coefficients must be provided for FIR filter")
                return apply_fir_filter(chunk, self.weights, self.center)
            else:
                raise ValueError(f"Unsupported filter type: {self.filter_type}")

        # Use map_overlap for proper boundary handling
        result = data.map_overlap(
            apply_filter,
            depth={-2: self._overlap_samples},  # Overlap along time dimension
            boundary="reflect",
            dtype=data.dtype,
        )

        # Restore original shape if needed
        if needs_reshape:
            result = result.reshape(-1)

        return result

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        summary_data = {
            "filter_type": self.filter_type,
            "window_size": self.window_size
            if self.filter_type != "exponential"
            else "N/A",
            "center": self.center,
            "accelerated": _HAS_RUST,
            "rust_implementation": _HAS_RUST,
        }

        # Add specific details for certain filter types
        if self.filter_type == "weighted" and self.weights is not None:
            summary_data["weights"] = f"Custom weights (length: {len(self.weights)})"

        if self.filter_type == "exponential":
            summary_data["alpha"] = self.alpha

        if self.filter_type == "savgol":
            summary_data["poly_order"] = self.poly_order

        base_summary.update(summary_data)
        return base_summary


@public_api
def create_moving_average(
    window_size: int = 11, center: bool = True
) -> FIRFilterProcessor:
    """
    Create a moving average filter processor.

    Args:
        window_size: Size of the moving window
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(
        filter_type="moving_average", window_size=window_size, center=center
    )


@public_api
def create_weighted_moving_average(
    weights: np.ndarray, center: bool = True
) -> FIRFilterProcessor:
    """
    Create a weighted moving average filter processor.

    Args:
        weights: Custom weights for moving average
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(filter_type="weighted", weights=weights, center=center)


@public_api
def create_moving_rms(window_size: int = 11, center: bool = True) -> FIRFilterProcessor:
    """
    Create a moving RMS filter processor.

    Args:
        window_size: Size of the moving window
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(filter_type="rms", window_size=window_size, center=center)


@public_api
def create_sinc_filter(
    cutoff_freq: float,
    window_size: int = 11,
    window_type: str = "hann",
    fs: float = 1.0,
    center: bool = True,
) -> FIRFilterProcessor:
    """
    Create a sinc window filter processor.

    Args:
        cutoff_freq: Cutoff frequency
        window_size: Size of the window
        window_type: Type of window function
        fs: Sampling frequency
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(
        filter_type="sinc",
        window_size=window_size,
        center=center,
        filter_kwargs={
            "cutoff_freq": cutoff_freq,
            "window_type": window_type,
            "fs": fs,
        },
    )


@public_api
def create_median_filter(
    window_size: int = 11, center: bool = True
) -> FIRFilterProcessor:
    """
    Create a median filter processor.

    Args:
        window_size: Size of the moving window
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(
        filter_type="median", window_size=window_size, center=center
    )


@public_api
def create_savgol_filter(
    window_size: int = 11, poly_order: int = 3, center: bool = True
) -> FIRFilterProcessor:
    """
    Create a Savitzky-Golay filter processor.

    Args:
        window_size: Size of the moving window
        poly_order: Polynomial order for smoothing
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(
        filter_type="savgol",
        window_size=window_size,
        poly_order=poly_order,
        center=center,
    )


@public_api
def create_exponential_moving_average(alpha: float = 0.3) -> FIRFilterProcessor:
    """
    Create an exponential moving average filter processor.

    Args:
        alpha: Smoothing factor (between 0 and 1)

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(filter_type="exponential", alpha=alpha)


@public_api
def create_fir_filter(
    coefficients: np.ndarray, center: bool = True
) -> FIRFilterProcessor:
    """
    Create a general FIR filter processor with custom coefficients.

    Args:
        coefficients: FIR filter coefficients
        center: Whether to center the window

    Returns:
        Configured FIRFilterProcessor
    """
    return FIRFilterProcessor(filter_type="fir", weights=coefficients, center=center)


@public_api
def generate_sinc_window(
    cutoff_freq: float, window_size: int, window_type: str = "hann", fs: float = 1.0
) -> np.ndarray:
    """
    Generate a sinc window for FIR filter design.

    Args:
        cutoff_freq: Cutoff frequency
        window_size: Size of the window
        window_type: Type of window function
        fs: Sampling frequency

    Returns:
        Generated sinc window coefficients
    """
    if not _HAS_RUST:
        raise ImportError("Rust extension not available for sinc window generation")

    return generate_sinc_window(cutoff_freq, window_size, window_type, fs)


__all__ = [
    "FIRFilterProcessor",
    "create_moving_average",
    "create_weighted_moving_average",
    "create_moving_rms",
    "create_sinc_filter",
    "create_median_filter",
    "create_savgol_filter",
    "create_exponential_moving_average",
    "create_fir_filter",
    "generate_sinc_window",
]
