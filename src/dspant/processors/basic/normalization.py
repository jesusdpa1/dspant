"""
Rust-accelerated Normalization implementations.

This module provides high-performance Rust implementations of normalization algorithms
with Python bindings, while preserving compatibility with Dask arrays for large datasets.
"""

from typing import Any, Dict, Literal

import dask.array as da
import numpy as np
from numba import jit

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor

try:
    from dspant._rs import (
        apply_mad,
        apply_mad_parallel,
        apply_minmax,
        apply_minmax_parallel,
        apply_robust,
        apply_robust_parallel,
        apply_zscore,
        apply_zscore_parallel,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )
    # Import the Python implementations
    from .normalization import _apply_minmax, _apply_robust, _apply_zscore

    # Define the MAD implementation since it's new
    @jit(nopython=True, cache=True)
    def _apply_mad(
        data: np.ndarray, median: float, mad: float, k: float = 1.4826
    ) -> np.ndarray:
        """
        Apply MAD (Median Absolute Deviation) normalization to a numpy array.

        Args:
            data: Input array
            median: Median value
            mad: Median absolute deviation
            k: Scale factor (1.4826 for normally distributed data)

        Returns:
            Normalized array
        """
        # Handle zero mad case
        if mad == 0:
            mad = 1.0

        return (data - median) / (k * mad)


@public_api
class NormalizationProcessor(BaseProcessor):
    """
    Normalization processor implementation with Rust and Numba acceleration.

    Normalizes data using various methods:
    - z-score (zero mean, unit variance)
    - min-max scaling (scales to range [0,1])
    - robust (median and interquartile range)
    - mad (median absolute deviation)
    """

    def __init__(self, method: Literal["zscore", "minmax", "robust", "mad"] = "zscore"):
        """
        Initialize the normalization processor.

        Args:
            method: Normalization method to use
                "zscore": zero mean, unit variance normalization
                "minmax": scales to range [0,1]
                "robust": uses median and interquartile range
                "mad": median absolute deviation normalization
        """
        self.method = method
        self._overlap_samples = 0

        # Stats for potential reuse
        self._stats = {}

        # MAD-specific parameters
        if self.method == "mad":
            self.k = 1.4826  # Default constant for normally distributed data

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply normalization to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments
                cache_stats: Whether to cache statistics (default: False)
                axis: Axis along which to normalize (default: None for global)
                use_rust: Whether to use Rust implementation if available (default: True)
                use_parallel: Whether to use parallel processing (default: True)
                k: Scale factor for MAD normalization (default: 1.4826)

        Returns:
            Normalized data array
        """
        cache_stats = kwargs.get("cache_stats", False)
        axis = kwargs.get("axis", None)
        use_rust = kwargs.get("use_rust", _HAS_RUST)
        use_parallel = kwargs.get("use_parallel", True)

        # Get MAD-specific parameters
        if self.method == "mad":
            self.k = kwargs.get("k", 1.4826)

        # Clear cached stats if not caching or dimensions changed
        if not cache_stats or (cache_stats and self._stats.get("shape") != data.shape):
            self._stats = {"shape": data.shape}

        # Ensure data is 2D for Rust implementation
        needs_reshape = False
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            needs_reshape = True

        # Define the map function for chunk-wise processing
        if self.method == "zscore":
            # Z-score normalization (zero mean, unit variance)
            if "mean" not in self._stats or "std" not in self._stats:
                self._stats["mean"] = data.mean(axis=axis, keepdims=True)
                self._stats["std"] = data.std(axis=axis, keepdims=True)

            # Use dask's map_blocks for parallel processing
            mean = self._stats["mean"]
            std = self._stats["std"]

            # Convert to scalars if they're arrays with single values
            if hasattr(mean, "compute") and mean.size == 1:
                mean = float(mean.compute())
            if hasattr(std, "compute") and std.size == 1:
                std = float(std.compute())
                std = std if std != 0 else 1.0

            # Define processing function
            def process_chunk(x):
                # Ensure array is contiguous and float32
                x = np.ascontiguousarray(x, dtype=np.float32)

                # Use Rust implementation if available and requested
                if _HAS_RUST and use_rust:
                    if use_parallel and x.shape[1] > 1:
                        return apply_zscore_parallel(x, mean, std)
                    else:
                        return apply_zscore(x, mean, std)
                else:
                    # Fall back to Python implementation
                    return _apply_zscore(x, mean, std)

            # Apply the function
            result = data.map_blocks(process_chunk, dtype=np.float32)

        elif self.method == "minmax":
            # Min-max scaling to [0,1]
            if "min" not in self._stats or "max" not in self._stats:
                self._stats["min"] = data.min(axis=axis, keepdims=True)
                self._stats["max"] = data.max(axis=axis, keepdims=True)

            min_val = self._stats["min"]
            max_val = self._stats["max"]

            # Convert to scalars if they're arrays with single values
            if hasattr(min_val, "compute") and min_val.size == 1:
                min_val = float(min_val.compute())
            if hasattr(max_val, "compute") and max_val.size == 1:
                max_val = float(max_val.compute())
                # Avoid division by zero
                if min_val == max_val:
                    max_val = min_val + 1

            # Define processing function
            def process_chunk(x):
                # Ensure array is contiguous and float32
                x = np.ascontiguousarray(x, dtype=np.float32)

                # Use Rust implementation if available and requested
                if _HAS_RUST and use_rust:
                    if use_parallel and x.shape[1] > 1:
                        return apply_minmax_parallel(x, min_val, max_val)
                    else:
                        return apply_minmax(x, min_val, max_val)
                else:
                    # Fall back to Python implementation
                    return _apply_minmax(x, min_val, max_val)

            # Apply the function
            result = data.map_blocks(process_chunk, dtype=np.float32)

        elif self.method == "robust":
            # Robust normalization using median and interquartile range
            if "median" not in self._stats or "iqr" not in self._stats:
                self._stats["median"] = da.percentile(
                    data, 50, axis=axis, keepdims=True
                )
                q75 = da.percentile(data, 75, axis=axis, keepdims=True)
                q25 = da.percentile(data, 25, axis=axis, keepdims=True)
                self._stats["iqr"] = q75 - q25

            median = self._stats["median"]
            iqr = self._stats["iqr"]

            # Convert to scalars if they're arrays with single values
            if hasattr(median, "compute") and median.size == 1:
                median = float(median.compute())
            if hasattr(iqr, "compute") and iqr.size == 1:
                iqr = float(iqr.compute())
                iqr = iqr if iqr != 0 else 1.0

            # Define processing function
            def process_chunk(x):
                # Ensure array is contiguous and float32
                x = np.ascontiguousarray(x, dtype=np.float32)

                # Use Rust implementation if available and requested
                if _HAS_RUST and use_rust:
                    if use_parallel and x.shape[1] > 1:
                        return apply_robust_parallel(x, median, iqr)
                    else:
                        return apply_robust(x, median, iqr)
                else:
                    # Fall back to Python implementation
                    return _apply_robust(x, median, iqr)

            # Apply the function
            result = data.map_blocks(process_chunk, dtype=np.float32)

        elif self.method == "mad":
            # MAD normalization using median and median absolute deviation
            if "median" not in self._stats or "mad" not in self._stats:
                self._stats["median"] = da.percentile(
                    data, 50, axis=axis, keepdims=True
                )
                # Calculate MAD: median of absolute deviations from the median
                abs_dev = da.abs(data - self._stats["median"])
                self._stats["mad"] = da.percentile(
                    abs_dev, 50, axis=axis, keepdims=True
                )

            median = self._stats["median"]
            mad_value = self._stats["mad"]

            # Convert to scalars if they're arrays with single values
            if hasattr(median, "compute") and median.size == 1:
                median = float(median.compute())
            if hasattr(mad_value, "compute") and mad_value.size == 1:
                mad_value = float(mad_value.compute())
                mad_value = mad_value if mad_value != 0 else 1.0

            # Define processing function
            def process_chunk(x):
                # Ensure array is contiguous and float32
                x = np.ascontiguousarray(x, dtype=np.float32)

                # Use Rust implementation if available and requested
                if _HAS_RUST and use_rust:
                    if use_parallel and x.shape[1] > 1:
                        return apply_mad_parallel(x, median, mad_value, self.k)
                    else:
                        return apply_mad(x, median, mad_value, self.k)
                else:
                    # Fall back to Python implementation
                    return _apply_mad(x, median, mad_value, self.k)

            # Apply the function
            result = data.map_blocks(process_chunk, dtype=np.float32)

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Restore original shape if needed
        if needs_reshape:
            result = result.reshape(-1)

        return result

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap (none for normalization)"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        summary_data = {
            "method": self.method,
            "cached_stats": list(self._stats.keys()) if self._stats else None,
            "accelerated": _HAS_RUST,
            "rust_implementation": _HAS_RUST,
        }

        # Add MAD-specific parameters if applicable
        if self.method == "mad":
            summary_data["k"] = self.k

        base_summary.update(summary_data)
        return base_summary


# Helper functions for direct use


def apply_normalization(
    data,
    method="zscore",
    mean=None,
    std=None,
    min_val=None,
    max_val=None,
    median=None,
    iqr=None,
    mad=None,
    k=1.4826,
    use_parallel=True,
    use_rust=True,
    axis=None,
):
    """
    Apply normalization algorithm with Rust acceleration if available.

    Args:
        data: Input signal (numpy array or dask array)
        method: Normalization method ('zscore', 'minmax', 'robust', or 'mad')
        mean: Mean value for z-score normalization (calculated from data if None)
        std: Standard deviation for z-score normalization (calculated from data if None)
        min_val: Minimum value for min-max normalization (calculated from data if None)
        max_val: Maximum value for min-max normalization (calculated from data if None)
        median: Median value for robust/mad normalization (calculated from data if None)
        iqr: Interquartile range for robust normalization (calculated from data if None)
        mad: Median absolute deviation for mad normalization (calculated from data if None)
        k: Scale factor for MAD normalization (default: 1.4826)
        use_parallel: Whether to use parallel optimized version
        use_rust: Whether to use Rust implementation if available
        axis: Axis along which to calculate statistics (default: None for global)

    Returns:
        Normalized signal
    """
    # Ensure data is a 2D array
    original_shape = data.shape
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Calculate statistics if not provided
    if method == "zscore":
        if mean is None:
            mean = np.mean(data, axis=axis, keepdims=True)
            # Convert to scalar if possible
            if hasattr(mean, "item") and mean.size == 1:
                mean = mean.item()
        if std is None:
            std = np.std(data, axis=axis, keepdims=True)
            # Convert to scalar if possible
            if hasattr(std, "item") and std.size == 1:
                std = std.item()
            # Handle zero std case
            if std == 0:
                std = 1.0

    elif method == "minmax":
        if min_val is None:
            min_val = np.min(data, axis=axis, keepdims=True)
            # Convert to scalar if possible
            if hasattr(min_val, "item") and min_val.size == 1:
                min_val = min_val.item()
        if max_val is None:
            max_val = np.max(data, axis=axis, keepdims=True)
            # Convert to scalar if possible
            if hasattr(max_val, "item") and max_val.size == 1:
                max_val = max_val.item()
            # Handle case where min equals max
            if min_val == max_val:
                max_val = min_val + 1

    elif method == "robust" or method == "mad":
        if median is None:
            median = np.median(data, axis=axis, keepdims=True)
            # Convert to scalar if possible
            if hasattr(median, "item") and median.size == 1:
                median = median.item()

        if method == "robust" and iqr is None:
            q75 = np.percentile(data, 75, axis=axis, keepdims=True)
            q25 = np.percentile(data, 25, axis=axis, keepdims=True)
            iqr = q75 - q25
            # Convert to scalar if possible
            if hasattr(iqr, "item") and iqr.size == 1:
                iqr = iqr.item()
            # Handle zero iqr case
            if iqr == 0:
                iqr = 1.0

        if method == "mad" and mad is None:
            # Calculate MAD: median of absolute deviations from the median
            abs_dev = np.abs(data - median)
            mad = np.median(abs_dev, axis=axis, keepdims=True)
            # Convert to scalar if possible
            if hasattr(mad, "item") and mad.size == 1:
                mad = mad.item()
            # Handle zero mad case
            if mad == 0:
                mad = 1.0

    # For numpy arrays, directly use Rust functions if available
    if isinstance(data, np.ndarray) and _HAS_RUST and use_rust:
        # Ensure array is contiguous and float32
        data = np.ascontiguousarray(data, dtype=np.float32)

        if method == "zscore":
            if use_parallel and data.shape[1] > 1:
                result = apply_zscore_parallel(data, mean, std)
            else:
                result = apply_zscore(data, mean, std)

        elif method == "minmax":
            if use_parallel and data.shape[1] > 1:
                result = apply_minmax_parallel(data, min_val, max_val)
            else:
                result = apply_minmax(data, min_val, max_val)

        elif method == "robust":
            if use_parallel and data.shape[1] > 1:
                result = apply_robust_parallel(data, median, iqr)
            else:
                result = apply_robust(data, median, iqr)

        elif method == "mad":
            if use_parallel and data.shape[1] > 1:
                result = apply_mad_parallel(data, median, mad, k)
            else:
                result = apply_mad(data, median, mad, k)

        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        # Fall back to Python implementation
        if method == "zscore":
            result = _apply_zscore(data, mean, std)

        elif method == "minmax":
            result = _apply_minmax(data, min_val, max_val)

        elif method == "robust":
            result = _apply_robust(data, median, iqr)

        elif method == "mad":
            result = _apply_mad(data, median, mad, k)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    # Restore original shape if needed
    if original_shape != data.shape:
        result = result.reshape(original_shape)

    return result


# Factory function
def create_normalizer(method: str = "zscore"):
    """Create a normalization processor with specified method."""
    return NormalizationProcessor(method=method)
