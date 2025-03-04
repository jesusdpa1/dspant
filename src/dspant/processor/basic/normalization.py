from typing import Any, Dict, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit

from ...engine.base import BaseProcessor


class NormalizationProcessor(BaseProcessor):
    """
    Normalization processor implementation with Numba acceleration.

    Normalizes data using various methods:
    - z-score (zero mean, unit variance)
    - min-max scaling (scales to range [0,1])
    - robust (median and interquartile range)
    """

    def __init__(self, method: Literal["zscore", "minmax", "robust"] = "zscore"):
        """
        Initialize the normalization processor.

        Args:
            method: Normalization method to use
                "zscore": zero mean, unit variance normalization
                "minmax": scales to range [0,1]
                "robust": uses median and interquartile range
        """
        self.method = method
        self._overlap_samples = 0

        # Stats for potential reuse
        self._stats = {}

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply normalization to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments
                cache_stats: Whether to cache statistics (default: False)
                axis: Axis along which to normalize (default: None for global)

        Returns:
            Normalized data array
        """
        cache_stats = kwargs.get("cache_stats", False)
        axis = kwargs.get("axis", None)

        # Clear cached stats if not caching or dimensions changed
        if not cache_stats or (cache_stats and self._stats.get("shape") != data.shape):
            self._stats = {"shape": data.shape}

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

            # Use accelerated function
            return data.map_blocks(
                lambda x, m=mean, s=std: _apply_zscore(x, m, s), dtype=data.dtype
            )

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

            # Use accelerated function
            return data.map_blocks(
                lambda x, min_v=min_val, max_v=max_val: _apply_minmax(x, min_v, max_v),
                dtype=data.dtype,
            )

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

            # Use accelerated function
            return data.map_blocks(
                lambda x, med=median, iqr_val=iqr: _apply_robust(x, med, iqr_val),
                dtype=data.dtype,
            )

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap (none for normalization)"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "method": self.method,
                "cached_stats": list(self._stats.keys()) if self._stats else None,
                "accelerated": True,
            }
        )
        return base_summary


@jit(nopython=True, cache=True)
def _apply_zscore(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Apply z-score normalization to a numpy array.

    Args:
        data: Input array
        mean: Mean value
        std: Standard deviation

    Returns:
        Normalized array
    """
    # Handle zero std case
    if std == 0:
        std = 1.0

    return (data - mean) / std


@jit(nopython=True, cache=True)
def _apply_minmax(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Apply min-max normalization to a numpy array.

    Args:
        data: Input array
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Normalized array
    """
    # Handle case where min equals max
    if min_val == max_val:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


@jit(nopython=True, cache=True)
def _apply_robust(data: np.ndarray, median: float, iqr: float) -> np.ndarray:
    """
    Apply robust normalization to a numpy array.

    Args:
        data: Input array
        median: Median value
        iqr: Interquartile range

    Returns:
        Normalized array
    """
    # Handle zero iqr case
    if iqr == 0:
        iqr = 1.0

    return (data - median) / iqr
