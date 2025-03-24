from typing import Any, Dict, Literal

import dask.array as da
import numpy as np
from numba import jit

from ...engine.base import BaseProcessor


class TKEOProcessor(BaseProcessor):
    """
    Teager-Kaiser Energy Operator (TKEO) processor implementation.

    The TKEO estimates the instantaneous energy of a signal by considering
    adjacent samples. It's useful for detecting abrupt changes and
    emphasizing high-frequency components with high energy.

    References:
    - Li et al., 2007 (classic 3-point TKEO)
    - Deburchgrave et al., 2008 (modified 4-point TKEO)
    """

    def __init__(self, method: Literal["classic", "modified"] = "classic"):
        """
        Initialize the TKEO processor.

        Args:
            method: TKEO algorithm to use
                "classic": 3-point algorithm (Li et al., 2007)
                "modified": 4-point algorithm (Deburchgrave et al., 2008)
        """
        self.method = method
        self._overlap_samples = 2 if method == "classic" else 3

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply the TKEO operation to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments

        Returns:
            Processed array with TKEO applied
        """
        # Use the appropriate accelerated function based on method
        if self.method == "classic":
            tkeo_func = _classic_tkeo
        else:
            tkeo_func = _modified_tkeo

        return data.map_overlap(
            tkeo_func,
            depth=(self.overlap_samples, 0),
            boundary="reflect",
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
                "method": self.method,
                "reference": "Li et al., 2007"
                if self.method == "classic"
                else "Deburchgrave et al., 2008",
                "accelerated": True,
            }
        )
        return base_summary


@jit(nopython=True, cache=True)
def _classic_tkeo(a: np.ndarray) -> np.ndarray:
    """
    Classic 3-point TKEO: y[n] = x[n]² - x[n-1] * x[n+1]

    Reference: Li et al., 2007

    Args:
        a: Input array

    Returns:
        Array with TKEO applied
    """
    if a.ndim == 1:
        # Create two temporary arrays of equal length, shifted 1 sample to the right
        # and left and squared:
        i = a[1:-1] * a[1:-1]
        j = a[2:] * a[:-2]
        # Calculate the difference between the two temporary arrays:
        a_tkeo = i - j
        return a_tkeo
    else:
        # Process each channel of multi-dimensional array
        result = np.empty((a.shape[0] - 2, a.shape[1]), dtype=a.dtype)
        for c in range(a.shape[1]):
            channel = a[:, c]
            i = channel[1:-1] * channel[1:-1]
            j = channel[2:] * channel[:-2]
            result[:, c] = i - j
        return result


@jit(nopython=True, cache=True)
def _modified_tkeo(a: np.ndarray) -> np.ndarray:
    """
    Modified 4-point TKEO: y[n] = x[n+1]*x[n-2] - x[n]*x[n-3]

    Reference: Deburchgrave et al., 2008

    Args:
        a: Input array

    Returns:
        Array with TKEO applied
    """
    if a.ndim == 1:
        l = 1
        p = 2
        q = 0
        s = 3

        a_tkeo = a[l:-p] * a[p:-l] - a[q:-s] * a[s:]
        return a_tkeo
    else:
        # Process each channel of multi-dimensional array
        result = np.empty((a.shape[0] - 3, a.shape[1]), dtype=a.dtype)
        for c in range(a.shape[1]):
            channel = a[:, c]
            l = 1
            p = 2
            q = 0
            s = 3
            result[:, c] = channel[l:-p] * channel[p:-l] - channel[q:-s] * channel[s:]
        return result
