from typing import Any, Dict, Literal, Optional, Union

import dask.array as da
import numpy as np
from numba import jit
from scipy import signal

from ...engine.base import BaseProcessor


class RectificationProcessor(BaseProcessor):
    """
    Signal rectification processor implementation with Numba acceleration.

    Rectifies signals using various methods:
    - abs: Full-wave rectification (absolute value)
    - square: Squared rectification
    - half: Half-wave rectification (keeps only positive values)
    - hilbert: Envelope extraction using Hilbert transform
    """

    def __init__(self, method: Literal["abs", "square", "half", "hilbert"] = "abs"):
        """
        Initialize the rectification processor.

        Args:
            method: Rectification method to use
                "abs": Full-wave rectification (absolute value)
                "square": Squared rectification
                "half": Half-wave rectification (keeps positive values)
                "hilbert": Envelope extraction using Hilbert transform
        """
        self.method = method

        # Set overlap based on method
        # Hilbert transform requires FFT, which works best with whole arrays
        # But we'll use a reasonable overlap to avoid edge effects
        self._overlap_samples = 0 if method != "hilbert" else 128

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply rectification to the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used for most methods)
            **kwargs: Additional keyword arguments

        Returns:
            Rectified data array
        """
        if self.method == "abs":
            # Full-wave rectification (absolute value)
            return data.map_blocks(_apply_abs_rectification, dtype=data.dtype)

        elif self.method == "square":
            # Squared rectification
            return data.map_blocks(_apply_square_rectification, dtype=data.dtype)

        elif self.method == "half":
            # Half-wave rectification
            return data.map_blocks(_apply_half_rectification, dtype=data.dtype)

        elif self.method == "hilbert":
            # Hilbert transform for envelope extraction
            return data.map_overlap(
                _apply_hilbert_envelope,
                depth=(self._overlap_samples, 0),
                boundary="reflect",
                dtype=data.dtype,
            )

        else:
            raise ValueError(f"Unknown rectification method: {self.method}")

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update({"method": self.method, "accelerated": True})
        return base_summary


@jit(nopython=True, cache=True)
def _apply_abs_rectification(data: np.ndarray) -> np.ndarray:
    """
    Apply absolute value rectification (full-wave rectification).

    Args:
        data: Input array

    Returns:
        Rectified array
    """
    return np.abs(data)


@jit(nopython=True, cache=True)
def _apply_square_rectification(data: np.ndarray) -> np.ndarray:
    """
    Apply squared rectification.

    Args:
        data: Input array

    Returns:
        Rectified array
    """
    return data * data  # Faster than data ** 2


@jit(nopython=True, cache=True)
def _apply_half_rectification(data: np.ndarray) -> np.ndarray:
    """
    Apply half-wave rectification (keep only positive values).

    Args:
        data: Input array

    Returns:
        Rectified array
    """
    result = np.empty_like(data)
    if data.ndim == 1:
        for i in range(data.shape[0]):
            result[i] = max(0, data[i])
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                result[i, j] = max(0, data[i, j])
    return result


def _apply_hilbert_envelope(data: np.ndarray) -> np.ndarray:
    """
    Extract signal envelope using Hilbert transform.

    The Hilbert transform creates an analytic signal where:
    - The real part is the original signal
    - The imaginary part is the 90° phase-shifted version

    The envelope is the magnitude of this complex analytic signal.

    Args:
        data: Input array

    Returns:
        Signal envelope
    """
    # Ensure data is contiguous for better performance
    data = np.ascontiguousarray(data)

    # Apply Hilbert transform and get envelope
    if data.ndim == 1:
        # Single channel case
        analytic_signal = signal.hilbert(data)
        return np.abs(analytic_signal)
    else:
        # Multi-channel case
        result = np.empty_like(data)
        for i in range(data.shape[1]):
            analytic_signal = signal.hilbert(data[:, i])
            result[:, i] = np.abs(analytic_signal)
        return result
