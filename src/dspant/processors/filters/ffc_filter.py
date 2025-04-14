"""
Feed Forward Comb (FFC) filter implementation.

This module provides an implementation of the FFC filter as described in [1].
The FFC filter is particularly effective for EMG envelope extraction and denoising
in human-machine interfaces.

References:
[1] D. Esposito, J. Centracchio, P. Bifulco, and E. Andreozzi,
"A smart approach to EMG envelope extraction and powerful denoising for human–machine interfaces,"
Sci. Rep., vol. 13, no. 1, p. 7768, 2023, doi: 10.1038/s41598-023-33319-4.
"""

from typing import Any, Dict, Optional, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from dspant.core.internals import public_api

from ...engine.base import BaseProcessor


@jit(nopython=True, cache=True)
def _apply_ffc_filter_single(
    data: np.ndarray, delay_samples: int, alpha: float
) -> np.ndarray:
    """
    Apply Feed Forward Comb filter to a single channel.

    Filter equation: y(k) = x(k) + (alpha * x)(k-N)
    where:
    - x is the input signal
    - N is the delay expressed in number of samples (fs/fc)
    - alpha regulates aspects of the filter behavior

    Args:
        data: Input 1D array
        delay_samples: Delay in samples (N in the equation)
        alpha: Filter coefficient (alpha in the equation)

    Returns:
        Filtered signal
    """
    n_samples = len(data)
    result = np.zeros_like(data)

    # Special case when delay is 0
    if delay_samples == 0:
        # Handle directly instead of return in the middle
        for i in range(n_samples):
            result[i] = data[i] * (1 + alpha)
        return result

    # Apply the filter: y(k) = x(k) + (alpha * x)(k-N)
    for i in range(n_samples):
        result[i] = data[i]
        if i >= delay_samples:
            result[i] += alpha * data[i - delay_samples]

    return result


@jit(nopython=True, parallel=True, cache=True)
def _apply_ffc_filter_multi(
    data: np.ndarray, delay_samples: int, alpha: float
) -> np.ndarray:
    """
    Apply Feed Forward Comb filter to multiple channels with parallelization.

    Args:
        data: Input 2D array (samples x channels)
        delay_samples: Delay in samples
        alpha: Filter coefficient

    Returns:
        Filtered signal
    """
    n_samples, n_channels = data.shape
    result = np.zeros_like(data)

    # Special case when delay is 0
    if delay_samples == 0:
        # Handle directly instead of return in the middle
        for c in prange(n_channels):
            for i in range(n_samples):
                result[i, c] = data[i, c] * (1 + alpha)
        return result

    # Process each channel in parallel
    for c in prange(n_channels):
        for i in range(n_samples):
            result[i, c] = data[i, c]
            if i >= delay_samples:
                result[i, c] += alpha * data[i - delay_samples, c]

    return result


@public_api
class FFCFilter(BaseProcessor):
    """
    Feed Forward Comb (FFC) filter processor implementation with Numba acceleration.

    The filter equation is: y(k) = x(k) + (alpha * x)(k-N)
    where:
    - x is the input signal
    - N is the delay expressed in number of samples (fs/fc)
    - alpha regulates aspects of the filter behavior

    This filter is particularly effective for EMG envelope extraction and denoising
    in human-machine interfaces as described in [1].

    References:
    [1] D. Esposito, J. Centracchio, P. Bifulco, and E. Andreozzi,
    "A smart approach to EMG envelope extraction and powerful denoising for human–machine interfaces,"
    Sci. Rep., vol. 13, no. 1, p. 7768, 2023, doi: 10.1038/s41598-023-33319-4.
    """

    def __init__(
        self,
        cutoff_frequency: float,
        alpha: float = -1.0,
        use_jit: bool = True,
    ):
        """
        Initialize the FFC filter processor.

        Args:
            cutoff_frequency: Cutoff frequency in Hz
            alpha: Filter coefficient, typically negative for notch behavior
                  and positive for enhancement
            use_jit: Whether to use JIT acceleration for processing
        """
        self.cutoff_frequency = cutoff_frequency
        self.alpha = alpha
        self.use_jit = use_jit
        self._delay_samples = None
        self._fs = None
        self._overlap_samples = 0  # Will be set when processing

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply the FFC filter to the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency in Hz (required)
            **kwargs: Additional keyword arguments

        Returns:
            Filtered data
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) is required for FFC filter")

        # Update sampling frequency if changed
        if self._fs != fs:
            self._fs = fs
            # Calculate delay in samples: N = fs/fc
            self._delay_samples = int(round(fs / self.cutoff_frequency))
            # Set overlap samples to delay length for proper filtering at chunk boundaries
            self._overlap_samples = self._delay_samples

        # Get acceleration flag (can be overridden in kwargs)
        use_jit = kwargs.get("use_jit", self.use_jit)

        # Apply filter
        if use_jit:
            # Select the appropriate JIT function based on input dimensions
            if data.ndim > 1:
                filter_func = _apply_ffc_filter_multi
            else:
                filter_func = _apply_ffc_filter_single

            # Define a wrapper function to avoid lambda
            def apply_filter_wrapper(x):
                return filter_func(x, self._delay_samples, self.alpha)

            return data.map_overlap(
                apply_filter_wrapper,
                depth={0: self._overlap_samples},
                boundary="reflect",
                dtype=data.dtype,
            )
        else:
            # Non-JIT implementation for when Numba is not available
            def apply_filter(x):
                result = np.zeros_like(x)

                # Special case when delay is 0
                if self._delay_samples == 0:
                    # Return direct multiplication
                    return x * (1 + self.alpha)

                if x.ndim == 1:
                    # Single channel case
                    for i in range(len(x)):
                        result[i] = x[i]
                        if i >= self._delay_samples:
                            result[i] += self.alpha * x[i - self._delay_samples]
                else:
                    # Multi-channel case
                    n_samples, n_channels = x.shape
                    for c in range(n_channels):
                        for i in range(n_samples):
                            result[i, c] = x[i, c]
                            if i >= self._delay_samples:
                                result[i, c] += (
                                    self.alpha * x[i - self._delay_samples, c]
                                )

                return result

        return data.map_overlap(
            apply_filter,
            depth={0: self._overlap_samples},
            boundary="reflect",
            dtype=data.dtype,
        )

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap (delay length)."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "cutoff_frequency": self.cutoff_frequency,
                "alpha": self.alpha,
                "delay_samples": self._delay_samples,
                "sampling_frequency": self._fs,
                "accelerated": self.use_jit,
            }
        )
        return base_summary


@public_api
def create_ffc_filter(
    cutoff_frequency: float, alpha: float = -1.0, use_jit: bool = True
) -> FFCFilter:
    """
    Create a Feed Forward Comb (FFC) filter processor.

    Args:
        cutoff_frequency: Cutoff frequency in Hz
        alpha: Filter coefficient, typically negative for notch behavior
              and positive for enhancement
        use_jit: Whether to use JIT acceleration for processing

    Returns:
        Configured FFCFilter instance
    """
    return FFCFilter(cutoff_frequency, alpha, use_jit)


@public_api
def create_ffc_notch(
    cutoff_frequency: float, alpha_strength: float = 1.0, use_jit: bool = True
) -> FFCFilter:
    """
    Create an FFC filter configured for notch behavior.

    Args:
        cutoff_frequency: Notch frequency in Hz
        alpha_strength: Strength of the notch effect (positive value)
        use_jit: Whether to use JIT acceleration for processing

    Returns:
        Configured FFCFilter instance for notch filtering
    """
    # For notch behavior, alpha should be negative
    alpha = -abs(alpha_strength)
    return FFCFilter(cutoff_frequency, alpha, use_jit)


@public_api
def create_ffc_enhancement(
    cutoff_frequency: float, alpha_strength: float = 1.0, use_jit: bool = True
) -> FFCFilter:
    """
    Create an FFC filter configured for enhancement behavior.

    Args:
        cutoff_frequency: Enhancement frequency in Hz
        alpha_strength: Strength of the enhancement effect (positive value)
        use_jit: Whether to use JIT acceleration for processing

    Returns:
        Configured FFCFilter instance for enhancement filtering
    """
    # For enhancement behavior, alpha should be positive
    alpha = abs(alpha_strength)
    return FFCFilter(cutoff_frequency, alpha, use_jit)
