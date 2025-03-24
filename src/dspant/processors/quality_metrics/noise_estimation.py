from typing import Any, Dict, Literal, Optional, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from ...engine.base import BaseProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_noise_mad(chunk: np.ndarray) -> np.ndarray:
    """
    Compute noise levels using Median Absolute Deviation (MAD) method.

    Args:
        chunk: Input data array

    Returns:
        Array of noise levels for each channel
    """
    if chunk.ndim == 1:
        chunk = chunk.reshape(-1, 1)

    n_channels = chunk.shape[1]
    noise_levels = np.zeros(n_channels, dtype=np.float32)

    for c in prange(n_channels):
        # Calculate median of the channel
        med = np.median(chunk[:, c])

        # Calculate MAD and scale to approximate standard deviation
        noise_levels[c] = np.median(np.abs(chunk[:, c] - med)) / 0.6744897501960817

    return noise_levels


@jit(nopython=True, parallel=True, cache=True)
def _compute_noise_std(chunk: np.ndarray) -> np.ndarray:
    """
    Compute noise levels using standard deviation method.

    Args:
        chunk: Input data array

    Returns:
        Array of noise levels for each channel
    """
    if chunk.ndim == 1:
        chunk = chunk.reshape(-1, 1)

    return np.std(chunk, axis=0, dtype=np.float32)


class NoiseEstimationProcessor(BaseProcessor):
    """
    Noise estimation processor for multi-channel signals.

    Supports two methods for noise level estimation:
    - MAD (Median Absolute Deviation): More robust to outliers
    - STD (Standard Deviation)
    """

    def __init__(
        self,
        method: Literal["mad", "std"] = "mad",
        relative_start: float = 0.0,
        relative_stop: float = 1.0,
        random_seed: Optional[int] = None,
        max_samples: int = 5000,
        sample_percentage: float = 0.1,
    ):
        """
        Initialize the noise estimation processor.

        Args:
            method: Noise estimation method ('mad' or 'std')
            relative_start: Relative start point for noise estimation (0.0-1.0)
            relative_stop: Relative end point for noise estimation (0.0-1.0)
            random_seed: Random seed for reproducibility
            max_samples: Maximum number of samples to use for estimation
            sample_percentage: Percentage of chunk to sample if chunk is large
        """
        if method not in ["mad", "std"]:
            raise ValueError("Method must be either 'mad' or 'std'")

        self.method = method
        self.relative_start = max(0.0, min(1.0, relative_start))
        self.relative_stop = max(0.0, min(1.0, relative_stop))
        self.random_seed = random_seed
        self.max_samples = max_samples
        self.sample_percentage = max(0.0, min(1.0, sample_percentage))

        # No overlap needed for this processor
        self._overlap_samples = 0

        # Cached noise levels to avoid recomputation
        self._noise_levels = None

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Estimate noise levels for the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, but kept for interface consistency)
            **kwargs: Additional keyword arguments

        Returns:
            Dask array with noise levels
        """
        # Ensure input is 2D
        if data.ndim == 1:
            data = data[:, np.newaxis]

        # Determine start and end samples
        total_samples = data.shape[0]
        start_sample = int(total_samples * self.relative_start)
        end_sample = int(total_samples * self.relative_stop)

        # Extract the slice for noise estimation
        noise_slice = data[start_sample:end_sample]

        def estimate_noise_chunk(chunk: np.ndarray) -> np.ndarray:
            """
            Process a chunk of data to estimate noise levels.

            Args:
                chunk: Input data chunk

            Returns:
                Noise levels for the chunk
            """
            # Ensure the input is a contiguous array
            chunk = np.ascontiguousarray(chunk)

            # Handle single-channel case
            if chunk.ndim == 1:
                chunk = chunk[:, np.newaxis]

            # Sample from the chunk
            rng = np.random.default_rng(seed=self.random_seed)

            # Determine sampling strategy
            if chunk.shape[0] > self.max_samples:
                # If chunk is large, sample a percentage or max_samples
                sample_size = min(
                    int(chunk.shape[0] * self.sample_percentage), self.max_samples
                )
                sample_indices = rng.choice(chunk.shape[0], sample_size, replace=False)
                sample_chunk = chunk[sample_indices]
            else:
                # Use entire chunk if small
                sample_chunk = chunk

            # Compute noise levels based on method
            if self.method == "mad":
                return _compute_noise_mad(sample_chunk)
            else:
                return _compute_noise_std(sample_chunk)

        # Use map_blocks for chunk-wise processing with proper chunk specification
        result = noise_slice.map_blocks(
            estimate_noise_chunk,
            dtype=np.float32,
            # Specify chunks for both dimensions
            chunks=((noise_slice.shape[1],) + (data.shape[1],)),
        )

        # Compute and cache the final noise levels
        noise_levels = result.mean(axis=0)
        self._noise_levels = noise_levels.compute()

        return da.from_array(self._noise_levels, chunks=(data.shape[1],))

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "method": self.method,
                "relative_start": self.relative_start,
                "relative_stop": self.relative_stop,
                "random_seed": self.random_seed,
                "max_samples": self.max_samples,
                "sample_percentage": self.sample_percentage,
                "cached": self._noise_levels is not None,
            }
        )
        return base_summary


def create_noise_estimation_processor(
    method: Literal["mad", "std"] = "mad",
    relative_start: float = 0.0,
    relative_stop: float = 1.0,
    random_seed: Optional[int] = None,
    max_samples: int = 5000,
    sample_percentage: float = 0.1,
) -> NoiseEstimationProcessor:
    """
    Create a noise estimation processor with standard parameters.

    Args:
        method: Noise estimation method ('mad' or 'std')
        relative_start: Relative start point for noise estimation (0.0-1.0)
        relative_stop: Relative end point for noise estimation (0.0-1.0)
        random_seed: Random seed for reproducibility
        max_samples: Maximum number of samples to use for estimation
        sample_percentage: Percentage of chunk to sample if chunk is large

    Returns:
        Configured NoiseEstimationProcessor
    """
    return NoiseEstimationProcessor(
        method=method,
        relative_start=relative_start,
        relative_stop=relative_stop,
        random_seed=random_seed,
        max_samples=max_samples,
        sample_percentage=sample_percentage,
    )
