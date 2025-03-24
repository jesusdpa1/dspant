"""
Rust-accelerated whitening implementations.

This module provides high-performance Rust implementations of whitening algorithms
with Python bindings, while preserving compatibility with Dask arrays for large datasets.
"""

from typing import Any, Dict, Optional

import dask.array as da
import numpy as np

from dspant.engine.base import BaseProcessor

try:
    from dspant._rs import (
        apply_whitening,
        apply_whitening_parallel,
        compute_covariance,
        compute_covariance_parallel,
        compute_mean,
        compute_mean_parallel,
        compute_whitening_matrix,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )


class WhiteningRustProcessor(BaseProcessor):
    """
    Rust-accelerated whitening processor implementation.

    Whitens the signal by decorrelating and normalizing the variance using
    Rust-accelerated functions for better performance.

    Parameters
    ----------
    apply_mean : bool, default: False
        Whether to subtract the mean before applying whitening matrix
    int_scale : float or None, default: None
        Apply a scaling factor to fit integer range if needed
    eps : float or None, default: None
        Small epsilon to regularize SVD. If None, it's automatically determined
    use_parallel : bool, default: True
        Whether to use parallel processing for better performance
    """

    def __init__(
        self,
        apply_mean: bool = False,
        int_scale: Optional[float] = None,
        eps: Optional[float] = None,
        use_parallel: bool = True,
    ):
        """Initialize the Rust-accelerated whitening processor."""
        if not _HAS_RUST:
            raise ImportError(
                "Rust extension not available. Install with 'pip install dspant[rust]'"
            )

        self.apply_mean = apply_mean
        self.int_scale = int_scale
        self.eps = eps if eps is not None else 1e-6
        self.use_parallel = use_parallel
        self._overlap_samples = 0  # No overlap needed for this operation

        # Will be initialized during processing
        self._whitening_matrix = None
        self._mean = None
        self._is_fitted = False

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply whitening to the data lazily using Rust implementation.

        Parameters
        ----------
        data : da.Array
            Input data as a Dask array
        fs : float, optional
            Sampling frequency (not used but required by interface)
        **kwargs : dict
            Additional keyword arguments
            compute_now: bool, default: False
                Whether to compute the whitening matrix immediately or defer
            sample_size: int, default: 10000
                Number of samples to use for computing covariance matrix
            use_parallel: bool
                Override the use_parallel setting from initialization

        Returns
        -------
        whitened_data : da.Array
            Whitened data as a Dask array
        """
        # Ensure the data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Get parameters from kwargs
        compute_now = kwargs.get("compute_now", False)
        sample_size = kwargs.get("sample_size", 10000)
        use_parallel = kwargs.get("use_parallel", self.use_parallel)

        # Select appropriate Rust functions based on parallel setting
        whitening_func = apply_whitening_parallel if use_parallel else apply_whitening
        cov_func = compute_covariance_parallel if use_parallel else compute_covariance
        mean_func = compute_mean_parallel if use_parallel else compute_mean

        # Check if we need to compute the whitening matrix
        if not self._is_fitted and compute_now:
            # For immediate computation, take a random subset
            total_samples = data.shape[0]
            if total_samples > sample_size:
                # Take samples from different parts of the data
                indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)
                sample_data = data[indices].compute().astype(np.float32)
            else:
                # Use all data if it's smaller than the sample size
                sample_data = data.compute().astype(np.float32)

            # Compute mean if needed
            if self.apply_mean:
                self._mean = mean_func(sample_data)
                data_centered = sample_data - self._mean
            else:
                self._mean = None
                data_centered = sample_data

            # Compute covariance matrix using Rust implementation
            cov = cov_func(data_centered)

            # Compute whitening matrix using Rust implementation
            self._whitening_matrix = compute_whitening_matrix(cov, self.eps)

            # Mark as fitted
            self._is_fitted = True

        # Define the whitening function for each chunk
        def apply_chunk_whitening(chunk: np.ndarray) -> np.ndarray:
            # Convert to float32 for computation
            chunk_float = chunk.astype(np.float32) if chunk.dtype.kind == "u" else chunk

            if not self._is_fitted:
                # Compute on-the-fly for this chunk
                if self.apply_mean:
                    chunk_mean = mean_func(chunk_float)
                    chunk_centered = chunk_float - chunk_mean
                else:
                    chunk_mean = None
                    chunk_centered = chunk_float

                # Compute covariance matrix using Rust implementation
                chunk_cov = cov_func(chunk_centered)

                # Compute whitening matrix using Rust implementation
                chunk_whitening = compute_whitening_matrix(chunk_cov, self.eps)

                # Apply whitening using Rust implementation
                return whitening_func(
                    chunk_float,
                    chunk_whitening,
                    chunk_mean if self.apply_mean else None,
                    self.int_scale,
                )
            else:
                # Use pre-computed whitening matrix
                return whitening_func(
                    chunk_float,
                    self._whitening_matrix,
                    self._mean if self.apply_mean else None,
                    self.int_scale,
                )

        # Use map_blocks to maintain laziness
        return data.map_blocks(apply_chunk_whitening, dtype=np.float32)

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a dictionary containing processor configuration details"""
        base_summary = super().summary
        base_summary.update(
            {
                "apply_mean": self.apply_mean,
                "eps": self.eps,
                "int_scale": self.int_scale,
                "is_fitted": self._is_fitted,
                "use_parallel": self.use_parallel,
                "rust_implementation": True,
            }
        )
        return base_summary


def create_whitening_processor_rs(
    apply_mean: bool = False,
    int_scale: Optional[float] = None,
    eps: Optional[float] = None,
    use_parallel: bool = True,
) -> WhiteningRustProcessor:
    """
    Create a Rust-accelerated whitening processor.

    Parameters
    ----------
    apply_mean : bool, default: False
        Whether to subtract the mean before whitening
    int_scale : float or None, default: None
        Apply a scaling factor if needed
    eps : float or None, default: None
        Small epsilon to regularize SVD
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    processor : WhiteningRustProcessor
        WhiteningRustProcessor configured for standard ZCA whitening
    """
    if not _HAS_RUST:
        from ..spatial.whiten import create_whitening_processor

        print(
            "Falling back to Python implementation as Rust extension is not available"
        )
        return create_whitening_processor(
            apply_mean=apply_mean,
            int_scale=int_scale,
            eps=eps,
        )

    return WhiteningRustProcessor(
        apply_mean=apply_mean,
        int_scale=int_scale,
        eps=eps,
        use_parallel=use_parallel,
    )


# Direct functions for immediate use without the processor class
def apply_whiten_rs(
    data: np.ndarray,
    apply_mean: bool = False,
    eps: float = 1e-6,
    int_scale: Optional[float] = None,
    use_parallel: bool = True,
) -> np.ndarray:
    """
    Apply whitening to data directly using Rust acceleration.

    Parameters
    ----------
    data : np.ndarray
        Input data
    apply_mean : bool, default: False
        Whether to subtract the mean
    eps : float, default: 1e-6
        Regularization parameter for eigenvalues
    int_scale : float or None, default: None
        Optional scaling factor for output
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    np.ndarray
        Whitened data
    """
    if not _HAS_RUST:
        # Fallback to numpy implementation
        from ..spatial.whiten import WhiteningProcessor

        print(
            "Falling back to Python implementation as Rust extension is not available"
        )
        processor = WhiteningProcessor(
            mode="global",
            apply_mean=apply_mean,
            int_scale=int_scale,
            eps=eps,
        )
        return processor._compute_whitening(data)

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        reshape_back = True
    else:
        reshape_back = False

    # Ensure float32 type
    data = data.astype(np.float32)

    # Select functions based on parallel setting
    whitening_func = apply_whitening_parallel if use_parallel else apply_whitening
    cov_func = compute_covariance_parallel if use_parallel else compute_covariance
    mean_func = compute_mean_parallel if use_parallel else compute_mean

    # Compute mean if needed
    if apply_mean:
        mean = mean_func(data)
        data_centered = data - mean
    else:
        mean = None
        data_centered = data

    # Compute covariance matrix
    cov = cov_func(data_centered)

    # Compute whitening matrix
    whitening_matrix = compute_whitening_matrix(cov, eps)

    # Apply whitening
    whitened = whitening_func(
        data, whitening_matrix, mean if apply_mean else None, int_scale
    )

    # Reshape back if needed
    if reshape_back:
        whitened = whitened.ravel()

    return whitened
