"""
Rust-accelerated whitening implementations.

This module provides high-performance Rust implementations of whitening algorithms
with Python bindings, while preserving compatibility with Dask arrays for large datasets.
"""

from typing import Any, Dict, Literal, Optional

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
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


@public_api
class WhiteningRustProcessor(BaseProcessor):
    """
    Rust-accelerated whitening processor implementation.

    Key modifications to ensure lazy evaluation and consistent behavior
    with the Python implementation.
    """

    def __init__(
        self,
        mode: Literal["global"] = "global",
        apply_mean: bool = False,
        int_scale: Optional[float] = None,
        eps: Optional[float] = None,
        W: Optional[np.ndarray] = None,
        M: Optional[np.ndarray] = None,
        regularize: bool = False,
        regularize_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize with more flexible configuration matching Python implementation.
        """
        # Add mode parameter to match Python implementation
        if mode != "global":
            raise ValueError("Only 'global' mode is currently supported")

        # Validate parameters similarly to Python implementation
        if not apply_mean and regularize:
            raise ValueError(
                "`apply_mean` must be `True` if regularizing. `assume_centered` is fixed to `True`."
            )

        # Initialization similar to Python implementation
        self.mode = mode
        self.apply_mean = apply_mean
        self.int_scale = int_scale
        self.eps = eps if eps is not None else 1e-6
        self.regularize = regularize
        self.regularize_kwargs = regularize_kwargs or {"method": "GraphicalLassoCV"}
        self._overlap_samples = 0  # No overlap needed for this operation

        # Pre-computed matrices handling
        if W is not None:
            self._whitening_matrix = np.asarray(W)
            self._mean = np.asarray(M) if M is not None else None
            self._is_fitted = True
        else:
            self._whitening_matrix = None
            self._mean = None
            self._is_fitted = False

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply whitening to the data lazily using Rust implementation.

        Enhanced to more closely match Python implementation's lazy evaluation.
        """
        # Ensure the data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Get parameters from kwargs with defaults matching Python implementation
        compute_now = kwargs.get("compute_now", False)
        sample_size = kwargs.get("sample_size", 10000)
        use_parallel = kwargs.get("use_parallel", True)

        # Select appropriate Rust functions based on parallel setting
        whitening_func = apply_whitening_parallel if use_parallel else apply_whitening
        cov_func = compute_covariance_parallel if use_parallel else compute_covariance
        mean_func = compute_mean_parallel if use_parallel else compute_mean

        # Computation strategy mirroring Python implementation
        def compute_whitening_matrix_for_data(sample_data):
            """
            Compute whitening matrix following Python implementation logic.
            """
            # Compute mean if needed
            if self.apply_mean:
                mean = np.mean(sample_data, axis=0)
                mean = mean[np.newaxis, :]
                data_centered = sample_data - mean
            else:
                mean = None
                data_centered = sample_data

            # Compute covariance matrix
            if not self.regularize:
                cov = data_centered.T @ data_centered / data_centered.shape[0]
            else:
                # Placeholder for robust covariance computation if needed
                cov = cov_func(data_centered)

            # Determine epsilon for regularization
            if self.eps is None:
                median_data_sqr = np.median(data_centered**2)
                eps = (
                    max(1e-16, median_data_sqr * 1e-3)
                    if 0 < median_data_sqr < 1
                    else 1e-16
                )
            else:
                eps = self.eps

            # Compute whitening matrix
            whitening_matrix = compute_whitening_matrix(cov.astype(np.float32), eps)

            return whitening_matrix, mean

        # Compute whitening matrix if not already fitted
        if not self._is_fitted:
            # For immediate computation or when specifically requested
            if compute_now:
                total_samples = data.shape[0]
                if total_samples > sample_size:
                    # Take samples from different parts of the data
                    indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)
                    sample_data = data[indices].compute().astype(np.float32)
                else:
                    # Use all data if it's smaller than the sample size
                    sample_data = data.compute().astype(np.float32)

                # Compute whitening matrix
                self._whitening_matrix, self._mean = compute_whitening_matrix_for_data(
                    sample_data
                )
                self._is_fitted = True

        # Define the whitening function for each chunk
        def apply_chunk_whitening(chunk: np.ndarray) -> np.ndarray:
            # Convert to float32 for computation
            chunk_float = chunk.astype(np.float32) if chunk.dtype.kind == "u" else chunk

            # Lazy computation of whitening matrix if not already fitted
            if not self._is_fitted:
                # Compute whitening matrix for this chunk
                whitening_matrix, mean = compute_whitening_matrix_for_data(chunk_float)
            else:
                whitening_matrix = self._whitening_matrix
                mean = self._mean

            # Apply whitening
            result = whitening_func(
                chunk_float,
                whitening_matrix,
                mean if self.apply_mean else None,
                self.int_scale,
            )

            return result

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


@public_api
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


@public_api
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
