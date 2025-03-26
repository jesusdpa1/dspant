"""
Rust-accelerated whitening implementations.

This module provides high-performance Rust implementations of whitening algorithms
with Python bindings, while preserving compatibility with Dask arrays for large datasets.
"""

from typing import Any, Dict, Literal, Optional, Tuple

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor

try:
    from dspant._rs import (
        apply_whitening_parallel,
        compute_covariance_parallel,
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
        use_parallel: bool = True,
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
        self.use_parallel = use_parallel

        # Pre-computed matrices handling
        if W is not None:
            self._whitening_matrix = np.asarray(W)
            self._mean = np.asarray(M) if M is not None else None
            self._is_fitted = True
        else:
            self._whitening_matrix = None
            self._mean = None
            self._is_fitted = False

    def _compute_whitening_matrix_for_data(
        self, sample_data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Compute whitening matrix from sample data.

        Args:
            sample_data: Representative data sample to compute whitening matrix

        Returns:
            Tuple of (whitening_matrix, mean)
        """
        # Compute mean if needed
        if self.apply_mean:
            if self.use_parallel:
                mean = compute_mean_parallel(sample_data)
            else:
                mean = np.mean(sample_data, axis=0, keepdims=True)
            data_centered = sample_data - mean
        else:
            mean = None
            data_centered = sample_data

        # Compute covariance matrix
        if not self.regularize:
            if self.use_parallel:
                cov = compute_covariance_parallel(data_centered)
            else:
                cov = data_centered.T @ data_centered / data_centered.shape[0]
        else:
            # For robust estimation, we'll use the Python implementation for now
            # since the Rust implementation doesn't support this yet
            from sklearn.covariance import (
                OAS,
                EmpiricalCovariance,
                GraphicalLassoCV,
                MinCovDet,
                ShrunkCovariance,
            )

            # Get method and create estimator
            method_name = self.regularize_kwargs.get("method", "GraphicalLassoCV")
            reg_kwargs = self.regularize_kwargs.copy()
            reg_kwargs["assume_centered"] = True

            estimator_map = {
                "EmpiricalCovariance": EmpiricalCovariance,
                "MinCovDet": MinCovDet,
                "OAS": OAS,
                "ShrunkCovariance": ShrunkCovariance,
                "GraphicalLassoCV": GraphicalLassoCV,
            }

            if method_name not in estimator_map:
                raise ValueError(f"Unknown covariance method: {method_name}")

            estimator_class = estimator_map[method_name]
            estimator = estimator_class(**reg_kwargs)

            # Fit estimator and get covariance
            estimator.fit(data_centered.astype(np.float64))
            cov = estimator.covariance_.astype(np.float32)

        # Determine epsilon for regularization
        if self.eps is None:
            median_data_sqr = np.median(data_centered**2)
            eps = (
                max(1e-16, median_data_sqr * 1e-3) if 0 < median_data_sqr < 1 else 1e-16
            )
        else:
            eps = self.eps

        # Compute whitening matrix using Rust implementation
        whitening_matrix = compute_whitening_matrix(cov.astype(np.float32), eps)

        return whitening_matrix, mean

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply whitening to the data lazily using Rust implementation.

        This improved implementation computes the whitening matrix once using
        a representative sample, rather than recomputing for each chunk.

        Args:
            data: Input data as a Dask array
            fs: Sampling frequency (not used but required by interface)
            **kwargs: Additional keyword arguments
                compute_now: Whether to compute the whitening matrix immediately (default: False)
                sample_size: Number of samples to use for computing whitening matrix (default: 10000)
                use_parallel: Whether to use parallel processing (default: True)

        Returns:
            Whitened data as a Dask array
        """
        # Ensure the data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Get parameters from kwargs
        compute_now = kwargs.get("compute_now", False)
        sample_size = kwargs.get("sample_size", 10000)
        use_parallel = kwargs.get("use_parallel", self.use_parallel)

        # If not already fitted, compute the whitening matrix once
        if not self._is_fitted:
            # Option 1: Compute immediately using a subset of data
            if compute_now:
                # Extract a subset of the data
                total_samples = data.shape[0]
                if total_samples > sample_size:
                    # Take samples from different parts of the data
                    indices = np.linspace(0, total_samples - 1, sample_size, dtype=int)
                    sample_data = data[indices].compute().astype(np.float32)
                else:
                    # Use all data if it's smaller than sample size
                    sample_data = data.compute().astype(np.float32)

                # Compute whitening matrix once from the sample data
                self._whitening_matrix, self._mean = (
                    self._compute_whitening_matrix_for_data(sample_data)
                )
                self._is_fitted = True
            else:
                # Option 2: Compute from first chunk only (will be triggered once)
                # To prevent recomputation for every chunk, we'll use a "first chunk" strategy
                # This flag lets us do lazy computation that happens only once
                self._compute_from_first_chunk = True

        # Define the whitening function for each chunk
        def apply_chunk_whitening(chunk: np.ndarray) -> np.ndarray:
            # Convert to float32 for computation
            chunk_float = chunk.astype(np.float32) if chunk.dtype.kind == "u" else chunk

            # For lazy computation, compute matrix from first chunk only
            if not self._is_fitted and getattr(
                self, "_compute_from_first_chunk", False
            ):
                # Compute whitening matrix from this chunk
                self._whitening_matrix, self._mean = (
                    self._compute_whitening_matrix_for_data(chunk_float)
                )
                self._is_fitted = True
                # Clear the flag to prevent recomputation
                self._compute_from_first_chunk = False

            # Now apply whitening using the matrix (either pre-computed or computed from first chunk)
            if self._is_fitted:
                # Use Rust parallel implementation if requested
                if use_parallel:
                    result = apply_whitening_parallel(
                        chunk_float,
                        self._whitening_matrix,
                        self._mean if self.apply_mean else None,
                        self.int_scale,
                    )
                else:
                    # Use non-parallel version
                    from dspant._rs import apply_whitening

                    result = apply_whitening(
                        chunk_float,
                        self._whitening_matrix,
                        self._mean if self.apply_mean else None,
                        self.int_scale,
                    )
                return result
            else:
                # This should never happen as we've ensured matrix computation above
                raise RuntimeError("Whitening matrix computation failed")

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
                "mode": self.mode,
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
    if use_parallel:
        mean_func = compute_mean_parallel
        cov_func = compute_covariance_parallel
        whitening_func = apply_whitening_parallel
    else:
        from dspant._rs import apply_whitening

        whitening_func = apply_whitening

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
