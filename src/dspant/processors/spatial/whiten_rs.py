"""
Rust-accelerated whitening implementations.

This module provides high-performance Rust implementations of whitening algorithms
with Python bindings, while preserving compatibility with Dask arrays for large datasets.
"""

from typing import Any, Dict, Literal, Optional, Union

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
        whiten_data,
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

    Whitens the signal by decorrelating and normalizing the variance.
    Uses optimized Rust implementation for improved performance, especially
    for multi-channel data.
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
        Initialize the Rust-accelerated whitening processor.

        Parameters
        ----------
        mode : "global", default: "global"
            Method to compute the whitening matrix, currently only "global" is supported
        apply_mean : bool, default: False
            Whether to subtract the mean before applying whitening matrix
        int_scale : float or None, default: None
            Apply a scaling factor to fit integer range if needed
        eps : float or None, default: None
            Small epsilon to regularize SVD. If None, it's automatically determined.
        W : np.ndarray or None, default: None
            Pre-computed whitening matrix
        M : np.ndarray or None, default: None
            Pre-computed means
        regularize : bool, default: False
            Whether to use scikit-learn covariance estimators (ignored in Rust version, for compatibility)
        regularize_kwargs : dict or None, default: None
            Parameters for scikit-learn covariance estimator (ignored in Rust version, for compatibility)
        use_parallel : bool, default: True
            Whether to use parallel processing in Rust implementation
        """
        if not _HAS_RUST:
            raise ImportError(
                "Rust extension not available. Install with 'pip install dspant[rust]'"
            )

        # Validate parameters
        if mode != "global":
            raise ValueError("Only 'global' mode is currently supported")

        self.mode = mode
        self.apply_mean = apply_mean
        self.int_scale = int_scale
        self.eps = eps if eps is not None else 1e-6
        self.use_parallel = use_parallel
        self._overlap_samples = 0  # No overlap needed for this operation

        # For compatibility with Python version
        self.regularize = regularize
        self.regularize_kwargs = regularize_kwargs

        # Pre-computed matrices handling
        if W is not None:
            self._whitening_matrix = np.asarray(W).astype(np.float32)
            self._mean = np.asarray(M).astype(np.float32) if M is not None else None
            self._is_fitted = True
        else:
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
                Whether to compute the whitening matrix immediately
            sample_size: int, default: 10000
                Number of samples to use for computing covariance matrix
            use_parallel: bool, default: True
                Whether to use parallel processing

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
        optimize_computation = kwargs.get("optimize_computation", True)

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

                # If we can use the optimized computation path (computes everything at once)
                if optimize_computation and self.apply_mean:
                    # Use the optimized function that computes everything in one go
                    whitened_sample = whiten_data(
                        sample_data, self.eps, self.apply_mean, self.int_scale
                    )
                    # We don't need to extract W and M since we'll recompute for each chunk
                    self._is_fitted = True
                else:
                    # Traditional calculation with separate steps
                    if self.apply_mean:
                        self._mean = compute_mean_parallel(sample_data)
                        data_centered = sample_data - self._mean
                    else:
                        self._mean = None
                        data_centered = sample_data

                    # Compute covariance matrix
                    cov = compute_covariance_parallel(data_centered)

                    # Compute whitening matrix
                    self._whitening_matrix = compute_whitening_matrix(cov, self.eps)
                    self._is_fitted = True

        # Define the whitening function for each chunk
        def apply_chunk_whitening(chunk: np.ndarray) -> np.ndarray:
            # Convert to float32 for computation
            chunk_float = chunk.astype(np.float32) if chunk.dtype.kind == "u" else chunk

            # If we're using the optimized approach and not pre-fitted
            if optimize_computation and not self._is_fitted:
                # Use all-in-one function for best performance
                return whiten_data(
                    chunk_float, self.eps, self.apply_mean, self.int_scale
                )

            # Traditional multi-step approach
            if not self._is_fitted:
                # Compute for this chunk on the fly
                if self.apply_mean:
                    mean = compute_mean_parallel(chunk_float)
                    data_centered = chunk_float - mean
                else:
                    mean = None
                    data_centered = chunk_float

                # Compute covariance and whitening matrix
                cov = compute_covariance_parallel(data_centered)
                whitening_matrix = compute_whitening_matrix(cov, self.eps)

                # Apply the whitening
                return apply_whitening_parallel(
                    chunk_float,
                    whitening_matrix,
                    mean if self.apply_mean else None,
                    self.int_scale,
                )
            else:
                # Use pre-computed matrices
                return apply_whitening_parallel(
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
        mode="global",
        apply_mean=apply_mean,
        int_scale=int_scale,
        eps=eps,
        use_parallel=use_parallel,
    )


@public_api
def create_robust_whitening_processor_rs(
    method: str = "MinCovDet",
    eps: Optional[float] = None,
    apply_mean: bool = True,
    use_parallel: bool = True,
) -> Union[WhiteningRustProcessor, BaseProcessor]:
    """
    Create a robust whitening processor.

    Note: This currently falls back to the Python implementation for robust estimation,
    as the Rust implementation doesn't support robust covariance estimators yet.

    Parameters
    ----------
    method : str, default: "MinCovDet"
        Covariance estimator to use
    eps : float or None, default: None
        Small epsilon to regularize SVD
    apply_mean : bool, default: True
        Whether to subtract the mean before whitening
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    processor : BaseProcessor
        Configured whitening processor with robust covariance estimation
    """
    # Fall back to Python implementation for robust methods
    from ..spatial.whiten import create_robust_whitening_processor

    print(
        "Using Python implementation for robust whitening as Rust doesn't support "
        "robust covariance estimators yet"
    )
    return create_robust_whitening_processor(
        method=method,
        eps=eps,
        apply_mean=apply_mean,
    )


@public_api
# Direct function for immediate use without the processor class
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
        return processor.process(data)

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        reshape_back = True
    else:
        reshape_back = False

    # Ensure float32 type
    data = data.astype(np.float32)

    # Use the optimized all-in-one function
    whitened = whiten_data(data, eps, apply_mean, int_scale)

    # Reshape back if needed
    if reshape_back:
        whitened = whitened.ravel()

    return whitened
