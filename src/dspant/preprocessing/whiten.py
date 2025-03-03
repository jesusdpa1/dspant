from typing import Any, Dict, Literal, Optional, Union

import dask.array as da
import numpy as np
from sklearn.covariance import (
    OAS,
    EmpiricalCovariance,
    GraphicalLassoCV,
    MinCovDet,
    ShrunkCovariance,
)

from ..core.nodes.stream_processing import BaseProcessor


class WhiteningProcessor(BaseProcessor):
    """
    Whitening processor implementation for dspAnt framework

    Whitens the signal by decorrelating and normalizing the variance.

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
        Whether to regularize the covariance matrix using scikit-learn
    regularize_kwargs : dict or None, default: None
        Parameters for the scikit-learn covariance estimator
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
        """Initialize the whitening processor."""
        # Validate parameters
        if mode != "global":
            raise ValueError("Only 'global' mode is currently supported")

        # Check for apply_mean when regularizing
        if not apply_mean and regularize:
            raise ValueError(
                "`apply_mean` must be `True` if regularizing. `assume_centered` is fixed to `True`."
            )

        self.mode = mode
        self.apply_mean = apply_mean
        self.int_scale = int_scale
        self.eps = eps
        self.regularize = regularize
        self.regularize_kwargs = regularize_kwargs or {"method": "GraphicalLassoCV"}
        self._overlap_samples = 0  # No overlap needed for this operation

        # Use pre-computed matrices if provided
        if W is not None:
            self._whitening_matrix = np.asarray(W)
            self._mean = np.asarray(M) if M is not None else None
            self._is_fitted = True
        else:
            self._whitening_matrix = None
            self._mean = None
            self._is_fitted = False

    def _compute_whitening_from_covariance(
        self, cov: np.ndarray, eps: float
    ) -> np.ndarray:
        """
        Compute the whitening matrix from the covariance matrix using ZCA whitening.

        Parameters
        ----------
        cov : np.ndarray
            Covariance matrix
        eps : float
            Small epsilon to regularize SVD

        Returns
        -------
        W : np.ndarray
            Whitening matrix
        """
        U, S, Ut = np.linalg.svd(cov, full_matrices=True)
        W = (U @ np.diag(1 / np.sqrt(S + eps))) @ Ut
        return W

    def fit(self, data: da.Array):
        """
        Compute the whitening matrix from the data.

        Parameters
        ----------
        data : da.Array
            Input data as a Dask array

        Returns
        -------
        self : WhiteningProcessor
            The fitted processor
        """
        # If matrices are already provided, no need to fit
        if self._is_fitted:
            return self

        # Get random data chunks
        random_data = data.compute().astype(np.float32)

        # Compute mean if needed
        if self.apply_mean:
            self._mean = np.mean(random_data, axis=0)
            self._mean = self._mean[np.newaxis, :]
            data_centered = random_data - self._mean
        else:
            self._mean = None
            data_centered = random_data

        # Compute covariance matrix
        if not self.regularize:
            cov = data_centered.T @ data_centered
            cov = cov / data_centered.shape[0]
        else:
            cov = self._compute_sklearn_covariance_matrix(
                data_centered, self.regularize_kwargs
            )
            cov = cov.astype(np.float32)

        # Determine epsilon for SVD regularization
        if self.eps is None:
            median_data_sqr = np.median(data_centered**2)
            if median_data_sqr < 1 and median_data_sqr > 0:
                eps = max(1e-16, median_data_sqr * 1e-3)
            else:
                eps = 1e-16
        else:
            eps = self.eps

        # Compute whitening matrix
        self._whitening_matrix = self._compute_whitening_from_covariance(cov, eps)
        self._is_fitted = True
        return self

    def _compute_sklearn_covariance_matrix(
        self, data: np.ndarray, regularize_kwargs: Dict[str, Any]
    ) -> np.ndarray:
        """
        Estimate the covariance matrix using scikit-learn functions.

        Parameters
        ----------
        data : np.ndarray
            Input data
        regularize_kwargs : dict
            Parameters for the scikit-learn covariance estimator

        Returns
        -------
        cov : np.ndarray
            Estimated covariance matrix
        """
        # Check for assume_centered
        if (
            "assume_centered" in regularize_kwargs
            and not regularize_kwargs["assume_centered"]
        ):
            raise ValueError(
                "Cannot use `assume_centered=False` for `regularize_kwargs`. Fixing to `True`."
            )

        # Get method and create estimator
        method = regularize_kwargs.pop("method")
        regularize_kwargs["assume_centered"] = True

        estimator_map = {
            "EmpiricalCovariance": EmpiricalCovariance,
            "MinCovDet": MinCovDet,
            "OAS": OAS,
            "ShrunkCovariance": ShrunkCovariance,
            "GraphicalLassoCV": GraphicalLassoCV,
        }

        if method not in estimator_map:
            raise ValueError(f"Unknown covariance method: {method}")

        estimator_class = estimator_map[method]
        estimator = estimator_class(**regularize_kwargs)

        # Fit estimator and get covariance
        estimator.fit(
            data.astype(np.float64)
        )  # sklearn covariance methods require float64
        cov = estimator.covariance_

        return cov

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply whitening to the data lazily.

        Parameters
        ----------
        data : da.Array
            Input data as a Dask array
        fs : float, optional
            Sampling frequency (not used but required by interface)
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        whitened_data : da.Array
            Whitened data as a Dask array
        """
        # Ensure the data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Fit if not already fitted
        if not self._is_fitted:
            self.fit(data)

        # Define the whitening function for each chunk
        def apply_whitening(chunk: np.ndarray) -> np.ndarray:
            # Convert to float32 for computation
            traces_dtype = chunk.dtype
            if traces_dtype.kind == "u":
                chunk = chunk.astype(np.float32)

            # Apply whitening
            if self._mean is not None:
                whitened = (chunk - self._mean) @ self._whitening_matrix
            else:
                whitened = chunk @ self._whitening_matrix

            # Apply scaling if needed
            if self.int_scale is not None:
                whitened *= self.int_scale

            return whitened

        # Use map_blocks to maintain laziness
        return data.map_blocks(apply_whitening, dtype=np.float32)

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
                "regularize": self.regularize,
                "regularize_method": self.regularize_kwargs.get("method", "None")
                if self.regularize
                else "None",
            }
        )
        return base_summary


def create_whitening_processor(
    apply_mean: bool = False,
    int_scale: Optional[float] = None,
    eps: Optional[float] = None,
    regularize: bool = False,
    regularize_kwargs: Optional[Dict[str, Any]] = None,
) -> WhiteningProcessor:
    """
    Create a whitening processor with standard parameters.

    Parameters
    ----------
    apply_mean : bool, default: False
        Whether to subtract the mean before whitening
    int_scale : float or None, default: None
        Apply a scaling factor if needed
    eps : float or None, default: None
        Small epsilon to regularize SVD
    regularize : bool, default: False
        Whether to regularize the covariance matrix
    regularize_kwargs : dict or None, default: None
        Parameters for the covariance estimator

    Returns
    -------
    processor : WhiteningProcessor
        WhiteningProcessor configured for standard ZCA whitening
    """
    return WhiteningProcessor(
        mode="global",
        apply_mean=apply_mean,
        int_scale=int_scale,
        eps=eps,
        regularize=regularize,
        regularize_kwargs=regularize_kwargs,
    )


def create_robust_whitening_processor(
    method: str = "MinCovDet",
    eps: Optional[float] = None,
    apply_mean: bool = True,
    estimator_kwargs: Optional[Dict[str, Any]] = None,
) -> WhiteningProcessor:
    """
    Create a whitening processor with robust covariance estimation.

    Parameters
    ----------
    method : str, default: "MinCovDet"
        Covariance estimator to use
    eps : float or None, default: None
        Small epsilon to regularize SVD
    apply_mean : bool, default: True
        Whether to subtract the mean before whitening
    estimator_kwargs : dict or None, default: None
        Additional parameters for the covariance estimator

    Returns
    -------
    processor : WhiteningProcessor
        WhiteningProcessor with robust covariance estimation
    """
    regularize_kwargs = {"method": method}
    if estimator_kwargs:
        regularize_kwargs.update(estimator_kwargs)

    return WhiteningProcessor(
        mode="global",
        apply_mean=apply_mean,
        eps=eps,
        regularize=True,
        regularize_kwargs=regularize_kwargs,
    )
