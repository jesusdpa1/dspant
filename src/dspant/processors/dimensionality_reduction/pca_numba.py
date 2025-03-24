# src/dspant/processor/dimensionality_reduction/pca_numba.py
"""
Numba-accelerated PCA implementation for dimensionality reduction.

This module provides a highly optimized PCA implementation using Numba for
acceleration, with no dependency on scikit-learn. This allows for better
performance and avoids type consistency issues when used in processing pipelines.
"""

from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from .base import BaseDimensionalityReductionProcessor


@jit(nopython=True, cache=True)
def _center_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center the data by subtracting the mean.

    Args:
        X: Input data array (n_samples, n_features)

    Returns:
        Tuple of (centered_data, mean)
    """
    mean = np.zeros(X.shape[1], dtype=np.float32)

    # Compute mean for each feature
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            mean[j] += X[i, j]
        mean[j] /= X.shape[0]

    # Center the data
    X_centered = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_centered[i, j] = X[i, j] - mean[j]

    return X_centered, mean


@jit(nopython=True, cache=True)
def _compute_covariance(X_centered: np.ndarray) -> np.ndarray:
    """
    Compute the covariance matrix from centered data.

    Args:
        X_centered: Centered data array (n_samples, n_features)

    Returns:
        Covariance matrix (n_features, n_features)
    """
    n_samples = X_centered.shape[0]
    n_features = X_centered.shape[1]
    cov = np.zeros((n_features, n_features), dtype=np.float32)

    # Compute covariance matrix
    for i in range(n_features):
        for j in range(i, n_features):
            cov_val = 0.0
            for k in range(n_samples):
                cov_val += X_centered[k, i] * X_centered[k, j]
            cov_val /= n_samples - 1
            cov[i, j] = cov_val
            cov[j, i] = cov_val  # Covariance matrix is symmetric

    return cov


@jit(nopython=True, cache=True)
def _sort_eigenvectors(
    eigenvals: np.ndarray, eigenvecs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort eigenvalues in descending order and reorder eigenvectors accordingly.

    Args:
        eigenvals: Array of eigenvalues
        eigenvecs: Matrix of eigenvectors (each column is an eigenvector)

    Returns:
        Tuple of (sorted_eigenvalues, sorted_eigenvectors)
    """
    # Get sorting indices in descending order
    sorted_indices = np.argsort(-eigenvals)

    # Sort eigenvalues
    sorted_eigenvals = np.zeros_like(eigenvals, dtype=np.float32)
    for i in range(len(eigenvals)):
        sorted_eigenvals[i] = eigenvals[sorted_indices[i]]

    # Sort eigenvectors
    n_features = eigenvecs.shape[0]
    sorted_eigenvecs = np.zeros_like(eigenvecs, dtype=np.float32)

    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        for j in range(n_features):
            sorted_eigenvecs[j, i] = eigenvecs[j, idx]

    return sorted_eigenvals, sorted_eigenvecs


@jit(nopython=True, cache=True)
def _select_components(
    eigenvals: np.ndarray, eigenvecs: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the top n_components eigenvectors and compute explained variance ratio.

    Args:
        eigenvals: Sorted eigenvalues in descending order
        eigenvecs: Sorted eigenvectors (each column is an eigenvector)
        n_components: Number of components to select

    Returns:
        Tuple of (components, explained_variance, explained_variance_ratio)
    """
    # Select top components
    n_components = min(n_components, len(eigenvals))

    # Compute explained variance and explained variance ratio
    total_var = np.sum(eigenvals)
    explained_variance = np.zeros(n_components, dtype=np.float32)
    explained_variance_ratio = np.zeros(n_components, dtype=np.float32)

    for i in range(n_components):
        explained_variance[i] = eigenvals[i]
        explained_variance_ratio[i] = eigenvals[i] / total_var

    # Select top eigenvectors as components
    components = np.zeros((n_components, eigenvecs.shape[0]), dtype=np.float32)
    for i in range(n_components):
        for j in range(eigenvecs.shape[0]):
            components[i, j] = eigenvecs[j, i]

    return components, explained_variance, explained_variance_ratio


@jit(nopython=True, cache=True)
def _whiten_components(
    components: np.ndarray, explained_variance: np.ndarray
) -> np.ndarray:
    """
    Apply whitening to the components.

    Args:
        components: PCA components
        explained_variance: Explained variance for each component

    Returns:
        Whitened components
    """
    n_components, n_features = components.shape
    whitened_components = np.zeros_like(components, dtype=np.float32)

    # Apply whitening transformation
    for i in range(n_components):
        scaling_factor = 1.0 / np.sqrt(explained_variance[i])
        for j in range(n_features):
            whitened_components[i, j] = components[i, j] * scaling_factor

    return whitened_components


@jit(nopython=True, cache=True)
def _transform_data(
    X: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """
    Transform data using PCA components.

    Args:
        X: Input data array (n_samples, n_features)
        mean: Mean values for each feature
        components: PCA components (n_components, n_features)

    Returns:
        Transformed data (n_samples, n_components)
    """
    n_samples = X.shape[0]
    n_components = components.shape[0]
    transformed = np.zeros((n_samples, n_components), dtype=np.float32)

    # Center the data
    X_centered = np.empty_like(X, dtype=np.float32)
    for i in range(n_samples):
        for j in range(X.shape[1]):
            X_centered[i, j] = X[i, j] - mean[j]

    # Project data onto components
    for i in range(n_samples):
        for j in range(n_components):
            for k in range(X.shape[1]):
                transformed[i, j] += X_centered[i, k] * components[j, k]

    return transformed


@jit(nopython=True, cache=True)
def _inverse_transform(
    X_transformed: np.ndarray, mean: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """
    Transform data back to its original space.

    Args:
        X_transformed: Transformed data (n_samples, n_components)
        mean: Mean values for each feature
        components: PCA components (n_components, n_features)

    Returns:
        Data in original space (n_samples, n_features)
    """
    n_samples = X_transformed.shape[0]
    n_features = components.shape[1]
    reconstructed = np.zeros((n_samples, n_features), dtype=np.float32)

    # Project back to original space
    for i in range(n_samples):
        for j in range(n_features):
            # Add contribution from each component
            for k in range(X_transformed.shape[1]):
                reconstructed[i, j] += X_transformed[i, k] * components[k, j]
            # Add mean
            reconstructed[i, j] += mean[j]

    return reconstructed


@jit(nopython=True, parallel=True, cache=True)
def _normalize_data_parallel(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for PCA processing with parallel processing.

    Args:
        data: Input data array (n_samples, n_features)

    Returns:
        Normalized data
    """
    n_samples = data.shape[0]
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in prange(n_samples):
        # Find min and max for each sample
        min_val = np.inf
        max_val = -np.inf
        for j in range(data.shape[1]):
            if data[i, j] < min_val:
                min_val = data[i, j]
            if data[i, j] > max_val:
                max_val = data[i, j]

        # Apply min-max normalization
        if max_val - min_val > 1e-10:
            for j in range(data.shape[1]):
                normalized[i, j] = (data[i, j] - min_val) / (max_val - min_val)
        else:
            # If range is too small, center the data
            sum_val = 0.0
            for j in range(data.shape[1]):
                sum_val += data[i, j]
            mean_val = sum_val / data.shape[1]

            for j in range(data.shape[1]):
                normalized[i, j] = data[i, j] - mean_val

    return normalized


@jit(nopython=True, parallel=True, cache=True)
def _flatten_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Efficiently flatten 3D data to 2D for PCA with parallelization.

    Args:
        data: Input 3D data array (samples × timepoints × channels)

    Returns:
        Flattened 2D data (samples × (timepoints*channels))
    """
    n_samples, n_timepoints, n_channels = data.shape
    flattened = np.zeros((n_samples, n_timepoints * n_channels), dtype=data.dtype)

    for i in prange(n_samples):
        for j in range(n_timepoints):
            for k in range(n_channels):
                flattened[i, j * n_channels + k] = data[i, j, k]

    return flattened


@jit(nopython=True, cache=True)
def fit_pca(
    X: np.ndarray, n_components: int, whiten: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA model to data.

    Args:
        X: Input data array (n_samples, n_features)
        n_components: Number of components to extract
        whiten: Whether to apply whitening

    Returns:
        Tuple of (components, mean, explained_variance, explained_variance_ratio)
    """
    # Center the data
    X_centered, mean = _center_data(X)

    # Compute covariance matrix
    cov = _compute_covariance(X_centered)

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors
    eigenvals, eigenvecs = _sort_eigenvectors(eigenvals, eigenvecs)

    # Select components and compute explained variance
    components, explained_variance, explained_variance_ratio = _select_components(
        eigenvals, eigenvecs, n_components
    )

    # Apply whitening if requested
    if whiten:
        components = _whiten_components(components, explained_variance)

    return components, mean, explained_variance, explained_variance_ratio


class NumbaRealPCAProcessor(BaseDimensionalityReductionProcessor):
    """
    Pure Numba-accelerated Principal Component Analysis (PCA) processor.

    This processor provides high-performance dimensionality reduction through PCA
    using pure Numba with no scikit-learn dependency, ensuring consistent types
    when used in processing pipelines.
    """

    def __init__(
        self,
        n_components: int = 10,
        normalize: bool = True,
        whiten: bool = False,
        parallel_threshold: int = 1000,
    ):
        """
        Initialize the Numba PCA processor.

        Args:
            n_components: Number of components to keep
            normalize: Whether to normalize data before PCA
            whiten: Whether to whiten the data
            parallel_threshold: Minimum number of samples to use parallel processing
        """
        super().__init__()
        self.n_components = n_components
        self.normalize = normalize
        self.whiten = whiten
        self.parallel_threshold = parallel_threshold

        # Model parameters
        self._components = None
        self._mean = None
        self._explained_variance = None
        self._explained_variance_ratio = None

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess data before PCA.

        Args:
            data: Input data array

        Returns:
            Preprocessed data
        """
        # Handle different input shapes
        if data.ndim > 2:
            data_flat = _flatten_3d_data(data)
        else:
            data_flat = data

        # Ensure data is float32 for consistent processing
        if data_flat.dtype != np.float32:
            data_flat = data_flat.astype(np.float32)

        # Normalize if requested
        if self.normalize:
            if data_flat.shape[0] >= self.parallel_threshold:
                data_flat = _normalize_data_parallel(data_flat)
            else:
                # For smaller datasets, use non-parallel version
                data_flat = _normalize_data_parallel(
                    data_flat
                )  # Still using parallel for now

        return data_flat

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data using PCA.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, but required by BaseProcessor interface)
            **kwargs: Additional keyword arguments
                compute_now: Whether to compute transformed data immediately

        Returns:
            Dask array with reduced dimensions
        """
        compute_now = kwargs.get("compute_now", True)

        if compute_now:
            # Process all data at once
            data_np = data.compute()
            result = self.fit_transform(data_np)
            return da.from_array(result)
        else:
            # Process in chunks
            def process_chunk(chunk):
                if not self._is_fitted:
                    # Fit and transform
                    return self.fit_transform(chunk)
                else:
                    # Only transform using existing model
                    return self.transform(chunk)

            # Determine output shape by processing a small sample
            if not self._is_fitted:
                sample_data = data[: min(10, data.shape[0])].compute()
                sample_output = self.fit_transform(sample_data)
                self._is_fitted = True  # Mark as fitted after processing sample
            else:
                sample_data = data[: min(10, data.shape[0])].compute()
                sample_output = self.transform(sample_data)

            # Apply function to dask array
            result = data.map_blocks(
                process_chunk,
                drop_axis=list(range(1, data.ndim)),  # Remove all but first dimension
                new_axis=list(range(1, len(sample_output.shape))),  # Add new dimensions
                dtype=np.float32,
            )

            return result

    def fit(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> "NumbaRealPCAProcessor":
        """
        Fit the PCA model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data
        data_flat = self._preprocess_data(data)

        # Determine number of components
        n_samples, n_features = data_flat.shape
        n_components = min(self.n_components, n_features, n_samples)

        # Fit PCA using Numba function
        components, mean, explained_variance, explained_variance_ratio = fit_pca(
            data_flat, n_components, self.whiten
        )

        # Store model parameters
        self._components = components
        self._mean = mean
        self._explained_variance = explained_variance
        self._explained_variance_ratio = explained_variance_ratio
        self._is_fitted = True

        return self

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data using the fitted PCA model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Transformed data with reduced dimensions
        """
        if not self._is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data
        data_flat = self._preprocess_data(data)

        # Transform data using Numba function
        return _transform_data(data_flat, self._mean, self._components)

    def fit_transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Fit the model and transform the data in one step.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Transformed data
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def inverse_transform(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> np.ndarray:
        """
        Transform data back to its original space.

        Args:
            data: Input data array in reduced dimensions
            **kwargs: Additional keyword arguments
                original_shape: Original shape to reshape result (if input was >2D)

        Returns:
            Data in original space
        """
        if not self._is_fitted:
            raise ValueError("PCA model not fitted. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Apply inverse transform
        reconstructed = _inverse_transform(data, self._mean, self._components)

        # Reshape if original shape is provided
        original_shape = kwargs.get("original_shape", None)
        if original_shape is not None and len(original_shape) > 2:
            reconstructed = reconstructed.reshape(original_shape)

        return reconstructed

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """
        Get the explained variance.

        Returns:
            Array of explained variance if available, None otherwise
        """
        return self._explained_variance

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """
        Get the explained variance ratio.

        Returns:
            Array of explained variance ratios if available, None otherwise
        """
        return self._explained_variance_ratio

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "n_components": self.n_components,
                "normalize": self.normalize,
                "whiten": self.whiten,
                "is_fitted": self._is_fitted,
                "explained_variance_sum": (
                    np.sum(self._explained_variance_ratio)
                    if self._explained_variance_ratio is not None
                    else None
                ),
                "implementation": "pure_numba",
            }
        )
        return base_summary


# Factory functions for easy creation
def create_numba_pca(
    n_components: int = 10,
    normalize: bool = True,
    whiten: bool = False,
) -> NumbaRealPCAProcessor:
    """
    Create a pure Numba PCA processor with common parameters.

    Args:
        n_components: Number of components to keep
        normalize: Whether to normalize data before PCA
        whiten: Whether to whiten the data

    Returns:
        Configured NumbaRealPCAProcessor
    """
    return NumbaRealPCAProcessor(
        n_components=n_components,
        normalize=normalize,
        whiten=whiten,
    )


def create_numba_whitening_pca(
    n_components: int = 10,
    normalize: bool = True,
) -> NumbaRealPCAProcessor:
    """
    Create a pure Numba PCA processor with whitening enabled.

    Whitening transforms the data to have uncorrelated components with unit variance,
    which can improve subsequent processing like clustering.

    Args:
        n_components: Number of components to keep
        normalize: Whether to normalize data before PCA

    Returns:
        Configured NumbaRealPCAProcessor with whitening enabled
    """
    return NumbaRealPCAProcessor(
        n_components=n_components,
        normalize=normalize,
        whiten=True,
    )


def create_fast_numba_pca(
    n_components: int = 10,
    normalize: bool = True,
) -> NumbaRealPCAProcessor:
    """
    Create a pure Numba PCA processor optimized for maximum speed.

    Args:
        n_components: Number of components to keep
        normalize: Whether to normalize data before PCA

    Returns:
        NumbaRealPCAProcessor optimized for speed
    """
    return NumbaRealPCAProcessor(
        n_components=n_components,
        normalize=normalize,
        whiten=False,  # Whitening adds computational overhead
        parallel_threshold=500,  # Lower threshold to use parallel processing more often
    )
