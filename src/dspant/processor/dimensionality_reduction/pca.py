# src/dspant/processor/dimensionality_reduction/pca.py
"""
PCA implementation for dimensionality reduction.

This module provides optimized PCA implementations using
dask and numba for acceleration.
"""

from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from sklearn.decomposition import PCA

from .base import BaseDimensionalityReductionProcessor


@jit(nopython=True, parallel=True, cache=True)
def _normalize_data_parallel(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for PCA processing with parallel processing.

    Args:
        data: Input data array (n_samples x n_features)

    Returns:
        Normalized data
    """
    n_samples = data.shape[0]
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in prange(n_samples):
        sample = data[i]
        min_val = np.min(sample)
        max_val = np.max(sample)

        if max_val - min_val > 1e-10:
            normalized[i] = (sample - min_val) / (max_val - min_val)
        else:
            # Calculate mean without np.mean for better numba performance
            mean_val = 0.0
            for j in range(len(sample)):
                mean_val += sample[j]
            mean_val /= len(sample)

            for j in range(len(sample)):
                normalized[i, j] = sample[j] - mean_val

    return normalized


@jit(nopython=True, cache=True)
def _normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for PCA processing (non-parallel version for smaller datasets).

    Args:
        data: Input data array

    Returns:
        Normalized data
    """
    n_samples = data.shape[0]
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in range(n_samples):
        sample = data[i]
        min_val = np.min(sample)
        max_val = np.max(sample)

        if max_val - min_val > 1e-10:
            normalized[i] = (sample - min_val) / (max_val - min_val)
        else:
            # Calculate mean without np.mean for better numba performance
            mean_val = 0.0
            for j in range(len(sample)):
                mean_val += sample[j]
            mean_val /= len(sample)

            for j in range(len(sample)):
                normalized[i, j] = sample[j] - mean_val

    return normalized


@jit(nopython=True, parallel=True, cache=True)
def _normalize_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize 3D data for PCA processing (waveforms × samples × channels).

    Args:
        data: Input 3D data array

    Returns:
        Normalized data
    """
    n_waveforms, n_samples, n_channels = data.shape
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in prange(n_waveforms):
        for c in range(n_channels):
            waveform = data[i, :, c]
            min_val = np.min(waveform)
            max_val = np.max(waveform)

            if max_val - min_val > 1e-10:
                for j in range(n_samples):
                    normalized[i, j, c] = (waveform[j] - min_val) / (max_val - min_val)
            else:
                # Calculate mean manually
                mean_val = 0.0
                for j in range(n_samples):
                    mean_val += waveform[j]
                mean_val /= n_samples

                for j in range(n_samples):
                    normalized[i, j, c] = waveform[j] - mean_val

    return normalized


@jit(nopython=True, parallel=True, cache=True)
def _flatten_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Efficiently flatten 3D data to 2D for PCA with parallelization.

    Args:
        data: Input 3D data array (waveforms × samples × channels)

    Returns:
        Flattened 2D data (waveforms × (samples*channels))
    """
    n_waveforms, n_samples, n_channels = data.shape
    flattened = np.zeros((n_waveforms, n_samples * n_channels), dtype=data.dtype)

    for i in prange(n_waveforms):
        for j in range(n_samples):
            for k in range(n_channels):
                flattened[i, j * n_channels + k] = data[i, j, k]

    return flattened


class PCAProcessor(BaseDimensionalityReductionProcessor):
    """
    Principal Component Analysis (PCA) processor implementation.

    This processor provides dimensionality reduction through PCA,
    optimized for large datasets with dask and numba acceleration.
    """

    def __init__(
        self,
        n_components: int = 10,
        normalize: bool = True,
        random_state: int = 42,
        whiten: bool = False,
        use_numba: bool = True,
        parallel_threshold: int = 1000,
    ):
        """
        Initialize the PCA processor.

        Args:
            n_components: Number of components to keep
            normalize: Whether to normalize data before PCA
            random_state: Random seed for reproducibility
            whiten: Whether to whiten the data
            use_numba: Whether to use Numba acceleration
            parallel_threshold: Minimum number of samples to use parallel processing
        """
        super().__init__()
        self.n_components = n_components
        self.normalize = normalize
        self.random_state = random_state
        self.whiten = whiten
        self.use_numba = use_numba
        self.parallel_threshold = parallel_threshold

        # Initialize model
        self._pca = None
        self._explained_variance_ratio = None
        self._mean = None

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess data before PCA with optimal Numba acceleration.

        Args:
            data: Input data array

        Returns:
            Preprocessed data ready for PCA
        """
        # Handle different input shapes
        if data.ndim > 2:
            n_samples = data.shape[0]

            # Use Numba acceleration if enabled
            if self.use_numba:
                data_flat = _flatten_3d_data(data)
            else:
                data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        # Normalize if requested
        if self.normalize:
            if data_flat.ndim == 2 and self.use_numba:
                # Use parallel version for larger datasets
                if data_flat.shape[0] >= self.parallel_threshold:
                    data_flat = _normalize_data_parallel(data_flat)
                else:
                    data_flat = _normalize_data(data_flat)
            else:
                # Fallback for other cases or when Numba is disabled
                data_flat = (data_flat - np.min(data_flat)) / (
                    np.max(data_flat) - np.min(data_flat) + 1e-10
                )

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

        # Define preprocessing and dimensionality reduction function
        def reduce_dimensions(data_chunk: np.ndarray) -> np.ndarray:
            """Process a chunk of data to perform PCA."""
            # Ensure input is a numpy array
            data_chunk = np.asarray(data_chunk)

            # Check if we have any data
            if len(data_chunk) == 0:
                return np.array([])

            # Preprocess data with optimized functions
            data_flat = self._preprocess_data(data_chunk)

            # Initialize or update PCA model
            if self._pca is None:
                n_samples, n_features = data_flat.shape
                n_components = min(self.n_components, n_features, n_samples)

                self._pca = PCA(
                    n_components=n_components,
                    random_state=self.random_state,
                    whiten=self.whiten,
                )

                # Fit PCA with the data
                transformed = self._pca.fit_transform(data_flat)
                self._explained_variance_ratio = self._pca.explained_variance_ratio_
                self._mean = self._pca.mean_
                self._components = self._pca.components_
                self._is_fitted = True
            else:
                # Use existing PCA model
                transformed = self._pca.transform(data_flat)

            return transformed

        # Convert Dask array to NumPy if compute_now is True
        if compute_now:
            data_np = data.compute()
            transformed = reduce_dimensions(data_np)
            return da.from_array(transformed)
        else:
            # Determine output shape by processing a small sample
            sample_data = data[: min(10, data.shape[0])].compute()
            sample_output = reduce_dimensions(sample_data)
            output_shape = (data.shape[0],) + sample_output.shape[1:]

            # Apply dimensionality reduction function to Dask array
            result = data.map_blocks(
                reduce_dimensions,
                drop_axis=list(range(1, data.ndim)),  # Remove all but first dimension
                new_axis=list(range(1, len(output_shape))),  # Add new dimensions
                dtype=np.float32,
            )

            return result

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "PCAProcessor":
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

        # Preprocess data with optimized functions
        data_flat = self._preprocess_data(data)

        # Determine number of components
        n_samples, n_features = data_flat.shape
        n_components = min(self.n_components, n_features, n_samples)

        # Initialize and fit PCA model
        self._pca = PCA(
            n_components=n_components,
            random_state=self.random_state,
            whiten=self.whiten,
        )

        self._pca.fit(data_flat)
        self._explained_variance_ratio = self._pca.explained_variance_ratio_
        self._mean = self._pca.mean_
        self._components = self._pca.components_
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

        # Preprocess data with optimized functions
        data_flat = self._preprocess_data(data)

        # Transform the data
        return self._pca.transform(data_flat)

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

        # Inverse transform
        reconstructed = self._pca.inverse_transform(data)

        # Reshape if original shape is provided
        original_shape = kwargs.get("original_shape", None)
        if original_shape is not None and len(original_shape) > 2:
            reconstructed = reconstructed.reshape(original_shape)

        return reconstructed

    def get_explained_variance(self) -> Optional[np.ndarray]:
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
                "use_numba": self.use_numba,
                "explained_variance_sum": (
                    np.sum(self._explained_variance_ratio)
                    if self._explained_variance_ratio is not None
                    else None
                ),
            }
        )
        return base_summary


# Factory functions for easy creation


def create_pca(
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
    whiten: bool = False,
    use_numba: bool = True,
) -> PCAProcessor:
    """
    Create a standard PCA processor with common parameters.

    Args:
        n_components: Number of components to keep
        normalize: Whether to normalize data before PCA
        random_state: Random seed for reproducibility
        whiten: Whether to whiten the data
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured PCAProcessor
    """
    return PCAProcessor(
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
        whiten=whiten,
        use_numba=use_numba,
    )


def create_whitening_pca(
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
    use_numba: bool = True,
) -> PCAProcessor:
    """
    Create a PCA processor with whitening enabled.

    Whitening transforms the data to have uncorrelated components with unit variance,
    which can improve subsequent processing like clustering.

    Args:
        n_components: Number of components to keep
        normalize: Whether to normalize data before PCA
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured PCAProcessor with whitening enabled
    """
    return PCAProcessor(
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
        whiten=True,
        use_numba=use_numba,
    )


def create_fast_pca(
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
) -> PCAProcessor:
    """
    Create a PCA processor optimized for maximum speed.

    This uses all Numba accelerations and is configured for
    optimal performance with larger datasets.

    Args:
        n_components: Number of components to keep
        normalize: Whether to normalize data before PCA
        random_state: Random seed for reproducibility

    Returns:
        PCAProcessor optimized for speed
    """
    return PCAProcessor(
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
        whiten=False,  # Whitening adds computational overhead
        use_numba=True,
        parallel_threshold=500,  # Lower threshold to use parallel processing more often
    )
