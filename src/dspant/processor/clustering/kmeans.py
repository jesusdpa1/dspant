"""
KMeans clustering implementation for large datasets.

This module provides a KMeans clustering implementation
optimized for processing large datasets using Dask arrays.
"""

import random
from typing import Any, Dict, Optional, Union

import dask.array as da
import numpy as np
from sklearn.cluster import KMeans

from .base import BaseClusteringProcessor


class KMeansProcessor(BaseClusteringProcessor):
    """
    KMeans clustering processor for large datasets.

    This implementation leverages scikit-learn's KMeans for efficient clustering
    and supports processing with Dask arrays.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        chunk_size: Optional[int] = None,
        compute_full_inertia: bool = False,
    ):
        """
        Initialize the KMeans processor.

        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations per initialization
            n_init: Number of initializations to try
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
            chunk_size: Size of chunks for Dask processing (None = auto)
            compute_full_inertia: Whether to compute inertia across all data
                                  (can be expensive for large datasets)
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.compute_full_inertia = compute_full_inertia

        # Create scikit-learn KMeans model
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            tol=tol,
            random_state=random_state,
        )

        # Initialize results
        self._inertia = None
        self._n_iter = None

    def _preprocess_data(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> np.ndarray:
        """
        Preprocess input data for clustering.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Preprocessed numpy array
        """
        # Convert dask array to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Apply custom preprocessing if provided
        preprocess_fn = kwargs.get("preprocess_fn", None)
        if preprocess_fn is not None:
            data = preprocess_fn(data)

        # Handle different input shapes
        if data.ndim > 2:
            n_samples = data.shape[0]
            data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        return data_flat

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "KMeansProcessor":
        """
        Fit the KMeans model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data before clustering

        Returns:
            Self for method chaining
        """
        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Fit the KMeans model
        self._kmeans.fit(data_flat)

        # Store results
        self._cluster_centers = self._kmeans.cluster_centers_
        self._cluster_labels = self._kmeans.labels_
        self._inertia = self._kmeans.inertia_
        self._n_iter = self._kmeans.n_iter_
        self._is_fitted = True

        return self

    def predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict cluster assignments for new data points.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data before prediction

        Returns:
            Array of cluster assignments
        """
        if not self._is_fitted:
            raise ValueError("KMeans model not fitted. Call fit() first.")

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Predict cluster assignments
        return self._kmeans.predict(data_flat)

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data to cluster distance space.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Array of distances to each cluster center
        """
        if not self._is_fitted:
            raise ValueError("KMeans model not fitted. Call fit() first.")

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Transform to cluster distances
        return self._kmeans.transform(data_flat)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform clustering using Dask.

        This method is optimized for large datasets using Dask's distributed computing.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used in clustering, but required by BaseProcessor interface)
            **kwargs: Additional algorithm-specific parameters
                compute_now: Whether to compute immediately (default: False)
                chunk_size: Size of chunks for processing (overrides instance setting)

        Returns:
            Dask array with cluster assignments
        """
        # Get processing parameters
        compute_now = kwargs.pop("compute_now", False)
        chunk_size = kwargs.pop("chunk_size", self.chunk_size)

        # Handle different input shapes for dask arrays
        if data.ndim > 2:
            n_samples = data.shape[0]
            data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        if compute_now:
            # Convert to numpy, fit, and convert back to dask
            data_np = data_flat.compute()
            labels = self.fit_predict(data_np, **kwargs)
            return da.from_array(labels)
        elif self._is_fitted:
            # If already fitted, use map_blocks for efficient prediction
            def predict_chunk(chunk, block_info=None):
                # Reshape multi-dimensional chunks if needed
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)
                return self.predict(chunk, **kwargs)

            # Apply to dask array with optimized chunk size if specified
            if chunk_size is not None:
                data_flat = data_flat.rechunk({0: chunk_size})

            result = data_flat.map_blocks(
                predict_chunk,
                drop_axis=list(
                    range(1, data_flat.ndim)
                ),  # Output has one dimension fewer
                dtype=np.int32,
            )

            return result
        else:
            # For initial fitting, we need to compute
            return self.process(data, fs, compute_now=True, **kwargs)

    def get_inertia(self) -> Optional[float]:
        """Get the inertia from the last clustering run."""
        return self._inertia

    def get_n_iter(self) -> Optional[int]:
        """Get the number of iterations from the best run."""
        return self._n_iter

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "n_clusters": self.n_clusters,
                "max_iter": self.max_iter,
                "n_init": self.n_init,
                "tol": self.tol,
                "is_fitted": self._is_fitted,
                "inertia": self._inertia,
                "n_iter": self._n_iter,
                "chunk_size": self.chunk_size,
            }
        )
        return base_summary


# Factory functions for easy creation


def create_kmeans(
    n_clusters: int = 8,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> KMeansProcessor:
    """
    Create a standard KMeans processor optimized for large datasets.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing (None = auto)

    Returns:
        Configured KMeansProcessor
    """
    return KMeansProcessor(
        n_clusters=n_clusters,
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_fast_kmeans(
    n_clusters: int = 8,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> KMeansProcessor:
    """
    Create a KMeans processor optimized for speed with large datasets.

    This configuration uses fewer initializations and iterations
    for faster clustering at some cost to quality.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing (None = auto)

    Returns:
        Configured KMeansProcessor for speed
    """
    return KMeansProcessor(
        n_clusters=n_clusters,
        max_iter=100,  # Fewer iterations
        n_init=3,  # Fewer initializations
        tol=1e-3,  # Less strict convergence
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_high_quality_kmeans(
    n_clusters: int = 8,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> KMeansProcessor:
    """
    Create a KMeans processor optimized for quality with large datasets.

    This configuration uses more initializations and stricter convergence
    criteria for higher quality clustering at some cost to speed.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing (None = auto)

    Returns:
        Configured KMeansProcessor for quality
    """
    return KMeansProcessor(
        n_clusters=n_clusters,
        max_iter=500,  # More iterations
        n_init=15,  # More initializations
        tol=1e-5,  # Stricter convergence
        random_state=random_state,
        chunk_size=chunk_size,
    )
