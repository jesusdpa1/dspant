"""
KMeans clustering implementation with Numba acceleration for large datasets.

This module provides a highly efficient KMeans clustering implementation
optimized for large datasets using Dask arrays and Numba acceleration.
"""

import random
from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from .base import BaseClusteringProcessor


@jit(nopython=True, cache=True, parallel=True)
def _compute_distances_parallel(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute distances from data points to cluster centers with Numba acceleration and parallelization.

    Args:
        X: Data points (n_samples, n_features)
        centers: Cluster centers (n_clusters, n_features)

    Returns:
        Distances matrix (n_samples, n_clusters)
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    n_features = X.shape[1]
    distances = np.zeros((n_samples, n_clusters), dtype=np.float32)

    for i in prange(n_samples):
        for j in range(n_clusters):
            dist = 0.0
            for k in range(n_features):
                diff = X[i, k] - centers[j, k]
                dist += diff * diff
            distances[i, j] = dist  # No sqrt for efficiency during distance comparisons

    return distances


@jit(nopython=True, cache=True)
def _assign_clusters(distances: np.ndarray) -> np.ndarray:
    """
    Assign data points to nearest cluster based on distance matrix.

    Args:
        distances: Distance matrix (n_samples, n_clusters)

    Returns:
        Cluster assignments (n_samples,)
    """
    return np.argmin(distances, axis=1)


@jit(nopython=True, parallel=True, cache=True)
def _update_centers(
    X: np.ndarray, labels: np.ndarray, n_clusters: int, n_features: int
) -> np.ndarray:
    """
    Update cluster centers based on current assignments with efficient parallelization.

    Args:
        X: Data points (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        n_clusters: Number of clusters
        n_features: Number of features

    Returns:
        Updated cluster centers (n_clusters, n_features)
    """
    centers = np.zeros((n_clusters, n_features), dtype=X.dtype)
    counts = np.zeros(n_clusters, dtype=np.int32)

    # Count points in each cluster and sum features
    for i in range(X.shape[0]):
        cluster = labels[i]
        counts[cluster] += 1
        for j in range(n_features):
            centers[cluster, j] += X[i, j]

    # Compute means (avoiding division by zero)
    for i in prange(n_clusters):
        if counts[i] > 0:
            for j in range(n_features):
                centers[i, j] /= counts[i]

    return centers, counts


@jit(nopython=True, cache=True)
def _compute_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute inertia (sum of squared distances to closest centroid).

    Args:
        X: Data points (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centers: Cluster centers (n_clusters, n_features)

    Returns:
        Inertia value
    """
    inertia = 0.0
    for i in range(X.shape[0]):
        cluster = labels[i]
        for j in range(X.shape[1]):
            diff = X[i, j] - centers[cluster, j]
            inertia += diff * diff

    return inertia


@jit(nopython=True, cache=True)
def _kmeans_plus_plus_init(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Implement k-means++ initialization with Numba.

    Args:
        X: Data points (n_samples, n_features)
        n_clusters: Number of clusters

    Returns:
        Initial centers (n_clusters, n_features)
    """
    n_samples, n_features = X.shape

    # Choose first center randomly
    first_idx = np.random.randint(0, n_samples)
    centers = np.zeros((n_clusters, n_features), dtype=X.dtype)
    centers[0] = X[first_idx].copy()

    # Distances to closest center for each point
    min_distances = np.zeros(n_samples, dtype=np.float32)

    # Choose remaining centers
    for c in range(1, n_clusters):
        # Compute squared distances to closest existing center
        for i in range(n_samples):
            min_dist = np.float32(1e10)
            for j in range(c):
                dist = 0.0
                for k in range(n_features):
                    diff = X[i, k] - centers[j, k]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
            min_distances[i] = min_dist

        # Choose next center with probability proportional to squared distance
        sum_distances = np.sum(min_distances)
        if sum_distances > 0:
            target = np.random.random() * sum_distances
            cumsum = 0.0
            for i in range(n_samples):
                cumsum += min_distances[i]
                if cumsum >= target:
                    centers[c] = X[i].copy()
                    break
        else:
            # Fallback if all points are very close to existing centers
            idx = np.random.randint(0, n_samples)
            centers[c] = X[idx].copy()

    return centers


@jit(nopython=True, cache=True)
def _kmeans_single_run(
    X: np.ndarray, n_clusters: int, max_iter: int = 100, tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Run KMeans clustering with Numba acceleration for a single initialization.

    Args:
        X: Data points (n_samples, n_features)
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (cluster_centers, labels, inertia, n_iter)
    """
    n_samples, n_features = X.shape

    # Initialize centers using k-means++
    centers = _kmeans_plus_plus_init(X, n_clusters)

    # Main loop
    labels = np.zeros(n_samples, dtype=np.int32)
    old_centers = np.zeros_like(centers)
    n_iter = 0

    for iteration in range(max_iter):
        n_iter = iteration + 1

        # Save old centers
        old_centers[:] = centers

        # Compute distances and assign clusters
        distances = _compute_distances_parallel(X, centers)
        labels = _assign_clusters(distances)

        # Update centers
        centers, counts = _update_centers(X, labels, n_clusters, n_features)

        # Handle empty clusters by reinitializing from random points
        for i in range(n_clusters):
            if counts[i] == 0:
                idx = np.random.randint(0, n_samples)
                centers[i] = X[idx].copy()

        # Check convergence
        center_shift = 0.0
        for i in range(n_clusters):
            for j in range(n_features):
                diff = centers[i, j] - old_centers[i, j]
                center_shift += diff * diff

        if np.sqrt(center_shift) < tol:
            break

    # Compute final inertia
    inertia = _compute_inertia(X, labels, centers)

    return centers, labels, inertia, n_iter


class KMeansProcessor(BaseClusteringProcessor):
    """
    Efficient KMeans clustering processor with Numba acceleration for large datasets.

    This implementation is optimized for processing large datasets with
    Dask arrays and Numba acceleration for high performance.
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

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        # Initialize results
        self._inertia = None
        self._n_iter = None
        self._best_centers = None
        self._best_labels = None

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

        # Run multiple initializations and keep the best
        best_inertia = float("inf")
        best_centers = None
        best_labels = None
        best_n_iter = 0

        for init in range(self.n_init):
            centers, labels, inertia, n_iter = _kmeans_single_run(
                data_flat, self.n_clusters, self.max_iter, self.tol
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        # Store results
        self._cluster_centers = best_centers
        self._cluster_labels = best_labels
        self._inertia = best_inertia
        self._n_iter = best_n_iter
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

        # Compute distances and assign clusters
        distances = _compute_distances_parallel(data_flat, self._cluster_centers)
        return _assign_clusters(distances)

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

        # Compute distances to all clusters (with sqrt for proper distances)
        distances = _compute_distances_parallel(data_flat, self._cluster_centers)
        return np.sqrt(distances)

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
