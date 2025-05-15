"""
Numba-accelerated KMeans clustering implementation.

This module provides a high-performance KMeans clustering implementation using
Numba for acceleration, with no dependency on scikit-learn. This allows for better
performance and type consistency when used in processing pipelines.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from sklearn.metrics import silhouette_score

from dspant.processors.clustering.base import BaseClusteringProcessor


@jit(nopython=True, cache=True)
def _init_random_centroids(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """
    Initialize cluster centroids randomly from the data points.

    Args:
        X: Input data array (n_samples, n_features)
        n_clusters: Number of clusters
        seed: Random seed

    Returns:
        Initial centroids (n_clusters, n_features)
    """
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Pick random indices for initial centroids
    indices = np.random.choice(n_samples, n_clusters, replace=False)

    # Create centroids array
    centroids = np.zeros((n_clusters, n_features), dtype=np.float32)

    # Fill with data from chosen indices
    for i in range(n_clusters):
        centroids[i] = X[indices[i]]

    return centroids


@jit(nopython=True, cache=True)
def _init_kmeans_pp(X: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    """
    Initialize cluster centroids using K-means++ algorithm.

    Args:
        X: Input data array (n_samples, n_features)
        n_clusters: Number of clusters
        seed: Random seed

    Returns:
        Initial centroids (n_clusters, n_features)
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features), dtype=np.float32)

    # Choose first centroid randomly
    first_idx = np.random.randint(0, n_samples)
    centroids[0] = X[first_idx].copy()

    # Choose remaining centroids
    for k in range(1, n_clusters):
        # Calculate squared distances to nearest centroid for each point
        min_distances = np.ones(n_samples, dtype=np.float32) * np.inf

        for i in range(n_samples):
            for j in range(k):
                # Compute distance to centroid j
                dist_sq = 0.0
                for f in range(n_features):
                    diff = X[i, f] - centroids[j, f]
                    dist_sq += diff * diff

                # Update minimum distance
                if dist_sq < min_distances[i]:
                    min_distances[i] = dist_sq

        # Choose next centroid with probability proportional to squared distance
        sum_distances = np.sum(min_distances)
        if sum_distances > 0:
            # Convert to probability
            probs = min_distances / sum_distances

            # Calculate cumulative probability
            cumsum = 0.0
            rand_val = np.random.random()
            for i in range(n_samples):
                cumsum += probs[i]
                if rand_val <= cumsum:
                    centroids[k] = X[i].copy()
                    break
        else:
            # If all points are at zero distance, choose randomly
            idx = np.random.randint(0, n_samples)
            centroids[k] = X[idx].copy()

    return centroids


@jit(nopython=True, parallel=True, cache=True)
def _compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances between data points and centroids.

    Args:
        X: Input data array (n_samples, n_features)
        centroids: Cluster centroids (n_clusters, n_features)

    Returns:
        Distances matrix (n_samples, n_clusters)
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_clusters = centroids.shape[0]

    distances = np.zeros((n_samples, n_clusters), dtype=np.float32)

    for i in prange(n_samples):
        for j in range(n_clusters):
            dist_sq = 0.0
            for f in range(n_features):
                diff = X[i, f] - centroids[j, f]
                dist_sq += diff * diff
            distances[i, j] = dist_sq

    return distances


@jit(nopython=True, cache=True)
def _assign_clusters(distances: np.ndarray) -> np.ndarray:
    """
    Assign data points to clusters based on minimum distance.

    Args:
        distances: Distances matrix (n_samples, n_clusters)

    Returns:
        Cluster assignments (n_samples,)
    """
    return np.argmin(distances, axis=1)


@jit(nopython=True, cache=True)
def _update_centroids(
    X: np.ndarray, labels: np.ndarray, n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Update centroids based on current cluster assignments.

    Args:
        X: Input data array (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        n_clusters: Number of clusters

    Returns:
        Tuple of (updated_centroids, counts_per_cluster)
    """
    n_samples, n_features = X.shape
    centroids = np.zeros((n_clusters, n_features), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int32)

    # Sum points in each cluster
    for i in range(n_samples):
        cluster = labels[i]
        counts[cluster] += 1
        for f in range(n_features):
            centroids[cluster, f] += X[i, f]

    # Compute means
    for j in range(n_clusters):
        if counts[j] > 0:
            for f in range(n_features):
                centroids[j, f] /= counts[j]
        else:
            # If a cluster is empty, reinitialize it (randomly)
            idx = np.random.randint(0, n_samples)
            centroids[j] = X[idx]

    return centroids, counts


@jit(nopython=True, cache=True)
def _compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Compute inertia (sum of squared distances to nearest centroid).

    Args:
        X: Input data array (n_samples, n_features)
        labels: Cluster assignments (n_samples,)
        centroids: Cluster centroids (n_clusters, n_features)

    Returns:
        Inertia value
    """
    n_samples, n_features = X.shape
    inertia = 0.0

    for i in range(n_samples):
        cluster = labels[i]
        dist_sq = 0.0
        for f in range(n_features):
            diff = X[i, f] - centroids[cluster, f]
            dist_sq += diff * diff
        inertia += dist_sq

    return inertia


@jit(nopython=True, cache=True)
def _has_converged(
    old_centroids: np.ndarray, new_centroids: np.ndarray, tol: float
) -> bool:
    """
    Check if centroids have converged.

    Args:
        old_centroids: Previous centroids (n_clusters, n_features)
        new_centroids: Updated centroids (n_clusters, n_features)
        tol: Tolerance for convergence

    Returns:
        True if converged, False otherwise
    """
    n_clusters, n_features = old_centroids.shape

    for j in range(n_clusters):
        dist_sq = 0.0
        for f in range(n_features):
            diff = old_centroids[j, f] - new_centroids[j, f]
            dist_sq += diff * diff

        if dist_sq > tol * tol:
            return False

    return True


@jit(nopython=True, cache=True)
def _kmeans_single_run(
    X: np.ndarray,  # Be explicit about array type
    n_clusters: np.int64,  # Specify exact integer type
    max_iter: np.int64,
    tol: np.float32,
    init_centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.float32, np.int32]:  # Explicit return types
    """Run K-means clustering for a single initialization."""
    # Ensure correct dtypes
    X = X.astype(np.float32)
    init_centroids = init_centroids.astype(np.float32)
    n_clusters = np.int64(n_clusters)
    max_iter = np.int64(max_iter)
    tol = np.float32(tol)

    centroids = init_centroids.copy()
    n_samples = np.int64(X.shape[0])  # Explicit cast

    labels = np.zeros(n_samples, dtype=np.int32)
    n_iter = np.int32(0)  # Explicit type

    for iteration in range(max_iter):
        # Calculate distances
        distances = _compute_distances(X, centroids).astype(np.float32)

        # Assign clusters
        new_labels = _assign_clusters(distances).astype(np.int32)

        # Check if labels changed
        if iteration > 0:
            changes = np.sum(labels != new_labels)
            if changes == 0:
                break

        labels = new_labels

        # Update centroids
        old_centroids = centroids.copy()
        centroids, _ = _update_centroids(X, labels, n_clusters)

        # Check convergence using centroid movement
        if _has_converged(old_centroids, centroids, tol):
            break

        n_iter = np.int32(iteration + 1)

    # Compute final inertia
    inertia = np.float32(_compute_inertia(X, labels, centroids))

    return centroids, labels, inertia, n_iter


@jit(nopython=True, parallel=False, cache=True)
def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 300,
    n_init: int = 10,
    tol: float = 1e-4,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Fit KMeans clustering model.

    Args:
        X: Input data array (n_samples, n_features)
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations per initialization
        n_init: Number of initializations to try
        tol: Tolerance for convergence
        random_state: Random seed

    Returns:
        Tuple of (best_centroids, best_labels, best_inertia, best_n_iter)
    """
    best_centroids = None
    best_labels = None
    best_inertia = np.inf
    best_n_iter = 0

    # Try multiple initializations
    for init in range(n_init):
        # Initialize centroids using k-means++
        init_centroids = _init_kmeans_pp(X, n_clusters, random_state + init)

        # Run k-means
        centroids, labels, inertia, n_iter = _kmeans_single_run(
            X, n_clusters, max_iter, tol, init_centroids
        )

        # Update best result if current run is better
        if inertia < best_inertia:
            best_centroids = centroids
            best_labels = labels
            best_inertia = inertia
            best_n_iter = n_iter

    return best_centroids, best_labels, best_inertia, best_n_iter


@jit(nopython=True, parallel=True, cache=True)
def _flatten_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Efficiently flatten 3D data to 2D for clustering with parallelization.

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


class NumbaKMeansProcessor(BaseClusteringProcessor):
    """
    Pure Numba-accelerated KMeans clustering processor.

    This processor provides high-performance clustering through KMeans
    using pure Numba with no scikit-learn dependency, ensuring consistent
    types when used in processing pipelines.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_state: Optional[int] = 42,
        chunk_size: Optional[int] = None,
        compute_silhouette: bool = False,
    ):
        """
        Initialize the Numba KMeans processor.

        Args:
            n_clusters: Number of clusters
            max_iter: Maximum number of iterations per initialization
            n_init: Number of initializations to try
            tol: Convergence tolerance
            random_state: Random seed for reproducibility
            chunk_size: Size of chunks for Dask processing
            compute_silhouette: Whether to compute silhouette score
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state or 42
        self.chunk_size = chunk_size
        self.compute_silhouette = compute_silhouette

        # Results storage
        self._centroids = None
        self._inertia = None
        self._n_iter = None
        self._silhouette_score = None

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
            if data.ndim == 3:
                data_flat = _flatten_3d_data(data)
            else:
                n_samples = data.shape[0]
                data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        # Ensure float32 type
        if data_flat.dtype != np.float32:
            data_flat = data_flat.astype(np.float32)

        return data_flat

    def fit(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> "NumbaKMeansProcessor":
        """
        Fit the KMeans model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data

        Returns:
            Self for method chaining
        """
        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Fit KMeans using Numba function
        centroids, labels, inertia, n_iter = fit_kmeans(
            data_flat,
            self.n_clusters,
            self.max_iter,
            self.n_init,
            self.tol,
            self.random_state,
        )

        # Store results
        self._cluster_centers = centroids
        self._cluster_labels = labels
        self._inertia = inertia
        self._n_iter = n_iter
        self._is_fitted = True

        # Compute silhouette score if requested
        if (
            self.compute_silhouette
            and self.n_clusters > 1
            and len(data_flat) > self.n_clusters
        ):
            try:
                self._silhouette_score = silhouette_score(data_flat, labels)
            except Exception:
                self._silhouette_score = None

        return self

    def predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict cluster assignments for new data points.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data

        Returns:
            Array of cluster assignments
        """
        if not self._is_fitted:
            raise ValueError("KMeans model not fitted. Call fit() first.")

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Compute distances to centroids
        distances = _compute_distances(data_flat, self._cluster_centers)

        # Assign to nearest centroid
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

        # Return distances to centroids
        return _compute_distances(data_flat, self._cluster_centers)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform clustering using Dask.
        """
        # Get processing parameters
        compute_now = kwargs.get("compute_now", False)
        chunk_size = kwargs.get("chunk_size", self.chunk_size)

        # Handle different input shapes for dask arrays
        if data.ndim > 2:
            n_samples = data.shape[0]
            data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        if compute_now or not self._is_fitted:
            # Convert to numpy, fit, and convert back to dask
            data_np = data_flat.compute()
            # Use fit and predict separately to avoid potential issues with fit_predict
            self.fit(data_np, **kwargs)  # Correct
            labels = self.predict(data_np, **kwargs)  # Fixed: use **kwargs
            return da.from_array(labels)
        else:
            # Already fitted, use map_blocks for efficient prediction
            def predict_chunk(chunk):
                # Handle multi-dimensional chunks
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)
                return self.predict(chunk, **kwargs)

            # Apply custom chunk size if specified
            if chunk_size is not None:
                data_flat = data_flat.rechunk({0: chunk_size})

            # Apply prediction to chunks
            result = data_flat.map_blocks(
                predict_chunk,
                drop_axis=list(range(1, data_flat.ndim)),
                dtype=np.int32,
            )

            return result

    def get_inertia(self) -> Optional[float]:
        """Get the inertia from the last clustering run."""
        return self._inertia

    def get_n_iter(self) -> Optional[int]:
        """Get the number of iterations from the best run."""
        return self._n_iter

    def get_silhouette_score(self) -> Optional[float]:
        """Get the silhouette score if computed."""
        return self._silhouette_score

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
                "silhouette_score": self._silhouette_score,
                "implementation": "pure_numba",
            }
        )
        return base_summary


# Factory functions for easy creation


def create_numba_kmeans(
    n_clusters: int = 8,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> NumbaKMeansProcessor:
    """
    Create a pure Numba KMeans processor with standard parameters.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaKMeansProcessor
    """
    return NumbaKMeansProcessor(
        n_clusters=n_clusters,
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_fast_numba_kmeans(
    n_clusters: int = 8,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> NumbaKMeansProcessor:
    """
    Create a pure Numba KMeans processor optimized for speed.

    This configuration uses fewer initializations and iterations for faster
    clustering at some cost to quality.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaKMeansProcessor for speed
    """
    return NumbaKMeansProcessor(
        n_clusters=n_clusters,
        max_iter=100,  # Fewer iterations
        n_init=3,  # Fewer initializations
        tol=1e-3,  # Less strict convergence
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_robust_numba_kmeans(
    n_clusters: int = 8,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> NumbaKMeansProcessor:
    """
    Create a pure Numba KMeans processor optimized for robustness.

    This configuration uses more initializations and stricter convergence
    for better clustering quality at the cost of speed.

    Args:
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaKMeansProcessor for robust clustering
    """
    return NumbaKMeansProcessor(
        n_clusters=n_clusters,
        max_iter=500,  # More iterations
        n_init=15,  # More initializations
        tol=1e-5,  # Stricter convergence
        random_state=random_state,
        chunk_size=chunk_size,
        compute_silhouette=True,  # Enable quality metrics
    )
