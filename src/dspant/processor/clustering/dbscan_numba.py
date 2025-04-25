"""
Numba-accelerated DBSCAN clustering implementation.

This module provides a high-performance DBSCAN implementation using
Numba for acceleration, with no dependency on scikit-learn. This allows for better
performance and type consistency when used in processing pipelines.

Part 1: Core mathematical functions for DBSCAN implementation.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

from ...processor.clustering.base import BaseClusteringProcessor

NOISE = -1  # Label for noise points
UNVISITED = -2  # Special value for unvisited points


@jit(nopython=True, parallel=True, cache=True)
def _compute_pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distances between all points.

    Args:
        X: Input data array (n_samples, n_features)
        metric: Distance metric ('euclidean' or 'manhattan')

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float32)

    if metric == "euclidean":
        for i in prange(n_samples):
            for j in range(i + 1, n_samples):
                # Euclidean distance
                d_sq = 0.0
                for k in range(X.shape[1]):
                    diff = X[i, k] - X[j, k]
                    d_sq += diff * diff

                d = np.sqrt(d_sq)
                distances[i, j] = d
                distances[j, i] = d
    elif metric == "manhattan":
        for i in prange(n_samples):
            for j in range(i + 1, n_samples):
                # Manhattan distance
                d = 0.0
                for k in range(X.shape[1]):
                    d += np.abs(X[i, k] - X[j, k])

                distances[i, j] = d
                distances[j, i] = d
    else:
        # Default to Euclidean if unknown metric
        for i in prange(n_samples):
            for j in range(i + 1, n_samples):
                d_sq = 0.0
                for k in range(X.shape[1]):
                    diff = X[i, k] - X[j, k]
                    d_sq += diff * diff

                d = np.sqrt(d_sq)
                distances[i, j] = d
                distances[j, i] = d

    return distances


@jit(nopython=True, cache=True)
def _compute_distance_euclidean(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.

    Args:
        x1: First point
        x2: Second point

    Returns:
        Euclidean distance
    """
    d_sq = 0.0
    for i in range(len(x1)):
        diff = x1[i] - x2[i]
        d_sq += diff * diff

    return np.sqrt(d_sq)


@jit(nopython=True, cache=True)
def _compute_distance_manhattan(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute Manhattan distance between two points.

    Args:
        x1: First point
        x2: Second point

    Returns:
        Manhattan distance
    """
    d = 0.0
    for i in range(len(x1)):
        d += np.abs(x1[i] - x2[i])

    return d


@jit(nopython=True, cache=True)
def _region_query(
    X: np.ndarray, point_idx: int, eps: float, metric: str = "euclidean"
) -> np.ndarray:
    """
    Find all points within eps distance of point_idx.

    Args:
        X: Input data array (n_samples, n_features)
        point_idx: Index of the point to query
        eps: Maximum distance for neighbors
        metric: Distance metric ('euclidean' or 'manhattan')

    Returns:
        Array of indices of points within eps distance
    """
    n_samples = X.shape[0]
    neighbors = []

    # Choose distance function based on metric
    if metric == "manhattan":
        distance_func = _compute_distance_manhattan
    else:
        distance_func = _compute_distance_euclidean

    # Find neighbors
    for i in range(n_samples):
        if i == point_idx:
            continue

        dist = distance_func(X[point_idx], X[i])
        if dist <= eps:
            neighbors.append(i)

    return np.array(neighbors, dtype=np.int32)


@jit(nopython=True, cache=True)
def _find_neighbors_from_matrix(distances: np.ndarray, eps: float) -> List[np.ndarray]:
    """
    Find neighbors within eps distance for each point using distance matrix.

    Args:
        distances: Pairwise distance matrix (n_samples, n_samples)
        eps: Maximum distance for neighbors

    Returns:
        List of neighbor arrays for each point
    """
    n_samples = distances.shape[0]
    neighbors_list = []

    for i in range(n_samples):
        # Find indices where distance <= eps
        neighbors = np.where(distances[i] <= eps)[0]
        # Remove self from neighbors (distance to self is 0)
        neighbors = neighbors[neighbors != i]
        neighbors_list.append(neighbors)

    return neighbors_list


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


@jit(nopython=True, cache=True)
def _expand_cluster(
    X: np.ndarray,
    labels: np.ndarray,
    point_idx: int,
    neighbors: np.ndarray,
    cluster_id: int,
    eps: float,
    min_samples: int,
    metric: str,
) -> None:
    """
    Expand a cluster from a core point.

    Args:
        X: Input data array (n_samples, n_features)
        labels: Array of cluster labels
        point_idx: Index of the core point
        neighbors: Array of indices of neighbors
        cluster_id: Current cluster ID
        eps: Maximum distance for neighbors
        min_samples: Minimum number of points to form a dense region
        metric: Distance metric

    Returns:
        None (modifies labels in-place)
    """
    # Assign current point to cluster
    labels[point_idx] = cluster_id

    # Process each neighbor
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        # If point was noise, add it to the cluster
        if labels[neighbor_idx] == NOISE:
            labels[neighbor_idx] = cluster_id

        # If point was unvisited
        elif labels[neighbor_idx] == UNVISITED:
            # Mark it as part of the cluster
            labels[neighbor_idx] = cluster_id

            # Find neighbors of this point
            new_neighbors = _region_query(X, neighbor_idx, eps, metric)

            # If it's a core point, add its neighbors to the processing queue
            if len(new_neighbors) >= min_samples:
                # Extend neighbors list with new neighbors
                for new_neighbor in new_neighbors:
                    if new_neighbor not in neighbors:
                        neighbors = np.append(neighbors, new_neighbor)

        i += 1


@jit(nopython=True, cache=True)
def _expand_cluster_precomputed(
    labels: np.ndarray,
    point_idx: int,
    neighbors_list: List[np.ndarray],
    cluster_id: int,
    min_samples: int,
) -> None:
    """
    Expand a cluster from a core point using precomputed neighbors.

    Args:
        labels: Array of cluster labels
        point_idx: Index of the core point
        neighbors_list: List of neighbor arrays for each point
        cluster_id: Current cluster ID
        min_samples: Minimum number of points to form a dense region

    Returns:
        None (modifies labels in-place)
    """
    # Get neighbors of current point
    neighbors = neighbors_list[point_idx].copy()

    # Assign current point to cluster
    labels[point_idx] = cluster_id

    # Initialize a list to process
    seeds = neighbors.copy()

    # Process each seed
    i = 0
    while i < len(seeds):
        curr_point = seeds[i]

        # If point was noise, add it to the cluster
        if labels[curr_point] == NOISE:
            labels[curr_point] = cluster_id

        # If point was unvisited
        elif labels[curr_point] == UNVISITED:
            # Mark it as part of the cluster
            labels[curr_point] = cluster_id

            # Get neighbors of this point
            curr_neighbors = neighbors_list[curr_point]

            # If it's a core point, add its neighbors to the seeds
            if len(curr_neighbors) >= min_samples:
                for j in range(len(curr_neighbors)):
                    neighbor = curr_neighbors[j]
                    if labels[neighbor] == UNVISITED or labels[neighbor] == NOISE:
                        seeds = np.append(seeds, neighbor)

        i += 1


@jit(nopython=True, cache=True)
def dbscan(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    metric: str = "euclidean",
    precompute_distances: bool = True,
) -> Tuple[np.ndarray, int, List[np.ndarray]]:
    """
    DBSCAN clustering algorithm.

    Args:
        X: Input data array (n_samples, n_features)
        eps: Maximum distance for neighbors
        min_samples: Minimum number of points to form a dense region
        metric: Distance metric ('euclidean' or 'manhattan')
        precompute_distances: Whether to precompute distance matrix

    Returns:
        Tuple of (labels, n_clusters, core_samples_idx)
    """
    n_samples = X.shape[0]

    # Initialize all points as unvisited
    labels = np.full(n_samples, UNVISITED, dtype=np.int32)

    # Precompute distance matrix if requested
    neighbors_list = None
    if precompute_distances:
        distances = _compute_pairwise_distances(X, metric)
        neighbors_list = _find_neighbors_from_matrix(distances, eps)

    # Current cluster ID (incremented for each new cluster)
    cluster_id = 0

    # Process each point
    for i in range(n_samples):
        # Skip points that have already been processed
        if labels[i] != UNVISITED:
            continue

        # Find neighbors
        if precompute_distances:
            neighbors = neighbors_list[i]
        else:
            neighbors = _region_query(X, i, eps, metric)

        # If point has fewer than min_samples neighbors, mark as noise
        if len(neighbors) < min_samples:
            labels[i] = NOISE
            continue

        # Otherwise, start a new cluster
        cluster_id += 1

        # Expand the cluster
        if precompute_distances:
            _expand_cluster_precomputed(
                labels, i, neighbors_list, cluster_id, min_samples
            )
        else:
            _expand_cluster(
                X, labels, i, neighbors, cluster_id, eps, min_samples, metric
            )

    # Get indices of core samples
    core_samples_idx = []
    for i in range(n_samples):
        if precompute_distances:
            if len(neighbors_list[i]) >= min_samples:
                core_samples_idx.append(i)
        else:
            neighbors = _region_query(X, i, eps, metric)
            if len(neighbors) >= min_samples:
                core_samples_idx.append(i)

    return labels, cluster_id, np.array(core_samples_idx, dtype=np.int32)


@jit(nopython=True, cache=True)
def _compute_silhouette_samples(
    X: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """
    Compute silhouette scores for each sample using Numba.

    Args:
        X: Data array (n_samples, n_features)
        labels: Cluster labels
        metric: Distance metric

    Returns:
        Silhouette scores for each sample
    """
    n_samples = X.shape[0]
    silhouette = np.zeros(n_samples, dtype=np.float32)

    # Choose distance function based on metric
    if metric == "manhattan":
        distance_func = _compute_distance_manhattan
    else:
        distance_func = _compute_distance_euclidean

    # Get unique cluster labels, excluding noise points (label -1)
    unique_labels = []
    for i in range(n_samples):
        if labels[i] >= 0 and labels[i] not in unique_labels:
            unique_labels.append(labels[i])

    # If no valid clusters or only one cluster, return zeros
    if len(unique_labels) <= 1:
        return silhouette

    # Compute silhouette for each point
    for i in range(n_samples):
        # Skip noise points
        if labels[i] < 0:
            silhouette[i] = 0.0
            continue

        # Find points in same cluster
        same_cluster_indices = []
        for j in range(n_samples):
            if j != i and labels[j] == labels[i]:
                same_cluster_indices.append(j)

        # If only one point in cluster, silhouette is 0
        if len(same_cluster_indices) == 0:
            silhouette[i] = 0.0
            continue

        # Compute a: mean distance to points in same cluster
        a = 0.0
        for j in same_cluster_indices:
            a += distance_func(X[i], X[j])

        a /= len(same_cluster_indices)

        # Compute b: minimum mean distance to points in different clusters
        b = np.inf
        for label in unique_labels:
            if label != labels[i]:
                # Find points in this cluster
                other_cluster_indices = []
                for j in range(n_samples):
                    if labels[j] == label:
                        other_cluster_indices.append(j)

                if len(other_cluster_indices) > 0:
                    # Compute mean distance to this cluster
                    mean_dist = 0.0
                    for j in other_cluster_indices:
                        mean_dist += distance_func(X[i], X[j])

                    mean_dist /= len(other_cluster_indices)

                    # Update b if this is the minimum
                    if mean_dist < b:
                        b = mean_dist

        # Compute silhouette
        if a < b:
            silhouette[i] = 1.0 - a / b
        elif a > b:
            silhouette[i] = b / a - 1.0
        else:
            silhouette[i] = 0.0

    return silhouette


@jit(nopython=True, cache=True)
def predict_dbscan(
    X: np.ndarray,
    core_samples: np.ndarray,
    core_sample_labels: np.ndarray,
    eps: float,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Predict cluster labels for new samples.

    This assigns each sample to the cluster of its nearest core sample
    if the distance is within eps, otherwise marks it as noise.

    Args:
        X: New data array (n_samples, n_features)
        core_samples: Array of core samples from the fitted model
        core_sample_labels: Cluster labels for core samples
        eps: Maximum distance for neighbors
        metric: Distance metric

    Returns:
        Predicted cluster labels
    """
    n_samples = X.shape[0]
    n_core_samples = core_samples.shape[0]

    # Initialize labels as noise
    labels = np.full(n_samples, NOISE, dtype=np.int32)

    # Choose distance function based on metric
    if metric == "manhattan":
        distance_func = _compute_distance_manhattan
    else:
        distance_func = _compute_distance_euclidean

    # Assign each sample to the nearest core sample's cluster if within eps
    for i in range(n_samples):
        min_dist = np.inf
        nearest_core_idx = -1

        # Find nearest core sample
        for j in range(n_core_samples):
            dist = distance_func(X[i], core_samples[j])

            if dist < min_dist:
                min_dist = dist
                nearest_core_idx = j

        # If within eps, assign to the cluster of the nearest core sample
        if min_dist <= eps and nearest_core_idx >= 0:
            labels[i] = core_sample_labels[nearest_core_idx]

    return labels


class NumbaDBSCANProcessor(BaseClusteringProcessor):
    """
    Pure Numba-accelerated DBSCAN processor.

    This processor provides high-performance clustering through DBSCAN
    using pure Numba with no scikit-learn dependency, ensuring consistent
    types when used in processing pipelines.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        precompute_distances: bool = True,
        compute_silhouette: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize the Numba DBSCAN processor.

        Args:
            eps: Maximum distance between two samples to be considered neighbors
            min_samples: Minimum number of samples in a neighborhood to be a core point
            metric: Distance metric ('euclidean' or 'manhattan')
            precompute_distances: Whether to precompute distance matrix
            compute_silhouette: Whether to compute silhouette score
            chunk_size: Size of chunks for Dask processing
        """
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.precompute_distances = precompute_distances
        self.compute_silhouette = compute_silhouette
        self.chunk_size = chunk_size

        # Initialize result variables
        self._distances = None
        self._neighbors_list = None
        self._core_samples = None
        self._core_sample_indices = None
        self._core_sample_labels = None
        self._n_clusters = 0
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
    ) -> "NumbaDBSCANProcessor":
        """
        Fit the DBSCAN model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data

        Returns:
            Self for method chaining
        """
        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Run DBSCAN using Numba function
        labels, n_clusters, core_sample_indices = dbscan(
            data_flat,
            self.eps,
            self.min_samples,
            self.metric,
            self.precompute_distances,
        )

        # Store results
        self._cluster_labels = labels
        self._n_clusters = n_clusters
        self._core_sample_indices = core_sample_indices

        # Store core samples and their labels
        if len(core_sample_indices) > 0:
            self._core_samples = data_flat[core_sample_indices]
            self._core_sample_labels = labels[core_sample_indices]
        else:
            self._core_samples = np.empty(
                (0, data_flat.shape[1]), dtype=data_flat.dtype
            )
            self._core_sample_labels = np.empty(0, dtype=np.int32)

        # Compute cluster centers as mean of points in each cluster (excluding noise)
        if n_clusters > 0:
            centers = []
            for cluster_id in range(1, n_clusters + 1):  # Cluster labels start at 1
                mask = labels == cluster_id
                if np.any(mask):
                    center = np.mean(data_flat[mask], axis=0)
                    centers.append(center)

            if centers:
                self._cluster_centers = np.vstack(centers)
            else:
                self._cluster_centers = np.empty(
                    (0, data_flat.shape[1]), dtype=data_flat.dtype
                )
        else:
            self._cluster_centers = np.empty(
                (0, data_flat.shape[1]), dtype=data_flat.dtype
            )

        self._is_fitted = True

        # Compute silhouette score if requested
        if (
            self.compute_silhouette
            and self._n_clusters > 1
            and len(data_flat) > self._n_clusters
        ):
            # Use only non-noise points for silhouette
            non_noise_mask = labels >= 0
            if np.sum(non_noise_mask) > self._n_clusters:
                try:
                    silhouette_samples = _compute_silhouette_samples(
                        data_flat[non_noise_mask], labels[non_noise_mask], self.metric
                    )
                    self._silhouette_score = np.mean(silhouette_samples)
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
            raise ValueError("DBSCAN model not fitted. Call fit() first.")

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Use Numba predict function
        if self._core_samples.shape[0] > 0:
            return predict_dbscan(
                data_flat,
                self._core_samples,
                self._core_sample_labels,
                self.eps,
                self.metric,
            )
        else:
            # If no core samples, all points are noise
            return np.full(data_flat.shape[0], NOISE, dtype=np.int32)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform DBSCAN clustering using Dask.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, required by BaseProcessor interface)
            **kwargs: Additional parameters
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

        if compute_now or not self._is_fitted:
            # Convert to numpy, fit, and convert back to dask
            data_np = data_flat.compute()

            if not self._is_fitted:
                self.fit(data_np, **kwargs)

            result = self.predict(data_np, **kwargs)
            return da.from_array(result)
        else:
            # DBSCAN doesn't work well with chunking, so we issue a warning
            import warnings

            warnings.warn(
                "DBSCAN generally works best with compute_now=True for the entire dataset. "
                "Chunked processing may give unexpected results."
            )

            # Already fitted, define a chunk processing function
            def process_chunk(chunk):
                # Handle multi-dimensional chunks
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)

                return self.predict(chunk, **kwargs)

            # Apply custom chunk size if specified
            if chunk_size is not None:
                data_flat = data_flat.rechunk({0: chunk_size})

            # Apply prediction to chunks
            result = data_flat.map_blocks(
                process_chunk,
                drop_axis=list(range(1, data_flat.ndim)),
                dtype=np.int32,
            )

            return result

    def get_noise_points_mask(self) -> Optional[np.ndarray]:
        """Get boolean mask indicating noise points (label -1)."""
        if self._cluster_labels is not None:
            return self._cluster_labels == NOISE
        return None

    def get_core_sample_indices(self) -> Optional[np.ndarray]:
        """Get indices of core samples."""
        return self._core_sample_indices

    def get_core_samples(self) -> Optional[np.ndarray]:
        """Get core samples."""
        return self._core_samples

    def get_n_clusters(self) -> int:
        """Get the number of clusters found (excluding noise)."""
        return self._n_clusters if self._n_clusters is not None else 0

    def get_silhouette_score(self) -> Optional[float]:
        """Get the silhouette score if computed."""
        return self._silhouette_score

    @property
    def core_sample_indices_(self) -> Optional[np.ndarray]:
        """Get indices of core samples (scikit-learn compatibility)."""
        return self._core_sample_indices

    @property
    def components_(self) -> Optional[np.ndarray]:
        """Get core samples (scikit-learn compatibility)."""
        return self._core_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "precompute_distances": self.precompute_distances,
                "is_fitted": self._is_fitted,
                "n_clusters": self.get_n_clusters(),
                "n_core_samples": len(self._core_sample_indices)
                if self._core_sample_indices is not None
                else 0,
                "n_noise_points": np.sum(self.get_noise_points_mask())
                if self.get_noise_points_mask() is not None
                else 0,
                "silhouette_score": self._silhouette_score,
                "implementation": "pure_numba",
            }
        )
        return base_summary


# Factory functions for easy creation


def create_numba_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    precompute_distances: bool = True,
    chunk_size: Optional[int] = None,
) -> NumbaDBSCANProcessor:
    """
    Create a pure Numba DBSCAN processor with standard parameters.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        min_samples: Minimum number of samples in a neighborhood to be a core point
        metric: Distance metric ('euclidean' or 'manhattan')
        precompute_distances: Whether to precompute distance matrix
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaDBSCANProcessor
    """
    return NumbaDBSCANProcessor(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        precompute_distances=precompute_distances,
        chunk_size=chunk_size,
    )


def create_dense_numba_dbscan(
    eps: float = 0.5,
    metric: str = "euclidean",
    chunk_size: Optional[int] = None,
) -> NumbaDBSCANProcessor:
    """
    Create a Numba DBSCAN processor for dense clusters.

    This configuration uses more strict density requirements,
    resulting in fewer, denser clusters.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        metric: Distance metric ('euclidean' or 'manhattan')
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaDBSCANProcessor for dense clusters
    """
    return NumbaDBSCANProcessor(
        eps=eps,
        min_samples=10,  # More points required for core points
        metric=metric,
        precompute_distances=True,
        compute_silhouette=True,
        chunk_size=chunk_size,
    )


def create_sparse_numba_dbscan(
    eps: float = 0.5,
    metric: str = "euclidean",
    chunk_size: Optional[int] = None,
) -> NumbaDBSCANProcessor:
    """
    Create a Numba DBSCAN processor for sparse clusters.

    This configuration is more relaxed about density requirements,
    potentially finding more clusters in sparse data.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        metric: Distance metric ('euclidean' or 'manhattan')
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaDBSCANProcessor for sparse clusters
    """
    return NumbaDBSCANProcessor(
        eps=eps,
        min_samples=3,  # Fewer points required for core points
        metric=metric,
        precompute_distances=True,
        compute_silhouette=True,
        chunk_size=chunk_size,
    )


def create_memory_efficient_numba_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    chunk_size: Optional[int] = None,
) -> NumbaDBSCANProcessor:
    """
    Create a Numba DBSCAN processor optimized for large datasets with memory constraints.

    This configuration avoids precomputing the distance matrix, which reduces memory usage
    at the cost of increased computation time.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        min_samples: Minimum number of samples in a neighborhood to be a core point
        metric: Distance metric ('euclidean' or 'manhattan')
        chunk_size: Size of chunks for Dask processing

    Returns:
        Memory-efficient NumbaDBSCANProcessor
    """
    return NumbaDBSCANProcessor(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        precompute_distances=False,  # Avoid precomputing distance matrix to save memory
        compute_silhouette=False,  # Skip silhouette computation to save memory
        chunk_size=chunk_size,
    )


def select_best_eps_dbscan(
    data: Union[np.ndarray, da.Array],
    min_samples: int = 5,
    eps_range: List[float] = None,
    metric: str = "euclidean",
    criterion: str = "silhouette",
    **kwargs,
) -> Tuple[NumbaDBSCANProcessor, float]:
    """
    Select the best eps parameter for DBSCAN by trying multiple values.

    Args:
        data: Input data array
        min_samples: Minimum number of samples in a neighborhood
        eps_range: List of eps values to try. If None, uses heuristic range.
        metric: Distance metric ('euclidean' or 'manhattan')
        criterion: Selection criterion ('silhouette', 'noise_ratio', or 'cluster_count')
        **kwargs: Additional parameters for NumbaDBSCANProcessor

    Returns:
        Tuple of (best_model, best_eps)
    """
    # Convert to numpy if needed
    if isinstance(data, da.Array):
        data = data.compute()

    # Preprocess data
    test_model = NumbaDBSCANProcessor(eps=0.5, min_samples=min_samples)
    data_flat = test_model._preprocess_data(data, **kwargs)

    # If no eps range provided, use heuristic
    if eps_range is None:
        # For heuristic, compute average distance to k-nearest neighbors
        k = min_samples
        distances = _compute_pairwise_distances(data_flat, metric)
        sorted_distances = np.sort(distances, axis=1)
        avg_knn_dist = np.mean(sorted_distances[:, k])

        # Create range around this value
        eps_range = [avg_knn_dist * factor for factor in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5]]

    best_model = None
    best_eps = None
    best_score = -np.inf if criterion != "noise_ratio" else np.inf

    print(f"Testing {len(eps_range)} eps values...")

    for eps in eps_range:
        # Create and fit model
        model = NumbaDBSCANProcessor(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            compute_silhouette=True,
            **kwargs,
        )
        model.fit(data_flat)

        # Get score based on criterion
        if criterion == "silhouette":
            score = model.get_silhouette_score() or -np.inf
        elif criterion == "noise_ratio":
            # Lower noise ratio is better
            n_noise = np.sum(model.get_noise_points_mask())
            score = n_noise / len(data_flat)
        elif criterion == "cluster_count":
            # Higher cluster count is better (up to a point)
            score = model.get_n_clusters()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Update best model if needed
        if (criterion != "noise_ratio" and score > best_score) or (
            criterion == "noise_ratio"
            and score < best_score
            and model.get_n_clusters() > 0
        ):
            best_score = score
            best_model = model
            best_eps = eps

    if best_model is None:
        print("No good models found. Using default eps.")
        best_eps = 0.5
        best_model = NumbaDBSCANProcessor(
            eps=best_eps, min_samples=min_samples, metric=metric, **kwargs
        )
        best_model.fit(data_flat)
    else:
        print(f"Selected eps={best_eps} with {criterion} score: {best_score}")
        if criterion == "cluster_count":
            print(f"Number of clusters: {best_model.get_n_clusters()}")
        elif criterion == "noise_ratio":
            noise_points = np.sum(best_model.get_noise_points_mask())
            print(
                f"Noise ratio: {noise_points / len(data_flat):.2f} ({noise_points}/{len(data_flat)} points)"
            )

    return best_model, best_eps
