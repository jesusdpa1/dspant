"""
DBSCAN clustering implementation with Numba acceleration.

This module provides a fast DBSCAN implementation optimized for
signal processing applications with Numba acceleration.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from sklearn.cluster import DBSCAN as SKLearnDBSCAN

from .base import BaseClusteringProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix with Numba acceleration.

    Args:
        X: Data array (n_samples, n_features)

    Returns:
        Distance matrix (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            # Compute Euclidean distance
            dist = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)

            # Store distance (symmetric matrix)
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


@jit(nopython=True, cache=True)
def _find_neighbors(distances: np.ndarray, eps: float) -> List[np.ndarray]:
    """
    Find neighbors within eps distance for each point.

    Args:
        distances: Distance matrix (n_samples, n_samples)
        eps: Maximum distance for neighbors

    Returns:
        List of neighbor indices for each point
    """
    n_samples = distances.shape[0]
    neighbors = []

    for i in range(n_samples):
        # Find indices where distance <= eps
        indices = np.where(distances[i] <= eps)[0]
        neighbors.append(indices)

    return neighbors


@jit(nopython=True, cache=True)
def _dbscan_inner(neighbors: List[np.ndarray], min_samples: int) -> np.ndarray:
    """
    Core DBSCAN algorithm implementation.

    Args:
        neighbors: List of neighbor arrays for each point
        min_samples: Minimum number of points to form a dense region

    Returns:
        Cluster labels for each point
    """
    n_samples = len(neighbors)
    labels = np.full(n_samples, -1, dtype=np.int32)

    # Find core points
    core_points = np.zeros(n_samples, dtype=np.bool_)
    for i in range(n_samples):
        if len(neighbors[i]) >= min_samples:
            core_points[i] = True

    # Cluster ID counter
    current_cluster = 0

    # Process each point
    for i in range(n_samples):
        # Skip processed points
        if labels[i] != -1:
            continue

        # Skip non-core points
        if not core_points[i]:
            # Noise point
            labels[i] = -1
            continue

        # Start new cluster
        labels[i] = current_cluster

        # Expand cluster (BFS)
        seed_queue = [i]
        seed_index = 0

        while seed_index < len(seed_queue):
            current_point = seed_queue[seed_index]
            seed_index += 1

            # Add neighbors to cluster
            for neighbor in neighbors[current_point]:
                # Skip already processed points
                if labels[neighbor] != -1:
                    continue

                # Add to cluster
                labels[neighbor] = current_cluster

                # If core point, add to queue for expansion
                if core_points[neighbor]:
                    seed_queue.append(neighbor)

        # Move to next cluster
        current_cluster += 1

    return labels


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
def _compute_silhouette_samples(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute silhouette scores for each sample.

    Args:
        X: Data array (n_samples, n_features)
        labels: Cluster labels

    Returns:
        Silhouette scores for each sample
    """
    n_samples = X.shape[0]
    silhouette = np.zeros(n_samples, dtype=np.float32)

    # Ignore noise points (label -1)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    # If no clusters or only one cluster, return zeros
    if len(unique_labels) <= 1:
        return silhouette

    # Compute pairwise distances
    distances = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Compute distance
            dist = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - X[j, k]
                dist += diff * diff
            dist = np.sqrt(dist)

            # Store distance (symmetric matrix)
            distances[i, j] = dist
            distances[j, i] = dist

    # Compute silhouette for each point
    for i in range(n_samples):
        # Skip noise points
        if labels[i] < 0:
            silhouette[i] = 0.0
            continue

        # Get points in same cluster
        same_cluster = np.where(labels == labels[i])[0]

        # If only one point in cluster, silhouette is 0
        if len(same_cluster) == 1:
            silhouette[i] = 0.0
            continue

        # Compute a: mean distance to points in same cluster
        a = np.mean([distances[i, j] for j in same_cluster if j != i])

        # Compute b: minimum mean distance to points in different clusters
        b = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = np.where(labels == label)[0]
                mean_dist = np.mean([distances[i, j] for j in other_cluster])
                b = min(b, mean_dist)

        # Compute silhouette
        if a < b:
            silhouette[i] = 1.0 - a / b
        elif a > b:
            silhouette[i] = b / a - 1.0
        else:
            silhouette[i] = 0.0

    return silhouette


class DBSCANProcessor(BaseClusteringProcessor):
    """
    DBSCAN clustering processor with Numba acceleration.

    This processor provides density-based clustering for signal processing,
    optimized with Numba for high performance, with an option to use
    scikit-learn's implementation for larger datasets.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        use_sklearn: bool = False,
        compute_silhouette: bool = False,
        leaf_size: int = 30,
        n_jobs: Optional[int] = None,
    ):
        """
        Initialize the DBSCAN processor.

        Args:
            eps: Maximum distance between two samples to be considered neighbors
            min_samples: Minimum number of samples in a neighborhood to be a core point
            metric: Distance metric (only 'euclidean' supported in Numba version)
            use_sklearn: Whether to use scikit-learn's implementation (recommended for large datasets)
            compute_silhouette: Whether to compute silhouette scores for evaluation
            leaf_size: Leaf size for ball tree or KD tree (only used with scikit-learn)
            n_jobs: Number of parallel jobs (only used with scikit-learn)
        """
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.use_sklearn = use_sklearn
        self.compute_silhouette = compute_silhouette
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

        # Initialize result variables
        self._sklearn_model = None
        self._distances = None
        self._neighbors = None
        self._silhouette_scores = None
        self._cluster_centers = None  # Note: DBSCAN doesn't compute centers by default
        self._n_clusters = None

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "DBSCANProcessor":
        """
        Fit the DBSCAN model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data before clustering

        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data if specified
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

        # Use scikit-learn implementation
        if self.use_sklearn:
            # Initialize scikit-learn model
            self._sklearn_model = SKLearnDBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric=self.metric,
                algorithm="auto",
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
            )

            # Fit model
            self._sklearn_model.fit(data_flat)
            self._cluster_labels = self._sklearn_model.labels_
            self._distances = None  # Not stored in scikit-learn
            self._neighbors = None  # Not stored in scikit-learn

        # Use Numba-accelerated implementation
        else:
            if self.metric != "euclidean":
                warnings.warn(
                    f"Metric '{self.metric}' not supported in Numba version. Using 'euclidean' instead."
                )

            # Compute distance matrix
            self._distances = _compute_distances(data_flat)

            # Find neighbors
            self._neighbors = _find_neighbors(self._distances, self.eps)

            # Run DBSCAN algorithm
            self._cluster_labels = _dbscan_inner(self._neighbors, self.min_samples)

        # Get number of clusters (excluding noise points)
        if len(self._cluster_labels) > 0:
            n_clusters = len(set(label for label in self._cluster_labels if label >= 0))
            self._n_clusters = n_clusters
        else:
            self._n_clusters = 0

        # Compute silhouette scores if requested
        if self.compute_silhouette and self._n_clusters > 1:
            self._silhouette_scores = _compute_silhouette_samples(
                data_flat, self._cluster_labels
            )
        else:
            self._silhouette_scores = None

        # Compute cluster centers as mean of points in each cluster
        if self._n_clusters > 0:
            centers = []
            for cluster_id in range(self._n_clusters):
                mask = self._cluster_labels == cluster_id
                if np.any(mask):
                    center = np.mean(data_flat[mask], axis=0)
                    centers.append(center)

            if centers:
                self._cluster_centers = np.vstack(centers)

        self._is_fitted = True
        return self

    def predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict cluster assignments for new data points.

        Note: This is an approximate method for DBSCAN, as the algorithm
        doesn't naturally support prediction for new points. This assigns
        each point to the same cluster as its nearest neighbor in the
        training data, or to noise if the nearest point is beyond eps.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data before prediction

        Returns:
            Array of cluster assignments
        """
        if not self._is_fitted:
            raise ValueError("DBSCAN model not fitted. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data if specified
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

        # For scikit-learn, use its transform method
        if self.use_sklearn and self._sklearn_model is not None:
            # Note: scikit-learn DBSCAN doesn't have a predict method,
            # we're using a custom approach based on core samples
            # This assigns each point to the cluster of its nearest core sample if within eps
            core_samples_mask = np.zeros_like(self._sklearn_model.labels_, dtype=bool)
            core_samples_mask[self._sklearn_model.core_sample_indices_] = True

            labels_new = np.full(data_flat.shape[0], -1, dtype=np.int32)

            # Get core samples and their labels
            core_samples = self._sklearn_model.components_
            core_labels = self._sklearn_model.labels_[
                self._sklearn_model.core_sample_indices_
            ]

            # For each new point, find nearest core point
            for i in range(data_flat.shape[0]):
                min_dist = np.inf
                min_idx = -1

                for j in range(core_samples.shape[0]):
                    dist = np.sqrt(np.sum((data_flat[i] - core_samples[j]) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                        min_idx = j

                # Assign to cluster if within eps
                if min_dist <= self.eps:
                    labels_new[i] = core_labels[min_idx]

            return labels_new

        # Custom implementation for Numba version
        # Similar approach but using the original data
        if self._distances is None or len(self._distances) == 0:
            warnings.warn(
                "No distance matrix available. Returning all points as noise."
            )
            return np.full(data_flat.shape[0], -1, dtype=np.int32)

        # Get original data and labels
        original_data = None
        for key, value in kwargs.items():
            if key == "original_data":
                original_data = value
                break

        if original_data is None:
            warnings.warn(
                "Original data not provided for prediction. Returning all points as noise."
            )
            return np.full(data_flat.shape[0], -1, dtype=np.int32)

        # Calculate distances to original data
        labels_new = np.full(data_flat.shape[0], -1, dtype=np.int32)

        for i in range(data_flat.shape[0]):
            min_dist = np.inf
            nearest_idx = -1

            for j in range(original_data.shape[0]):
                dist = 0.0
                for k in range(data_flat.shape[1]):
                    diff = data_flat[i, k] - original_data[j, k]
                    dist += diff * diff
                dist = np.sqrt(dist)

                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = j

            # Assign to cluster if within eps
            if min_dist <= self.eps:
                labels_new[i] = self._cluster_labels[nearest_idx]

        return labels_new

    def fit_predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster assignments in one step.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Array of cluster assignments
        """
        self.fit(data, **kwargs)
        # For DBSCAN, we can just return the labels from the fit
        return self._cluster_labels

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform DBSCAN clustering.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used in clustering, but required by BaseProcessor interface)
            **kwargs: Additional algorithm-specific parameters
                compute_now: Whether to compute immediately (default: False)

        Returns:
            Dask array with cluster assignments
        """
        # Get processing parameters
        compute_now = kwargs.pop("compute_now", False)

        if compute_now:
            # Convert to numpy, fit, and convert back to dask
            data_np = data.compute()
            labels = self.fit_predict(data_np, **kwargs)
            return da.from_array(labels)
        else:
            # DBSCAN generally needs all data at once
            warnings.warn(
                "DBSCAN generally works best with compute_now=True for the entire dataset."
                " Using chunked processing may give suboptimal results."
            )

            # Define function to apply to chunks
            def apply_dbscan(chunk, block_info=None):
                # Create a new processor instance for each chunk to avoid state issues
                local_processor = DBSCANProcessor(
                    eps=self.eps,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    use_sklearn=self.use_sklearn,
                    compute_silhouette=False,
                )
                return local_processor.fit_predict(chunk, **kwargs)

            # Apply to dask array
            result = data.map_blocks(
                apply_dbscan,
                drop_axis=list(range(1, data.ndim)),  # Output has one dimension fewer
                dtype=np.int32,
            )

            return result

    def get_silhouette_scores(self) -> Optional[np.ndarray]:
        """Get silhouette scores if computed during fitting."""
        return self._silhouette_scores

    def get_average_silhouette(self) -> Optional[float]:
        """Get average silhouette score across all samples."""
        if self._silhouette_scores is not None:
            # Exclude noise points (label -1) from average
            valid_mask = self._cluster_labels >= 0
            if np.any(valid_mask):
                return np.mean(self._silhouette_scores[valid_mask])
        return None

    def get_n_clusters(self) -> int:
        """Get the number of clusters found (excluding noise)."""
        return self._n_clusters if self._n_clusters is not None else 0

    def get_noise_points_mask(self) -> Optional[np.ndarray]:
        """Get boolean mask indicating noise points (label -1)."""
        if self._cluster_labels is not None:
            return self._cluster_labels == -1
        return None

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration and results."""
        base_summary = super().summary
        base_summary.update(
            {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "use_sklearn": self.use_sklearn,
                "n_clusters": self.get_n_clusters(),
                "avg_silhouette": self.get_average_silhouette(),
                "n_noise_points": np.sum(self.get_noise_points_mask())
                if self.get_noise_points_mask() is not None
                else None,
            }
        )
        return base_summary


# Factory functions for easy creation


def create_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    use_sklearn: bool = False,
) -> DBSCANProcessor:
    """
    Create a standard DBSCAN processor.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        min_samples: Minimum number of samples in a neighborhood to be a core point
        use_sklearn: Whether to use scikit-learn's implementation

    Returns:
        Configured DBSCANProcessor
    """
    return DBSCANProcessor(
        eps=eps,
        min_samples=min_samples,
        use_sklearn=use_sklearn,
    )


def create_dense_dbscan(
    eps: float = 0.5,
    use_sklearn: bool = False,
) -> DBSCANProcessor:
    """
    Create a DBSCAN processor for dense clusters.

    This configuration uses more strict density requirements,
    resulting in fewer, denser clusters.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        use_sklearn: Whether to use scikit-learn's implementation

    Returns:
        Configured DBSCANProcessor for dense clusters
    """
    return DBSCANProcessor(
        eps=eps,
        min_samples=10,  # More samples required for core points
        use_sklearn=use_sklearn,
    )


def create_sparse_dbscan(
    eps: float = 0.5,
    use_sklearn: bool = False,
) -> DBSCANProcessor:
    """
    Create a DBSCAN processor for sparse clusters.

    This configuration is more relaxed about density requirements,
    potentially finding more clusters in sparse data.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        use_sklearn: Whether to use scikit-learn's implementation

    Returns:
        Configured DBSCANProcessor for sparse clusters
    """
    return DBSCANProcessor(
        eps=eps,
        min_samples=3,  # Fewer samples required for core points
        use_sklearn=use_sklearn,
    )


def create_sklearn_dbscan(
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "euclidean",
    n_jobs: int = -1,
) -> DBSCANProcessor:
    """
    Create a DBSCAN processor using scikit-learn's implementation.

    This is recommended for larger datasets where the full distance
    matrix would be too memory-intensive. Scikit-learn uses more
    efficient algorithms for large datasets.

    Args:
        eps: Maximum distance between two samples to be considered neighbors
        min_samples: Minimum number of samples in a neighborhood to be a core point
        metric: Distance metric ('euclidean', 'manhattan', etc.)
        n_jobs: Number of parallel jobs (-1 to use all processors)

    Returns:
        Configured DBSCANProcessor using scikit-learn
    """
    return DBSCANProcessor(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        use_sklearn=True,
        n_jobs=n_jobs,
    )
