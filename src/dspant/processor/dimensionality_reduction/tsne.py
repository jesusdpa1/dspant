# src/dspant/processor/dimensionality_reduction/tsne.py
"""
t-SNE implementation for dimensionality reduction and visualization.

This module provides t-SNE (t-Distributed Stochastic Neighbor Embedding)
implementations for high-dimensional data visualization with Numba acceleration.
"""

from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from sklearn.manifold import TSNE

from .base import BaseDimensionalityReductionProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distances with Numba acceleration.

    Args:
        X: Input data matrix (n_samples × n_features)
        metric: Distance metric ('euclidean' or 'manhattan')

    Returns:
        Pairwise distance matrix (n_samples × n_samples)
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float32)

    if metric == "euclidean":
        for i in prange(n_samples):
            for j in range(i + 1, n_samples):
                # Euclidean distance
                d = 0.0
                for k in range(X.shape[1]):
                    d += (X[i, k] - X[j, k]) ** 2
                d = np.sqrt(d)
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

    return distances


@jit(nopython=True, cache=True)
def _flatten_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Efficiently flatten 3D data to 2D for t-SNE.

    Args:
        data: Input 3D data array (samples × timepoints × channels)

    Returns:
        Flattened 2D data (samples × (timepoints*channels))
    """
    n_samples, n_timepoints, n_channels = data.shape
    flattened = np.zeros((n_samples, n_timepoints * n_channels), dtype=data.dtype)

    for i in range(n_samples):
        for j in range(n_timepoints):
            for k in range(n_channels):
                flattened[i, j * n_channels + k] = data[i, j, k]

    return flattened


@jit(nopython=True, parallel=True, cache=True)
def _normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for t-SNE processing.

    Args:
        data: Input data array (n_samples × n_features)

    Returns:
        Normalized data
    """
    n_samples = data.shape[0]
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in prange(n_samples):
        # Calculate mean and std for each sample
        mean_val = 0.0
        for j in range(data.shape[1]):
            mean_val += data[i, j]
        mean_val /= data.shape[1]

        var_val = 0.0
        for j in range(data.shape[1]):
            var_val += (data[i, j] - mean_val) ** 2
        var_val /= data.shape[1]
        std_val = np.sqrt(var_val) if var_val > 1e-10 else 1.0

        # Normalize sample
        for j in range(data.shape[1]):
            normalized[i, j] = (data[i, j] - mean_val) / std_val

    return normalized


@jit(nopython=True, cache=True)
def _stratified_sample(labels: np.ndarray, max_samples: int) -> np.ndarray:
    """
    Perform stratified sampling with Numba acceleration.

    Args:
        labels: Array of cluster labels
        max_samples: Maximum number of samples to return

    Returns:
        Indices of selected samples
    """
    # Count unique labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)

    # Calculate samples per label
    samples_per_label = max(1, max_samples // n_labels)

    # Collect indices for each label
    result_indices = []
    for label in unique_labels:
        # Find indices for this label
        label_indices = np.where(labels == label)[0]

        # If we have more indices than needed, select randomly
        if len(label_indices) > samples_per_label:
            # Simple deterministic selection (can't use np.random in numba)
            step = len(label_indices) // samples_per_label
            selected_indices = label_indices[::step][:samples_per_label]
        else:
            selected_indices = label_indices

        # Add to result
        for idx in selected_indices:
            result_indices.append(idx)

    # Convert list to array (limited to max_samples)
    result = np.zeros(min(len(result_indices), max_samples), dtype=np.int64)
    for i in range(len(result)):
        if i < len(result_indices):
            result[i] = result_indices[i]

    return result


class TSNEProcessor(BaseDimensionalityReductionProcessor):
    """
    t-SNE processor for dimensionality reduction and visualization.

    This processor implements t-SNE for reducing high-dimensional data to
    2D or 3D for visualization, preserving local relationships.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
        metric: str = "euclidean",
        use_numba: bool = True,
        init: str = "pca",
    ):
        """
        Initialize the t-SNE processor.

        Args:
            n_components: Number of dimensions in output (typically 2 or 3)
            perplexity: Related to number of nearest neighbors in manifold learning
            early_exaggeration: Controls how tight clusters are in the embedding
            learning_rate: Learning rate for optimization
            n_iter: Number of iterations for optimization
            random_state: Random seed for reproducibility
            metric: Distance metric to use
            use_numba: Whether to use Numba acceleration
            init: Initialization method ('pca', 'random')
        """
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.metric = metric
        self.use_numba = use_numba
        self.init = init

        # Initialize model
        self._tsne = None
        self._embedding = None
        self._distance_matrix = None
        self._labels = None

    def _preprocess_data(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess data for t-SNE with optional Numba acceleration.

        Args:
            data: Input data array
            normalize: Whether to normalize the data

        Returns:
            Processed data ready for t-SNE
        """
        # Handle different input shapes
        if data.ndim > 2:
            if self.use_numba:
                data_flat = _flatten_3d_data(data)
            else:
                n_samples = data.shape[0]
                data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        # Normalize if requested
        if normalize:
            if self.use_numba:
                data_flat = _normalize_data(data_flat)
            else:
                # Standard normalization
                data_flat = (data_flat - np.mean(data_flat, axis=1, keepdims=True)) / (
                    np.std(data_flat, axis=1, keepdims=True) + 1e-10
                )

        return data_flat

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data using t-SNE.

        Note: t-SNE is not well-suited for large dask arrays due to its
        computational complexity. This method will compute the input data
        and perform t-SNE on the entire dataset.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, but required by BaseProcessor interface)
            **kwargs: Additional keyword arguments
                max_samples: Maximum number of samples to use
                normalize: Whether to normalize the data (default: True)
                precompute_distances: Whether to precompute distances with Numba (default: False)
                use_existing_labels: Whether to use existing labels for stratified sampling
                labels: Optional array of labels for stratified sampling

        Returns:
            Dask array with reduced dimensions
        """
        # Extract parameters from kwargs
        max_samples = kwargs.get("max_samples", 10000)
        normalize = kwargs.get("normalize", True)
        precompute_distances = kwargs.get("precompute_distances", False)
        use_existing_labels = kwargs.get("use_existing_labels", False)
        labels = kwargs.get("labels", None)

        # Convert to numpy (t-SNE is not scalable for large dask arrays)
        data_np = data.compute()

        # Preprocess data with Numba acceleration if enabled
        data_flat = self._preprocess_data(data_np, normalize)

        # Store labels if provided
        if labels is not None:
            self._labels = labels

        # Subsample if too many points
        if data_flat.shape[0] > max_samples:
            if use_existing_labels and self._labels is not None:
                # Use stratified sampling based on existing labels
                if self.use_numba:
                    indices = _stratified_sample(self._labels, max_samples)
                else:
                    # Standard stratified sampling
                    unique_labels = np.unique(self._labels)
                    indices = []
                    samples_per_label = max(1, max_samples // len(unique_labels))

                    for label in unique_labels:
                        label_indices = np.where(self._labels == label)[0]
                        if len(label_indices) > samples_per_label:
                            selected = np.random.choice(
                                label_indices, samples_per_label, replace=False
                            )
                            indices.extend(selected)
                        else:
                            indices.extend(label_indices)

                    indices = np.array(indices)[:max_samples]
            else:
                # Regular random sampling
                indices = np.random.choice(
                    data_flat.shape[0], max_samples, replace=False
                )

            data_flat = data_flat[indices]
            print(
                f"Subsampled t-SNE input from {data_np.shape[0]} to {len(indices)} samples"
            )

        # Precompute distance matrix if requested and using Numba
        if (
            precompute_distances
            and self.use_numba
            and self.metric in ["euclidean", "manhattan"]
        ):
            self._distance_matrix = _compute_pairwise_distances(data_flat, self.metric)
            # Adjust kwargs for fit_transform
            kwargs["metric"] = "precomputed"
            # Use the distance matrix instead of data
            embedding = self.fit_transform(self._distance_matrix, **kwargs)
        else:
            # Regular t-SNE
            embedding = self.fit_transform(data_flat, **kwargs)

        # Convert back to dask array
        return da.from_array(embedding)

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "TSNEProcessor":
        """
        Fit the t-SNE model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        # Not implemented directly since t-SNE doesn't have a separate fit method
        self._is_fitted = True
        return self

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data using the fitted t-SNE model.

        Note: t-SNE doesn't have a separate transform method for new data.
        This will compute a new embedding for the provided data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Embedded data in lower dimensions
        """
        # t-SNE doesn't support separate transform for new data
        # We'll fit a new model with the same parameters
        return self.fit_transform(data, **kwargs)

    def fit_transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Fit the t-SNE model and transform the data in one step.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                metric: Override the distance metric
                precomputed: Whether the input is a precomputed distance matrix

        Returns:
            Embedded data in lower dimensions
        """
        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Check if input is a precomputed distance matrix
        metric = kwargs.get("metric", self.metric)
        is_precomputed = kwargs.get("precomputed", False) or metric == "precomputed"

        if not is_precomputed:
            # Handle different input shapes if not precomputed
            if data.ndim > 2:
                if self.use_numba:
                    data_flat = _flatten_3d_data(data)
                else:
                    n_samples = data.shape[0]
                    data_flat = data.reshape(n_samples, -1)
            else:
                data_flat = data
        else:
            # Use data directly if it's precomputed
            data_flat = data

        # Adjust perplexity if needed
        actual_perplexity = min(self.perplexity, data_flat.shape[0] - 1)

        # Initialize t-SNE model with optimizations
        tsne_params = {
            "n_components": self.n_components,
            "perplexity": actual_perplexity,
            "early_exaggeration": self.early_exaggeration,
            "learning_rate": self.learning_rate,
            "n_iter": self.n_iter,
            "random_state": self.random_state,
            "metric": metric,
            "init": self.init,
            # Add any additional parameters that may help performance
            "method": "barnes_hut" if data_flat.shape[0] > 2000 else "exact",
            "n_jobs": -1
            if data_flat.shape[0] > 500
            else None,  # Use parallelism for larger datasets
        }

        # Update with any kwargs
        for key, value in kwargs.items():
            if key not in ["metric", "precomputed"]:
                tsne_params[key] = value

        # Create and fit model
        tsne = TSNE(**tsne_params)
        embedding = tsne.fit_transform(data_flat)

        # Store the model and embedding
        self._tsne = tsne
        self._embedding = embedding
        self._is_fitted = True

        return embedding

    def inverse_transform(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> np.ndarray:
        """
        t-SNE does not support inverse transform.

        This method raises NotImplementedError.
        """
        raise NotImplementedError("t-SNE does not support inverse transformation.")

    @property
    def embedding(self) -> Optional[np.ndarray]:
        """Get the last computed embedding if available."""
        return self._embedding

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "n_components": self.n_components,
                "perplexity": self.perplexity,
                "n_iter": self.n_iter,
                "metric": self.metric,
                "is_fitted": self._is_fitted,
                "use_numba": self.use_numba,
                "init": self.init,
            }
        )
        return base_summary


# Factory functions for easy creation


def create_tsne(
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    use_numba: bool = True,
) -> TSNEProcessor:
    """
    Create a standard t-SNE processor.

    Args:
        n_components: Number of dimensions in output (typically 2 or 3)
        perplexity: Related to number of nearest neighbors
        n_iter: Number of iterations for optimization
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured TSNEProcessor
    """
    return TSNEProcessor(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        use_numba=use_numba,
    )


def create_visualization_tsne(
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    use_numba: bool = True,
) -> TSNEProcessor:
    """
    Create a t-SNE processor optimized for 2D visualization.

    Args:
        perplexity: Related to number of nearest neighbors
        n_iter: Number of iterations for optimization
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured TSNEProcessor for visualization
    """
    return TSNEProcessor(
        n_components=2,  # Fixed at 2D for visualization
        perplexity=perplexity,
        early_exaggeration=12.0,  # Slightly higher for better cluster separation
        learning_rate=200.0,  # Standard value
        n_iter=n_iter,
        random_state=random_state,
        use_numba=use_numba,
        init="pca",  # PCA initialization is usually better for visualization
    )


def create_fast_tsne(
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> TSNEProcessor:
    """
    Create a t-SNE processor optimized for speed.

    This configuration uses fewer iterations and optimizes for performance.

    Args:
        n_components: Number of dimensions in output (typically 2 or 3)
        perplexity: Related to number of nearest neighbors
        random_state: Random seed for reproducibility

    Returns:
        Configured TSNEProcessor for faster processing
    """
    return TSNEProcessor(
        n_components=n_components,
        perplexity=perplexity,
        early_exaggeration=4.0,  # Lower value for faster convergence
        learning_rate=200.0,
        n_iter=500,  # Fewer iterations
        random_state=random_state,
        use_numba=True,  # Always use Numba
        init="pca",  # PCA initialization converges faster
    )
