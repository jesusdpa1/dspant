# src/dspant/processor/dimensionality_reduction/umap.py
"""
UMAP implementation for dimensionality reduction and visualization.

This module provides UMAP (Uniform Manifold Approximation and Projection)
implementations for effective high-dimensional data reduction and visualization,
optimized with Numba acceleration.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from .base import BaseDimensionalityReductionProcessor


@jit(nopython=True, parallel=True, cache=True)
def _flatten_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Efficiently flatten 3D data to 2D for UMAP with parallelization.

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


@jit(nopython=True, parallel=True, cache=True)
def _normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for UMAP processing with parallel acceleration.

    Args:
        data: Input data array (n_samples × n_features)

    Returns:
        Normalized data
    """
    n_samples = data.shape[0]
    normalized = np.zeros_like(data, dtype=np.float32)

    for i in prange(n_samples):
        # Find min and max for this sample
        min_val = np.inf
        max_val = -np.inf

        for j in range(data.shape[1]):
            val = data[i, j]
            if val < min_val:
                min_val = val
            if val > max_val:
                max_val = val

        # Min-max normalization
        range_val = max_val - min_val
        if range_val > 1e-10:
            for j in range(data.shape[1]):
                normalized[i, j] = (data[i, j] - min_val) / range_val
        else:
            # If range is too small, center the data
            mean_val = 0.0
            for j in range(data.shape[1]):
                mean_val += data[i, j]
            mean_val /= data.shape[1]

            for j in range(data.shape[1]):
                normalized[i, j] = data[i, j] - mean_val

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
        indices = np.where(labels == label)[0]

        # If we have more indices than needed, select deterministically
        if len(indices) > samples_per_label:
            step = max(1, len(indices) // samples_per_label)
            selected = indices[::step][:samples_per_label]
        else:
            selected = indices

        # Add to result
        for idx in selected:
            result_indices.append(idx)

    # Convert to array and ensure we don't exceed max_samples
    result = np.array(result_indices[:max_samples])
    return result


class UMAPProcessor(BaseDimensionalityReductionProcessor):
    """
    UMAP processor for dimensionality reduction and visualization.

    This processor implements UMAP for reducing high-dimensional data while
    preserving both local and global structure better than t-SNE.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        learning_rate: float = 1.0,
        random_state: int = 42,
        low_memory: bool = False,
        use_numba: bool = True,
        n_epochs: Optional[int] = None,
        init: str = "spectral",
        precomputed_knn: bool = False,
    ):
        """
        Initialize the UMAP processor.

        Args:
            n_components: Number of dimensions in output (typically 2 or 3)
            n_neighbors: Number of nearest neighbors to consider
            min_dist: Minimum distance between points in embedding
            metric: Distance metric to use
            learning_rate: Learning rate for optimization
            random_state: Random seed for reproducibility
            low_memory: Use memory-efficient implementation
            use_numba: Whether to use additional Numba optimizations
            n_epochs: Number of training epochs (None for auto)
            init: Initialization method ('spectral' or 'random')
            precomputed_knn: Whether to use precomputed k-nearest neighbors
        """
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.low_memory = low_memory
        self.use_numba = use_numba
        self.n_epochs = n_epochs
        self.init = init
        self.precomputed_knn = precomputed_knn

        # Initialize model
        self._umap = None
        self._embedding = None
        self._knn_indices = None
        self._knn_dists = None
        self._labels = None

        # Check if UMAP is available
        try:
            import umap

            self._has_umap = True
        except ImportError:
            self._has_umap = False
            warnings.warn(
                "UMAP package not found. Install it with 'pip install umap-learn'."
                "UMAPProcessor will raise errors when used."
            )

    def _preprocess_data(self, data: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Preprocess data for UMAP with optional Numba acceleration.

        Args:
            data: Input data array
            normalize: Whether to normalize the data

        Returns:
            Processed data ready for UMAP
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
                # Standard min-max normalization
                data_min = np.min(data_flat, axis=1, keepdims=True)
                data_max = np.max(data_flat, axis=1, keepdims=True)
                data_range = data_max - data_min
                data_range[data_range < 1e-10] = 1.0  # Avoid division by zero
                data_flat = (data_flat - data_min) / data_range

        return data_flat

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data using UMAP.

        Note: UMAP requires the entire dataset to be in memory.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, but required by BaseProcessor interface)
            **kwargs: Additional keyword arguments
                max_samples: Maximum number of samples to use
                normalize: Whether to normalize the data (default: True)
                use_existing_labels: Whether to use existing labels for stratified sampling
                labels: Optional array of labels for stratified sampling

        Returns:
            Dask array with reduced dimensions
        """
        self._check_umap_available()

        # Get parameters from kwargs
        max_samples = kwargs.get("max_samples", 10000)
        normalize = kwargs.get("normalize", True)
        use_existing_labels = kwargs.get("use_existing_labels", False)
        labels = kwargs.get("labels", None)

        # Convert to numpy
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
                f"Subsampled UMAP input from {data_np.shape[0]} to {len(indices)} samples"
            )

        # Fit and transform
        embedding = self.fit_transform(data_flat, **kwargs)

        # Convert back to dask array
        return da.from_array(embedding)

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "UMAPProcessor":
        """
        Fit the UMAP model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        self._check_umap_available()

        # Import UMAP here to avoid import issues if package is missing
        import umap

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Handle different input shapes
        if data.ndim > 2:
            if self.use_numba:
                data_flat = _flatten_3d_data(data)
            else:
                n_samples = data.shape[0]
                data_flat = data.reshape(n_samples, -1)
        else:
            data_flat = data

        # Adjust n_neighbors if needed
        actual_n_neighbors = min(self.n_neighbors, data_flat.shape[0] - 1)
        if actual_n_neighbors < self.n_neighbors:
            print(
                f"Reducing n_neighbors from {self.n_neighbors} to {actual_n_neighbors} due to sample size"
            )

        # Extract UMAP specific parameters from kwargs
        normalize = kwargs.get("normalize", False)  # We already normalize in preprocess
        local_connectivity = kwargs.get("local_connectivity", 1.0)
        set_op_mix_ratio = kwargs.get("set_op_mix_ratio", 1.0)

        # Create the UMAP parameters
        umap_params = {
            "n_components": self.n_components,
            "n_neighbors": actual_n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "low_memory": self.low_memory,
            "n_epochs": self.n_epochs,
            "init": self.init,
            "spread": kwargs.get("spread", 1.0),
            "local_connectivity": local_connectivity,
            "set_op_mix_ratio": set_op_mix_ratio,
            "transform_seed": self.random_state,
            "force_approximation_algorithm": kwargs.get(
                "force_approximation_algorithm", False
            ),
            "verbose": kwargs.get("verbose", False),
        }

        # Use precomputed KNN if available
        if (
            self.precomputed_knn
            and self._knn_indices is not None
            and self._knn_dists is not None
        ):
            umap_params["precomputed_knn"] = (self._knn_indices, self._knn_dists)

        # Initialize UMAP model
        self._umap = umap.UMAP(**umap_params)

        # Fit UMAP without transforming
        self._umap.fit(data_flat)
        self._is_fitted = True

        return self

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data using the fitted UMAP model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Embedded data in lower dimensions
        """
        self._check_umap_available()

        if not self._is_fitted:
            raise ValueError("UMAP model not fitted. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data
        normalize = kwargs.get("normalize", True)
        data_flat = self._preprocess_data(data, normalize)

        # Transform data using fitted model
        embedding = self._umap.transform(data_flat)
        return embedding

    def fit_transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Fit the UMAP model and transform the data in one step.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Embedded data in lower dimensions
        """
        self._check_umap_available()

        # Import UMAP here to avoid import issues if package is missing
        import umap

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data
        normalize = kwargs.get("normalize", True)
        data_flat = self._preprocess_data(data, normalize)

        # Adjust n_neighbors if needed
        actual_n_neighbors = min(self.n_neighbors, data_flat.shape[0] - 1)

        # Extract UMAP specific parameters from kwargs
        local_connectivity = kwargs.get("local_connectivity", 1.0)
        set_op_mix_ratio = kwargs.get("set_op_mix_ratio", 1.0)

        # Create the UMAP parameters
        umap_params = {
            "n_components": self.n_components,
            "n_neighbors": actual_n_neighbors,
            "min_dist": self.min_dist,
            "metric": self.metric,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "low_memory": self.low_memory,
            "n_epochs": self.n_epochs,
            "init": self.init,
            "spread": kwargs.get("spread", 1.0),
            "local_connectivity": local_connectivity,
            "set_op_mix_ratio": set_op_mix_ratio,
            "transform_seed": self.random_state,
            "force_approximation_algorithm": kwargs.get(
                "force_approximation_algorithm", False
            ),
            "verbose": kwargs.get("verbose", False),
        }

        # Use precomputed KNN if available
        if (
            self.precomputed_knn
            and self._knn_indices is not None
            and self._knn_dists is not None
        ):
            umap_params["precomputed_knn"] = (self._knn_indices, self._knn_dists)

        # Initialize and fit UMAP model
        umap_model = umap.UMAP(**umap_params)

        # Get UMAP embedding
        embedding = umap_model.fit_transform(data_flat)

        # Store the model and embedding
        self._umap = umap_model
        self._embedding = embedding
        self._is_fitted = True

        # Store the KNN indices and distances for possible reuse
        self._knn_indices = getattr(umap_model, "_knn_indices", None)
        self._knn_dists = getattr(umap_model, "_knn_dists", None)

        return embedding

    def compute_knn(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute k-nearest neighbors for faster UMAP embedding.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (knn_indices, knn_distances)
        """
        self._check_umap_available()

        # Import UMAP here to avoid import issues if package is missing
        import umap

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Preprocess data
        normalize = kwargs.get("normalize", True)
        data_flat = self._preprocess_data(data, normalize)

        # Adjust n_neighbors if needed
        actual_n_neighbors = min(self.n_neighbors, data_flat.shape[0] - 1)

        # Use UMAP's nearest_neighbors function to compute KNN
        from umap.umap_ import nearest_neighbors

        # Compute nearest neighbors
        knn_indices, knn_dists = nearest_neighbors(
            data_flat,
            n_neighbors=actual_n_neighbors,
            metric=self.metric,
            metric_kwds={},
            angular=False,
            random_state=self.random_state,
        )

        # Store for later use
        self._knn_indices = knn_indices
        self._knn_dists = knn_dists

        return knn_indices, knn_dists

    def inverse_transform(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> np.ndarray:
        """
        Transform data from the embedding space back to the original space.

        Note: UMAP inverse transform is an approximation and may not be accurate.

        Args:
            data: Input data array in reduced dimensions
            **kwargs: Additional keyword arguments

        Returns:
            Approximated data in original space
        """
        self._check_umap_available()

        if not self._is_fitted:
            raise ValueError("UMAP model not fitted. Call fit() first.")

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Check if UMAP implementation supports inverse_transform
        if hasattr(self._umap, "inverse_transform"):
            return self._umap.inverse_transform(data)
        else:
            raise NotImplementedError(
                "Your UMAP version does not support inverse_transform. "
                "Update to the latest version of umap-learn if needed."
            )

    def _check_umap_available(self):
        """Check if UMAP package is available."""
        if not self._has_umap:
            raise ImportError(
                "UMAP package not found. Install it with 'pip install umap-learn'."
            )

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
                "n_neighbors": self.n_neighbors,
                "min_dist": self.min_dist,
                "metric": self.metric,
                "is_fitted": self._is_fitted,
                "has_umap_package": self._has_umap,
                "use_numba": self.use_numba,
                "init": self.init,
                "precomputed_knn": self.precomputed_knn
                and self._knn_indices is not None
                and self._knn_dists is not None,
            }
        )
        return base_summary


# Factory functions for easy creation


def create_umap(
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    use_numba: bool = True,
) -> UMAPProcessor:
    """
    Create a standard UMAP processor.

    Args:
        n_components: Number of dimensions in output (typically 2 or 3)
        n_neighbors: Number of nearest neighbors to consider
        min_dist: Minimum distance between points in embedding
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured UMAPProcessor
    """
    return UMAPProcessor(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        use_numba=use_numba,
    )


def create_visualization_umap(
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    use_numba: bool = True,
) -> UMAPProcessor:
    """
    Create a UMAP processor optimized for 2D visualization.

    Args:
        n_neighbors: Number of nearest neighbors to consider
        min_dist: Minimum distance between points in embedding
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured UMAPProcessor for visualization
    """
    return UMAPProcessor(
        n_components=2,  # Fixed at 2D for visualization
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        use_numba=use_numba,
        init="spectral",  # Spectral initialization often works better for visualization
    )


def create_preserving_umap(
    n_components: int = 2,
    random_state: int = 42,
    use_numba: bool = True,
) -> UMAPProcessor:
    """
    Create a UMAP processor that preserves global structure.

    This configuration uses more neighbors and a larger min_dist to better
    preserve global structure in the data.

    Args:
        n_components: Number of dimensions in output
        random_state: Random seed for reproducibility
        use_numba: Whether to use Numba acceleration

    Returns:
        Configured UMAPProcessor that preserves global structure
    """
    return UMAPProcessor(
        n_components=n_components,
        n_neighbors=50,  # More neighbors for global structure
        min_dist=0.5,  # Larger min_dist for better global structure
        random_state=random_state,
        use_numba=use_numba,
        set_op_mix_ratio=0.8,  # Prefer global structure (higher values → more global)
    )


def create_fast_umap(
    n_components: int = 2,
    random_state: int = 42,
) -> UMAPProcessor:
    """
    Create a UMAP processor optimized for speed.

    This configuration uses fewer neighbors, fewer epochs, and other optimizations
    to speed up UMAP embedding at some cost to quality.

    Args:
        n_components: Number of dimensions in output
        random_state: Random seed for reproducibility

    Returns:
        Configured UMAPProcessor optimized for speed
    """
    return UMAPProcessor(
        n_components=n_components,
        n_neighbors=10,  # Fewer neighbors for speed
        min_dist=0.1,
        random_state=random_state,
        use_numba=True,  # Always use Numba for speed
        n_epochs=100,  # Fewer epochs for speed
        low_memory=False,  # Trade memory for speed
        init="random",  # Random initialization is faster
    )


def create_supervised_umap(
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> UMAPProcessor:
    """
    Create a UMAP processor for supervised dimensionality reduction.

    This processor preserves structure guided by class labels for better
    class separation.

    Args:
        n_components: Number of dimensions in output
        n_neighbors: Number of nearest neighbors to consider
        min_dist: Minimum distance between points in embedding
        random_state: Random seed for reproducibility

    Returns:
        Configured UMAPProcessor for supervised learning
    """
    # Note: You'll need to pass y=labels when calling fit or fit_transform
    # UMAP will use this for supervised dimensionality reduction
    return UMAPProcessor(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        use_numba=True,
        metric="euclidean",
        # Set up supervised parameters, but target_metric will be set by UMAP automatically
        # based on the label type when y is passed to fit_transform
    )
