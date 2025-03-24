# src/dspant/processor/dimensionality_reduction/tsne_numba.py
"""
Numba-accelerated t-SNE implementation for dimensionality reduction.

This module provides a high-performance t-SNE implementation using Numba for
acceleration, with no dependency on scikit-learn. This allows for better
performance and avoids type consistency issues when used in processing pipelines.
"""

import time
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from .base import BaseDimensionalityReductionProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances with Numba acceleration.

    Args:
        X: Input data matrix (n_samples × n_features)

    Returns:
        Pairwise distance matrix (n_samples × n_samples)
    """
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            # Euclidean distance
            d = 0.0
            for k in range(X.shape[1]):
                d += (X[i, k] - X[j, k]) ** 2
            d = np.sqrt(d)
            distances[i, j] = d
            distances[j, i] = d

    return distances


@jit(nopython=True, cache=True)
def _compute_joint_probabilities(
    distances: np.ndarray, perplexity: float, tol: float = 1e-5
) -> np.ndarray:
    """
    Compute joint probabilities for t-SNE.

    Args:
        distances: Pairwise distance matrix (n_samples × n_samples)
        perplexity: Perplexity parameter related to effective neighbors
        tol: Tolerance for perplexity binary search

    Returns:
        Joint probability matrix (n_samples × n_samples)
    """
    n_samples = distances.shape[0]
    P = np.zeros((n_samples, n_samples), dtype=np.float32)
    beta = np.ones(n_samples, dtype=np.float32)
    logU = np.log(perplexity)

    # For each data point find optimal sigma (precision)
    for i in range(n_samples):
        # Initialize bounds for binary search
        betamin = -np.inf
        betamax = np.inf

        # Get distances from point i to all other points
        dist_i = distances[i]

        # Compute H for initial beta
        sum_Pi = 0.0
        for j in range(n_samples):
            if i != j:
                P[i, j] = np.exp(-dist_i[j] * beta[i])
                sum_Pi += P[i, j]

        if sum_Pi == 0.0:
            sum_Pi = 1e-8

        sum_disti_Pi = 0.0
        for j in range(n_samples):
            if i != j:
                P[i, j] /= sum_Pi
                sum_disti_Pi += dist_i[j] * P[i, j]

        H = np.log(sum_Pi) + beta[i] * sum_disti_Pi

        # Binary search for beta that gives desired perplexity
        Hdiff = H - logU
        tries = 0

        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf:
                    beta[i] = beta[i] * 2.0
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i]
                if betamin == -np.inf:
                    beta[i] = beta[i] / 2.0
                else:
                    beta[i] = (beta[i] + betamin) / 2.0

            # Recompute
            sum_Pi = 0.0
            for j in range(n_samples):
                if i != j:
                    P[i, j] = np.exp(-dist_i[j] * beta[i])
                    sum_Pi += P[i, j]

            if sum_Pi == 0.0:
                sum_Pi = 1e-8

            sum_disti_Pi = 0.0
            for j in range(n_samples):
                if i != j:
                    P[i, j] /= sum_Pi
                    sum_disti_Pi += dist_i[j] * P[i, j]

            H = np.log(sum_Pi) + beta[i] * sum_disti_Pi
            Hdiff = H - logU
            tries += 1

    # Symmetrize the joint probability matrix
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            P[i, j] = (P[i, j] + P[j, i]) / (2.0 * n_samples)
            P[j, i] = P[i, j]

    # Apply early exaggeration
    P = P * 4.0  # Default early exaggeration factor

    return P


@jit(nopython=True, cache=True)
def _compute_q_joint(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Q joint probability matrix for the output embedding.

    Args:
        Y: Current embedding (n_samples × n_components)

    Returns:
        Tuple of (Q matrix, distances in embedding space)
    """
    n_samples = Y.shape[0]

    # Compute squared Euclidean distances
    embedding_distances = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            # Squared Euclidean distance
            d = 0.0
            for k in range(Y.shape[1]):
                d += (Y[i, k] - Y[j, k]) ** 2
            embedding_distances[i, j] = d
            embedding_distances[j, i] = d

    # Add 1 to get student's t-distribution denominator (1 + ||y_i - y_j||²)
    Q = 1.0 / (1.0 + embedding_distances)

    # Set diagonal to zero
    for i in range(n_samples):
        Q[i, i] = 0.0

    # Normalize Q
    sum_Q = np.sum(Q)
    if sum_Q == 0.0:
        sum_Q = 1e-8

    Q = Q / sum_Q

    return Q, embedding_distances


@jit(nopython=True, cache=True)
def _compute_gradient(
    P: np.ndarray, Q: np.ndarray, Y: np.ndarray, embedding_distances: np.ndarray
) -> np.ndarray:
    """
    Compute the gradient for t-SNE optimization.

    Args:
        P: Joint probability matrix from high-dimensional space
        Q: Joint probability matrix from low-dimensional space
        Y: Current embedding (n_samples × n_components)
        embedding_distances: Distances in embedding space

    Returns:
        Gradient (n_samples × n_components)
    """
    n_samples, n_components = Y.shape
    grad = np.zeros_like(Y, dtype=np.float32)

    # Precompute some values
    inv_distances = np.zeros_like(embedding_distances, dtype=np.float32)
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # (1 + ||y_i - y_j||²)⁻¹
                inv_distances[i, j] = 1.0 / (1.0 + embedding_distances[i, j])

    # Compute gradient
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # (p_ij - q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||²)⁻¹
                coeff = 4.0 * (P[i, j] - Q[i, j]) * inv_distances[i, j]
                for k in range(n_components):
                    grad[i, k] += coeff * (Y[i, k] - Y[j, k])

    return grad


@jit(nopython=True, cache=True)
def _gradient_descent(
    P: np.ndarray,
    Y: np.ndarray,
    n_iter: int,
    learning_rate: float,
    momentum: float = 0.8,
    min_gain: float = 0.01,
    early_exaggeration_iter: int = 250,
    early_exaggeration: float = 4.0,
) -> np.ndarray:
    """
    Perform gradient descent optimization for t-SNE.

    Args:
        P: Joint probability matrix from high-dimensional space
        Y: Initial embedding (n_samples × n_components)
        n_iter: Number of iterations
        learning_rate: Learning rate for gradient descent
        momentum: Momentum for gradient descent
        min_gain: Minimum gain for adaptive learning rate
        early_exaggeration_iter: Number of iterations with early exaggeration
        early_exaggeration: Early exaggeration factor

    Returns:
        Optimized embedding (n_samples × n_components)
    """
    n_samples, n_components = Y.shape

    # Initialize variables
    Y_inplace = Y.copy()  # Copy to avoid modifying the input
    gains = np.ones_like(Y_inplace, dtype=np.float32)
    update = np.zeros_like(Y_inplace, dtype=np.float32)

    # Exaggerated P for early iterations
    P_early = P.copy() * early_exaggeration

    # Main optimization loop
    for iteration in range(n_iter):
        # Choose appropriate P matrix
        current_P = P_early if iteration < early_exaggeration_iter else P

        # Compute Q and distances
        Q, embedding_distances = _compute_q_joint(Y_inplace)

        # Compute gradient
        grad = _compute_gradient(current_P, Q, Y_inplace, embedding_distances)

        # Update gains with adaptive learning rate
        for i in range(n_samples):
            for j in range(n_components):
                if np.sign(grad[i, j]) != np.sign(update[i, j]):
                    gains[i, j] += 0.2
                else:
                    gains[i, j] *= 0.8

                if gains[i, j] < min_gain:
                    gains[i, j] = min_gain

        # Update with momentum and adaptive learning rate
        for i in range(n_samples):
            for j in range(n_components):
                update[i, j] = (
                    momentum * update[i, j] - learning_rate * gains[i, j] * grad[i, j]
                )
                Y_inplace[i, j] += update[i, j]

        # Center embedding to avoid drifting
        mean_Y = np.zeros(n_components, dtype=np.float32)
        for i in range(n_samples):
            for j in range(n_components):
                mean_Y[j] += Y_inplace[i, j]

        for j in range(n_components):
            mean_Y[j] /= n_samples

        for i in range(n_samples):
            for j in range(n_components):
                Y_inplace[i, j] -= mean_Y[j]

    return Y_inplace


@jit(nopython=True, cache=True)
def _initialize_embedding(
    n_samples: int, n_components: int, random_state: int
) -> np.ndarray:
    """
    Initialize the output embedding randomly.

    Args:
        n_samples: Number of samples
        n_components: Number of components in the embedding
        random_state: Random seed

    Returns:
        Initial embedding (n_samples × n_components)
    """
    # Set random seed
    np.random.seed(random_state)

    # Initialize with small random values from normal distribution
    return np.random.normal(0.0, 0.0001, (n_samples, n_components)).astype(np.float32)


@jit(nopython=True, parallel=True, cache=True)
def _flatten_3d_data(data: np.ndarray) -> np.ndarray:
    """
    Efficiently flatten 3D data to 2D for t-SNE with parallelization.

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
    Normalize data for t-SNE processing with parallel acceleration.

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


def fit_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int = 42,
    early_exaggeration: float = 4.0,
    early_exaggeration_iter: int = 250,
    verbose: bool = False,
) -> np.ndarray:
    """
    Fit t-SNE embedding to data using Numba-accelerated functions.

    Args:
        X: Input data array (n_samples × n_features)
        n_components: Number of dimensions in output
        perplexity: Perplexity parameter (related to nearest neighbors)
        learning_rate: Learning rate for gradient descent
        n_iter: Number of iterations
        random_state: Random seed
        early_exaggeration: Early exaggeration factor
        early_exaggeration_iter: Number of iterations with early exaggeration
        verbose: Whether to print progress

    Returns:
        Embedding in low-dimensional space
    """
    n_samples, n_features = X.shape

    if verbose:
        start_time = time.time()
        print(f"Computing {n_samples}x{n_samples} distance matrix...")

    # Compute pairwise distances
    distances = _compute_pairwise_distances(X)

    if verbose:
        print(f"Computing joint probabilities (perplexity={perplexity})...")

    # Compute joint probabilities
    P = _compute_joint_probabilities(distances, perplexity)

    if verbose:
        print(f"Initializing {n_samples}x{n_components} embedding...")

    # Initialize embedding
    Y = _initialize_embedding(n_samples, n_components, random_state)

    if verbose:
        print(f"Optimizing embedding for {n_iter} iterations...")

    # Perform gradient descent
    Y = _gradient_descent(
        P,
        Y,
        n_iter,
        learning_rate=learning_rate,
        early_exaggeration_iter=early_exaggeration_iter,
        early_exaggeration=early_exaggeration,
    )

    if verbose:
        elapsed = time.time() - start_time
        print(f"t-SNE embedding complete in {elapsed:.2f} seconds")

    return Y


class NumbaRealTSNEProcessor(BaseDimensionalityReductionProcessor):
    """
    Pure Numba-accelerated t-SNE processor implementation.

    This processor provides high-performance dimensionality reduction through t-SNE
    using pure Numba with no scikit-learn dependency, ensuring consistent types
    when used in processing pipelines.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
        normalize: bool = True,
        early_exaggeration: float = 4.0,
        early_exaggeration_iter: int = 250,
        verbose: bool = False,
        max_samples: int = 10000,
    ):
        """
        Initialize the Numba t-SNE processor.

        Args:
            n_components: Number of dimensions in output (typically 2 or 3)
            perplexity: Perplexity parameter (related to nearest neighbors)
            learning_rate: Learning rate for gradient descent
            n_iter: Number of iterations
            random_state: Random seed for reproducibility
            normalize: Whether to normalize data before processing
            early_exaggeration: Early exaggeration factor
            early_exaggeration_iter: Number of iterations with early exaggeration
            verbose: Whether to print progress
            max_samples: Maximum number of samples to use (t-SNE doesn't scale well)
        """
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.normalize = normalize
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.verbose = verbose
        self.max_samples = max_samples

        # Model state
        self._embedding = None

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess data before t-SNE.

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
            data_flat = _normalize_data(data_flat)

        return data_flat

    def _subsample_data(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample data if there are too many points for t-SNE.

        Args:
            data: Input data array
            labels: Optional cluster labels for stratified sampling

        Returns:
            Tuple of (subsampled_data, indices)
        """
        n_samples = data.shape[0]

        if n_samples <= self.max_samples:
            return data, np.arange(n_samples)

        if self.verbose:
            print(f"Subsampling {n_samples} points to {self.max_samples} for t-SNE...")

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        if labels is not None:
            # Stratified sampling
            unique_labels = np.unique(labels)
            n_labels = len(unique_labels)

            # Calculate samples per label
            samples_per_label = max(1, self.max_samples // n_labels)

            indices = []
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]

                if len(label_indices) > samples_per_label:
                    # Random sampling for this label
                    sampled_indices = np.random.choice(
                        label_indices, samples_per_label, replace=False
                    )
                    indices.extend(sampled_indices)
                else:
                    # Take all samples for this label
                    indices.extend(label_indices)

            # Ensure we don't exceed max_samples
            indices = np.array(indices)[: self.max_samples]
        else:
            # Random sampling
            indices = np.random.choice(n_samples, self.max_samples, replace=False)

        return data[indices], indices

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data using t-SNE.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, but required by BaseProcessor interface)
            **kwargs: Additional keyword arguments
                labels: Optional cluster labels for stratified sampling
                max_samples: Override for maximum number of samples
                perplexity: Override for perplexity parameter
                n_iter: Override for number of iterations

        Returns:
            Dask array with reduced dimensions
        """
        # Extract parameters from kwargs
        labels = kwargs.get("labels", None)
        max_samples = kwargs.get("max_samples", self.max_samples)
        perplexity = kwargs.get("perplexity", self.perplexity)
        n_iter = kwargs.get("n_iter", self.n_iter)

        # Update instance parameters
        self.max_samples = max_samples
        self.perplexity = perplexity
        self.n_iter = n_iter

        # Convert to numpy (t-SNE doesn't work well with dask)
        data_np = data.compute()

        # Preprocess data
        data_flat = self._preprocess_data(data_np)

        # Subsample if needed
        if labels is not None:
            # Ensure labels are numpy array
            if isinstance(labels, da.Array):
                labels = labels.compute()

        data_subset, indices = self._subsample_data(data_flat, labels)

        # Check perplexity against dataset size
        effective_perplexity = min(perplexity, data_subset.shape[0] - 1)
        if effective_perplexity < perplexity:
            warnings.warn(
                f"Perplexity ({perplexity}) is too large for the number of samples "
                f"({data_subset.shape[0]}). Using {effective_perplexity} instead."
            )
            self.perplexity = effective_perplexity

        # Fit t-SNE using Numba function
        embedding = fit_tsne(
            data_subset,
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state,
            early_exaggeration=self.early_exaggeration,
            early_exaggeration_iter=self.early_exaggeration_iter,
            verbose=self.verbose,
        )

        # Store embedding and mark as fitted
        self._embedding = embedding
        self._subset_indices = indices
        self._is_fitted = True

        # Return as dask array
        return da.from_array(embedding)

    def fit(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> "NumbaRealTSNEProcessor":
        """
        Fit the t-SNE model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Extract parameters from kwargs
        labels = kwargs.get("labels", None)

        # Preprocess data
        data_flat = self._preprocess_data(data)

        # Subsample if needed
        data_subset, indices = self._subsample_data(data_flat, labels)

        # Check perplexity against dataset size
        effective_perplexity = min(self.perplexity, data_subset.shape[0] - 1)
        if effective_perplexity < self.perplexity:
            warnings.warn(
                f"Perplexity ({self.perplexity}) is too large for the number of samples "
                f"({data_subset.shape[0]}). Using {effective_perplexity} instead."
            )
            self.perplexity = effective_perplexity

        # Fit t-SNE
        embedding = fit_tsne(
            data_subset,
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state,
            early_exaggeration=self.early_exaggeration,
            early_exaggeration_iter=self.early_exaggeration_iter,
            verbose=self.verbose,
        )

        # Store embedding and mark as fitted
        self._embedding = embedding
        self._subset_indices = indices
        self._is_fitted = True

        return self

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data using the fitted t-SNE model.

        Note: t-SNE doesn't support transform for out-of-sample data.
        This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "t-SNE does not support transforming new data. "
            "Use fit_transform() on the entire dataset instead."
        )

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
        return self._embedding

    def inverse_transform(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> np.ndarray:
        """
        t-SNE does not support inverse transform.

        This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "t-SNE does not support inverse transformation as it is a non-linear method."
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
                "perplexity": self.perplexity,
                "learning_rate": self.learning_rate,
                "n_iter": self.n_iter,
                "normalize": self.normalize,
                "is_fitted": self._is_fitted,
                "max_samples": self.max_samples,
                "implementation": "pure_numba",
            }
        )
        return base_summary


# Factory functions for easy creation


def create_numba_tsne(
    n_components: int = 2,
    perplexity: float = 30.0,
    normalize: bool = True,
    random_state: int = 42,
) -> NumbaRealTSNEProcessor:
    """
    Create a pure Numba t-SNE processor with standard parameters.

    Args:
        n_components: Number of dimensions in output (typically 2 or 3)
        perplexity: Perplexity parameter (related to nearest neighbors)
        normalize: Whether to normalize data before processing
        random_state: Random seed for reproducibility

    Returns:
        Configured NumbaRealTSNEProcessor
    """
    return NumbaRealTSNEProcessor(
        n_components=n_components,
        perplexity=perplexity,
        normalize=normalize,
        random_state=random_state,
    )


def create_numba_visualization_tsne(
    perplexity: float = 30.0,
    random_state: int = 42,
    verbose: bool = True,
) -> NumbaRealTSNEProcessor:
    """
    Create a pure Numba t-SNE processor optimized for 2D visualization.

    Args:
        perplexity: Perplexity parameter (related to nearest neighbors)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Configured NumbaRealTSNEProcessor for visualization
    """
    return NumbaRealTSNEProcessor(
        n_components=2,  # Fixed at 2D for visualization
        perplexity=perplexity,
        learning_rate=200.0,
        n_iter=1000,
        random_state=random_state,
        normalize=True,
        verbose=verbose,
    )


def create_fast_numba_tsne(
    n_components: int = 2,
    random_state: int = 42,
) -> NumbaRealTSNEProcessor:
    """
    Create a pure Numba t-SNE processor optimized for speed.

    This configuration uses fewer iterations and a smaller perplexity
    value for faster execution at some cost to quality.

    Args:
        n_components: Number of dimensions in output
        random_state: Random seed for reproducibility

    Returns:
        NumbaRealTSNEProcessor optimized for speed
    """
    return NumbaRealTSNEProcessor(
        n_components=n_components,
        perplexity=15.0,  # Lower perplexity for faster computation
        learning_rate=300.0,  # Higher learning rate for faster convergence
        n_iter=500,  # Fewer iterations
        random_state=random_state,
        normalize=True,
        early_exaggeration=4.0,
        early_exaggeration_iter=100,  # Fewer early exaggeration iterations
        verbose=False,
        max_samples=5000,  # Limit to fewer samples for speed
    )


def create_high_quality_numba_tsne(
    n_components: int = 2,
    random_state: int = 42,
    max_samples: int = 10000,
) -> NumbaRealTSNEProcessor:
    """
    Create a pure Numba t-SNE processor optimized for quality.

    This configuration uses more iterations and a carefully tuned
    perplexity for better visualization quality.

    Args:
        n_components: Number of dimensions in output
        random_state: Random seed for reproducibility
        max_samples: Maximum number of samples to use

    Returns:
        NumbaRealTSNEProcessor optimized for quality
    """
    return NumbaRealTSNEProcessor(
        n_components=n_components,
        perplexity=50.0,  # Higher perplexity for better global structure
        learning_rate=150.0,  # Lower learning rate for more stable convergence
        n_iter=2000,  # More iterations for better convergence
        random_state=random_state,
        normalize=True,
        early_exaggeration=12.0,  # Stronger early exaggeration
        early_exaggeration_iter=500,  # More early exaggeration iterations
        verbose=True,
        max_samples=max_samples,
    )


def create_cluster_preservation_numba_tsne(
    perplexity: float = 30.0,
    random_state: int = 42,
) -> NumbaRealTSNEProcessor:
    """
    Create a pure Numba t-SNE processor that preserves cluster structure.

    This configuration is tuned to better preserve cluster separation
    in the low-dimensional space, which can be useful for visualization
    of clustered data.

    Args:
        perplexity: Perplexity parameter (related to nearest neighbors)
        random_state: Random seed for reproducibility

    Returns:
        NumbaRealTSNEProcessor optimized for cluster preservation
    """
    return NumbaRealTSNEProcessor(
        n_components=2,  # 2D for visualization
        perplexity=perplexity,
        learning_rate=200.0,
        n_iter=1500,  # More iterations for better separation
        random_state=random_state,
        normalize=True,
        early_exaggeration=16.0,  # Higher exaggeration to separate clusters
        early_exaggeration_iter=400,  # Longer early phase
        verbose=True,
    )


def create_incremental_numba_tsne(
    n_components: int = 2,
    chunk_size: int = 1000,
    random_state: int = 42,
) -> NumbaRealTSNEProcessor:
    """
    Create a pure Numba t-SNE processor for incremental processing.

    This configuration is designed to work well with chunked data processing,
    though t-SNE is inherently a batch algorithm.

    Args:
        n_components: Number of dimensions in output
        chunk_size: Size of data chunks to process at once
        random_state: Random seed for reproducibility

    Returns:
        NumbaRealTSNEProcessor for incremental processing
    """
    return NumbaRealTSNEProcessor(
        n_components=n_components,
        perplexity=min(30.0, chunk_size // 3),  # Scale perplexity to chunk size
        learning_rate=200.0,
        n_iter=750,  # Moderate number of iterations
        random_state=random_state,
        normalize=True,
        max_samples=chunk_size,  # Limit to chunk size
        verbose=False,
    )
