"""
Gaussian Mixture Model implementation with Numba acceleration and scikit-learn fallback.

This module provides an efficient GMM implementation for signal processing
applications, optimized with Numba for high performance, with an option
to fall back to scikit-learn's implementation.
"""

from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from sklearn.mixture import GaussianMixture as SKLearnGMM

from .base import BaseClusteringProcessor


@jit(nopython=True, cache=True)
def _log_multivariate_normal_density_diag(
    X: np.ndarray, means: np.ndarray, precisions: np.ndarray
) -> np.ndarray:
    """
    Compute Gaussian log-density for diagonal covariance matrices.

    Args:
        X: Data array (n_samples, n_features)
        means: Mean vectors (n_components, n_features)
        precisions: Diagonal of precision matrices (n_components, n_features)

    Returns:
        Log-density array (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_det = np.zeros(n_components, dtype=np.float64)

    # Compute log determinant of precision matrices
    for k in range(n_components):
        log_det[k] = 0.0
        for j in range(n_features):
            log_det[k] += np.log(precisions[k, j])

    # Allocate result array
    log_prob = np.zeros((n_samples, n_components), dtype=np.float64)

    # Constant term in log-density
    const = -0.5 * n_features * np.log(2.0 * np.pi)

    # Compute log-density for each sample and component
    for i in range(n_samples):
        for k in range(n_components):
            # Mahalanobis distance (diagonal case)
            mahalanobis = 0.0
            for j in range(n_features):
                diff = X[i, j] - means[k, j]
                mahalanobis += diff * diff * precisions[k, j]

            # Log-density
            log_prob[i, k] = const + 0.5 * log_det[k] - 0.5 * mahalanobis

    return log_prob


@jit(nopython=True, cache=True)
def _estimate_responsibilities(
    X: np.ndarray, means: np.ndarray, precisions: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Estimate responsibilities and log-likelihood.

    Args:
        X: Data array (n_samples, n_features)
        means: Mean vectors (n_components, n_features)
        precisions: Diagonal of precision matrices (n_components, n_features)
        weights: Component weights (n_components,)

    Returns:
        Tuple of (responsibilities, log_likelihood)
    """
    n_samples = X.shape[0]
    n_components = means.shape[0]

    # Compute weighted log probabilities
    weighted_log_prob = _log_multivariate_normal_density_diag(X, means, precisions)
    for k in range(n_components):
        weighted_log_prob[:, k] += np.log(weights[k])

    # Compute log likelihood and responsibilities
    log_prob_norm = np.zeros(n_samples, dtype=np.float64)
    resp = np.zeros((n_samples, n_components), dtype=np.float64)

    for i in range(n_samples):
        # Find max log probability for numerical stability
        max_log_prob = weighted_log_prob[i, 0]
        for k in range(1, n_components):
            if weighted_log_prob[i, k] > max_log_prob:
                max_log_prob = weighted_log_prob[i, k]

        # Compute normalized probabilities
        sum_exp = 0.0
        for k in range(n_components):
            resp[i, k] = np.exp(weighted_log_prob[i, k] - max_log_prob)
            sum_exp += resp[i, k]

        # Normalize responsibilities
        for k in range(n_components):
            resp[i, k] /= sum_exp

        # Compute log probability
        log_prob_norm[i] = max_log_prob + np.log(sum_exp)

    # Compute total log likelihood
    log_likelihood = np.sum(log_prob_norm)

    return resp, log_likelihood


@jit(nopython=True, parallel=True, cache=True)
def _estimate_gaussian_parameters(
    X: np.ndarray, resp: np.ndarray, reg_covar: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate Gaussian parameters from data and responsibilities.

    Args:
        X: Data array (n_samples, n_features)
        resp: Responsibility matrix (n_samples, n_components)
        reg_covar: Regularization added to covariance

    Returns:
        Tuple of (weights, means, covariances)
    """
    n_samples, n_features = X.shape
    n_components = resp.shape[1]

    # Initialize parameters
    weights = np.zeros(n_components, dtype=np.float64)
    means = np.zeros((n_components, n_features), dtype=np.float64)
    covariances = np.zeros((n_components, n_features), dtype=np.float64)

    # Estimate parameters for each component in parallel
    for k in prange(n_components):
        # Compute weight (sum of responsibilities)
        weights[k] = 0.0
        for i in range(n_samples):
            weights[k] += resp[i, k]

        # Normalize weight
        weights[k] /= n_samples

        # Compute mean
        if weights[k] > 0:
            for j in range(n_features):
                means[k, j] = 0.0
                for i in range(n_samples):
                    means[k, j] += resp[i, k] * X[i, j]
                means[k, j] /= weights[k] * n_samples

        # Compute covariance (diagonal only)
        if weights[k] > 0:
            for j in range(n_features):
                covariances[k, j] = 0.0
                for i in range(n_samples):
                    diff = X[i, j] - means[k, j]
                    covariances[k, j] += resp[i, k] * diff * diff
                covariances[k, j] /= weights[k] * n_samples

                # Add regularization
                covariances[k, j] += reg_covar

    return weights, means, covariances


@jit(nopython=True, cache=True)
def _compute_precision_cholesky(covariances: np.ndarray) -> np.ndarray:
    """
    Compute precision matrices from covariance matrices.

    Args:
        covariances: Diagonal covariance matrices (n_components, n_features)

    Returns:
        Diagonal of precision matrices (n_components, n_features)
    """
    n_components, n_features = covariances.shape
    precisions = np.zeros_like(covariances)

    # For diagonal matrices, precision is 1/variance
    for k in range(n_components):
        for j in range(n_features):
            precisions[k, j] = 1.0 / np.sqrt(covariances[k, j])

    return precisions


@jit(nopython=True, cache=True)
def _kmeans_init(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Initialize cluster centers using K-means++ algorithm.

    Args:
        X: Data array (n_samples, n_features)
        n_components: Number of components

    Returns:
        Initial centers (n_components, n_features)
    """
    n_samples, n_features = X.shape
    centers = np.zeros((n_components, n_features), dtype=X.dtype)

    # Choose first center randomly
    first_idx = np.random.randint(0, n_samples)
    centers[0] = X[first_idx].copy()

    # Choose remaining centers
    for k in range(1, n_components):
        # Compute squared distances to nearest center
        min_dists = np.ones(n_samples) * np.inf

        for i in range(n_samples):
            for j in range(k):
                # Squared distance to center j
                dist = 0.0
                for l in range(n_features):
                    diff = X[i, l] - centers[j, l]
                    dist += diff * diff

                # Update minimum distance
                if dist < min_dists[i]:
                    min_dists[i] = dist

        # Choose next center with probability proportional to squared distance
        sum_dists = np.sum(min_dists)
        if sum_dists > 0:
            # Convert to probabilities
            probs = min_dists / sum_dists

            # Compute cumulative probabilities
            cumprobs = np.zeros(n_samples)
            cumsum = 0.0
            for i in range(n_samples):
                cumsum += probs[i]
                cumprobs[i] = cumsum

            # Sample according to probabilities
            rand_val = np.random.random()
            for i in range(n_samples):
                if rand_val <= cumprobs[i]:
                    centers[k] = X[i].copy()
                    break
        else:
            # If all points are at zero distance, choose randomly
            idx = np.random.randint(0, n_samples)
            centers[k] = X[idx].copy()

    return centers


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


class GMMProcessor(BaseClusteringProcessor):
    """
    Gaussian Mixture Model processor with Numba acceleration and scikit-learn fallback.

    This processor provides GMM clustering for signal processing,
    optimized with Numba for high performance or using scikit-learn's
    implementation depending on the configuration.
    """

    def __init__(
        self,
        n_components: int = 8,
        max_iter: int = 100,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = None,
        init_params: str = "kmeans",
        warm_start: bool = False,
        use_sklearn: bool = False,
        covariance_type: str = "diag",
    ):
        """
        Initialize the GMM processor.

        Args:
            n_components: Number of Gaussian components
            max_iter: Maximum number of EM iterations
            tol: Convergence tolerance (relative change in log-likelihood)
            reg_covar: Regularization added to covariance
            random_state: Random seed for reproducibility
            init_params: Initialization method ('kmeans', 'random')
            warm_start: Whether to reuse previous results to initialize
            use_sklearn: Whether to use scikit-learn's GMM implementation
            covariance_type: Type of covariance (only used with scikit-learn,
                             options: 'full', 'tied', 'diag', 'spherical')
        """
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_params = init_params
        self.warm_start = warm_start
        self.use_sklearn = use_sklearn
        self.covariance_type = covariance_type

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize model parameters
        self._weights = None
        self._means = None
        self._covariances = None
        self._precisions = None
        self._converged = False
        self._log_likelihood = None
        self._sklearn_model = None

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "GMMProcessor":
        """
        Fit the GMM model to the data.

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

        # Use scikit-learn if specified
        if self.use_sklearn:
            # Initialize scikit-learn model
            self._sklearn_model = SKLearnGMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                tol=self.tol,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
                n_init=1,  # sklearn uses multiple initializations by default
                init_params=self.init_params,
                weights_init=self._weights
                if self.warm_start and self._weights is not None
                else None,
                means_init=self._means
                if self.warm_start and self._means is not None
                else None,
                precisions_init=None,  # Not supporting custom precisions for now
                random_state=self.random_state,
                warm_start=self.warm_start,
            )

            # Fit model
            self._sklearn_model.fit(data_flat)

            # Extract parameters
            self._weights = self._sklearn_model.weights_
            self._means = self._sklearn_model.means_
            self._covariances = self._sklearn_model.covariances_
            if self.covariance_type == "diag":
                self._precisions = 1.0 / np.sqrt(self._covariances)
            self._converged = self._sklearn_model.converged_
            self._log_likelihood = (
                self._sklearn_model.score(data_flat) * data_flat.shape[0]
            )
            self._cluster_centers = self._means
            self._cluster_labels = self._sklearn_model.predict(data_flat)

        # Use Numba-accelerated implementation
        else:
            n_samples, n_features = data_flat.shape

            # Initialize parameters
            if not self.warm_start or self._means is None:
                # Initialize weights uniformly
                self._weights = (
                    np.ones(self.n_components, dtype=np.float64) / self.n_components
                )

                # Initialize means with k-means++ or random
                if self.init_params == "kmeans":
                    self._means = _kmeans_init(data_flat, self.n_components)
                else:
                    # Random initialization
                    indices = np.random.choice(
                        n_samples, self.n_components, replace=False
                    )
                    self._means = data_flat[indices].copy()

                # Initialize diagonal covariances as data variance
                data_var = np.var(data_flat, axis=0)
                self._covariances = np.zeros(
                    (self.n_components, n_features), dtype=np.float64
                )
                for k in range(self.n_components):
                    self._covariances[k] = data_var + self.reg_covar

            # Compute initial precision matrix
            self._precisions = _compute_precision_cholesky(self._covariances)

            # Run EM algorithm
            log_likelihood_previous = -np.inf
            self._converged = False

            for n_iter in range(self.max_iter):
                # E-step: compute responsibilities
                resp, log_likelihood = _estimate_responsibilities(
                    data_flat, self._means, self._precisions, self._weights
                )

                # Check convergence
                if abs(log_likelihood - log_likelihood_previous) < self.tol * abs(
                    log_likelihood
                ):
                    self._converged = True
                    break

                log_likelihood_previous = log_likelihood

                # M-step: update parameters
                self._weights, self._means, self._covariances = (
                    _estimate_gaussian_parameters(data_flat, resp, self.reg_covar)
                )

                # Update precision matrices
                self._precisions = _compute_precision_cholesky(self._covariances)

            # Store results
            self._log_likelihood = log_likelihood
            self._cluster_centers = self._means

            # Get cluster assignments
            resp, _ = _estimate_responsibilities(
                data_flat, self._means, self._precisions, self._weights
            )
            self._cluster_labels = np.argmax(resp, axis=1)

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
            raise ValueError("GMM model not fitted. Call fit() first.")

        # Get probabilities and return most likely component
        proba = self.predict_proba(data, **kwargs)
        return np.argmax(proba, axis=1)

    def predict_proba(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict posterior probabilities for each component.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data before prediction

        Returns:
            Array of posterior probabilities (n_samples, n_components)
        """
        if not self._is_fitted:
            raise ValueError("GMM model not fitted. Call fit() first.")

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

        # Use scikit-learn if model exists
        if self.use_sklearn and self._sklearn_model is not None:
            return self._sklearn_model.predict_proba(data_flat)

        # Use Numba implementation
        resp, _ = _estimate_responsibilities(
            data_flat, self._means, self._precisions, self._weights
        )
        return resp

    def score_samples(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Compute the log likelihood of each sample under the model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Log likelihood array (n_samples,)
        """
        if not self._is_fitted:
            raise ValueError("GMM model not fitted. Call fit() first.")

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

        # Use scikit-learn if model exists
        if self.use_sklearn and self._sklearn_model is not None:
            return self._sklearn_model.score_samples(data_flat)

        # Compute log probabilities using Numba implementation
        weighted_log_prob = _log_multivariate_normal_density_diag(
            data_flat, self._means, self._precisions
        )
        for k in range(self.n_components):
            weighted_log_prob[:, k] += np.log(self._weights[k])

        # Compute log likelihood for each sample
        log_prob_norm = np.zeros(data_flat.shape[0], dtype=np.float64)
        for i in range(data_flat.shape[0]):
            max_log_prob = np.max(weighted_log_prob[i])
            log_prob_norm[i] = max_log_prob + np.log(
                np.sum(np.exp(weighted_log_prob[i] - max_log_prob))
            )

        return log_prob_norm

    def score(self, data: Union[np.ndarray, da.Array], **kwargs) -> float:
        """
        Compute the average log likelihood of data under the model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Average log likelihood per sample
        """
        return np.mean(self.score_samples(data, **kwargs))

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform GMM clustering.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used in clustering, but required by BaseProcessor interface)
            **kwargs: Additional algorithm-specific parameters
                return_proba: Whether to return probabilities instead of assignments

        Returns:
            Dask array with cluster assignments or probabilities
        """
        # GMM works better with global processing due to statistical nature
        # Convert to numpy, process, and convert back to dask
        data_np = data.compute()

        # Get processing option
        return_proba = kwargs.pop("return_proba", False)

        # Process data
        if return_proba:
            self.fit(data_np, **kwargs)
            result = self.predict_proba(data_np, **kwargs)
        else:
            self.fit(data_np, **kwargs)
            result = self.predict(data_np, **kwargs)

        # Return as dask array
        return da.from_array(result)

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Get the component weights."""
        return self._weights

    @property
    def means(self) -> Optional[np.ndarray]:
        """Get the component means."""
        return self._means

    @property
    def covariances(self) -> Optional[np.ndarray]:
        """Get the component covariances."""
        return self._covariances

    @property
    def converged(self) -> bool:
        """Check if the EM algorithm converged."""
        return self._converged

    @property
    def log_likelihood(self) -> Optional[float]:
        """Get the log likelihood from the last fitting."""
        return self._log_likelihood

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "n_components": self.n_components,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "reg_covar": self.reg_covar,
                "init_params": self.init_params,
                "converged": self._converged,
                "log_likelihood": self._log_likelihood,
                "use_sklearn": self.use_sklearn,
                "covariance_type": self.covariance_type if self.use_sklearn else "diag",
            }
        )
        return base_summary


# Factory functions for easy creation


def create_gmm(
    n_components: int = 8,
    random_state: int = 42,
    use_sklearn: bool = False,
) -> GMMProcessor:
    """
    Create a standard GMM processor.

    Args:
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility
        use_sklearn: Whether to use scikit-learn's implementation

    Returns:
        Configured GMMProcessor
    """
    return GMMProcessor(
        n_components=n_components,
        random_state=random_state,
        init_params="kmeans",
        use_sklearn=use_sklearn,
    )


def create_fast_gmm(
    n_components: int = 8,
    random_state: int = 42,
    use_sklearn: bool = False,
) -> GMMProcessor:
    """
    Create a GMM processor optimized for speed.

    This configuration uses fewer iterations and relaxed convergence criteria
    for faster clustering at some cost to quality.

    Args:
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility
        use_sklearn: Whether to use scikit-learn's implementation

    Returns:
        Configured GMMProcessor for speed
    """
    return GMMProcessor(
        n_components=n_components,
        max_iter=50,  # Fewer iterations
        tol=1e-2,  # Less strict convergence
        random_state=random_state,
        init_params="kmeans",  # K-means initialization usually converges faster
        use_sklearn=use_sklearn,
    )


def create_robust_gmm(
    n_components: int = 8,
    random_state: int = 42,
    use_sklearn: bool = False,
) -> GMMProcessor:
    """
    Create a robust GMM processor.

    This configuration is more resistant to initialization issues and
    numerical instabilities in the EM algorithm.

    Args:
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility
        use_sklearn: Whether to use scikit-learn's implementation

    Returns:
        Configured GMMProcessor for robust performance
    """
    return GMMProcessor(
        n_components=n_components,
        max_iter=200,  # More iterations for better convergence
        tol=1e-4,  # Stricter convergence
        reg_covar=1e-4,  # Slightly higher regularization
        random_state=random_state,
        init_params="kmeans",
        use_sklearn=use_sklearn,
    )


def create_sklearn_gmm(
    n_components: int = 8,
    random_state: int = 42,
    covariance_type: str = "full",
) -> GMMProcessor:
    """
    Create a GMM processor using scikit-learn's implementation.

    This can be useful for comparison or when advanced covariance types
    such as full, tied, or spherical are needed.

    Args:
        n_components: Number of Gaussian components
        random_state: Random seed for reproducibility
        covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')

    Returns:
        Configured GMMProcessor using scikit-learn
    """
    return GMMProcessor(
        n_components=n_components,
        random_state=random_state,
        use_sklearn=True,
        covariance_type=covariance_type,
    )
