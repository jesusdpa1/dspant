"""
Numba-accelerated Gaussian Mixture Model implementation.

This module provides a high-performance GMM implementation using
Numba for acceleration, with no dependency on scikit-learn. This allows for better
performance and type consistency when used in processing pipelines.

"""

from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from sklearn.metrics import silhouette_score

from ...processors.clustering.base import BaseClusteringProcessor

EPSILON = 1e-10  # Small value to avoid numerical issues


@jit(nopython=True, cache=True)
def _log_multivariate_normal_pdf_diag(
    X: np.ndarray, means: np.ndarray, precisions: np.ndarray
) -> np.ndarray:
    """
    Compute log PDF of multivariate normal with diagonal covariance.

    Args:
        X: Data array (n_samples, n_features)
        means: Mean vectors (n_components, n_features)
        precisions: Diagonal of precision matrices (n_components, n_features)

    Returns:
        Log-density array (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_det = np.zeros(n_components, dtype=np.float32)

    # Compute log determinant of precision matrices
    for k in range(n_components):
        for j in range(n_features):
            log_det[k] += np.log(max(precisions[k, j], EPSILON))

    # Initialize result array
    log_prob = np.zeros((n_samples, n_components), dtype=np.float32)

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
    weighted_log_prob = _log_multivariate_normal_pdf_diag(X, means, precisions)
    for k in range(n_components):
        weighted_log_prob[:, k] += np.log(max(weights[k], EPSILON))

    # Compute log likelihood and responsibilities
    log_prob_norm = np.zeros(n_samples, dtype=np.float32)
    resp = np.zeros((n_samples, n_components), dtype=np.float32)

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
            if sum_exp > EPSILON:
                resp[i, k] /= sum_exp
            else:
                # Avoid division by zero
                resp[i, k] = 1.0 / n_components

        # Compute log probability
        log_prob_norm[i] = max_log_prob + np.log(max(sum_exp, EPSILON))

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
    weights = np.zeros(n_components, dtype=np.float32)
    means = np.zeros((n_components, n_features), dtype=np.float32)
    covariances = np.zeros((n_components, n_features), dtype=np.float32)

    # Estimate parameters for each component in parallel
    for k in prange(n_components):
        # Compute weight (sum of responsibilities)
        weights[k] = 0.0
        for i in range(n_samples):
            weights[k] += resp[i, k]

        # Normalize weight
        weights[k] /= max(n_samples, 1)

        # Compute mean
        if weights[k] > EPSILON:
            for j in range(n_features):
                means[k, j] = 0.0
                for i in range(n_samples):
                    means[k, j] += resp[i, k] * X[i, j]
                means[k, j] /= max(weights[k] * n_samples, EPSILON)

        # Compute covariance (diagonal only)
        if weights[k] > EPSILON:
            for j in range(n_features):
                covariances[k, j] = 0.0
                for i in range(n_samples):
                    diff = X[i, j] - means[k, j]
                    covariances[k, j] += resp[i, k] * diff * diff
                covariances[k, j] /= max(weights[k] * n_samples, EPSILON)

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
    precisions = np.zeros_like(covariances, dtype=np.float32)

    # For diagonal matrices, precision is 1/variance
    for k in range(n_components):
        for j in range(n_features):
            precisions[k, j] = 1.0 / np.sqrt(max(covariances[k, j], EPSILON))

    return precisions


@jit(nopython=True, cache=True)
def _kmeans_init(X: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    """
    Initialize cluster centers using K-means++ algorithm.

    Args:
        X: Data array (n_samples, n_features)
        n_components: Number of components
        random_state: Random seed

    Returns:
        Initial centers (n_components, n_features)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    centers = np.zeros((n_components, n_features), dtype=np.float32)

    # Choose first center randomly
    first_idx = np.random.randint(0, n_samples)
    centers[0] = X[first_idx].copy()

    # Choose remaining centers
    for k in range(1, n_components):
        # Compute squared distances to nearest center
        min_dists = np.ones(n_samples, dtype=np.float32) * np.inf

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
        if sum_dists > EPSILON:
            # Convert to probabilities
            probs = min_dists / sum_dists

            # Compute cumulative probabilities
            cumprobs = np.zeros(n_samples, dtype=np.float32)
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


@jit(nopython=True, cache=True)
def _random_init(
    X: np.ndarray, n_components: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize GMM parameters randomly.

    Args:
        X: Data array (n_samples, n_features)
        n_components: Number of components
        random_state: Random seed

    Returns:
        Tuple of (weights, means, covariances)
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    # Initialize weights uniformly
    weights = np.ones(n_components, dtype=np.float32) / n_components

    # Initialize means randomly from data points
    indices = np.random.choice(
        n_samples, n_components, replace=n_samples < n_components
    )
    means = X[indices].copy().astype(np.float32)

    # Estimate data variance for covariance initialization
    variances = np.zeros(n_features, dtype=np.float32)
    for j in range(n_features):
        mean_val = 0.0
        for i in range(n_samples):
            mean_val += X[i, j]
        mean_val /= n_samples

        var_val = 0.0
        for i in range(n_samples):
            diff = X[i, j] - mean_val
            var_val += diff * diff
        variances[j] = var_val / n_samples

    # Initialize each component's covariance
    covariances = np.zeros((n_components, n_features), dtype=np.float32)
    for k in range(n_components):
        for j in range(n_features):
            # Randomize the variance within a reasonable range
            covariances[k, j] = variances[j] * (0.5 + np.random.random())

    return weights, means, covariances


@jit(nopython=True, cache=True)
def _check_convergence(
    log_likelihood: float,
    old_log_likelihood: float,
    tol: float,
    n_iter: int,
    max_iter: int,
) -> bool:
    """
    Check if the EM algorithm has converged.

    Args:
        log_likelihood: Current log likelihood
        old_log_likelihood: Previous log likelihood
        tol: Tolerance for convergence
        n_iter: Current iteration number
        max_iter: Maximum number of iterations

    Returns:
        True if converged, False otherwise
    """
    # Check if reached maximum iterations
    if n_iter >= max_iter:
        return True

    # Check if log likelihood is not finite
    if not np.isfinite(log_likelihood):
        return True

    # Calculate relative change in log likelihood
    if old_log_likelihood == 0:
        # Avoid division by zero
        return abs(log_likelihood) < tol
    else:
        change = abs(log_likelihood - old_log_likelihood) / abs(old_log_likelihood)
        return change < tol


@jit(nopython=True, cache=True)
def _bic_score(X: np.ndarray, log_likelihood: float, n_components: int) -> float:
    """
    Calculate the Bayesian Information Criterion (BIC) score.

    Args:
        X: Data array (n_samples, n_features)
        log_likelihood: Log-likelihood of the model
        n_components: Number of components

    Returns:
        BIC score (lower is better)
    """
    n_samples, n_features = X.shape

    # Number of free parameters
    # For each component: mean (n_features) + diagonal covariance (n_features) + weight (1)
    n_params = n_components * (2 * n_features + 1) - 1  # -1 because weights sum to 1

    # Calculate BIC
    return -2 * log_likelihood + n_params * np.log(n_samples)


@jit(nopython=True, cache=True)
def _aic_score(X: np.ndarray, log_likelihood: float, n_components: int) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) score.

    Args:
        X: Data array (n_samples, n_features)
        log_likelihood: Log-likelihood of the model
        n_components: Number of components

    Returns:
        AIC score (lower is better)
    """
    n_samples, n_features = X.shape

    # Number of free parameters
    # For each component: mean (n_features) + diagonal covariance (n_features) + weight (1)
    n_params = n_components * (2 * n_features + 1) - 1  # -1 because weights sum to 1

    # Calculate AIC
    return -2 * log_likelihood + 2 * n_params


@jit(nopython=True, cache=True)
def _fit_gmm_single(
    X: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    max_iter: int,
    tol: float,
    reg_covar: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool]:
    """
    Fit a single GMM with given initialization.

    Args:
        X: Data array (n_samples, n_features)
        weights: Initial component weights
        means: Initial component means
        covariances: Initial component covariances
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        reg_covar: Regularization added to covariance

    Returns:
        Tuple of (weights, means, covariances, responsibilities, log_likelihood, n_iter, converged)
    """
    n_samples, n_features = X.shape
    n_components = weights.shape[0]

    # Make copies to avoid modifying input arrays
    weights = weights.copy()
    means = means.copy()
    covariances = covariances.copy()

    # Initialize
    log_likelihood = -np.inf
    old_log_likelihood = -np.inf
    n_iter = 0
    converged = False
    resp = np.zeros((n_samples, n_components), dtype=np.float32)

    # EM algorithm
    for n_iter in range(max_iter):
        old_log_likelihood = log_likelihood

        # E-step: compute responsibilities and log-likelihood
        precisions = _compute_precision_cholesky(covariances)
        resp, log_likelihood = _estimate_responsibilities(X, means, precisions, weights)

        # Check for convergence
        if n_iter > 0:
            if _check_convergence(
                log_likelihood, old_log_likelihood, tol, n_iter, max_iter
            ):
                converged = True
                break

        # M-step: update parameters
        weights, means, covariances = _estimate_gaussian_parameters(X, resp, reg_covar)

        # Handle degenerate cases
        for k in range(n_components):
            if weights[k] < EPSILON:
                # Reset component parameters if weight becomes too small
                weights[k] = EPSILON
                # Select a random data point for the mean
                idx = np.random.randint(0, n_samples)
                means[k] = X[idx].copy()
                # Use global variance for covariance
                for j in range(n_features):
                    covariances[k, j] = np.var(X[:, j]) + reg_covar

    return weights, means, covariances, resp, log_likelihood, n_iter + 1, converged


@jit(nopython=True, cache=True)
def fit_gmm(
    X: np.ndarray,
    n_components: int,
    max_iter: int = 100,
    tol: float = 1e-3,
    reg_covar: float = 1e-6,
    n_init: int = 10,
    init_params: str = "kmeans",  # Only 'kmeans' or 'random' supported
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int, bool, float]:
    """
    Fit a Gaussian Mixture Model to data.

    Args:
        X: Data array (n_samples, n_features)
        n_components: Number of mixture components
        max_iter: Maximum number of EM iterations
        tol: Tolerance for convergence
        reg_covar: Regularization added to covariance
        n_init: Number of initializations to perform
        init_params: Initialization method ('kmeans' or 'random')
        random_state: Random seed

    Returns:
        Tuple of (weights, means, covariances, responsibilities, log_likelihood, n_iter, converged, bic)
    """
    # Convert inputs to float32
    X = X.astype(np.float32)
    tol = np.float32(tol)
    reg_covar = np.float32(reg_covar)

    # Initialize best parameters
    best_weights = None
    best_means = None
    best_covariances = None
    best_resp = None
    best_log_likelihood = -np.inf
    best_n_iter = 0
    best_converged = False

    # Try multiple initializations
    for init in range(n_init):
        # Initialize parameters
        init_seed = random_state + init

        if init_params == "kmeans":
            # Initialize means using KMeans++
            means = _kmeans_init(X, n_components, init_seed)

            # Initialize weights uniformly
            weights = np.ones(n_components, dtype=np.float32) / n_components

            # Initialize covariances based on data variance
            n_samples, n_features = X.shape
            covariances = np.zeros((n_components, n_features), dtype=np.float32)

            # Compute global variance
            global_var = np.zeros(n_features, dtype=np.float32)
            global_mean = np.zeros(n_features, dtype=np.float32)

            # Compute mean
            for j in range(n_features):
                for i in range(n_samples):
                    global_mean[j] += X[i, j]
                global_mean[j] /= n_samples

            # Compute variance
            for j in range(n_features):
                for i in range(n_samples):
                    diff = X[i, j] - global_mean[j]
                    global_var[j] += diff * diff
                global_var[j] /= n_samples

            # Initialize covariances
            for k in range(n_components):
                for j in range(n_features):
                    covariances[k, j] = global_var[j]
        else:
            # Random initialization
            weights, means, covariances = _random_init(X, n_components, init_seed)

        # Add regularization
        for k in range(n_components):
            for j in range(n_features):
                covariances[k, j] += reg_covar

        # Fit single GMM
        weights, means, covariances, resp, log_likelihood, n_iter, converged = (
            _fit_gmm_single(X, weights, means, covariances, max_iter, tol, reg_covar)
        )

        # Update best parameters if current model is better
        if log_likelihood > best_log_likelihood:
            best_weights = weights
            best_means = means
            best_covariances = covariances
            best_resp = resp
            best_log_likelihood = log_likelihood
            best_n_iter = n_iter
            best_converged = converged

    # Calculate BIC for the best model
    bic = _bic_score(X, best_log_likelihood, n_components)

    return (
        best_weights,
        best_means,
        best_covariances,
        best_resp,
        best_log_likelihood,
        best_n_iter,
        best_converged,
        bic,
    )


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
def predict_gmm(
    X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """
    Predict cluster assignments for data points.

    Args:
        X: Data array (n_samples, n_features)
        weights: Component weights
        means: Component means
        covariances: Component covariances

    Returns:
        Cluster assignments (n_samples,)
    """
    # Compute precisions
    precisions = _compute_precision_cholesky(covariances)

    # Compute responsibilities
    resp, _ = _estimate_responsibilities(X, means, precisions, weights)

    # Return most likely component for each sample
    return np.argmax(resp, axis=1)


@jit(nopython=True, cache=True)
def predict_proba_gmm(
    X: np.ndarray, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """
    Predict posterior probabilities for components.

    Args:
        X: Data array (n_samples, n_features)
        weights: Component weights
        means: Component means
        covariances: Component covariances

    Returns:
        Posterior probabilities (n_samples, n_components)
    """
    # Compute precisions
    precisions = _compute_precision_cholesky(covariances)

    # Compute responsibilities
    resp, _ = _estimate_responsibilities(X, means, precisions, weights)

    return resp


class NumbaGMMProcessor(BaseClusteringProcessor):
    """
    Pure Numba-accelerated Gaussian Mixture Model processor.

    This processor provides high-performance clustering through GMM
    using pure Numba with no scikit-learn dependency, ensuring consistent
    types when used in processing pipelines.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 100,
        n_init: int = 10,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        random_state: Optional[int] = 42,
        init_params: str = "kmeans",
        chunk_size: Optional[int] = None,
        compute_silhouette: bool = False,
    ):
        """
        Initialize the Numba GMM processor.

        Args:
            n_components: Number of mixture components
            max_iter: Maximum number of EM iterations
            n_init: Number of initializations to try
            tol: Convergence tolerance
            reg_covar: Regularization added to covariance
            random_state: Random seed for reproducibility
            init_params: Initialization method ('kmeans' or 'random')
            chunk_size: Size of chunks for Dask processing
            compute_silhouette: Whether to compute silhouette score
        """
        super().__init__()
        self.n_components = n_components
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state or 42
        self.init_params = init_params
        self.chunk_size = chunk_size
        self.compute_silhouette = compute_silhouette

        # Initialize result variables
        self._weights = None
        self._means = None
        self._covariances = None
        self._resp = None
        self._log_likelihood = None
        self._n_iter = None
        self._converged = False
        self._bic = None
        self._aic = None
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

    def fit(self, data: Union[np.ndarray, da.Array], **kwargs) -> "NumbaGMMProcessor":
        """
        Fit the GMM model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data

        Returns:
            Self for method chaining
        """
        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Fit GMM using Numba function
        (
            self._weights,
            self._means,
            self._covariances,
            self._resp,
            self._log_likelihood,
            self._n_iter,
            self._converged,
            self._bic,
        ) = fit_gmm(
            data_flat,
            self.n_components,
            self.max_iter,
            self.tol,
            self.reg_covar,
            self.n_init,
            self.init_params,
            self.random_state,
        )

        # Compute AIC score
        self._aic = _aic_score(data_flat, self._log_likelihood, self.n_components)

        # Set cluster labels based on responsibilities
        self._cluster_labels = np.argmax(self._resp, axis=1)
        self._cluster_centers = self._means
        self._is_fitted = True

        # Compute silhouette score if requested
        if (
            self.compute_silhouette
            and self.n_components > 1
            and len(data_flat) > self.n_components
        ):
            try:
                self._silhouette_score = silhouette_score(
                    data_flat, self._cluster_labels
                )
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
            raise ValueError("GMM model not fitted. Call fit() first.")

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Use Numba predict function
        return predict_gmm(data_flat, self._weights, self._means, self._covariances)

    def predict_proba(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict posterior probabilities for each component.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments
                preprocess_fn: Optional function to preprocess data

        Returns:
            Array of posterior probabilities (n_samples, n_components)
        """
        if not self._is_fitted:
            raise ValueError("GMM model not fitted. Call fit() first.")

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Use Numba predict_proba function
        return predict_proba_gmm(
            data_flat, self._weights, self._means, self._covariances
        )

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

        # Preprocess data
        data_flat = self._preprocess_data(data, **kwargs)

        # Compute precisions
        precisions = _compute_precision_cholesky(self._covariances)

        # Compute weighted log probabilities
        weighted_log_prob = _log_multivariate_normal_pdf_diag(
            data_flat, self._means, precisions
        )

        for k in range(self.n_components):
            weighted_log_prob[:, k] += np.log(max(self._weights[k], EPSILON))

        # Compute log likelihood for each sample
        log_prob = np.zeros(data_flat.shape[0], dtype=np.float32)

        for i in range(data_flat.shape[0]):
            max_log_prob = weighted_log_prob[i, 0]
            for k in range(1, self.n_components):
                if weighted_log_prob[i, k] > max_log_prob:
                    max_log_prob = weighted_log_prob[i, k]

            sum_exp = 0.0
            for k in range(self.n_components):
                sum_exp += np.exp(weighted_log_prob[i, k] - max_log_prob)

            log_prob[i] = max_log_prob + np.log(max(sum_exp, EPSILON))

        return log_prob

    def score(self, data: Union[np.ndarray, da.Array], **kwargs) -> float:
        """
        Compute the average log likelihood of data under the model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Average log likelihood per sample
        """
        log_prob = self.score_samples(data, **kwargs)
        return np.mean(log_prob)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform GMM clustering using Dask.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, required by BaseProcessor interface)
            **kwargs: Additional parameters
                compute_now: Whether to compute immediately (default: False)
                chunk_size: Size of chunks for processing (overrides instance setting)
                return_proba: Whether to return probabilities instead of labels

        Returns:
            Dask array with cluster assignments or probabilities
        """
        # Get processing parameters
        compute_now = kwargs.pop("compute_now", False)
        chunk_size = kwargs.pop("chunk_size", self.chunk_size)
        return_proba = kwargs.pop("return_proba", False)

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

            if return_proba:
                result = self.predict_proba(data_np, **kwargs)
            else:
                result = self.predict(data_np, **kwargs)

            return da.from_array(result)
        else:
            # Already fitted, use map_blocks for efficient prediction
            def process_chunk(chunk):
                # Handle multi-dimensional chunks
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)

                if return_proba:
                    return self.predict_proba(chunk, **kwargs)
                else:
                    return self.predict(chunk, **kwargs)

            # Apply custom chunk size if specified
            if chunk_size is not None:
                data_flat = data_flat.rechunk({0: chunk_size})

            # Apply prediction to chunks
            if return_proba:
                # For probabilities, output has an extra dimension
                result = data_flat.map_blocks(
                    process_chunk,
                    drop_axis=list(range(1, data_flat.ndim)),
                    new_axis=1,  # Add component dimension
                    chunks=(data_flat.chunks[0], (self.n_components,)),
                    dtype=np.float32,
                )
            else:
                # For labels, output is 1D per chunk
                result = data_flat.map_blocks(
                    process_chunk,
                    drop_axis=list(range(1, data_flat.ndim)),
                    dtype=np.int32,
                )

            return result

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
    def bic(self) -> Optional[float]:
        """Get the BIC score (lower is better)."""
        return self._bic

    @property
    def aic(self) -> Optional[float]:
        """Get the AIC score (lower is better)."""
        return self._aic

    @property
    def silhouette_score(self) -> Optional[float]:
        """Get the silhouette score (higher is better)."""
        return self._silhouette_score

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "n_components": self.n_components,
                "max_iter": self.max_iter,
                "n_init": self.n_init,
                "tol": self.tol,
                "reg_covar": self.reg_covar,
                "init_params": self.init_params,
                "is_fitted": self._is_fitted,
                "log_likelihood": self._log_likelihood,
                "n_iter": self._n_iter,
                "converged": self._converged,
                "bic": self._bic,
                "aic": self._aic,
                "silhouette_score": self._silhouette_score,
                "implementation": "pure_numba",
            }
        )
        return base_summary


# Factory functions for easy creation


def create_numba_gmm(
    n_components: int = 3,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> NumbaGMMProcessor:
    """
    Create a pure Numba GMM processor with standard parameters.

    Args:
        n_components: Number of mixture components
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaGMMProcessor
    """
    return NumbaGMMProcessor(
        n_components=n_components,
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_fast_numba_gmm(
    n_components: int = 3,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> NumbaGMMProcessor:
    """
    Create a pure Numba GMM processor optimized for speed.

    This configuration uses fewer initializations and iterations for faster
    clustering at some cost to quality.

    Args:
        n_components: Number of mixture components
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaGMMProcessor for speed
    """
    return NumbaGMMProcessor(
        n_components=n_components,
        max_iter=50,  # Fewer iterations
        n_init=3,  # Fewer initializations
        tol=1e-2,  # Less strict convergence
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_robust_numba_gmm(
    n_components: int = 3,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> NumbaGMMProcessor:
    """
    Create a pure Numba GMM processor optimized for robustness.

    This configuration uses more initializations and stricter convergence
    for better clustering quality at the cost of speed.

    Args:
        n_components: Number of mixture components
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured NumbaGMMProcessor for robust clustering
    """
    return NumbaGMMProcessor(
        n_components=n_components,
        max_iter=200,  # More iterations
        n_init=15,  # More initializations
        tol=1e-4,  # Stricter convergence
        reg_covar=1e-4,  # Slightly higher regularization
        random_state=random_state,
        chunk_size=chunk_size,
        compute_silhouette=True,  # Enable quality metrics
    )


def select_best_gmm(
    data: Union[np.ndarray, da.Array],
    min_components: int = 2,
    max_components: int = 10,
    criterion: str = "bic",
    random_state: int = 42,
    **kwargs,
) -> NumbaGMMProcessor:
    """
    Select the best GMM by automatically determining the optimal number of components.

    Args:
        data: Input data array
        min_components: Minimum number of components to try
        max_components: Maximum number of components to try
        criterion: Model selection criterion ('bic', 'aic', or 'silhouette')
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for NumbaGMMProcessor

    Returns:
        Fitted NumbaGMMProcessor with the optimal number of components
    """
    best_model = None
    best_score = np.inf if criterion in ["bic", "aic"] else -np.inf
    best_n_components = min_components

    # Convert to numpy if needed
    if isinstance(data, da.Array):
        data = data.compute()

    # Create a preprocessor once and reuse
    test_gmm = NumbaGMMProcessor(n_components=min_components, random_state=random_state)
    data_flat = test_gmm._preprocess_data(data, **kwargs)

    for n in range(min_components, max_components + 1):
        # Create and fit model
        gmm = NumbaGMMProcessor(
            n_components=n,
            random_state=random_state,
            compute_silhouette=criterion == "silhouette",
            **kwargs,
        )
        gmm.fit(data_flat)

        # Get score based on criterion
        if criterion == "bic":
            score = gmm.bic
        elif criterion == "aic":
            score = gmm.aic
        elif criterion == "silhouette":
            score = gmm.silhouette_score or -np.inf
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Update best model if needed
        if (criterion in ["bic", "aic"] and score < best_score) or (
            criterion == "silhouette" and score > best_score
        ):
            best_score = score
            best_model = gmm
            best_n_components = n

    print(
        f"Selected model with {best_n_components} components using {criterion} criterion."
    )
    return best_model
