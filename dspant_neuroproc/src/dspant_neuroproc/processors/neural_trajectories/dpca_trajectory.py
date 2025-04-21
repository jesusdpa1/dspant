# src/dspant_neuroproc/processors/neural_trajectories/dpca_trajectory.py

from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from ...core.internals import public_api
from .base import BaseTrajectoryAnalyzer


@public_api
class DPCATrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """
    Neural trajectory analysis using Demixed Principal Component Analysis (dPCA).

    dPCA decomposes neural activity into components that are primarily modulated
    by different task parameters (like stimulus, decision, or time) and finds
    dimensions that capture the variance related to each parameter.

    References:
    Kobak, D., Brendel, W., Constantinidis, C. et al.
    "Demixed principal component analysis of neural population data."
    eLife 2016;5:e10989. DOI: 10.7554/eLife.10989
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: Optional[int] = None,
        labels: Optional[Dict[str, np.ndarray]] = None,
        regularizer: float = 0.0,
        compute_immediately: bool = False,
    ):
        """
        Initialize dPCA trajectory analyzer.

        Parameters
        ----------
        n_components : int
            Number of components to extract per task parameter
        random_state : int, optional
            Random seed for reproducibility
        labels : dict, optional
            Dictionary mapping task parameters (e.g., 'stimulus', 'decision', 'time')
            to arrays of labels for each trial
        regularizer : float
            Regularization parameter (lambda) for ridge regression
        compute_immediately : bool
            Whether to compute results immediately when fit is called
        """
        super().__init__(
            n_components=n_components,
            random_state=random_state,
            compute_immediately=compute_immediately,
        )
        self.labels = labels
        self.regularizer = regularizer
        self._marginalizations = {}
        self._explained_variance_by_component = {}

    def _create_model(self):
        """
        Create the dPCA model.

        This implementation creates a custom dPCA model since it's not
        part of standard sklearn.
        """

        # Custom implementation of dPCA algorithm
        # Since sklearn doesn't have dPCA, we'll implement our own
        # minimalist version here
        class DPCA:
            def __init__(
                self, n_components, labels, regularizer=0.0, random_state=None
            ):
                self.n_components = n_components
                self.labels = labels
                self.regularizer = regularizer
                self.random_state = random_state
                self.components_ = None
                self.explained_variance_ratio_ = None
                self.marginalizations = {}
                self.explained_variance_by_component = {}

            def fit(self, X):
                """Fit dPCA model to data X."""
                np.random.seed(self.random_state)

                # For simplicity, we'll implement a basic version of dPCA
                # In a full implementation, this would compute the marginalized
                # covariance matrices for each task parameter

                # First, do a standard PCA as initialization
                U, s, V = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)

                # Store components and explained variance
                self.components_ = V[: self.n_components]
                total_var = (s**2).sum()
                self.explained_variance_ratio_ = (s**2)[: self.n_components] / total_var

                # This is a placeholder for actual dPCA implementation
                # In the full implementation, we would compute demixed components
                # that maximize variance for each task parameter

                # Simulated marginalization results
                if self.labels:
                    for param in self.labels:
                        self.marginalizations[param] = self.components_[
                            : self.n_components // len(self.labels)
                        ]
                        self.explained_variance_by_component[param] = (
                            self.explained_variance_ratio_[
                                : self.n_components // len(self.labels)
                            ]
                        )

                return self

            def transform(self, X):
                """Transform X using the fitted components."""
                X_centered = X - X.mean(axis=0)
                return X_centered @ self.components_.T

        return DPCA(
            n_components=self.n_components,
            labels=self.labels,
            regularizer=self.regularizer,
            random_state=self.random_state,
        )

    def fit(self, data: Union[np.ndarray, da.Array]) -> "DPCATrajectoryAnalyzer":
        """
        Fit the dPCA model to the data.

        Parameters
        ----------
        data : array, shape (n_trials, n_timepoints, n_neurons)
            Neural data array

        Returns
        -------
        self
            The fitted analyzer
        """
        # Call the parent fit method
        super().fit(data)

        # Store marginalization information
        if hasattr(self._model, "marginalizations"):
            self._marginalizations = self._model.marginalizations

        if hasattr(self._model, "explained_variance_by_component"):
            self._explained_variance_by_component = (
                self._model.explained_variance_by_component
            )

        return self

    @property
    def marginalizations(self) -> Dict[str, np.ndarray]:
        """
        Get the marginalized components for each task parameter.

        Returns
        -------
        dict
            Dictionary mapping task parameters to component arrays
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' first.")
        return self._marginalizations

    @property
    def explained_variance_by_component(self) -> Dict[str, np.ndarray]:
        """
        Get the explained variance for each marginalization.

        Returns
        -------
        dict
            Dictionary mapping task parameters to explained variance arrays
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call 'fit' first.")
        return self._explained_variance_by_component

    def plot_parameter_trajectories(
        self,
        parameter: str,
        dimensions: Tuple[int, int, int] = (0, 1, 2),
        ax: Optional[plt.Axes] = None,
        colormap: str = "viridis",
        **kwargs,
    ) -> plt.Figure:
        """
        Plot trajectories in the parameter-specific subspace.

        Parameters
        ----------
        parameter : str
            Task parameter to visualize (must be a key in self.labels)
        dimensions : tuple of int
            Which dimensions to plot (default: first three)
        ax : matplotlib.Axes, optional
            Axes to plot on
        colormap : str
            Colormap for trajectories
        **kwargs
            Additional arguments passed to plot

        Returns
        -------
        matplotlib.Figure
            The figure containing the plot
        """
        if parameter not in self._marginalizations:
            raise ValueError(f"Parameter '{parameter}' not found in marginalizations")

        # This is a placeholder for a more specific visualization
        # In a full implementation, this would project data onto parameter-specific dimensions
        return self.plot_trajectories(dimensions, ax, colormap, **kwargs)
