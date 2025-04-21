# src/dspant/processors/neural_trajectories/pca.py

import numpy as np
from sklearn.decomposition import PCA

from ...core.internals import public_api
from .base import BaseTrajectoryAnalyzer


@public_api
class PCATrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """
    Neural trajectory analysis using Principal Component Analysis (PCA).

    PCA finds orthogonal dimensions that maximize variance in the data.
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = None,
        whiten: bool = False,
        compute_immediately: bool = False,
        **kwargs,
    ):
        """
        Initialize PCA trajectory analyzer.

        Parameters
        ----------
        n_components : int
            Number of components to extract
        random_state : int, optional
            Random seed for reproducibility
        whiten : bool
            Whether to whiten the data
        compute_immediately : bool
            Whether to compute results immediately when fit is called
        **kwargs
            Additional arguments passed to sklearn PCA
        """
        super().__init__(
            n_components=n_components,
            random_state=random_state,
            compute_immediately=compute_immediately,
        )
        self.whiten = whiten
        self.kwargs = kwargs

    def _create_model(self):
        """Create the PCA model."""
        return PCA(
            n_components=self.n_components,
            random_state=self.random_state,
            whiten=self.whiten,
            **self.kwargs,
        )
