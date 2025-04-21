# src/dspant/processors/neural_trajectories/factor_analysis.py

import numpy as np
from sklearn.decomposition import FactorAnalysis

from ...core.internals import public_api
from .base import BaseTrajectoryAnalyzer


@public_api
class FATrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """
    Neural trajectory analysis using Factor Analysis (FA).

    FA models data as a combination of latent factors plus noise,
    which can be more appropriate for neural data than PCA.
    """

    def __init__(
        self,
        n_components: int = 3,
        random_state: int = None,
        compute_immediately: bool = False,
        rotation: str = None,
        max_iter: int = 1000,
        **kwargs,
    ):
        """
        Initialize FA trajectory analyzer.

        Parameters
        ----------
        n_components : int
            Number of components to extract
        random_state : int, optional
            Random seed for reproducibility
        compute_immediately : bool
            Whether to compute results immediately when fit is called
        rotation : str, optional
            Type of rotation to perform
        max_iter : int
            Maximum number of iterations
        **kwargs
            Additional arguments passed to sklearn FactorAnalysis
        """
        super().__init__(
            n_components=n_components,
            random_state=random_state,
            compute_immediately=compute_immediately,
        )
        self.rotation = rotation
        self.max_iter = max_iter
        self.kwargs = kwargs

    def _create_model(self):
        """Create the FA model."""
        return FactorAnalysis(
            n_components=self.n_components,
            random_state=self.random_state,
            rotation=self.rotation,
            max_iter=self.max_iter,
            **self.kwargs,
        )
