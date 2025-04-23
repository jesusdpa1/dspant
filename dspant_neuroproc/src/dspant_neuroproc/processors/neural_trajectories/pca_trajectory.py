# src/dspant_neuroproc/processors/neural_trajectories/pca_trajectory.py

import numpy as np
from sklearn.decomposition import IncrementalPCA

from ...core.internals import public_api
from .base import BaseTrajectoryAnalyzer


@public_api
class PCATrajectoryAnalyzer(BaseTrajectoryAnalyzer):
    """
    Neural trajectory analysis using Incremental Principal Component Analysis (PCA).

    IncrementalPCA performs PCA with memory efficiency for large datasets by
    using batches of samples instead of the whole dataset at once.
    """

    def __init__(
        self,
        n_components: int = 3,
        whiten: bool = False,
        compute_immediately: bool = False,
        batch_size: int = None,
        **kwargs,
    ):
        """
        Initialize PCA trajectory analyzer using IncrementalPCA.

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
        batch_size : int, optional
            Size of batches for incremental learning. If None, uses
            the larger of 5 times n_components and 100.
        **kwargs
            Additional arguments passed to sklearn IncrementalPCA
        """
        super().__init__(
            n_components=n_components,
            compute_immediately=compute_immediately,
        )
        self.whiten = whiten
        self.batch_size = batch_size
        self.kwargs = kwargs

    def _create_model(self):
        """Create the IncrementalPCA model."""
        return IncrementalPCA(
            n_components=self.n_components,
            whiten=self.whiten,
            batch_size=self.batch_size,
            **self.kwargs,
        )
