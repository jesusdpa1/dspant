# src/dspant/processors/neural_trajectories/__init__.py

from typing import Literal, Optional, Union

import dask.array as da
import numpy as np

# Import other analyzers as they are implemented
from dspant.core.internals import public_api

from .base import BaseTrajectoryAnalyzer
from .fa_trajectory import FATrajectoryAnalyzer
from .pca_trajectory import PCATrajectoryAnalyzer


@public_api
def create_trajectory_analyzer(
    method: Literal["pca", "fa", "gpfa", "lle", "tsne", "umap", "dpca"] = "pca",
    n_components: int = 3,
    random_state: Optional[int] = None,
    compute_immediately: bool = False,
    **kwargs,
) -> BaseTrajectoryAnalyzer:
    """
    Create a neural trajectory analyzer.

    Parameters
    ----------
    method : str
        Dimensionality reduction method to use:
        - "pca": Principal Component Analysis
        - "fa": Factor Analysis
        - "gpfa": Gaussian Process Factor Analysis
        - "lle": Locally Linear Embedding
        - "tsne": t-SNE
        - "umap": Uniform Manifold Approximation and Projection
        - "dpca": Demixed Principal Component Analysis
    n_components : int
        Number of components to extract
    random_state : int, optional
        Random seed for reproducibility
    compute_immediately : bool
        Whether to compute results immediately when fit is called
    **kwargs
        Additional arguments passed to the specific analyzer

    Returns
    -------
    BaseTrajectoryAnalyzer
        The configured trajectory analyzer
    """
    if method == "pca":
        return PCATrajectoryAnalyzer(
            n_components=n_components,
            random_state=random_state,
            compute_immediately=compute_immediately,
            **kwargs,
        )
    elif method == "fa":
        return FATrajectoryAnalyzer(
            n_components=n_components,
            random_state=random_state,
            compute_immediately=compute_immediately,
            **kwargs,
        )
    # Add other methods as they are implemented
    else:
        raise ValueError(f"Unknown method: {method}")


@public_api
def analyze_trajectories(
    data: Union[np.ndarray, da.Array],
    method: str = "pca",
    n_components: int = 3,
    random_state: Optional[int] = None,
    compute_immediately: bool = True,
    **kwargs,
) -> Union[np.ndarray, da.Array]:
    """
    Analyze neural trajectories in a single function call.

    Parameters
    ----------
    data : array, shape (n_trials, n_timepoints, n_neurons)
        Neural data array
    method : str
        Dimensionality reduction method to use
    n_components : int
        Number of components to extract
    random_state : int, optional
        Random seed for reproducibility
    compute_immediately : bool
        Whether to compute results immediately
    **kwargs
        Additional arguments passed to the specific analyzer

    Returns
    -------
    array
        Reduced-dimension neural trajectories
    """
    analyzer = create_trajectory_analyzer(
        method=method,
        n_components=n_components,
        random_state=random_state,
        compute_immediately=compute_immediately,
        **kwargs,
    )

    return analyzer.fit_transform(data)


__all__ = [
    "BaseTrajectoryAnalyzer",
    "PCATrajectoryAnalyzer",
    "FATrajectoryAnalyzer",
    "create_trajectory_analyzer",
    "analyze_trajectories",
]
