# src/dspant_neuroproc/processors/neural_trajectories/__init__.py

from typing import Literal, Optional, Union

import dask.array as da
import numpy as np

# Import analyzers
from dspant.core.internals import public_api

from .base import BaseTrajectoryAnalyzer
from .dpca_trajectory import DPCATrajectoryAnalyzer
from .fa_trajectory import FATrajectoryAnalyzer
from .pca_trajectory import PCATrajectoryAnalyzer


@public_api
def create_trajectory_analyzer(
    method: Literal["pca", "fa", "dpca", "gpfa", "lle", "tsne", "umap"] = "pca",
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
        - "pca": Principal Component Analysis (Incremental implementation)
        - "fa": Factor Analysis
        - "dpca": Demixed Principal Component Analysis
        - "gpfa": Gaussian Process Factor Analysis (not yet implemented)
        - "lle": Locally Linear Embedding (not yet implemented)
        - "tsne": t-SNE (not yet implemented)
        - "umap": Uniform Manifold Approximation and Projection (not yet implemented)
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
    elif method == "dpca":
        return DPCATrajectoryAnalyzer(
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
) -> Union[BaseTrajectoryAnalyzer, np.ndarray, da.Array]:
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
    BaseTrajectoryAnalyzer
        The fitted analyzer with stored trajectories
    """
    analyzer = create_trajectory_analyzer(
        method=method,
        n_components=n_components,
        random_state=random_state,
        compute_immediately=compute_immediately,
        **kwargs,
    )

    # Fit and transform data, storing results within the analyzer
    analyzer.fit_transform(data)

    # Return the analyzer, which now contains the trajectories
    return analyzer


__all__ = [
    "BaseTrajectoryAnalyzer",
    "PCATrajectoryAnalyzer",
    "FATrajectoryAnalyzer",
    "DPCATrajectoryAnalyzer",
    "create_trajectory_analyzer",
    "analyze_trajectories",
]
