"""
PCA-KMeans clustering module for neural waveform analysis.

This module provides implementations of PCA-KMeans clustering optimized for
neural spike waveform analysis, with support for dask and numba acceleration.
"""

from .pca_clustering import (
    PCAKMeansProcessor,
    create_adaptive_clustering,
    create_incremental_clustering,
    create_multichannel_clustering,
    create_pca_kmeans,
    create_quality_metrics_clustering,
    create_visualization_clustering,
)

__all__ = [
    # Main processor class
    "PCAKMeansProcessor",
    # Factory functions
    "create_pca_kmeans",
    "create_adaptive_clustering",
    "create_visualization_clustering",
    "create_incremental_clustering",
    "create_quality_metrics_clustering",
    "create_multichannel_clustering",
]
