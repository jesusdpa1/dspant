"""
Neural signal sorting algorithms.

This submodule contains specialized clustering algorithms optimized for
sorting neural signals, such as spike sorting and feature-based clustering.
"""

from .pca_kmeans import (
    ComposedPCAKMeansProcessor,
    create_adaptive_clustering,
    create_high_quality_clustering,
    create_incremental_clustering,
    create_multichannel_clustering,
    create_pca_kmeans,
    create_quality_metrics_clustering,
)
from .pca_kmeans_numba import (
    NumbaComposedPCAKMeansProcessor,
    create_adaptive_numba_pca_kmeans,
    create_fast_numba_pca_kmeans,
    create_multichannel_numba_pca_kmeans,
    create_numba_pca_kmeans,
)

__all__ = [
    # PCA-KMeans
    "ComposedPCAKMeansProcessor",
    "create_pca_kmeans",
    "create_adaptive_clustering",
    "create_high_quality_clustering",
    "create_incremental_clustering",
    "create_quality_metrics_clustering",
    "create_multichannel_clustering",
    # Numba-accelerated PCA-KMeans
    "NumbaComposedPCAKMeansProcessor",
    "create_numba_pca_kmeans",
    "create_fast_numba_pca_kmeans",
    "create_high_quality_numba_pca_kmeans",
    "create_adaptive_numba_pca_kmeans",
    "create_multichannel_numba_pca_kmeans",
]
