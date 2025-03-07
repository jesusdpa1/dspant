"""
Clustering module for neural and signal processing applications.

This module provides various clustering implementations optimized for
signal processing applications, including K-means, GMM, DBSCAN, and PCA-KMeans,
with support for Dask and Numba acceleration.
"""

# Import from KMeans module
# Import from base module
from .base import BaseClusteringProcessor

# Import from DBSCAN module
from .dbscan import (
    DBSCANProcessor,
    create_dbscan,
    create_dense_dbscan,
    create_sklearn_dbscan,
    create_sparse_dbscan,
)

# Import from GMM module
from .gmm import (
    GMMProcessor,
    create_fast_gmm,
    create_gmm,
    create_robust_gmm,
    create_sklearn_gmm,
)
from .kmeans import (
    KMeansProcessor,
    create_fast_kmeans,
    create_high_quality_kmeans,
    create_kmeans,
)

# Import from PCA-KMeans module
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
    # Base class
    "BaseClusteringProcessor",
    # KMeans
    "KMeansProcessor",
    "create_kmeans",
    "create_fast_kmeans",
    "create_high_quality_kmeans",
    # GMM
    "GMMProcessor",
    "create_gmm",
    "create_fast_gmm",
    "create_robust_gmm",
    "create_sklearn_gmm",
    # DBSCAN
    "DBSCANProcessor",
    "create_dbscan",
    "create_dense_dbscan",
    "create_sparse_dbscan",
    "create_sklearn_dbscan",
    # PCA-KMeans
    "PCAKMeansProcessor",
    "create_pca_kmeans",
    "create_adaptive_clustering",
    "create_visualization_clustering",
    "create_incremental_clustering",
    "create_quality_metrics_clustering",
    "create_multichannel_clustering",
]
