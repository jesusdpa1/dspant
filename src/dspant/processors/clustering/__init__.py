"""
Clustering algorithms for signal processing.

This submodule contains various clustering algorithms optimized for
signal processing tasks, including KMeans implementations with different
performance characteristics.
"""

from .kmeans import (
    KMeansProcessor,
    create_fast_kmeans,
    create_high_quality_kmeans,
    create_kmeans,
)
from .kmeans_numba import (
    NumbaKMeansProcessor,
    create_fast_numba_kmeans,
    create_numba_kmeans,
    create_robust_numba_kmeans,
)

__all__ = [
    # Standard KMeans
    "KMeansProcessor",
    "create_kmeans",
    "create_fast_kmeans",
    "create_high_quality_kmeans",
    # Numba-accelerated KMeans
    "NumbaKMeansProcessor",
    "create_numba_kmeans",
    "create_fast_numba_kmeans",
    "create_robust_numba_kmeans",
]
