# src/dspant/processor/dimensionality_reduction/__init__.py
"""
Dimensionality reduction module for signal processing.

This module provides implementations of various dimensionality reduction
techniques optimized for signal processing applications, including:
- PCA (Principal Component Analysis)
- t-SNE (t-Distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)

These techniques can be used for feature extraction, visualization,
and preprocessing for machine learning algorithms. All implementations
are optimized with Numba acceleration for improved performance.
"""

from .base import BaseDimensionalityReductionProcessor
from .pca import (
    PCAProcessor,
    create_fast_pca,
    create_pca,
    create_whitening_pca,
)
from .tsne import (
    TSNEProcessor,
    create_fast_tsne,
    create_tsne,
    create_visualization_tsne,
)
from .umap import (
    UMAPProcessor,
    create_fast_umap,
    create_preserving_umap,
    create_supervised_umap,
    create_umap,
    create_visualization_umap,
)

__all__ = [
    # Base class
    "BaseDimensionalityReductionProcessor",
    # PCA
    "PCAProcessor",
    "create_pca",
    "create_whitening_pca",
    "create_fast_pca",
    # t-SNE
    "TSNEProcessor",
    "create_tsne",
    "create_visualization_tsne",
    "create_fast_tsne",
    # UMAP
    "UMAPProcessor",
    "create_umap",
    "create_visualization_umap",
    "create_preserving_umap",
    "create_fast_umap",
    "create_supervised_umap",
]
