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

# Import Numba-accelerated implementations
try:
    from .pca_numba import (
        NumbaRealPCAProcessor,
        create_fast_numba_pca,
        create_numba_pca,
        create_numba_whitening_pca,
    )

    has_numba_pca = True
except ImportError:
    has_numba_pca = False

try:
    from .tsne_numba import (
        NumbaRealTSNEProcessor,
        create_cluster_preservation_numba_tsne,
        create_fast_numba_tsne,
        create_high_quality_numba_tsne,
        create_incremental_numba_tsne,
        create_numba_tsne,
        create_numba_visualization_tsne,
    )

    has_numba_tsne = True
except ImportError:
    has_numba_tsne = False

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

# Add Numba implementations to __all__ if available
if has_numba_pca:
    __all__.extend(
        [
            "NumbaRealPCAProcessor",
            "create_numba_pca",
            "create_numba_whitening_pca",
            "create_fast_numba_pca",
        ]
    )

if has_numba_tsne:
    __all__.extend(
        [
            "NumbaRealTSNEProcessor",
            "create_numba_tsne",
            "create_numba_visualization_tsne",
            "create_fast_numba_tsne",
            "create_high_quality_numba_tsne",
            "create_cluster_preservation_numba_tsne",
            "create_incremental_numba_tsne",
        ]
    )
