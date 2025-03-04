"""
Basic signal processing operations for dspant.

This module provides fundamental signal processing operations:
- Energy operators (TKEO)
- Normalization (z-score, min-max, robust)
- Rectification (abs, square, half-wave, hilbert)
"""

from .energy import TKEOProcessor
from .normalization import NormalizationProcessor
from .rectification import RectificationProcessor


# Factory functions for easier creation
def create_tkeo(method="classic"):
    """Create a TKEO processor with specified method."""
    return TKEOProcessor(method=method)


def create_normalizer(method="zscore"):
    """Create a normalization processor with specified method."""
    return NormalizationProcessor(method=method)


def create_rectifier(method="abs"):
    """Create a rectification processor with specified method."""
    return RectificationProcessor(method=method)


__all__ = [
    # Processor classes
    "TKEOProcessor",
    "NormalizationProcessor",
    "RectificationProcessor",
    # Factory functions
    "create_tkeo",
    "create_normalizer",
    "create_rectifier",
]
