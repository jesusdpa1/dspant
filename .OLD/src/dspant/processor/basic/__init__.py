"""
Basic signal processing operations for dspant.

This module provides fundamental signal processing operations:
- Energy operators (TKEO)
- Normalization (z-score, min-max, robust)
- Rectification (abs, square, half-wave, hilbert)
- Moving statistics (average, RMS)
- Envelope detection (multiple methods)
"""

from .energy import TKEOProcessor

# Import moving stats classes
from .moving_stats import MovingAverageProcessor, MovingRMSProcessor
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


def create_moving_average(window_size=11, method="simple", **kwargs):
    """Create a moving average processor with specified parameters."""
    # Direct implementation instead of importing from moving_stats
    return MovingAverageProcessor(window_size=window_size, method=method, **kwargs)


def create_moving_rms(window_size=11, center=True):
    """Create a moving RMS processor with specified parameters."""
    # Direct implementation instead of importing from moving_stats
    return MovingRMSProcessor(window_size=window_size, center=center)


# Import envelope functions
from .envelope import (
    create_hilbert_envelope,
    create_peak_envelope,
    create_rectify_smooth_envelope,
    create_rms_envelope,
    create_tkeo_envelope,
)

__all__ = [
    # Processor classes
    "TKEOProcessor",
    "MovingAverageProcessor",
    "MovingRMSProcessor",
    "NormalizationProcessor",
    "RectificationProcessor",
    # Factory functions
    "create_tkeo",
    "create_normalizer",
    "create_rectifier",
    "create_moving_average",
    "create_moving_rms",
    # Envelope detection
    "create_rectify_smooth_envelope",
    "create_hilbert_envelope",
    "create_rms_envelope",
    "create_tkeo_envelope",
    "create_peak_envelope",
]
