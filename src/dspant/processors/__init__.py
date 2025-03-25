"""
Processor modules for dspant.

This module provides various signal processing components:
- Basic: Fundamental operations (rectification, normalization, etc.)
- Filters: Signal filtering operations
- Segments: Segment extraction from continuous data
- Spatial: Spatial operations for multi-channel data
- Spectral: Frequency domain operations
- Utils: Utility processors
"""

# Import factory functions for easier access
# Expose all sub-packages
from . import basic, filters, quality_metrics, spatial, spectral
from .basic import (
    create_normalizer,
    create_rectifier,
    create_tkeo,
)
from .filters import (
    create_bandpass_filter,
    create_filter_processor,
    create_highpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from .quality_metrics import create_noise_estimation_processor
from .spatial import (
    create_car_processor,
    create_cmr_processor,
    create_whitening_processor,
)
from .spectral import (
    create_lfcc,
    create_mfcc,
    create_spectrogram,
)

__all__ = [
    # Sub-packages
    "basic",
    "filters",
    "segments",
    "spatial",
    "spectral",
    "quality_metrics",
    # Basic processors
    "create_moving_average",
    "create_normalizer",
    "create_rectifier",
    "create_tkeo",
    # Filter processors
    "create_bandpass_filter",
    "create_filter_processor",
    "create_highpass_filter",
    "create_lowpass_filter",
    "create_notch_filter",
    # Spatial processors
    "create_car_processor",
    "create_cmr_processor",
    "create_whitening_processor",
    # Spectral processors
    "create_lfcc",
    "create_mfcc",
    "create_spectrogram",
    # Qualty metrics processor
    "create_noise_estimation_processor",
]
