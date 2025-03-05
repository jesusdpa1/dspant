"""
Spatial processors for dspant.

This module provides processors for spatial operations on multi-channel data:
- Common Reference (CAR, CMR)
- Whitening and decorrelation
"""

from .common_reference import (
    CommonReferenceProcessor,
    create_car_processor,
    create_cmr_processor,
)
from .whiten import (
    WhiteningProcessor,
    create_robust_whitening_processor,
    create_whitening_processor,
)

__all__ = [
    # Common reference
    "CommonReferenceProcessor",
    "create_car_processor",
    "create_cmr_processor",
    # Whitening
    "WhiteningProcessor",
    "create_whitening_processor",
    "create_robust_whitening_processor",
]
