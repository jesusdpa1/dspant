"""
Neural spike detection module.

This module provides tools for detecting action potentials (spikes) in
extracellular neural recordings using various methods.
"""

from .base import BaseDetector, ThresholdDetector
from .peak_detector import (
    PeakDetector,
    create_bipolar_peak_detector,
    create_negative_peak_detector,
    create_positive_peak_detector,
    create_threshold_detector,
)

# Export all public components
__all__ = [
    # Base classes
    "BaseDetector",
    "ThresholdDetector",
    # Detector implementations
    "PeakDetector",
    # Factory functions
    "create_threshold_detector",
    "create_negative_peak_detector",
    "create_positive_peak_detector",
    "create_bipolar_peak_detector",
]
