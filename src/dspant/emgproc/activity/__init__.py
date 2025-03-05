"""
EMG activity detection module for identifying muscle activation patterns.

This module provides methods for detecting muscle activity in EMG signals,
particularly onset and offset detection using various threshold-based approaches.
"""

from .threshold_base import (
    EMGOnsetDetector,
    create_absolute_threshold_detector,
    create_rms_onset_detector,
    create_std_onset_detector,
)

# You can add additional imports if needed
# from .onset_detection import OtherDetectorClass

__all__ = [
    # Detector classes
    "EMGOnsetDetector",
    # Factory functions
    "create_std_onset_detector",
    "create_rms_onset_detector",
    "create_absolute_threshold_detector",
]
