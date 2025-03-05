"""
EMG processing module for analyzing electromyography signals.

This module contains tools for processing and analyzing EMG data, including:
- Activity detection (onset/offset detection)
- Fatigue analysis
- Feature extraction
- Quality metrics
"""

# Re-export key functionality from submodules
from .activity import (
    EMGOnsetDetector,
    create_absolute_threshold_detector,
    create_rms_onset_detector,
    create_std_onset_detector,
)

# You can add additional imports from other submodules as needed
# For example:
# from .fatigue import FatigueAnalyzer
# from .metrics import EMGQualityMetrics

__all__ = [
    # Activity detection
    "EMGOnsetDetector",
    "create_std_onset_detector",
    "create_rms_onset_detector",
    "create_absolute_threshold_detector",
]
