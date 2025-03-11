"""
EMG activity detection module for identifying muscle activation patterns.

This module provides methods for detecting muscle activity in EMG signals,
particularly onset and offset detection using various approaches including
threshold-based and Bayesian changepoint detection methods.
"""

from .bayesian_detector import (
    BayesianChangepointDetector,
    create_realtime_changepoint_detector,
    create_robust_changepoint_detector,
    create_sensitive_changepoint_detector,
)
from .single_threshold import (
    EMGOnsetDetector,
    create_absolute_threshold_detector,
    create_rms_onset_detector,
    create_std_onset_detector,
)

__all__ = [
    # Threshold-based detectors
    "EMGOnsetDetector",
    "create_std_onset_detector",
    "create_rms_onset_detector",
    "create_absolute_threshold_detector",
    # Bayesian changepoint detectors
    "BayesianChangepointDetector",
    "create_sensitive_changepoint_detector",
    "create_robust_changepoint_detector",
    "create_realtime_changepoint_detector",
]
