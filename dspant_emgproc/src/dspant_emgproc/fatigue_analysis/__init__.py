"""
EMG fatigue analysis module.

This module provides tools for analyzing muscle fatigue from electromyographic (EMG) signals.
It includes implementations of spectral and temporal fatigue indices, conduction velocity
estimation, and fatigue trend analysis.

Various methods are provided:
- Spectral indices: Median frequency, mean frequency, spectral entropy, etc.
- Conduction velocity: Estimation of muscle fiber conduction velocity
- Fatigue indices: Metrics derived from spectral changes over time
"""

# Import spectral indices functions
from .spectral_indices import (
    FatigueIndicesProcessor,
    SpectralIndicesProcessor,
    analyze_fatigue_from_spectrogram,
    compare_fatigue_between_segments,
    create_fatigue_analysis_processor,
    create_spectral_indices_processor,
    extract_fatigue_metrics_from_segments,
)

__all__ = [
    # Processor classes
    "SpectralIndicesProcessor",
    "FatigueIndicesProcessor",
    # Factory functions
    "create_spectral_indices_processor",
    "create_fatigue_analysis_processor",
    # Analysis functions
    "analyze_fatigue_from_spectrogram",
    "extract_fatigue_metrics_from_segments",
    "compare_fatigue_between_segments",
]
