"""
Segment extraction module for neural time-series data.

This module provides functionality for extracting segments of data based on onsets,
with various methods for determining segment boundaries.
"""

from .waveform_extraction import (
    SegmentExtractionProcessor,
    create_centered_extractor,
    create_fixed_window_extractor,
    create_onset_offset_extractor,
)

__all__ = [
    # Main processor class
    "SegmentExtractionProcessor",
    # Factory functions
    "create_fixed_window_extractor",
    "create_onset_offset_extractor",
    "create_centered_extractor",
]
