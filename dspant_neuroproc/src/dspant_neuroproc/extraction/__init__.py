"""
Waveform extraction module for neural data analysis.

This module provides functionality for extracting and aligning spike waveforms
from continuous neural recordings, optimized for large datasets.

Functions:
    extract_spike_waveforms: Extract waveforms around specified spike times
    align_waveforms: Align existing waveforms to their peaks
    batch_extract_waveforms: Process large datasets in memory-efficient batches
"""

from .waveform_extractor import (
    align_waveforms,
    extract_spike_waveforms,
)

__all__ = [
    "extract_spike_waveforms",
    "align_waveforms",
]
