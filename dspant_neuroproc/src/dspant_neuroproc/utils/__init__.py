"""
Utility functions for neural signal processing.

This module provides various utility functions supporting the neural processing
capabilities of dspant, including spike data handling and signal preprocessing.
"""

from .spike_utilities import (
    compute_spike_positions,
    estimate_noise_levels,
    extract_spike_amplitudes,
    extract_spike_times_from_binary,
    extract_templates,
    prepare_spike_data_for_metrics,
)

__all__ = [
    # Spike utilities
    "extract_spike_times_from_binary",
    "extract_spike_amplitudes",
    "extract_templates",
    "estimate_noise_levels",
    "compute_spike_positions",
    "prepare_spike_data_for_metrics",
]
