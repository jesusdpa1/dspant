"""
Metrics module for neural signal processing.

This module provides tools for evaluating the quality of neural signal processing
results, including spike sorting quality metrics and general signal quality assessment.
"""

from .quality_metrics import (
    compute_amplitude_cutoffs,
    compute_drift_metrics,
    compute_firing_ranges,
    compute_firing_rates,
    compute_isi_violations,
    compute_num_spikes,
    compute_presence_ratios,
    compute_quality_metrics,
    compute_refrac_period_violations,
    compute_snrs,
    compute_synchrony_metrics,
)

__all__ = [
    # High-level functions
    "compute_quality_metrics",
    # Individual metric functions
    "compute_amplitude_cutoffs",
    "compute_drift_metrics",
    "compute_firing_ranges",
    "compute_firing_rates",
    "compute_isi_violations",
    "compute_num_spikes",
    "compute_presence_ratios",
    "compute_refrac_period_violations",
    "compute_snrs",
    "compute_synchrony_metrics",
]
