"""
Visualization module for neural data analysis.

This module provides visualization tools for neural signals, spike data,
and clustering results, with specialized plots for different data types.
"""

from .general_plots import (
    plot_multi_channel_data,
    plot_spike_events,
    plot_spike_raster,
    plot_spikes,
)

__all__ = [
    # Spike visualization
    "plot_spikes",
    "plot_spike_raster",
    "plot_spike_events",
    "plot_multi_channel_data",
]
