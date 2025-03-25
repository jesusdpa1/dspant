"""
Visualization module for neural data analysis.

This module provides visualization tools for neural signals, spike data,
and clustering results, with specialized plots for different data types.
"""

from .general_plots import (
    plot_multi_channel_data,
)

__all__ = [
    # Spike visualization
    "plot_multi_channel_data",
]
