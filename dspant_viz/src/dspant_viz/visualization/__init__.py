# src/dspant_viz/visualization/__init__.py
"""
Visualization module for neural data analysis.

This module provides visualization tools for neural signals, spike data,
and clustering results, with specialized plots for different data types.
"""

from dspant_viz.core.internals import public_api

from .spike.correlogram import CorrelogramPlot
from .spike.waveforms import WaveformPlot
from .stream.time_series import TimeSeriesPlot
from .stream.ts_area import TimeSeriesAreaPlot

# Direct export without explicit __all__ modification

public_api()(CorrelogramPlot)
public_api()(WaveformPlot)
public_api()(TimeSeriesPlot)
public_api()(TimeSeriesAreaPlot)
