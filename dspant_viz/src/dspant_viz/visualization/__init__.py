# src/dspant_viz/visualization/__init__.py
"""
Visualization module for neural data analysis.

This module provides visualization tools for neural signals, spike data,
and clustering results, with specialized plots for different data types.
"""

from dspant_viz.core.internals import register_module_components

from .spike.correlogram import CorrelogramPlot
from .spike.psth import PSTHPlot
from .spike.raster import RasterPlot
from .spike.waveforms import WaveformPlot
from .stream.time_series import TimeSeriesPlot
from .stream.ts_area import TimeSeriesAreaPlot
from .stream.ts_raster import TimeSeriesRasterPlot

# Register all imported components at once
register_module_components("dspant_viz.visualization")(__import__(__name__))
