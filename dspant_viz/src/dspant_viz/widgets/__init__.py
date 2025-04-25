# src/dspant_viz/widgets/__init__.py
"""
widgets module for neural data analysis.

This module provides visualization tools for neural signals, spike data,
and clustering results, with specialized plots for different data types.
"""

from dspant_viz.core.internals import register_module_components

from .correlogram_inspector import CorrelogramInspector
from .psth_raster_inspector import PSTHRasterInspector
from .waveform_inspector import WaveformInspector

# Register all imported components at once
register_module_components("dspant_viz.widgets")(__import__(__name__))
