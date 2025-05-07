# src/dspant_viz/__init__.py
"""
dspant_viz: Multi-backend visualization library for electrophysiology data
"""

# Import public components from submodules
from .core.data_models import MultiChannelData, SpikeData, TimeSeriesData

# Register all imported components at the package level
from .core.internals import register_module_components
from .visualization import (
    CorrelogramPlot,
    PSTHPlot,
    RasterPlot,
    TimeSeriesAreaPlot,
    TimeSeriesPlot,
    TimeSeriesRasterPlot,
    WaveformPlot,
)
from .widgets import CorrelogramInspector, PSTHRasterInspector, WaveformInspector

register_module_components("dspant_viz.widgets")(__import__(__name__))
