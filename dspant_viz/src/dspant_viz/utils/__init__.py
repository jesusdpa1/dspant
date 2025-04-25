# src/dspant_viz/__init__.py
"""
dspant_viz: Multi-backend visualization library for electrophysiology data
"""

# Register all imported components at the package level
from dspant_viz.core.internals import register_module_components

# Import public components from submodules
from .normalization import normalize_data

register_module_components("dspant_viz.utils")(__import__(__name__))
