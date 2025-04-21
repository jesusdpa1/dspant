"""
IO module for data loading and conversion in dspant.

This module provides functionality for:
- Converting data from various formats (TDT, etc.) to dspant's ANT format
- Loading data from various file formats
"""

# Re-export key functions for easier access
from .exporters.tdt2ant import convert_tdt_to_ant, drvPathCarpenter
from .loaders.tdt_loader import load_tdt_block

__all__ = [
    # Conversion utilities
    "convert_tdt_to_ant",
    "drvPathCarpenter",
    # Loading utilities
    "load_tdt_block",
]
