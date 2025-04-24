# components/spike/correlogram.py
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.internals import public_api


@public_api
class CorrelogramPlot(VisualizationComponent):
    """Component for auto/cross-correlogram visualization"""

    def __init__(
        self,
        time_bins: List[float],
        counts: List[float],
        is_autocorrelogram: bool = True,
        unit_id1: Optional[int] = None,
        unit_id2: Optional[int] = None,
        bar_color: str = "#4878d0",
        edge_color: str = "black",
        show_grid: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize correlogram plot component

        Parameters
        ----------
        time_bins : List[float]
            Time bin centers in seconds
        counts : List[float]
            Spike counts for each bin
        is_autocorrelogram : bool
            Whether this is an autocorrelogram (True) or cross-correlogram (False)
        unit_id1 : int, optional
            ID of the first unit
        unit_id2 : int, optional
            ID of the second unit (for cross-correlogram)
        bar_color : str
            Color for the histogram bars
        edge_color : str
            Color for the bar edges
        show_grid : bool
            Whether to show grid lines
        xlim : Tuple[float, float], optional
            X-axis limits as (min, max)
        ylim : Tuple[float, float], optional
            Y-axis limits as (min, max)
        title : str, optional
            Plot title. If None, a default title will be generated
        """
        # Validate inputs
        if len(time_bins) != len(counts):
            raise ValueError("time_bins and counts must have the same length")

        # Store data
        self.time_bins = time_bins
        self.counts = counts
        self.is_autocorrelogram = is_autocorrelogram
        self.unit_id1 = unit_id1
        self.unit_id2 = unit_id2

        # Store visualization parameters
        self.bar_color = bar_color
        self.edge_color = edge_color
        self.show_grid = show_grid
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.additional_params = kwargs

    def get_data(self) -> Dict[str, Any]:
        """Get correlogram data in format ready for rendering"""
        # Calculate bin width
        if len(self.time_bins) > 1:
            bin_width = self.time_bins[1] - self.time_bins[0]
        else:
            bin_width = 0.001  # Default 1ms

        # Generate a default title if none provided
        if self.title is None:
            if self.is_autocorrelogram:
                if self.unit_id1 is not None:
                    self.title = f"Autocorrelogram for Unit {self.unit_id1}"
                else:
                    self.title = "Autocorrelogram"
            else:
                if self.unit_id1 is not None and self.unit_id2 is not None:
                    self.title = f"Cross-correlogram: Unit {self.unit_id1} × Unit {self.unit_id2}"
                else:
                    self.title = "Cross-correlogram"

        # Build the data dictionary
        result = {
            "data": {
                "time_bins": self.time_bins,
                "counts": self.counts,
                "bin_width": bin_width,
                "is_autocorrelogram": self.is_autocorrelogram,
                "unit_id1": self.unit_id1,
                "unit_id2": self.unit_id2
            },
            "params": {
                "bar_color": self.bar_color,
                "edge_color": self.edge_color,
                "show_grid": self.show_grid,
                "title": self.title,
                **self.additional_params
            }
        }

        # Add optional parameters
        if self.xlim is not None:
            result["params"]["xlim"] = self.xlim
        if self.ylim is not None:
            result["params"]["ylim"] = self.ylim

        return result

    def update(self, **kwargs) -> None:
        """Update correlogram plot parameters"""
        for key, value in kwargs.items():
            if key in ["time_bins", "counts", "is_autocorrelogram", "unit_id1", "unit_id2"]:
                setattr(self, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value
