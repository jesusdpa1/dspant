# components/spike/psth.py
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import PSTHData
from dspant_viz.core.internals import public_api


@public_api
class PSTHPlot(VisualizationComponent):
    """Component for PSTH visualization"""

    def __init__(
        self,
        time_bins: List[float],
        firing_rates: List[float],
        sem: Optional[List[float]] = None,
        unit_id: Optional[int] = None,
        line_color: str = "orange",
        line_width: float = 2.0,
        show_sem: bool = True,
        sem_alpha: float = 0.3,
        baseline_window: Optional[Tuple[float, float]] = None,
        show_grid: bool = True,
        show_event_onset: bool = True,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PSTH plot component

        Parameters
        ----------
        time_bins : List[float]
            Time bin centers in seconds
        firing_rates : List[float]
            Firing rates in Hz for each time bin
        sem : List[float], optional
            Standard error of mean for each bin
        unit_id : int, optional
            Unit identifier
        line_color : str
            Color for the PSTH line
        line_width : float
            Width of the PSTH line
        show_sem : bool
            Whether to show standard error of mean shading
        sem_alpha : float
            Alpha transparency for SEM shading
        baseline_window : Tuple[float, float], optional
            Time window for baseline period as (start, end)
        show_grid : bool
            Whether to show grid lines
        show_event_onset : bool
            Whether to show vertical line at time=0
        xlim : Tuple[float, float], optional
            X-axis limits as (min, max)
        ylim : Tuple[float, float], optional
            Y-axis limits as (min, max)
        title : str, optional
            Plot title. If None, a default title may be generated
        """
        # Validate and store the PSTH data
        self.data = PSTHData(
            time_bins=time_bins,
            firing_rates=firing_rates,
            sem=sem,
            unit_id=unit_id,
            baseline_window=baseline_window
        )

        # Store visualization parameters
        self.line_color = line_color
        self.line_width = line_width
        self.show_sem = show_sem
        self.sem_alpha = sem_alpha
        self.show_grid = show_grid
        self.show_event_onset = show_event_onset
        self.xlim = xlim
        self.ylim = ylim
        self.title = title
        self.additional_params = kwargs

    def get_data(self) -> Dict[str, Any]:
        """Get PSTH data in format ready for rendering"""
        # Build the data dictionary
        result = {
            "data": {
                "time_bins": self.data.time_bins,
                "firing_rates": self.data.firing_rates,
                "sem": self.data.sem,
                "unit_id": self.data.unit_id,
                "baseline_window": self.data.baseline_window
            },
            "params": {
                "line_color": self.line_color,
                "line_width": self.line_width,
                "show_sem": self.show_sem,
                "sem_alpha": self.sem_alpha,
                "show_grid": self.show_grid,
                "show_event_onset": self.show_event_onset,
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
        """Update PSTH plot parameters"""
        for key, value in kwargs.items():
            if key in ["time_bins", "firing_rates", "sem", "unit_id", "baseline_window"]:
                # Update data attribute directly
                setattr(self.data, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_params[key] = value
