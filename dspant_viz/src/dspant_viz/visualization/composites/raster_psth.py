# src/dspant_viz/visualization/composites/raster_psth.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dspant_viz.core.base import CompositeVisualization
from dspant_viz.core.data_models import PSTHData, SpikeData
from dspant_viz.visualization.spike.psth import PSTHPlot
from dspant_viz.visualization.spike.raster import RasterPlot


class RasterPSTHComposite(CompositeVisualization):
    """
    Composite visualization that combines a raster plot and PSTH.

    This component creates a synchronized visualization with the raster plot
    on top and the PSTH below, both aligned to the same time scale.
    """

    def __init__(
        self,
        spike_data: SpikeData,
        bin_width: float = 0.05,
        time_window: Tuple[float, float] = (-1.0, 1.0),
        raster_color: str = "#2D3142",
        raster_alpha: float = 0.7,
        psth_color: str = "orange",
        show_sem: bool = True,
        sem_alpha: float = 0.3,
        show_smoothed: bool = True,
        marker_size: float = 4,
        marker_type: str = "|",
        title: Optional[str] = None,
        show_grid: bool = True,
        normalize_psth: bool = False,
        ylim_raster: Optional[Tuple[float, float]] = None,
        ylim_psth: Optional[Tuple[float, float]] = None,
        raster_height_ratio: float = 2.0,
        unit_id: Optional[int] = None,  # Add unit_id parameter
        **kwargs,
    ):
        """
        Initialize the RasterPSTH composite visualization.

        Parameters
        ----------
        spike_data : SpikeData
            Spike data containing spike times for different trials/units
        bin_width : float
            Width of PSTH bins in seconds
        time_window : tuple of (float, float)
            Time window to display (start_time, end_time) in seconds
        raster_color : str
            Color for raster plot markers
        raster_alpha : float
            Alpha transparency for raster markers
        psth_color : str
            Color for PSTH line
        show_sem : bool
            Whether to show standard error of mean
        sem_alpha : float
            Alpha transparency for SEM shading
        show_smoothed : bool
            Whether to show smoothed PSTH if available
        marker_size : float
            Size of raster markers
        marker_type : str
            Type of marker for raster plot
        title : str or None
            Title for the figure
        show_grid : bool
            Whether to show grid lines
        normalize_psth : bool
            Whether to normalize PSTH
        ylim_raster : tuple or None
            Y-axis limits for raster plot
        ylim_psth : tuple or None
            Y-axis limits for PSTH plot
        raster_height_ratio : float
            Ratio of raster height to PSTH height
        unit_id : int or None
            Specific unit ID to display. If None, uses the first unit in the data.
        **kwargs
            Additional configuration parameters
        """
        # Determine unit_id if not provided
        if unit_id is None and spike_data.spikes:
            unit_id = next(iter(spike_data.spikes.keys()))

        # Create individual components with the same unit_id
        raster_plot = RasterPlot(
            data=spike_data,
            marker_size=marker_size,
            marker_color=raster_color,
            marker_alpha=raster_alpha,
            marker_type=marker_type,
            unit_id=unit_id,  # Pass unit_id to RasterPlot
        )

        psth_plot = PSTHPlot(
            data=spike_data,
            bin_width=bin_width,
            time_window=time_window,
            line_color=psth_color,
            line_width=2.0,
            show_sem=show_sem,
            sem_alpha=sem_alpha,
            unit_id=unit_id,  # Pass unit_id to PSTHPlot
        )

        # Initialize base class
        super().__init__(components=[raster_plot, psth_plot], **kwargs)

        # Store configuration
        self.time_window = time_window
        self.title = title
        self.show_grid = show_grid
        self.normalize_psth = normalize_psth
        self.ylim_raster = ylim_raster
        self.ylim_psth = ylim_psth
        self.raster_height_ratio = raster_height_ratio
        self.show_smoothed = show_smoothed
        self.unit_id = unit_id  # Store unit_id
        self.spike_data = spike_data

    def get_data(self) -> Dict:
        """
        Prepare data from all components for rendering.

        Returns
        -------
        dict
            Combined data and parameters for rendering
        """
        raster_plot = self.components[0]
        psth_plot = self.components[1]

        # Get data from individual components
        raster_data = raster_plot.get_data()
        psth_data = psth_plot.get_data()

        # Build title if not provided
        computed_title = self.title
        if not computed_title and self.unit_id is not None:
            computed_title = f"Unit {self.unit_id}"

        # Combine into a single data structure
        return {
            "raster": raster_data,
            "psth": psth_data,
            "params": {
                "time_window": self.time_window,
                "title": computed_title,
                "show_grid": self.show_grid,
                "normalize_psth": self.normalize_psth,
                "ylim_raster": self.ylim_raster,
                "ylim_psth": self.ylim_psth,
                "raster_height_ratio": self.raster_height_ratio,
                "show_smoothed": self.show_smoothed,
                "unit_id": self.unit_id,  # Include unit_id in params
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update composite and component parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        # Update composite parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

        # Update individual components if necessary
        raster_plot = self.components[0]
        psth_plot = self.components[1]

        # Update raster plot parameters
        raster_updates = {}
        for key in [
            "marker_size",
            "marker_color",
            "marker_alpha",
            "marker_type",
            "unit_id",
        ]:
            if key in kwargs:
                raster_updates[key] = kwargs[key]

        if raster_updates:
            raster_plot.update(**raster_updates)

        # Update PSTH plot parameters
        psth_updates = {}
        for key in [
            "bin_width",
            "time_window",
            "line_color",
            "line_width",
            "show_sem",
            "sem_alpha",
            "unit_id",  # Include unit_id in PSTH updates
        ]:
            if key in kwargs:
                psth_updates[key] = kwargs[key]

        if psth_updates:
            psth_plot.update(**psth_updates)

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate the composite visualization using the specified backend.

        Parameters
        ----------
        backend : str
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs : dict
            Additional parameters for the backend

        Returns
        -------
        Any
            Composite figure from the specified backend
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.composite.raster_psth import render_raster_psth
        elif backend == "plotly":
            from dspant_viz.backends.plotly.composite.raster_psth import (
                render_raster_psth,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_raster_psth(self.get_data(), **kwargs)
