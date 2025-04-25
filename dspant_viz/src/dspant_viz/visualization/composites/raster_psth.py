# src/dspant_viz/visualization/composites/raster_psth.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dspant.core.internals import public_api
from dspant_viz.core.base import CompositeVisualization
from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.spike.psth import PSTHPlot
from dspant_viz.visualization.spike.raster import RasterPlot


@public_api(module_override="dspant_viz.visualization")
class RasterPSTHComposite(CompositeVisualization):
    """
    Composite visualization that combines a trial-based raster plot and PSTH.

    This component creates a synchronized visualization with the raster plot
    on top and the PSTH below, both aligned to the same time scale around event times.
    """

    def __init__(
        self,
        spike_data: SpikeData,
        event_times: np.ndarray,
        pre_time: float,
        post_time: float,
        bin_width: float = 0.05,
        raster_color: str = "#2D3142",
        raster_alpha: float = 0.7,
        psth_color: str = "orange",
        show_sem: bool = True,
        sem_alpha: float = 0.3,
        sigma: Optional[float] = None,
        marker_size: float = 4,
        marker_type: str = "|",
        title: Optional[str] = None,
        show_grid: bool = True,
        ylim_raster: Optional[Tuple[float, float]] = None,
        ylim_psth: Optional[Tuple[float, float]] = None,
        raster_height_ratio: float = 2.0,
        unit_id: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the RasterPSTH composite visualization.

        Parameters
        ----------
        spike_data : SpikeData
            Spike data containing spike times for different units
        event_times : ndarray
            Event/trigger times in seconds
        pre_time : float
            Time before each event to include (seconds)
        post_time : float
            Time after each event to include (seconds)
        bin_width : float
            Width of PSTH bins in seconds
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
        sigma : float, optional
            Standard deviation for Gaussian smoothing (seconds)
        marker_size : float
            Size of raster markers
        marker_type : str
            Type of marker for raster plot
        title : str or None
            Title for the figure
        show_grid : bool
            Whether to show grid lines
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
        # Input validation
        if not isinstance(spike_data, SpikeData):
            raise TypeError("spike_data must be a SpikeData instance")

        if event_times is None or len(event_times) == 0:
            raise ValueError("event_times must be provided for RasterPSTHComposite")

        if pre_time <= 0 or post_time <= 0:
            raise ValueError("pre_time and post_time must be positive values")

        # Determine unit_id if not provided
        if unit_id is None and spike_data.spikes:
            unit_id = list(spike_data.spikes.keys())[0]

        # Store time window for convenience
        time_window = (-pre_time, post_time)

        # Create individual components with shared parameters
        raster_plot = RasterPlot(
            data=spike_data,
            event_times=event_times,
            pre_time=pre_time,
            post_time=post_time,
            marker_size=marker_size,
            marker_color=raster_color,
            marker_alpha=raster_alpha,
            marker_type=marker_type,
            unit_id=unit_id,
        )

        psth_plot = PSTHPlot(
            data=spike_data,
            event_times=event_times,
            pre_time=pre_time,
            post_time=post_time,
            bin_width=bin_width,
            line_color=psth_color,
            line_width=2.0,
            show_sem=show_sem,
            sem_alpha=sem_alpha,
            unit_id=unit_id,
            sigma=sigma,
        )

        # Initialize base class
        super().__init__(components=[raster_plot, psth_plot], **kwargs)

        # Store configuration
        self.title = title
        self.show_grid = show_grid
        self.ylim_raster = ylim_raster
        self.ylim_psth = ylim_psth
        self.raster_height_ratio = raster_height_ratio
        self.unit_id = unit_id
        self.time_window = time_window

        # Store parameters for update method
        self.spike_data = spike_data
        self.event_times = event_times
        self.pre_time = pre_time
        self.post_time = post_time
        self.bin_width = bin_width
        self.sigma = sigma

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
                "title": computed_title,
                "show_grid": self.show_grid,
                "ylim_raster": self.ylim_raster,
                "ylim_psth": self.ylim_psth,
                "raster_height_ratio": self.raster_height_ratio,
                "unit_id": self.unit_id,
                "time_window": self.time_window,
                "show_event_onset": True,  # Always show event onset in this composite
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
        # Special handling for time window parameters
        if "pre_time" in kwargs or "post_time" in kwargs:
            pre_time = kwargs.get("pre_time", self.pre_time)
            post_time = kwargs.get("post_time", self.post_time)
            self.time_window = (-pre_time, post_time)

        # Update composite parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

        # Update individual components if necessary
        raster_plot = self.components[0]
        psth_plot = self.components[1]

        # Prepare parameter updates for each component
        raster_updates = {}
        psth_updates = {}

        # Map parameters to appropriate components
        for key, value in kwargs.items():
            # Parameters for both components
            if key in ["event_times", "pre_time", "post_time", "unit_id"]:
                raster_updates[key] = value
                psth_updates[key] = value

            # Raster-specific parameters
            elif key in ["marker_size", "marker_color", "marker_alpha", "marker_type"]:
                raster_updates[key] = value

            # PSTH-specific parameters
            elif key in ["bin_width", "line_color", "show_sem", "sem_alpha", "sigma"]:
                psth_updates[key] = value

        # Apply updates to components
        if raster_updates:
            raster_plot.update(**raster_updates)

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

        # Pass WebGL option if using plotly and not explicitly set
        if backend == "plotly" and "use_webgl" not in kwargs:
            kwargs["use_webgl"] = True

        return render_raster_psth(self.get_data(), **kwargs)
