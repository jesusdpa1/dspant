from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.spike.base import BaseSpikeVisualization


class RasterPlot(BaseSpikeVisualization):
    """Component for spike raster visualization with multi-unit support"""

    def __init__(
        self,
        data: SpikeData,
        event_times: Optional[np.ndarray] = None,
        pre_time: Optional[float] = None,
        post_time: Optional[float] = None,
        marker_size: float = 4,
        marker_color: str = "#2D3142",
        marker_alpha: float = 0.7,
        marker_type: str = "|",
        unit_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            event_times=event_times,
            pre_time=pre_time,
            post_time=post_time,
            **kwargs,
        )

        self.marker_size = marker_size
        self.marker_color = marker_color
        self.marker_alpha = marker_alpha
        self.marker_type = marker_type
        self.unit_id = unit_id or (list(data.spikes.keys())[0] if data.spikes else None)

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        # Check if we have a valid unit ID
        if self.unit_id is None:
            return {
                "data": {
                    "spike_times": [],
                    "y_values": [],
                    "unit_id": None,
                    "is_trial_based": False,
                },
                "params": {
                    "marker_size": self.marker_size,
                    "marker_color": self.marker_color,
                    "marker_alpha": self.marker_alpha,
                    "marker_type": self.marker_type,
                    **self.config,
                },
            }

        # Prepare data based on whether we're in trial-based or continuous mode
        if self.is_trial_based:
            # Trial-based mode - each trial gets its own row (y-value)
            trial_data = self.get_trial_data(self.unit_id)[self.unit_id]

            spike_times = []
            trial_indices = []

            for trial_idx, spikes in trial_data.items():
                spike_times.extend(spikes)
                trial_indices.extend([trial_idx] * len(spikes))

            return {
                "data": {
                    "spike_times": spike_times,
                    "y_values": trial_indices,
                    "unit_id": self.unit_id,
                    "is_trial_based": True,
                    "n_trials": len(trial_data),
                },
                "params": {
                    "marker_size": self.marker_size,
                    "marker_color": self.marker_color,
                    "marker_alpha": self.marker_alpha,
                    "marker_type": self.marker_type,
                    **self.config,
                },
            }

        else:
            # Continuous mode - each unit gets its own row (y-value)
            # For a single unit, all spikes are on the same row
            unit_spikes = self.get_continuous_data(self.unit_id)[self.unit_id]

            return {
                "data": {
                    "spike_times": unit_spikes.tolist(),
                    "y_values": [0]
                    * len(unit_spikes),  # All at position 0 for single unit
                    "unit_id": self.unit_id,
                    "is_trial_based": False,
                },
                "params": {
                    "marker_size": self.marker_size,
                    "marker_color": self.marker_color,
                    "marker_alpha": self.marker_alpha,
                    "marker_type": self.marker_type,
                    **self.config,
                },
            }

    def update(self, **kwargs) -> None:
        """
        Update component parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        # Handle special case for updating event-related parameters
        if "event_times" in kwargs or "pre_time" in kwargs or "post_time" in kwargs:
            # Get current values for missing parameters
            event_times = kwargs.get("event_times", self.event_times)
            pre_time = kwargs.get("pre_time", self.pre_time)
            post_time = kwargs.get("post_time", self.post_time)

            # Update trial-based status
            self.is_trial_based = event_times is not None
            self.event_times = event_times
            self.pre_time = pre_time
            self.post_time = post_time

            # Reorganize data if we're in trial-based mode
            if self.is_trial_based:
                self._trial_data = self._organize_by_trials()

        # Update other parameters
        for key, value in kwargs.items():
            if key not in ["event_times", "pre_time", "post_time"]:
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.config[key] = value

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a plot using the specified backend.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl', 'plotly')
        **kwargs : dict
            Additional parameters for the backend

        Returns
        -------
        Any
            Plot figure from the specified backend
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.raster import render_raster
        elif backend == "plotly":
            from dspant_viz.backends.plotly.raster import render_raster
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_raster(self.get_data(), **kwargs)
