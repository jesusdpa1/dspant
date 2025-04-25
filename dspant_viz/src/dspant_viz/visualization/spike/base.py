from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.data_models import SpikeData


class BaseSpikeVisualization(VisualizationComponent, ABC):
    """
    Base class for spike visualization components with common data organization logic.

    This base class provides utilities for organizing spike data in both continuous
    and trial-based formats, allowing derived components to handle both use cases.
    """

    def __init__(
        self,
        data: SpikeData,
        event_times: Optional[np.ndarray] = None,
        pre_time: Optional[float] = None,
        post_time: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the base spike visualization.

        Parameters
        ----------
        data : SpikeData
            Spike data containing spike times for different units
        event_times : ndarray, optional
            Event/trigger times in seconds. If provided, data will be organized around these events.
        pre_time : float, optional
            Time before each event to include (seconds)
        post_time : float, optional
            Time after each event to include (seconds)
        **kwargs : dict
            Additional configuration parameters for the visualization
        """
        super().__init__(data, **kwargs)
        self.event_times = event_times
        self.pre_time = pre_time
        self.post_time = post_time

        # Flag to track if data is in trial-based format
        self.is_trial_based = event_times is not None

        # Store organized data if using trial-based view
        self._trial_data = self._organize_by_trials() if self.is_trial_based else None

    def _organize_by_trials(self) -> Dict[int, Dict[int, List[float]]]:
        """
        Organize continuous spike data into trial-based format.

        Returns
        -------
        Dict[int, Dict[int, List[float]]]
            Nested dictionary: {unit_id: {trial_id: [spike_times]}}
            where spike_times are relative to each event onset
        """
        if (
            self.event_times is None
            or len(self.event_times) == 0
            or self.pre_time is None
            or self.post_time is None
        ):
            raise ValueError("Event times and pre/post time windows must be provided")

        trial_data = {}

        # Process each unit
        for unit_id, spike_times in self.data.spikes.items():
            unit_trials = {}

            # Process each event/trial
            for trial_idx, event_time in enumerate(self.event_times):
                # Define window boundaries
                window_start = event_time - self.pre_time
                window_end = event_time + self.post_time

                # Find spikes within this window
                mask = (spike_times >= window_start) & (spike_times < window_end)
                window_spikes = spike_times[mask]

                # Convert to times relative to event onset
                rel_times = window_spikes - event_time

                # Store for this trial
                unit_trials[trial_idx] = rel_times.tolist()

            # Store trials for this unit
            trial_data[unit_id] = unit_trials

        return trial_data

    def get_continuous_data(
        self, unit_id: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Get data in continuous format.

        Parameters
        ----------
        unit_id : int, optional
            Specific unit ID to retrieve. If None, returns all units.

        Returns
        -------
        Dict[int, np.ndarray]
            Dictionary mapping unit IDs to arrays of spike times
        """
        if unit_id is not None:
            return {unit_id: self.data.spikes.get(unit_id, np.array([]))}

        return self.data.spikes

    def get_trial_data(
        self, unit_id: Optional[int] = None
    ) -> Dict[int, Dict[int, List[float]]]:
        """
        Get data in trial-based format.

        Parameters
        ----------
        unit_id : int, optional
            Specific unit ID to retrieve. If None, returns all units.

        Returns
        -------
        Dict[int, Dict[int, List[float]]]
            Nested dictionary: {unit_id: {trial_id: [spike_times]}}
        """
        if not self.is_trial_based:
            raise ValueError(
                "Data is not in trial-based format. Provide event_times to enable trial-based view."
            )

        if unit_id is not None:
            return {unit_id: self._trial_data.get(unit_id, {})}

        return self._trial_data
