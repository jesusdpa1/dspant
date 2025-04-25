# src/dspant_viz/visualization/stream/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.internals import public_api


@public_api(module_override="dspant_viz.visualization")
class BaseStreamVisualization(VisualizationComponent, ABC):
    """
    Base class for time series visualization components with common functionality.

    This base class provides utilities for handling time windows, channel/unit selection,
    and data extraction that are common across different time series visualizations.
    """

    def __init__(
        self,
        data: Union[da.Array, Dict[int, np.ndarray]],
        sampling_rate: float,
        elements: Optional[List[int]] = None,  # Generic name for selectable elements
        time_window: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        """
        Initialize the base stream visualization.

        Parameters
        ----------
        data : Union[da.Array, Dict[int, np.ndarray]]
            Either:
            - Dask array with data in the format (samples × elements)
            - Dictionary mapping element IDs to time series data
        sampling_rate : float
            Sampling frequency in Hz
        elements : list of int, optional
            Specific elements to display. If None, uses all available.
        time_window : tuple of (float, float), optional
            Time window to display (start_time, end_time) in seconds
        **kwargs : dict
            Additional configuration parameters
        """
        super().__init__(data, **kwargs)
        self.sampling_rate = sampling_rate
        self.time_window = time_window

        # Set up element IDs based on data type
        if isinstance(data, da.Array):
            # Dask array - elements are dimensions
            available_elements = data.shape[1] if data.ndim > 1 else 1
            all_elements = list(range(available_elements))
        elif isinstance(data, dict):
            # Dictionary - keys are element IDs
            all_elements = list(data.keys())
        else:
            raise TypeError("Data must be either Dask array or dictionary")

        # Set elements to display
        self.elements = elements if elements is not None else all_elements

    def _get_element_data(self, element_id: int) -> np.ndarray:
        """
        Get data for a specific element.

        Parameters
        ----------
        element_id : int
            Element ID to retrieve

        Returns
        -------
        np.ndarray
            Time series data for the specified element
        """
        if isinstance(self.data, da.Array):
            # Extract from Dask array
            if self.data.ndim == 1:
                return self.data.compute()
            else:
                if 0 <= element_id < self.data.shape[1]:
                    return self.data[:, element_id].compute()
                else:
                    return np.array([])
        elif isinstance(self.data, dict):
            # Extract from dictionary
            return self.data.get(element_id, np.array([]))
        else:
            return np.array([])

    def _get_time_array(self, data_length: int = None) -> np.ndarray:
        """
        Get time array based on data length and sampling rate.

        Parameters
        ----------
        data_length : int, optional
            Length of data array. If None, determines from data.

        Returns
        -------
        np.ndarray
            Time values in seconds
        """
        if data_length is None:
            if isinstance(self.data, da.Array):
                data_length = self.data.shape[0]
            elif isinstance(self.data, dict) and self.elements:
                # Use first available element for length
                first_element = self._get_element_data(self.elements[0])
                data_length = len(first_element)
            else:
                data_length = 0

        return np.arange(data_length) / self.sampling_rate

    def _apply_time_window(
        self, data: np.ndarray, time: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply time window to data and time arrays.

        Parameters
        ----------
        data : np.ndarray
            Data array to filter
        time : np.ndarray
            Time array to filter

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered data and time arrays
        """
        if self.time_window is None:
            return data, time

        start, end = self.time_window
        mask = (time >= start) & (time <= end)
        return data[mask], time[mask]

    def _downsample_if_needed(
        self, data: np.ndarray, time: np.ndarray, max_points: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Downsample data if it exceeds max_points.

        Parameters
        ----------
        data : np.ndarray
            Data array to downsample
        time : np.ndarray
            Time array to downsample
        max_points : int
            Maximum number of points to display

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Downsampled data and time arrays
        """
        if len(time) <= max_points:
            return data, time

        # Calculate downsampling factor
        downsample_factor = len(time) // max_points

        # Apply downsampling
        return data[::downsample_factor], time[::downsample_factor]
