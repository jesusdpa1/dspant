"""
Base classes for spike analytics in dspant neuroproc.

This module provides base classes for various spike train transformations
and analysis methods, establishing a consistent interface for all spike
analytics components.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np

from dspant.nodes.sorter import SorterNode


class BaseSpikeTransform(ABC):
    """
    Base class for all spike train transformations and analyses.

    This abstract class defines the common interface for classes that
    transform spike train data into other representations or metrics.
    """

    @abstractmethod
    def transform(
        self,
        sorter: SorterNode,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        unit_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> Tuple:
        """
        Transform spike data from a sorter node.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        start_time_s : float
            Start time for analysis in seconds
        end_time_s : float or None
            End time for analysis in seconds. If None, use the end of the recording.
        unit_ids : list of int or None
            Units to include. If None, use all units.
        **kwargs : dict
            Additional keyword arguments for specific transformations

        Returns
        -------
        Tuple
            Transformation results, specific to each implementation
        """
        pass

    def _validate_sorter(self, sorter: SorterNode) -> None:
        """
        Validate that the sorter node has required properties.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode to validate

        Raises
        ------
        ValueError
            If the sorter node lacks required properties
        """
        if sorter.sampling_frequency is None:
            raise ValueError("Sorter node must have sampling frequency set")

        if sorter.spike_times is None or sorter.spike_clusters is None:
            raise ValueError("Sorter node must have spike data loaded")

    def _get_time_range_samples(
        self, sorter: SorterNode, start_time_s: float, end_time_s: Optional[float]
    ) -> Tuple[int, int]:
        """
        Convert time range from seconds to samples.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        start_time_s : float
            Start time in seconds
        end_time_s : float or None
            End time in seconds. If None, use the end of the recording.

        Returns
        -------
        start_frame : int
            Start time in samples
        end_frame : int
            End time in samples
        """
        sampling_rate = sorter.sampling_frequency

        # Convert times to samples
        start_frame = int(start_time_s * sampling_rate)

        if end_time_s is None:
            # Use the last spike time as the end time
            end_frame = int(np.max(sorter.spike_times))
        else:
            end_frame = int(end_time_s * sampling_rate)

        return start_frame, end_frame

    def _get_filtered_unit_ids(
        self, sorter: SorterNode, unit_ids: Optional[List[int]]
    ) -> List[int]:
        """
        Get filtered list of unit IDs.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        unit_ids : list of int or None
            Units to include. If None, use all units.

        Returns
        -------
        List[int]
            Filtered list of unit IDs
        """
        if unit_ids is None:
            return sorter.unit_ids

        # Filter to only include units that exist in the sorter
        return [u for u in unit_ids if u in sorter.unit_ids]
