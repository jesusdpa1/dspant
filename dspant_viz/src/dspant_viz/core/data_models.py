# core/data_models.py
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class SpikeData(BaseModel):
    """Data model for spike times organized by units"""

    spikes: Dict[int, np.ndarray] = Field(
        ..., description="Dictionary mapping unit IDs to spike times"
    )
    unit_labels: Optional[Dict[int, str]] = Field(
        None, description="Optional custom labels for units"
    )

    def get_unit_ids(self) -> List[int]:
        """Get available unit IDs"""
        return list(self.spikes.keys())

    def get_unit_spikes(self, unit_id: int) -> np.ndarray:
        """
        Get spike times for a specific unit

        Parameters
        ----------
        unit_id : int
            ID of the unit to retrieve

        Returns
        -------
        np.ndarray
            Array of spike times for the unit
        """
        return self.spikes.get(unit_id, np.array([]))

    class Config:
        arbitrary_types_allowed = True


class PSTHData(BaseModel):
    """Data model for PSTH results"""

    time_bins: List[float] = Field(..., description="Time bin centers in seconds")
    firing_rates: List[float] = Field(..., description="Firing rates in Hz")
    sem: Optional[List[float]] = None
    unit_id: Optional[int] = None
    baseline_window: Optional[Tuple[float, float]] = None


class TimeSeriesData(BaseModel):
    """Data model for time series"""

    times: List[float] = Field(..., description="Time points in seconds")
    values: List[float] = Field(..., description="Signal values")
    sampling_rate: Optional[float] = None
    channel_id: Optional[Union[int, str]] = None
    channel_name: Optional[str] = None


class MultiChannelData(BaseModel):
    """Data model for multi-channel time series"""

    times: List[float] = Field(..., description="Time points in seconds")
    channels: Dict[Union[int, str], List[float]] = Field(
        ..., description="Dictionary mapping channel IDs to signal values"
    )
    sampling_rate: Optional[float] = None
