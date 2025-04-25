# core/data_models.py
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class SpikeData(BaseModel):
    """Data model for spike times organized by unit and trial"""

    spikes: Dict[int, Dict[Union[str, int], List[float]]] = Field(
        ..., description="Dictionary mapping unit IDs to trial labels to spike times"
    )
    unit_labels: Optional[Dict[int, str]] = Field(
        None, description="Optional custom labels for units"
    )

    def get_unit_ids(self) -> List[int]:
        """Get available unit IDs"""
        return list(self.spikes.keys())

    def get_unit_spikes(self, unit_id: int) -> Dict[Union[str, int], List[float]]:
        """
        Get spike data for a specific unit

        Parameters
        ----------
        unit_id : int
            ID of the unit to retrieve

        Returns
        -------
        Dict[Union[str, int], List[float]]
            Dictionary mapping trial labels to spike times
        """
        return self.spikes.get(unit_id, {})

    def get_unit_label(self, unit_id: int) -> str:
        """
        Get label for a specific unit

        Parameters
        ----------
        unit_id : int
            ID of the unit to get label for

        Returns
        -------
        str
            Label for the unit, or default if not set
        """
        if self.unit_labels and unit_id in self.unit_labels:
            return self.unit_labels[unit_id]
        return f"Unit {unit_id}"

    def get_trial_count(self, unit_id: int) -> int:
        """
        Get number of trials for a specific unit

        Parameters
        ----------
        unit_id : int
            ID of the unit

        Returns
        -------
        int
            Number of trials for the unit
        """
        unit_data = self.get_unit_spikes(unit_id)
        return len(unit_data)


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
