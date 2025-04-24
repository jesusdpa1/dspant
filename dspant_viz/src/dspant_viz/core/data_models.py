# core/data_models.py
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field


class SpikeData(BaseModel):
    """Data model for spike times organized by groups"""
    spikes: Dict[Union[str, int], List[float]] = Field(
        ...,
        description="Dictionary mapping labels (trial IDs, neuron IDs, etc.) to spike times"
    )
    unit_id: Optional[int] = None

    def flatten(self) -> Tuple[List[float], List[Union[int, str]], Dict[int, str]]:
        """
        Flatten the spike data for rendering

        Returns:
            spike_times: List of all spike times
            y_values: List of numeric y-values for each spike
            y_labels: Dictionary mapping y-values to original labels
        """
        spike_times = []
        y_values = []
        y_labels = {}

        for i, (label, times) in enumerate(self.spikes.items()):
            spike_times.extend(times)
            y_values.extend([i] * len(times))
            y_labels[i] = str(label)

        return spike_times, y_values, y_labels


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
        ...,
        description="Dictionary mapping channel IDs to signal values"
    )
    sampling_rate: Optional[float] = None
