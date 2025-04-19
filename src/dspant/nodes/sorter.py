import json
from pathlib import Path
from typing import Dict, List, Optional, Self, Union

import numpy as np

from .base import BaseNode


class BaseSorterNode(BaseNode):
    """Base class for handling sorted spike data"""

    name: Optional[str] = None
    sampling_frequency: Optional[float] = None
    unit_ids: Optional[List[int]] = None


class SorterNode(BaseSorterNode):
    """Class for loading and accessing sorted spike data"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.data = None
        self.spike_times = None
        self.spike_clusters = None
        self.unit_properties = {}
        self._spike_indices = {}  # For efficient access to spikes by unit ID

    def load_data(self, force_reload: bool = False) -> bool:
        """
        Load spike times and cluster data

        Args:
            force_reload: Whether to reload even if data is already loaded

        Returns:
            True if data was loaded successfully
        """
        if self.data is not None and not force_reload:
            return True

        try:
            # Ensure files are validated before loading
            if self.parquet_path is None:
                self.validate_files()

            # Load metadata if not loaded
            if self.metadata is None:
                self.load_metadata()

            # In a real implementation, we would load data from parquet,
            # but for now we'll just set placeholders
            self.data = True  # Dummy value to indicate data is loaded

            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load sorter data: {e}") from e

    def get_unit_spike_train(
        self,
        unit_id: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get spike times for a specific unit

        Args:
            unit_id: Unit ID to retrieve
            start_frame: Optional start frame for filtering spikes
            end_frame: Optional end frame for filtering spikes

        Returns:
            Array of spike times in samples
        """
        if self.spike_times is None or self.spike_clusters is None:
            raise RuntimeError("Spike data not loaded. Call load_data() first.")

        if unit_id not in self.unit_ids:
            raise ValueError(f"Unit ID {unit_id} not found")

        # Use cached indices if available
        if unit_id not in self._spike_indices:
            self._spike_indices[unit_id] = np.where(self.spike_clusters == unit_id)[0]

        unit_indices = self._spike_indices[unit_id]
        unit_spikes = self.spike_times[unit_indices]

        # Apply time filter if requested
        if start_frame is not None or end_frame is not None:
            start = 0 if start_frame is None else start_frame
            end = np.inf if end_frame is None else end_frame

            mask = (unit_spikes >= start) & (unit_spikes < end)
            unit_spikes = unit_spikes[mask]

        return unit_spikes

    def summarize(self):
        """Print a summary of the sorter node configuration and metadata"""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Create main table
        table = Table(title="Sorter Node Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row(
            "Parquet Path",
            str(self.parquet_path) if self.parquet_path else "Not validated",
        )
        table.add_row(
            "Metadata Path",
            str(self.metadata_path) if self.metadata_path else "Not validated",
        )

        # Add metadata information
        if self.metadata:
            table.add_section()
            table.add_row("Name", str(self.name))
            table.add_row(
                "Sampling Rate",
                f"{self.sampling_frequency} Hz"
                if self.sampling_frequency
                else "Not set",
            )

            if self.unit_ids:
                table.add_row("Number of Units", str(len(self.unit_ids)))

            # Add unit properties summary if available
            if self.unit_properties:
                props_str = ", ".join(list(self.unit_properties.keys()))
                table.add_row("Unit Properties", props_str)

        # Add data information if loaded
        if self.spike_times is not None and self.spike_clusters is not None:
            table.add_section()
            table.add_row("Number of Spikes", str(len(self.spike_times)))

            # Calculate mean firing rates if possible
            if self.sampling_frequency:
                duration_s = np.max(self.spike_times) / self.sampling_frequency
                spikes_per_unit = {
                    uid: np.sum(self.spike_clusters == uid) for uid in self.unit_ids
                }

                mean_fr = np.mean(
                    [spikes / duration_s for spikes in spikes_per_unit.values()]
                )
                max_fr = np.max(
                    [spikes / duration_s for spikes in spikes_per_unit.values()]
                )

                table.add_row("Recording Duration", f"{duration_s:.2f} s")
                table.add_row("Mean Firing Rate", f"{mean_fr:.2f} Hz")
                table.add_row("Max Firing Rate", f"{max_fr:.2f} Hz")

        console.print(table)
