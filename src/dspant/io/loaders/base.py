"""
Base reader classes for dspant data loading
Provides common functionality for different data format readers
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dask.array as da
import numpy as np
from rich.console import Console
from rich.table import Table

from dspant.core.internals import public_api
from dspant.nodes.base import BaseNode


class BaseReader(BaseNode, ABC):
    """Abstract base class for all data readers in dspant"""

    def __init__(
        self,
        data_path: str,
        chunk_size: Union[int, str] = "auto",
        **kwargs,
    ):
        """
        Initialize the BaseReader.

        Args:
            data_path: Path to the data
            chunk_size: Size of chunks for Dask array or "auto" for automatic sizing
            **kwargs: Additional keyword arguments passed to BaseNode
        """
        super().__init__(data_path=data_path, **kwargs)
        self.chunk_size = chunk_size
        self.data = None

        # Common metadata attributes
        self.name: Optional[str] = None
        self.fs: Optional[float] = None
        self.sampling_frequency: Optional[float] = None
        self.number_of_samples: Optional[int] = None
        self.data_shape: Optional[List[int]] = None
        self.channel_numbers: Optional[int] = None
        self.channel_names: Optional[List[str]] = None
        self.channel_types: Optional[List[str]] = None
        self.dtype: Optional[str] = None

        # Additional metadata storage
        self.properties: Optional[Dict[str, Any]] = None
        self.annotations: Optional[Dict[str, Any]] = None

    @abstractmethod
    def _init_paths(self) -> None:
        """Initialize and validate data paths. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def validate_files(self) -> None:
        """
        Ensure required files exist and are accessible.
        Must be implemented by subclasses.

        Raises:
            FileNotFoundError: If required files are not found
        """
        pass

    @abstractmethod
    def load_metadata(self) -> "BaseReader":
        """
        Load metadata from the data source.
        Must be implemented by subclasses.

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def load_data(self, force_reload: bool = False) -> da.Array:
        """
        Load data into a Dask array.
        Must be implemented by subclasses.

        Args:
            force_reload: Whether to reload data even if already loaded

        Returns:
            Dask array containing the data
        """
        pass

    def _calculate_auto_chunk_size(
        self, n_samples: int, n_channels: int, dtype_size: int
    ) -> int:
        """
        Calculate optimal chunk size based on data characteristics and system resources.

        Args:
            n_samples: Total number of samples
            n_channels: Number of channels
            dtype_size: Size of data type in bytes

        Returns:
            Optimal chunk size in samples
        """
        # Estimate memory per sample row (all channels)
        bytes_per_row = dtype_size * n_channels

        # Target chunk size around 100MB (adjustable)
        target_chunk_size = 100 * 1024 * 1024  # 100MB
        samples_per_chunk = int(target_chunk_size / bytes_per_row)

        # Ensure reasonable parallelism
        import os

        num_cores = os.cpu_count() or 4
        samples_per_chunk = min(
            samples_per_chunk,
            max(1000, n_samples // (num_cores * 2)),  # At least 2 chunks per core
        )

        # Ensure chunk size is reasonable and doesn't exceed dataset size
        chunk_size = min(n_samples, max(1000, samples_per_chunk))

        return chunk_size

    def _update_chunk_size(self, calculated_chunk_size: int) -> None:
        """Update the chunk_size attribute with calculated value"""
        if self.chunk_size == "auto":
            self.chunk_size = calculated_chunk_size

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded data.

        Returns:
            Dictionary containing data information
        """
        info = {
            "data_path": str(self.data_path),
            "name": self.name,
            "sampling_frequency": self.fs or self.sampling_frequency,
            "number_of_samples": self.number_of_samples,
            "data_shape": self.data_shape,
            "channel_numbers": self.channel_numbers,
            "channel_names": self.channel_names,
            "channel_types": self.channel_types,
            "dtype": self.dtype,
            "chunk_size": self.chunk_size,
        }

        # Add data array info if loaded
        if self.data is not None:
            info.update(
                {
                    "data_array_shape": self.data.shape,
                    "data_chunks": self.data.chunks,
                    "data_dtype": str(self.data.dtype),
                    "data_nbytes": self.data.nbytes,
                }
            )

        return info

    def summarize(self) -> None:
        """Print a comprehensive summary of the reader and loaded data"""
        console = Console()

        # Determine reader type from class name
        reader_type = self.__class__.__name__.replace("Reader", "").upper()

        # Create main table
        table = Table(title=f"{reader_type} Reader Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add basic information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row("Reader Type", reader_type)
        table.add_row("Chunk Size", str(self.chunk_size))

        # Add metadata information if available
        if any([self.name, self.fs, self.sampling_frequency]):
            table.add_section()
            if self.name:
                table.add_row("Name", str(self.name))
            if self.fs or self.sampling_frequency:
                fs_value = self.fs or self.sampling_frequency
                table.add_row("Sampling Rate", f"{fs_value} Hz")
            if self.number_of_samples:
                table.add_row("Number of Samples", str(self.number_of_samples))
            if self.data_shape:
                table.add_row("Data Shape", str(self.data_shape))
            if self.channel_numbers:
                table.add_row("Channel Numbers", str(self.channel_numbers))
            if self.dtype:
                table.add_row("Data Type", str(self.dtype))

        # Add channel information if available
        if self.channel_names or self.channel_types:
            table.add_section()
            if self.channel_names:
                # Truncate long channel name lists
                if len(self.channel_names) > 10:
                    displayed_names = (
                        ", ".join(self.channel_names[:10])
                        + f", ... ({len(self.channel_names)} total)"
                    )
                else:
                    displayed_names = ", ".join(self.channel_names)
                table.add_row("Channel Names", displayed_names)
            if self.channel_types:
                unique_types = (
                    list(set(self.channel_types)) if self.channel_types else []
                )
                table.add_row("Channel Types", ", ".join(unique_types))

        # Add data array information if loaded
        if self.data is not None:
            table.add_section()
            table.add_row("Data Array Shape", str(self.data.shape))
            table.add_row("Data Chunks", str(self.data.chunks))
            table.add_row("Data Array Type", str(self.data.dtype))

            # Add memory information
            try:
                nbytes_mb = self.data.nbytes / (1024 * 1024)
                table.add_row("Data Size", f"{nbytes_mb:.1f} MB")
            except:
                table.add_row("Data Size", "Unknown")

        console.print(table)

    def __repr__(self) -> str:
        """String representation of the reader"""
        class_name = self.__class__.__name__
        status = "loaded" if self.data is not None else "not loaded"
        shape_info = f", shape={self.data.shape}" if self.data is not None else ""
        return f"{class_name}(path='{self.data_path}', status={status}{shape_info})"


class StreamReaderMixin:
    """Mixin class providing stream-specific functionality for time-series data readers"""

    def get_traces(
        self,
        start_sample: Optional[int] = None,
        end_sample: Optional[int] = None,
        channel_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Get raw traces data from the loaded Dask array.

        Args:
            start_sample: Start sample index
            end_sample: End sample index
            channel_indices: List of channel indices to load

        Returns:
            Raw traces as numpy array
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Handle slicing
        start_sample = start_sample or 0
        end_sample = end_sample or self.data.shape[0]

        # Create slice objects
        sample_slice = slice(start_sample, end_sample)
        channel_slice = slice(None) if channel_indices is None else channel_indices

        # Extract data and compute (load into memory)
        traces = self.data[sample_slice, channel_slice].compute()

        return traces

    def get_time_vector(
        self, start_sample: Optional[int] = None, end_sample: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate time vector for the specified sample range.

        Args:
            start_sample: Start sample index
            end_sample: End sample index

        Returns:
            Time vector in seconds
        """
        if not (self.fs or self.sampling_frequency):
            raise ValueError("Sampling frequency not available")

        fs = self.fs or self.sampling_frequency
        start_sample = start_sample or 0
        end_sample = end_sample or self.number_of_samples

        n_samples = end_sample - start_sample
        time_vector = (np.arange(n_samples) + start_sample) / fs

        return time_vector

    def get_duration(self) -> Optional[float]:
        """
        Get total duration of the recording in seconds.

        Returns:
            Duration in seconds or None if information not available
        """
        if self.number_of_samples and (self.fs or self.sampling_frequency):
            fs = self.fs or self.sampling_frequency
            return self.number_of_samples / fs
        return None
