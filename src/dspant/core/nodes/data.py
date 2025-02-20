from typing import List, Optional, Union

import dask.array as da
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from .loading import BaseNode


class BaseStreamNode(BaseNode):
    """Base class for handling time-series data"""

    name: Optional[str] = None
    fs: Optional[float] = None
    number_of_samples: Optional[int] = None
    data_shape: Optional[List[int]] = None
    channel_numbers: Optional[int] = None
    channel_names: Optional[List[str]] = None
    channel_types: Optional[List[str]] = None


class StreamNode(BaseStreamNode):
    """Class for loading and accessing stream data"""

    def __init__(self, data_path: str, chunk_size: Union[int, str] = "auto", **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.chunk_size = chunk_size
        self.data = None

    def load_data(self, force_reload: bool = False) -> da.Array:
        """Load data into a Dask array"""
        if self.data is not None and not force_reload:
            return self.data

        try:
            with pa.memory_map(str(self.parquet_path), "r") as mmap:
                table = pq.read_table(mmap)
                self.data = da.from_array(
                    table.to_pandas().values, chunks=self.chunk_size
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}") from e

        return self.data

    def summarize(self):
        """Print a summary of the stream node configuration and metadata"""
        console = Console()

        # Create main table
        table = Table(title="Stream Node Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row("Parquet Path", str(self.parquet_path))
        table.add_row("Metadata Path", str(self.metadata_path))
        table.add_row("Chunk Size", str(self.chunk_size))

        # Add metadata information
        if self.metadata:
            table.add_section()
            table.add_row("Name", str(self.name))
            table.add_row("Sampling Rate", f"{self.fs} Hz" if self.fs else "Not set")
            table.add_row("Number of Samples", str(self.number_of_samples))
            table.add_row("Data Shape", str(self.data_shape))
            table.add_row("Channel Numbers", str(self.channel_numbers))

            if self.channel_names:
                table.add_row("Channel Names", ", ".join(self.channel_names))
            if self.channel_types:
                table.add_row("Channel Types", ", ".join(self.channel_types))

        # Add data information if loaded
        if self.data is not None:
            table.add_section()
            table.add_row("Data Array Shape", str(self.data.shape))
            table.add_row("Data Chunks", str(self.data.chunks))
            table.add_row("Data Type", str(self.data.dtype))

        console.print(table)


class BaseEpocNode(BaseNode):
    """Base class for handling event-based data"""

    name: Optional[str] = None


class EpocNode(BaseEpocNode):
    """Class for loading and accessing epoch data"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.data = None

    def load_data(self, force_reload: bool = False) -> pl.DataFrame:
        """Load data into a Polars DataFrame"""
        if self.data is not None and not force_reload:
            return self.data

        try:
            self.data = pl.read_parquet(str(self.parquet_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load epoch data: {e}") from e

        return self.data

    def summarize(self):
        """Print a summary of the epoch node configuration and metadata"""
        console = Console()

        # Create main table
        table = Table(title="Epoch Node Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row("Parquet Path", str(self.parquet_path))
        table.add_row("Metadata Path", str(self.metadata_path))

        # Add metadata information
        if self.metadata:
            table.add_section()
            table.add_row("Name", str(self.name))

            # Add any other metadata fields from base
            for key, value in self.metadata.get("base", {}).items():
                if key not in ["data_path", "metadata", "name"]:
                    table.add_row(key, str(value))

        # Add data information if loaded
        if self.data is not None:
            table.add_section()
            table.add_row("Number of Events", str(len(self.data)))
            table.add_row("Columns", ", ".join(self.data.columns))
            table.add_row("Data Schema", str(self.data.schema))

        console.print(table)
