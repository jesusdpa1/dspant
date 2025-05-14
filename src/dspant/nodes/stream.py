"""
Adding loading zarr file option
Author: Jesus Penaloza
"""

import glob
import json
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from rich.console import Console
from rich.table import Table

from dspant.core.internals import public_api
from dspant.nodes.base import BaseNode


class BaseStreamNode(BaseNode):
    """Base class for handling time-series data"""

    name: Optional[str] = None
    fs: Optional[float] = None
    number_of_samples: Optional[int] = None
    data_shape: Optional[List[int]] = None
    channel_numbers: Optional[int] = None
    channel_names: Optional[List[str]] = None
    channel_types: Optional[List[str]] = None


@public_api
class StreamNode(BaseStreamNode):
    """Class for loading and accessing stream data from Parquet or Zarr formats"""

    def __init__(
        self,
        data_path: str,
        source: Literal["auto", "parquet", "zarr"] = "auto",
        chunk_size: Union[int, str] = "auto",
        **kwargs,
    ):
        """
        Initialize the StreamNode.

        Args:
            data_path: Base path to the data directory
            source: Data source format - "auto", "parquet", or "zarr"
            chunk_size: Size of chunks for Dask array or "auto" for automatic sizing
            **kwargs: Additional keyword arguments passed to BaseNode
        """
        super().__init__(data_path=data_path, **kwargs)
        self.chunk_size = chunk_size
        self.source = source
        self.data = None
        self.zarr_path = None

        # Initialize paths based on the chosen source
        self._init_paths()

    def _init_paths(self):
        """Initialize data paths based on source type"""
        base_path = Path(self.data_path)

        # Check if this is a direct path to a file or directory
        if str(base_path).endswith(".zarr"):
            self.zarr_path = base_path
            if self.source == "auto":
                self.source = "zarr"
            return

        # Check if this is a direct path to a parquet file
        if base_path.is_file() and base_path.suffix == ".parquet":
            self.parquet_path = base_path
            self.source = "parquet"
            return

        # Handle directory paths
        # For .ant directory (parquet)
        if base_path.suffix == ".ant" or base_path.name.endswith(".ant"):
            ant_path = base_path
            # Look for parquet files
            parquet_files = list(ant_path.glob("data_*.parquet"))
            if parquet_files:
                self.parquet_path = parquet_files[0]
                self.metadata_path = (
                    ant_path
                    / f"metadata_{parquet_files[0].stem.replace('data_', '')}.json"
                )
                if self.source == "auto":
                    self.source = "parquet"

        # If source is explicitly set to "zarr" or auto-detection reaches this point
        if self.source == "zarr" or (
            self.source == "auto" and not hasattr(self, "parquet_path")
        ):
            # We're using a zarr source, set the path accordingly
            if str(base_path).endswith(".zarr"):
                self.zarr_path = base_path
            else:
                # Check if there's a .zarr directory with matching name
                potential_zarr = base_path / f"{base_path.stem}.zarr"
                if potential_zarr.exists():
                    self.zarr_path = potential_zarr
                else:
                    # Look in parent directory
                    parent_dir = base_path.parent
                    potential_zarr = parent_dir / f"data_{base_path.stem}.zarr"
                    if potential_zarr.exists():
                        self.zarr_path = potential_zarr

            if self.zarr_path:
                self.source = "zarr"

        # If source is still "auto", set default based on discovered paths
        if self.source == "auto":
            if hasattr(self, "parquet_path") and self.parquet_path:
                self.source = "parquet"
            elif hasattr(self, "zarr_path") and self.zarr_path:
                self.source = "zarr"
            else:
                # Default to parquet if nothing found yet
                self.source = "parquet"

    # Override the validate_files method to handle both Parquet and Zarr sources
    def validate_files(self) -> None:
        """
        Ensure required files exist for the selected source type.
        Sets paths if files are found.

        Raises:
            FileNotFoundError: If required files are not found
        """
        if self.source == "zarr":
            if self.zarr_path is None:
                raise FileNotFoundError(f"No Zarr data found at: {self.data_path}")

            # Check if the Zarr store exists and is valid
            if not Path(self.zarr_path).exists():
                raise FileNotFoundError(f"Zarr path does not exist: {self.zarr_path}")

            # Check if it's a valid Zarr array
            try:
                zarr_array = zarr.open(str(self.zarr_path), mode="r")
                # Try to get metadata from Zarr attributes
                if hasattr(zarr_array, "attrs"):
                    meta_str = zarr_array.attrs.get("metadata")
                    if meta_str:
                        self.metadata = json.loads(meta_str)
                        # Extract key metadata and set attributes
                        if "base" in self.metadata:
                            base_meta = self.metadata["base"]
                            for key, value in base_meta.items():
                                setattr(self, key, value)
            except Exception as e:
                raise FileNotFoundError(f"Invalid Zarr array at {self.zarr_path}: {e}")

        else:
            # Default to the original behavior for Parquet
            import glob

            path_ = Path(self.data_path)

            # Use glob to find matching files
            data_pattern = str(path_ / f"data_*.parquet")
            metadata_pattern = str(path_ / f"metadata_*.json")

            data_files = glob.glob(data_pattern)
            metadata_files = glob.glob(metadata_pattern)

            if not data_files:
                raise FileNotFoundError(f"No data file found matching: {data_pattern}")
            if not metadata_files:
                raise FileNotFoundError(
                    f"No metadata file found matching: {metadata_pattern}"
                )

            # Use the first matching file
            self.parquet_path = data_files[0]
            self.metadata_path = metadata_files[0]

    def load_metadata(self) -> "StreamNode":
        """
        Load metadata appropriate for the selected source type.

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If metadata cannot be found
            json.JSONDecodeError: If metadata contains invalid JSON
        """
        if self.source == "zarr":
            # For Zarr sources, metadata might already be loaded in validate_files
            if self.metadata is None:
                self.validate_files()

            # If still no metadata, try to extract it from Zarr attributes
            if self.metadata is None and self.zarr_path is not None:
                try:
                    zarr_array = zarr.open(str(self.zarr_path), mode="r")
                    if hasattr(zarr_array, "attrs"):
                        meta_str = zarr_array.attrs.get("metadata")
                        if meta_str:
                            self.metadata = json.loads(meta_str)
                            # Extract key metadata
                            if "base" in self.metadata:
                                base_meta = self.metadata["base"]
                                for key, value in base_meta.items():
                                    setattr(self, key, value)
                except Exception as e:
                    print(f"Warning: Failed to load metadata from Zarr: {e}")
        else:
            # Use original method for Parquet
            super().load_metadata()

        return self

    def load_data(self, force_reload: bool = False) -> da.Array:
        """
        Load data into a Dask array with optimized chunks.

        Args:
            force_reload: Whether to reload data even if already loaded

        Returns:
            Dask array containing the data

        Raises:
            RuntimeError: If data loading fails
        """
        if self.data is not None and not force_reload:
            return self.data

        try:
            if self.source == "parquet":
                return self._load_from_parquet()
            elif self.source == "zarr":
                return self._load_from_zarr()
            else:
                raise ValueError(f"Unsupported source type: {self.source}")
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.source}: {e}") from e

    def _load_from_parquet(self) -> da.Array:
        """Load data from Parquet file"""
        # Ensure files are validated before loading
        if self.parquet_path is None:
            self.validate_files()

        with pa.memory_map(str(self.parquet_path), "r") as mmap:
            table = pq.read_table(mmap)
            data_array = table.to_pandas().values

            # Auto-chunk optimization based on data shape and available memory
            if self.chunk_size == "auto":
                # Get data dimensions (samples × channels)
                n_samples, n_channels = data_array.shape

                # Estimate memory per sample row (all channels)
                bytes_per_row = data_array.itemsize * n_channels

                # Choose chunk size to keep chunks around 100MB (adjustable)
                target_chunk_size = 100 * 1024 * 1024  # 100MB
                samples_per_chunk = int(target_chunk_size / bytes_per_row)

                # Ensure reasonable parallelism
                num_cores = os.cpu_count() or 4
                samples_per_chunk = min(
                    samples_per_chunk,
                    max(
                        1000, n_samples // (num_cores * 2)
                    ),  # At least 2 chunks per core
                )

                # Ensure chunk size is at least 1000 samples and doesn't exceed dataset size
                chunk_size = min(n_samples, max(1000, samples_per_chunk))
                self.chunk_size = chunk_size

                # Use samples dimension for chunking, keep channels dimension intact
                self.data = da.from_array(data_array, chunks=(chunk_size, n_channels))
            else:
                # Use specified chunk size
                self.data = da.from_array(data_array, chunks=(self.chunk_size, -1))

        return self.data

    def _load_from_zarr(self) -> da.Array:
        """Load data from Zarr format"""
        if self.zarr_path is None:
            self.validate_files()

        if self.zarr_path is None:
            raise ValueError(
                "Zarr path not found. Check the data path or source setting."
            )

        try:
            # Auto-chunk determination for Zarr
            chunks = "auto"
            if isinstance(self.chunk_size, int):
                # If an integer was specified, use it for chunking samples
                # but keep channel dimension as-is
                chunks = (self.chunk_size, -1)

            # Load as Dask array with optimized chunking
            self.data = da.from_zarr(str(self.zarr_path), chunks=chunks)

            # Update metadata with array info
            if self.data is not None:
                if self.data_shape is None or not isinstance(self.data_shape, list):
                    self.data_shape = list(self.data.shape)
                if self.number_of_samples is None:
                    self.number_of_samples = self.data.shape[0]
                if self.channel_numbers is None:
                    self.channel_numbers = self.data.shape[1]

        except Exception as e:
            raise RuntimeError(f"Failed to load Zarr data: {e}") from e

        return self.data

    def summarize(self):
        """Print a summary of the stream node configuration and metadata"""
        console = Console()

        # Create main table
        table = Table(title=f"Stream Node Summary ({self.source.upper()} Source)")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row("Source Type", self.source.upper())

        if self.source == "parquet":
            table.add_row(
                "Parquet Path",
                str(self.parquet_path)
                if hasattr(self, "parquet_path") and self.parquet_path
                else "Not validated",
            )
            table.add_row(
                "Metadata Path",
                str(self.metadata_path)
                if hasattr(self, "metadata_path") and self.metadata_path
                else "Not validated",
            )
        else:
            table.add_row(
                "Zarr Path",
                str(self.zarr_path) if self.zarr_path else "Not found",
            )

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
