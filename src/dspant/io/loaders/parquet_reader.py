"""
ParquetReader - Implementation for loading Parquet-based time-series data
Handles .ant directories and standalone .parquet files with Dask optimization
"""

import glob
import json
from pathlib import Path
from typing import Optional, Union

import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq

from dspant.core.internals import public_api

from .base import BaseReader, StreamReaderMixin


@public_api
class ParquetReader(BaseReader, StreamReaderMixin):
    """Class for loading and accessing stream data from Parquet format"""

    def __init__(
        self,
        data_path: str,
        chunk_size: Union[int, str] = "auto",
        **kwargs,
    ):
        """
        Initialize the ParquetReader.

        Args:
            data_path: Base path to the data directory or direct path to .parquet file
            chunk_size: Size of chunks for Dask array or "auto" for automatic sizing
            **kwargs: Additional keyword arguments passed to BaseReader
        """
        super().__init__(data_path=data_path, chunk_size=chunk_size, **kwargs)

        # Parquet-specific paths
        self.parquet_path: Optional[Path] = None
        self.metadata_path: Optional[Path] = None

        # Initialize paths
        self._init_paths()

    def _init_paths(self) -> None:
        """Initialize data paths for Parquet files"""
        base_path = Path(self.data_path)

        # Case 1: Direct path to a parquet file
        if base_path.is_file() and base_path.suffix == ".parquet":
            self.parquet_path = base_path
            self._find_metadata_file(base_path)
            return

        # Case 2: .ant directory (dspant convention)
        if base_path.suffix == ".ant" or base_path.name.endswith(".ant"):
            self._handle_ant_directory(base_path)
            return

        # Case 3: Regular directory with data_*.parquet pattern
        if base_path.is_dir():
            self._handle_regular_directory(base_path)
            return

        # If we get here, no valid paths were found
        # Will be caught in validate_files()

    def _find_metadata_file(self, parquet_path: Path) -> None:
        """Find corresponding metadata file for a parquet file"""
        metadata_candidates = [
            # Same name with .json extension
            parquet_path.with_suffix(".json"),
            # metadata_<stem>.json pattern
            parquet_path.parent / f"metadata_{parquet_path.stem}.json",
            # metadata_<stem_without_data_prefix>.json pattern
            parquet_path.parent
            / f"metadata_{parquet_path.stem.replace('data_', '')}.json",
        ]

        for candidate in metadata_candidates:
            if candidate.exists():
                self.metadata_path = candidate
                break

    def _handle_ant_directory(self, ant_path: Path) -> None:
        """Handle .ant directory structure"""
        # Look for parquet files in .ant directory
        parquet_files = list(ant_path.glob("data_*.parquet"))
        if parquet_files:
            # Use the first parquet file found
            self.parquet_path = parquet_files[0]
            # Look for corresponding metadata
            self.metadata_path = (
                ant_path
                / f"metadata_{self.parquet_path.stem.replace('data_', '')}.json"
            )

    def _handle_regular_directory(self, directory: Path) -> None:
        """Handle regular directory with data_*.parquet pattern"""
        parquet_files = list(directory.glob("data_*.parquet"))
        if parquet_files:
            # Use the first parquet file found
            self.parquet_path = parquet_files[0]
            self._find_metadata_file(self.parquet_path)

    def validate_files(self) -> None:
        """
        Ensure required Parquet files exist and are accessible.

        Raises:
            FileNotFoundError: If required files are not found
        """
        # Check if parquet path was found
        if self.parquet_path is None:
            # Try glob search as fallback
            path_ = Path(self.data_path)
            data_pattern = str(path_ / "data_*.parquet")
            data_files = glob.glob(data_pattern)

            if not data_files:
                raise FileNotFoundError(
                    f"No Parquet data file found. Searched for:\n"
                    f"  - Direct file: {self.data_path}\n"
                    f"  - Pattern: {data_pattern}\n"
                    f"  - .ant directory structure"
                )

            self.parquet_path = Path(data_files[0])
            self._find_metadata_file(self.parquet_path)

        # Validate parquet file exists and is accessible
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file does not exist: {self.parquet_path}")

        try:
            # Test if we can read the parquet file
            with pa.memory_map(str(self.parquet_path), "r") as mmap:
                table = pq.read_table(mmap, columns=[])  # Read just metadata
        except Exception as e:
            raise FileNotFoundError(
                f"Cannot read Parquet file {self.parquet_path}: {e}"
            )

        # Metadata file is optional but warn if not found
        if self.metadata_path is None or not self.metadata_path.exists():
            import warnings

            warnings.warn(f"No metadata file found for {self.parquet_path}")

    def load_metadata(self) -> "ParquetReader":
        """
        Load metadata from JSON file if available.

        Returns:
            Self for method chaining

        Raises:
            json.JSONDecodeError: If metadata contains invalid JSON
        """
        if self.metadata_path and self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)

                # Extract common metadata fields
                if isinstance(self.metadata, dict):
                    # Handle nested metadata structure
                    base_meta = self.metadata.get("base", self.metadata)

                    # Map common fields
                    field_mapping = {
                        "name": "name",
                        "fs": "fs",
                        "sampling_frequency": "sampling_frequency",
                        "number_of_samples": "number_of_samples",
                        "data_shape": "data_shape",
                        "channel_numbers": "channel_numbers",
                        "channel_names": "channel_names",
                        "channel_types": "channel_types",
                        "dtype": "dtype",
                    }

                    for meta_key, attr_key in field_mapping.items():
                        if meta_key in base_meta:
                            setattr(self, attr_key, base_meta[meta_key])

                    # Handle alternative field names
                    if "fs" not in base_meta and "sampling_frequency" in base_meta:
                        self.fs = base_meta["sampling_frequency"]
                    elif "sampling_frequency" not in base_meta and "fs" in base_meta:
                        self.sampling_frequency = base_meta["fs"]

            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in metadata file {self.metadata_path}: {e}"
                )
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to load metadata from {self.metadata_path}: {e}")

        return self

    def load_data(self, force_reload: bool = False) -> da.Array:
        """
        Load Parquet data into a Dask array with optimized chunks.

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
            return self._load_from_parquet()
        except Exception as e:
            raise RuntimeError(f"Failed to load Parquet data: {e}") from e

    def _load_from_parquet(self) -> da.Array:
        """Load data from Parquet file with optimized chunking"""
        # Ensure files are validated before loading
        if self.parquet_path is None:
            self.validate_files()

        with pa.memory_map(str(self.parquet_path), "r") as mmap:
            table = pq.read_table(mmap)
            data_array = table.to_pandas().values

            # Update metadata from actual data if not loaded from metadata file
            if self.data_shape is None:
                self.data_shape = list(data_array.shape)
            if self.number_of_samples is None:
                self.number_of_samples = data_array.shape[0]
            if self.channel_numbers is None:
                self.channel_numbers = data_array.shape[1]
            if self.dtype is None:
                self.dtype = str(data_array.dtype)

            # Handle chunk size calculation
            if self.chunk_size == "auto":
                n_samples, n_channels = data_array.shape
                chunk_size = self._calculate_auto_chunk_size(
                    n_samples, n_channels, data_array.itemsize
                )
                self._update_chunk_size(chunk_size)

                # Create Dask array with calculated chunks
                self.data = da.from_array(data_array, chunks=(chunk_size, n_channels))
            else:
                # Use specified chunk size
                chunk_spec = (
                    (self.chunk_size, -1)
                    if isinstance(self.chunk_size, int)
                    else self.chunk_size
                )
                self.data = da.from_array(data_array, chunks=chunk_spec)

        return self.data

    def get_parquet_info(self) -> dict:
        """
        Get detailed information about the Parquet file.

        Returns:
            Dictionary with Parquet file information
        """
        if self.parquet_path is None:
            return {}

        try:
            with pa.memory_map(str(self.parquet_path), "r") as mmap:
                parquet_file = pq.ParquetFile(mmap)

                info = {
                    "file_path": str(self.parquet_path),
                    "metadata_path": str(self.metadata_path)
                    if self.metadata_path
                    else None,
                    "num_rows": parquet_file.metadata.num_rows,
                    "num_columns": parquet_file.metadata.num_columns,
                    "num_row_groups": parquet_file.metadata.num_row_groups,
                    "file_size_bytes": self.parquet_path.stat().st_size,
                    "schema": str(parquet_file.schema),
                }

                return info
        except Exception as e:
            return {"error": f"Failed to read Parquet info: {e}"}

    def __repr__(self) -> str:
        """String representation of the ParquetReader"""
        status = "loaded" if self.data is not None else "not loaded"
        shape_info = f", shape={self.data.shape}" if self.data is not None else ""
        parquet_file = self.parquet_path.name if self.parquet_path else "unknown"
        return f"ParquetReader(file='{parquet_file}', status={status}{shape_info})"
