"""
ZarrReader - Implementation for loading Zarr-based time-series data
Handles SpikeInterface Zarr directories with Dask optimization
"""

import json
import warnings
from pathlib import Path
from typing import Optional, Union

import dask.array as da
import numpy as np
import zarr
import zarrs

from dspant.core.internals import public_api

from .base import BaseReader, StreamReaderMixin

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def is_path_remote(path: str) -> bool:
    """Check if path is remote (s3://, gcs://, etc.)"""
    return path.startswith(("s3://", "gcs://", "http://", "https://"))


def super_zarr_open(
    folder_path: Union[str, Path],
    mode: str = "r",
    storage_options: Optional[dict] = None,
):
    """
    Open a zarr folder with fallback strategies.
    Based on SpikeInterface's super_zarr_open function.
    """

    if mode in ("a", "r+"):
        open_funcs = (zarr.open,)
    else:
        open_funcs = (zarr.open_consolidated, zarr.open)

    if storage_options is None or storage_options == {}:
        storage_options_to_test = ({"anon": True}, {"anon": False})
    else:
        storage_options_to_test = (storage_options,)

    root = None
    exception = None

    if is_path_remote(str(folder_path)):
        for open_func in open_funcs:
            if root is not None:
                break
            for storage_opts in storage_options_to_test:
                try:
                    root = open_func(
                        str(folder_path), mode=mode, storage_options=storage_opts
                    )
                    break
                except Exception as e:
                    exception = e
                    pass
    else:
        if not Path(folder_path).is_dir():
            raise ValueError(f"Folder {folder_path} does not exist")
        for open_func in open_funcs:
            try:
                root = open_func(
                    str(folder_path), mode=mode, storage_options=storage_options
                )
                break
            except Exception as e:
                exception = e
                pass

    if root is None:
        raise ValueError(
            f"Cannot open {folder_path} in mode {mode} with storage_options {storage_options}.\nException: {exception}"
        )
    return root


@public_api
class ZarrReader(BaseReader, StreamReaderMixin):
    """Class for loading and accessing stream data from Zarr format"""

    def __init__(
        self,
        data_path: str,
        chunk_size: Union[int, str] = "auto",
        storage_options: Optional[dict] = None,
        segment_index: int = 0,
        **kwargs,
    ):
        """
        Initialize the ZarrReader.

        Args:
            data_path: Base path to the Zarr directory or direct path to .zarr directory
            chunk_size: Size of chunks for Dask array or "auto" for automatic sizing
            storage_options: Storage options for zarr (e.g., S3 credentials)
            segment_index: Which recording segment to load (default: 0)
            **kwargs: Additional keyword arguments passed to BaseReader
        """
        super().__init__(data_path=data_path, chunk_size=chunk_size, **kwargs)

        # Zarr-specific paths and options
        self.zarr_path: Optional[Path] = None
        self.storage_options = storage_options or {}
        self.segment_index = segment_index

        # Initialize paths
        self._init_paths()

    def _init_paths(self) -> None:
        """Initialize data paths for Zarr files"""
        base_path = Path(self.data_path)

        # Case 1: Direct path to a zarr directory
        if base_path.is_dir() and (
            base_path.suffix == ".zarr" or str(base_path).endswith(".zarr")
        ):
            self.zarr_path = base_path
            return

        # Case 2: Directory containing .zarr subdirectories
        if base_path.is_dir():
            zarr_dirs = list(base_path.glob("*.zarr"))
            if zarr_dirs:
                self.zarr_path = zarr_dirs[0]  # Use first .zarr directory found
                return
            else:
                # Assume the directory itself is a zarr store
                self.zarr_path = base_path
                return

        # Case 3: Path that should be a zarr directory
        if str(base_path).endswith(".zarr"):
            self.zarr_path = base_path
            return

        # If we get here, no valid paths were found
        # Will be caught in validate_files()

    def validate_files(self) -> None:
        """
        Ensure required Zarr files exist and are accessible.

        Raises:
            FileNotFoundError: If required files are not found
        """
        # Check if zarr path was found
        if self.zarr_path is None:
            raise FileNotFoundError(
                f"No Zarr directory found. Searched for:\n"
                f"  - Direct zarr directory: {self.data_path}\n"
                f"  - .zarr subdirectories in: {self.data_path}"
            )

        # Validate zarr directory exists
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr directory does not exist: {self.zarr_path}")

        try:
            # Test if we can open the zarr store
            zarr_root = super_zarr_open(
                self.zarr_path, mode="r", storage_options=self.storage_options
            )

            # Check for SpikeInterface format requirements
            if "channel_ids" not in zarr_root.keys():
                raise ValueError(
                    f"Invalid SpikeInterface Zarr format. 'channel_ids' not found. "
                    f"Available keys: {list(zarr_root.keys())}"
                )

            # Check for required attributes
            required_attrs = ["sampling_frequency", "num_segments"]
            missing_attrs = [
                attr for attr in required_attrs if attr not in zarr_root.attrs
            ]
            if missing_attrs:
                raise ValueError(
                    f"Invalid SpikeInterface Zarr format. Missing attributes: {missing_attrs}"
                )

            # Check if requested segment exists
            trace_name = f"traces_seg{self.segment_index}"
            if trace_name not in zarr_root.keys():
                available_traces = [
                    key for key in zarr_root.keys() if key.startswith("traces_seg")
                ]
                raise ValueError(
                    f"Segment {self.segment_index} not found. Available segments: {available_traces}"
                )

        except Exception as e:
            raise FileNotFoundError(f"Cannot read Zarr store {self.zarr_path}: {e}")

    def load_metadata(self) -> "ZarrReader":
        """
        Load metadata from Zarr attributes and datasets.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If metadata loading fails
        """
        try:
            # Open zarr store
            zarr_root = super_zarr_open(
                self.zarr_path, mode="r", storage_options=self.storage_options
            )

            # Load basic metadata from root attributes
            self.sampling_frequency = float(zarr_root.attrs["sampling_frequency"])
            self.fs = self.sampling_frequency  # Alias for compatibility
            self.num_segments = int(zarr_root.attrs["num_segments"])

            # Load channel information
            channel_ids = zarr_root["channel_ids"][:]
            self.channel_ids = np.array(channel_ids)
            self.channel_numbers = len(self.channel_ids)
            self.channel_names = [str(ch_id) for ch_id in self.channel_ids]

            # Get traces dataset info for the specified segment
            trace_name = f"traces_seg{self.segment_index}"
            traces_dataset = zarr_root[trace_name]
            self.data_shape = list(traces_dataset.shape)
            self.number_of_samples = traces_dataset.shape[0]
            self.dtype = str(traces_dataset.dtype)

            # Load properties if available
            if "properties" in zarr_root:
                properties = {}
                prop_group = zarr_root["properties"]
                for key in prop_group.keys():
                    try:
                        properties[key] = np.array(prop_group[key][:])
                    except Exception as e:
                        warnings.warn(f"Failed to load property '{key}': {e}")

                self.properties = properties

                # Extract channel names and types if available
                if "channel_names" in properties:
                    self.channel_names = [
                        str(name) for name in properties["channel_names"]
                    ]
                if "channel_types" in properties:
                    self.channel_types = [
                        str(type_) for type_ in properties["channel_types"]
                    ]

            # Load annotations if available
            annotations = zarr_root.attrs.get("annotations", {})
            if annotations:
                self.annotations = annotations

        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {self.zarr_path}: {e}")

        return self

    def load_data(self, force_reload: bool = False) -> da.Array:
        """
        Load Zarr data into a Dask array with optimized chunks.

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
            return self._load_from_zarr()
        except Exception as e:
            raise RuntimeError(f"Failed to load Zarr data: {e}") from e

    def _load_from_zarr(self) -> da.Array:
        """Load data from Zarr file with optimized chunking"""
        # Ensure files are validated before loading
        if self.zarr_path is None:
            self.validate_files()

        # Open zarr store and get traces dataset
        zarr_root = super_zarr_open(
            self.zarr_path, mode="r", storage_options=self.storage_options
        )
        trace_name = f"traces_seg{self.segment_index}"
        traces_dataset = zarr_root[trace_name]

        # Update metadata from actual data if not loaded from metadata
        if self.data_shape is None:
            self.data_shape = list(traces_dataset.shape)
        if self.number_of_samples is None:
            self.number_of_samples = traces_dataset.shape[0]
        if self.channel_numbers is None:
            self.channel_numbers = traces_dataset.shape[1]
        if self.dtype is None:
            self.dtype = str(traces_dataset.dtype)

        # Handle chunk size calculation
        if self.chunk_size == "auto":
            # Use existing zarr chunks if available
            if hasattr(traces_dataset, "chunks") and traces_dataset.chunks:
                chunks = traces_dataset.chunks
                chunk_size = chunks[0] if isinstance(chunks, tuple) else chunks
            else:
                # Calculate optimal chunk size
                n_samples, n_channels = traces_dataset.shape
                chunk_size = self._calculate_auto_chunk_size(
                    n_samples, n_channels, traces_dataset.dtype.itemsize
                )
                chunks = (chunk_size, n_channels)

            self._update_chunk_size(chunk_size)

            # Create Dask array from Zarr
            self.data = da.from_zarr(traces_dataset, chunks=chunks)
        else:
            # Use specified chunk size
            chunk_spec = (
                (self.chunk_size, -1)
                if isinstance(self.chunk_size, int)
                else self.chunk_size
            )
            self.data = da.from_zarr(traces_dataset, chunks=chunk_spec)

        return self.data

    def get_zarr_info(self) -> dict:
        """
        Get detailed information about the Zarr store.

        Returns:
            Dictionary with Zarr store information
        """
        if self.zarr_path is None:
            return {}

        try:
            zarr_root = super_zarr_open(
                self.zarr_path, mode="r", storage_options=self.storage_options
            )

            info = {
                "zarr_path": str(self.zarr_path),
                "segment_index": self.segment_index,
                "storage_options": self.storage_options,
                "root_keys": list(zarr_root.keys()),
                "root_attrs": dict(zarr_root.attrs),
            }

            # Add traces dataset info
            trace_name = f"traces_seg{self.segment_index}"
            if trace_name in zarr_root:
                traces_dataset = zarr_root[trace_name]
                info["traces_info"] = {
                    "shape": traces_dataset.shape,
                    "dtype": str(traces_dataset.dtype),
                    "chunks": getattr(traces_dataset, "chunks", None),
                    "compressor": str(getattr(traces_dataset, "compressor", None)),
                    "nbytes": getattr(traces_dataset, "nbytes", None),
                }

            return info
        except Exception as e:
            return {"error": f"Failed to read Zarr info: {e}"}

    def __repr__(self) -> str:
        """String representation of the ZarrReader"""
        status = "loaded" if self.data is not None else "not loaded"
        shape_info = f", shape={self.data.shape}" if self.data is not None else ""
        zarr_name = self.zarr_path.name if self.zarr_path else "unknown"
        return f"ZarrReader(zarr='{zarr_name}', segment={self.segment_index}, status={status}{shape_info})"
