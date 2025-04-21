"""
Utility for saving array data to ANT format.

This module provides functions to save NumPy or Dask arrays
with associated metadata into the ANT format data structure.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dspant.core.internals import public_api


@public_api
class ArrayToANT:
    """
    Handles conversion of array data to ANT format.
    Extracts data and metadata from NumPy or Dask arrays.
    """

    def __init__(
        self,
        data: Union[np.ndarray, da.Array],
        metadata: Optional[Dict[str, Any]] = None,
        fs: Optional[float] = None,
        name: Optional[str] = "data",
    ):
        """
        Initialize the ArrayToANT converter.

        Args:
            data: Input data array (samples x channels)
            metadata: Optional metadata dictionary to include
            fs: Sampling frequency in Hz
            name: Name for the dataset
        """
        self.data = data
        self.user_metadata = metadata or {}
        self.fs = fs
        self.name = name

        # Ensure data is in correct format [samples, channels]
        self._validate_data()

        # Initialize metadata containers
        self.base_metadata = {}
        self.other_metadata = {}

    def _validate_data(self):
        """Validates that the data is in correct format and dimensions."""
        # Check if Dask array
        if isinstance(self.data, da.Array):
            # Ensure it's 2D
            if self.data.ndim == 1:
                # Reshape to 2D (samples, 1)
                self.data = self.data.reshape(-1, 1)
            elif self.data.ndim > 2:
                raise ValueError(f"Data array must be 1D or 2D, got {self.data.ndim}D")
        # Check if NumPy array
        elif isinstance(self.data, np.ndarray):
            # Ensure it's 2D
            if self.data.ndim == 1:
                # Reshape to 2D (samples, 1)
                self.data = self.data.reshape(-1, 1)
            elif self.data.ndim > 2:
                raise ValueError(f"Data array must be 1D or 2D, got {self.data.ndim}D")
        else:
            raise TypeError(f"Data must be NumPy or Dask array, got {type(self.data)}")

    def prepare_metadata(self) -> Dict[str, Any]:
        """
        Prepare metadata from the array and user-provided metadata.

        Returns:
            Dictionary containing organized metadata
        """
        # Get array shape
        if isinstance(self.data, da.Array):
            data_shape = self.data.shape
            data_dtype = str(self.data.dtype)
        else:
            data_shape = self.data.shape
            data_dtype = str(self.data.dtype)

        n_samples, n_channels = data_shape

        # Check if user provided metadata has the expected structure
        if "base" in self.user_metadata and "other" in self.user_metadata:
            # Use existing metadata structure
            self.base_metadata = self.user_metadata.get("base", {}).copy()
            self.other_metadata = self.user_metadata.get("other", {}).copy()
            source = self.user_metadata.get("source", "ArrayToANT")

            # Update critical fields based on actual array
            self.base_metadata["number_of_samples"] = n_samples
            self.base_metadata["data_shape"] = [
                n_channels,
                n_samples,
            ]  # Preserve TDT format [channels, samples]

            # Update fs if provided
            if self.fs is not None:
                self.base_metadata["fs"] = float(self.fs)

            # Update name if provided and not already in metadata
            if self.name and "name" not in self.base_metadata:
                self.base_metadata["name"] = self.name

            # Ensure channel information is correct for the array
            # Only update if channel count doesn't match
            if self.base_metadata.get("channel_numbers") != n_channels:
                self.base_metadata["channel_numbers"] = n_channels
                self.base_metadata["channel_names"] = [
                    str(i) for i in range(n_channels)
                ]
                self.base_metadata["channel_types"] = [data_dtype] * n_channels
        else:
            # Build new metadata structure
            source = "ArrayToANT"
            self.base_metadata = {
                "name": self.name,
                "fs": float(self.fs) if self.fs is not None else None,
                "number_of_samples": n_samples,
                "data_shape": [
                    n_channels,
                    n_samples,
                ],  # [channels, samples] to match TDT format
                "channel_numbers": n_channels,
                "channel_names": [str(i) for i in range(n_channels)],
                "channel_types": [data_dtype] * n_channels,
            }

            # Add all user-provided metadata to other_metadata
            self.other_metadata = self.user_metadata.copy()

        # Combine metadata
        metadata = {
            "source": source,
            "base": self.base_metadata,
            "other": self.other_metadata,
        }

        return metadata

    def data_to_parquet(
        self,
        save: bool = False,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Dict[str, Any], pa.Table]:
        """
        Converts array data into a PyArrow Table with embedded metadata.

        Args:
            save: Whether to save the data to disk
            save_path: Path where to save the data

        Returns:
            Tuple of (metadata_dict, data_table)
        """
        # Prepare metadata
        metadata_dict = self.prepare_metadata()
        base_metadata = metadata_dict["base"]
        other_metadata = metadata_dict["other"]

        # Convert metadata for embedding in Parquet
        metadata_parquet = {
            key: json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            for key, value in {**base_metadata, **other_metadata}.items()
        }

        if save and save_path:
            # Convert to Path object if string
            save_path = Path(save_path)

            # Get relative path
            try:
                relative_path = save_path.relative_to(save_path.anchor)
                metadata_parquet["save_path"] = str(relative_path)
            except ValueError:
                # Handle case where path is not relative
                metadata_parquet["save_path"] = str(save_path)

        # Convert data to NumPy array if it's a Dask array
        if isinstance(self.data, da.Array):
            data = self.data.compute()
        else:
            data = self.data

        # Column names as string indices
        column_names = [str(i) for i in range(data.shape[1])]

        # Convert NumPy arrays to PyArrow arrays (transposing to make it channels x samples for saving)
        pa_arrays = [pa.array(data[:, i]) for i in range(data.shape[1])]

        # Create PyArrow Table with metadata
        data_table = pa.Table.from_arrays(
            pa_arrays,
            names=column_names,
            metadata={
                key.encode(): value.encode() for key, value in metadata_parquet.items()
            },
        )

        if save and save_path:
            if save_path.exists():
                # Use save_path.stem for filename
                save_name = save_path.stem

                data_path = save_path / f"data_{save_name}.parquet"
                pq.write_table(data_table, data_path, compression="snappy")
                print(f"✅ Data saved to {data_path}")

                # Save metadata separately
                metadata_path = save_path / f"metadata_{save_name}.json"
                with open(metadata_path, "w") as metadata_file:
                    json.dump(metadata_dict, metadata_file, indent=4)
                print(f"✅ Metadata saved to {metadata_path}")
            else:
                print("❌ Save path does not exist")

        return metadata_dict, data_table


@public_api
def save_to_ant(
    data: Union[np.ndarray, da.Array],
    output_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    fs: Optional[float] = None,
    name: str = "data",
    create_dirs: bool = True,
) -> Path:
    """
    Save array data to ANT format.

    Args:
        data: Input data array (samples x channels)
        output_path: Path where to save the data. If None, uses current directory.
        metadata: Optional metadata dictionary to include
        fs: Sampling frequency in Hz
        name: Name for the dataset
        create_dirs: Whether to create output directories if they don't exist

    Returns:
        Path to the directory containing the saved data
    """
    # Use current directory if output_path is None
    if output_path is None:
        output_path = Path.cwd()

    # Convert to Path object
    output_path = Path(output_path)

    # Create .ant directory if it doesn't end with .ant
    if not output_path.name.endswith(".ant"):
        output_path = output_path / f"{name}.ant"

    # Create directories if they don't exist
    if create_dirs and not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {output_path}")

    # Create converter and save data
    converter = ArrayToANT(data=data, metadata=metadata, fs=fs, name=name)
    converter.data_to_parquet(save=True, save_path=output_path)

    return output_path


@public_api
def save_to_ant_multi(
    data_dict: Dict[str, Union[np.ndarray, da.Array]],
    output_path: Optional[Union[str, Path]] = None,
    metadata_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    fs_dict: Optional[Dict[str, float]] = None,
    create_dirs: bool = True,
) -> Path:
    """
    Save multiple arrays to ANT format in separate directories.

    Args:
        data_dict: Dictionary of named arrays {name: array}
        output_path: Base path where to save the data. If None, uses current directory.
        metadata_dict: Optional dictionary of metadata for each array
        fs_dict: Optional dictionary of sampling frequencies for each array
        create_dirs: Whether to create output directories if they don't exist

    Returns:
        Path to the base directory containing the saved data
    """
    # Use current directory if output_path is None
    if output_path is None:
        output_path = Path.cwd()

    # Convert to Path object
    base_path = Path(output_path)

    # Create base directory if it doesn't exist
    if create_dirs and not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created base directory: {base_path}")

    # Initialize metadata and fs dictionaries if not provided
    metadata_dict = metadata_dict or {}
    fs_dict = fs_dict or {}

    # Process each array
    for name, data in data_dict.items():
        # Get metadata and fs for this array
        metadata = metadata_dict.get(name, {})
        fs = fs_dict.get(name, None)

        # Create path for this array
        array_path = base_path / f"{name}.ant"

        # Save array
        save_to_ant(data, array_path, metadata, fs, name, create_dirs)

    return base_path
