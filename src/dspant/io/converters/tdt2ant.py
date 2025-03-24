"""
Conversion utilities for TDT data to ANT format.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tdt
from pydantic import BaseModel, Field, field_validator

from dspant.core.internals import public_api


@public_api
class drvPathCarpenter(BaseModel):
    """
    Helper class for managing derived data directories.
    Creates consistent directory structures for processed data.
    """

    base_path: Path = Field(..., description="Base directory path for data")
    drv_path: Optional[Path] = None
    drv_sub_path: Optional[Path] = None

    def build_drv_directory(self, drv_base_path: Optional[Path] = None) -> Path:
        """
        Creates a derived directory for data processing.

        Args:
            drv_base_path: Optional base path for derived data. If None, uses base_path

        Returns:
            Path to the created derived directory
        """
        recording_name = self.base_path.name
        parent_name = self.base_path.parent.name

        # Use drv_base_path if provided, else fall back to base_path
        if drv_base_path:
            self.drv_path = drv_base_path / f"{parent_name}" / f"drv_{recording_name}"
        else:
            self.drv_path = self.base_path / f"drv_{recording_name}"

        if not self.drv_path.exists():
            self.drv_path.mkdir(exist_ok=False, parents=True)
            print(f"✅ drv folder created at {self.drv_path}")
        else:
            print(f"✅ drv already exists at {self.drv_path}")

        return self.drv_path

    def build_recording_directory(self, name: str) -> Path:
        """
        Creates a sub-directory for a specific recording.

        Args:
            name: Name of the recording/data source

        Returns:
            Path to the created recording directory
        """
        if self.drv_path is None:
            raise RuntimeError(
                "drv_path is not set. Call `build_drv_directory()` first."
            )

        self.drv_sub_path = self.drv_path / f"{name}.ant"
        if not self.drv_sub_path.exists():
            self.drv_sub_path.mkdir(exist_ok=False)
            print(f"✅ recording folder created at {self.drv_sub_path}")
        else:
            print(f"✅ recording folder already exists at {self.drv_sub_path}")

        return self.drv_sub_path

    @field_validator("base_path", mode="before")
    @classmethod
    def _check_path(cls, path_):
        """Validates that the base path exists."""
        path_ = Path(path_) if not isinstance(path_, Path) else path_
        if path_.exists():
            return path_
        raise ValueError(f"❌ Path does not exist: {path_}")


@public_api
class tdtStream(BaseModel):
    """
    Handles conversion of TDT stream data to ANT format.
    Extracts data and metadata from a TDT stream structure.
    """

    tdt_struct: tdt.StructType = Field(..., description="TDT data structure")
    base_metadata: Optional[Dict[str, Any]] = None
    other_metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def data_to_parquet(
        self,
        save: bool = False,
        save_path: Optional[Path] = None,
        time_segment: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Dict[str, Any], pa.Table]:
        """
        Converts TDT stream data into a PyArrow Table with embedded metadata.

        Args:
            save: Whether to save the data to disk
            save_path: Path where to save the data
            time_segment: Optional tuple of (start_time, end_time) in seconds to extract only a portion of the data.
                          If None, the entire recording is used.

        Returns:
            Tuple of (metadata_dict, data_table)
        """
        # Extract the relevant portion of data based on time_segment
        data = self.tdt_struct.data
        fs = self.tdt_struct.fs

        # Handle single-channel data (1D array)
        if data.ndim == 1:
            # Reshape to 2D (1, samples)
            data = data.reshape(1, -1)
            original_shape = data.shape
        else:
            original_shape = data.shape

        if time_segment is not None:
            start_time, end_time = time_segment
            # Convert time in seconds to sample indices
            start_sample = max(0, int(start_time * fs))
            end_sample = min(data.shape[1], int(end_time * fs))

            # Update data to only include the requested segment
            data = data[:, start_sample:end_sample]

            # Add time segment info to metadata
            segment_info = {
                "original_samples": original_shape[1],
                "segment_start_time": start_time,
                "segment_end_time": end_time,
                "segment_start_sample": start_sample,
                "segment_end_sample": end_sample,
                "segment_duration": end_time - start_time,
            }
        else:
            segment_info = None

        # Get metadata with potentially modified data shape
        metadata_dict = self.metadata_to_dict(save, save_path, data.shape, segment_info)

        base_metadata = metadata_dict["base"]
        other_metadata = metadata_dict["other"]

        # Convert metadata for embedding in Parquet
        metadata_parquet = {
            key: json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            for key, value in {**base_metadata, **other_metadata}.items()
        }

        if save and save_path:
            # Get relative path (remove the home directory)
            relative_path = save_path.relative_to(save_path.anchor)
            metadata_parquet["save_path"] = str(relative_path)

        column_names = [
            str(i) for i in range(data.shape[0])
        ]  # Column names as string indices

        # Convert NumPy arrays to PyArrow arrays
        pa_arrays = [pa.array(data[i, :]) for i in range(data.shape[0])]

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
                # Use save_path.stem to strip the .ant extension
                save_name = save_path.stem  # Strips the .ant extension

                # Add time segment info to filename if provided
                if time_segment is not None:
                    start_str = f"{time_segment[0]:.1f}".replace(".", "_")
                    end_str = f"{time_segment[1]:.1f}".replace(".", "_")
                    save_name = f"{save_name}_t{start_str}-{end_str}"

                data_path = save_path / f"data_{save_name}.parquet"
                pq.write_table(data_table, data_path, compression="snappy")
                print(f"✅ Data saved to {data_path}")
            else:
                print("❌ Save path does not exist")

        return metadata_dict, data_table

    def metadata_to_dict(
        self,
        save: bool = False,
        save_path: Optional[Path] = None,
        data_shape: Optional[Tuple[int, int]] = None,
        segment_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts metadata from a TDT stream and returns it as a dictionary.

        Args:
            save: Whether to save the metadata to disk
            save_path: Path where to save the metadata
            data_shape: The shape of the data being saved (may differ from original if time_segment is used)
            segment_info: Optional information about time segment extraction

        Returns:
            Dictionary containing organized metadata
        """
        # Get the original data for metadata generation
        orig_data = self.tdt_struct.data

        # Handle single-channel data (1D array)
        if orig_data.ndim == 1:
            # Reshape to 2D (1, samples) for consistent handling
            orig_data = orig_data.reshape(1, -1)

        # Use the provided data shape if available, otherwise use the original shape
        shape = data_shape if data_shape is not None else orig_data.shape

        # Get channel information
        # If channel is a scalar or 1D array with one element, adjust accordingly
        channel_info = self.tdt_struct.channel
        if not isinstance(channel_info, (list, np.ndarray)) or len(channel_info) == 1:
            # Convert to list with single element if scalar
            if not isinstance(channel_info, (list, np.ndarray)):
                channel_list = [channel_info]
            else:
                channel_list = list(channel_info)
        else:
            channel_list = list(channel_info)

        self.base_metadata = {
            "name": self.tdt_struct.name,
            "fs": float(self.tdt_struct.fs),
            "number_of_samples": shape[1],
            "data_shape": shape,
            "channel_numbers": len(channel_list),
            "channel_names": [str(ch) for ch in range(shape[0])],
            "channel_types": [str(orig_data[i, :].dtype) for i in range(shape[0])],
        }

        self.other_metadata = {
            "code": int(self.tdt_struct.code),
            "size": int(self.tdt_struct.size),
            "type": int(self.tdt_struct.type),
            "type_str": self.tdt_struct.type_str,
            "ucf": str(self.tdt_struct.ucf),
            "dform": int(self.tdt_struct.dform),
            "start_time": float(self.tdt_struct.start_time),
            "channel": [str(ch) for ch in channel_list],
        }

        # Add segment information if available
        if segment_info:
            self.other_metadata["segment_info"] = segment_info

        # Add relative save path to metadata if save is True
        if save and save_path:
            # Get relative path (remove the home directory)
            relative_path = save_path.relative_to(save_path.anchor)
            self.other_metadata["save_path"] = str(relative_path)

        metadata = {
            "source": type(self.tdt_struct).__name__,
            "base": self.base_metadata,
            "other": self.other_metadata,
        }

        if save and save_path:
            if save_path.exists():
                # Use save_path.stem to strip the .ant extension
                save_name = save_path.stem  # Strips the .ant extension

                # Add time segment info to filename if provided
                if segment_info is not None:
                    start_str = f"{segment_info['segment_start_time']:.1f}".replace(
                        ".", "_"
                    )
                    end_str = f"{segment_info['segment_end_time']:.1f}".replace(
                        ".", "_"
                    )
                    save_name = f"{save_name}_t{start_str}-{end_str}"

                metadata_path = save_path / f"metadata_{save_name}.json"
                with open(metadata_path, "w") as metadata_file:
                    json.dump(metadata, metadata_file, indent=4)
                print(f"✅ Metadata saved to {metadata_path}")
            else:
                print("❌ Save path does not exist")

        return metadata

    @field_validator("tdt_struct", mode="before")
    @classmethod
    def _check_type_str(cls, tdt_struct: Any):
        """Validates that the provided data is a TDT stream."""
        if (
            not isinstance(tdt_struct, tdt.StructType)
            or tdt_struct.type_str != "streams"
        ):
            raise ValueError("Provided data is not a valid TDT stream.")
        return tdt_struct


@public_api
class tdtEpoc(BaseModel):
    """
    Handles conversion of TDT epoc data to ANT format.
    Extracts data and metadata from a TDT epoc structure.
    """

    tdt_struct: tdt.StructType = Field(..., description="TDT data structure")
    base_metadata: Optional[Dict[str, Any]] = None
    other_metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def data_to_parquet(
        self, save: bool = False, save_path: Optional[Path] = None
    ) -> pa.Table:
        """
        Converts TDT epoc data into a PyArrow Table with embedded metadata.

        Args:
            save: Whether to save the data to disk
            save_path: Path where to save the data

        Returns:
            PyArrow table containing the epoc data
        """
        metadata_dict = self.metadata_to_dict(save, save_path)
        base_metadata = metadata_dict["base"]
        other_metadata = metadata_dict["other"]

        # Convert metadata for embedding in Parquet
        metadata_parquet = {
            key: json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            for key, value in {**base_metadata, **other_metadata}.items()
        }

        data = self.tdt_struct.data  # Epoc data
        onset = self.tdt_struct.onset  # Onset timestamps
        offset = self.tdt_struct.offset  # Offset timestamps

        # Convert NumPy arrays to PyArrow arrays
        concat_array = np.vstack([data, onset, offset])

        pa_arrays = [pa.array(concat_array[i, :]) for i in range(concat_array.shape[0])]

        column_names = ["data", "onset", "offset"]

        # Create PyArrow Table with metadata
        data_table = pa.Table.from_arrays(
            pa_arrays,
            names=column_names,
            metadata={
                key.encode(): value.encode() for key, value in metadata_parquet.items()
            },
        )

        if save and save_path:
            # Add relative save path to metadata if provided and save is True
            relative_path = save_path.relative_to(save_path.anchor)
            metadata_parquet["save_path"] = str(relative_path)

            # Strip the .ant extension from the filename
            save_name = save_path.stem  # Strips the .ant extension

            # Save Parquet data
            if save_path.exists():
                data_path = save_path / f"data_{save_name}.parquet"
                pq.write_table(data_table, data_path, compression="snappy")
                print(f"✅ Data saved to {data_path}")
            else:
                print("❌ Save path does not exist")

        return data_table

    def metadata_to_dict(
        self, save: bool = False, save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Returns metadata as a dictionary, adds save path if save is True.

        Args:
            save: Whether to save the metadata to disk
            save_path: Path where to save the metadata

        Returns:
            Dictionary containing organized metadata
        """
        # Validate and collect attributes for base_metadata
        self.base_metadata = {
            "name": self._validate_attribute(self.tdt_struct, "name"),
            "data_shape": self._validate_attribute(self.tdt_struct, "data").shape
            if self._validate_attribute(self.tdt_struct, "data") is not None
            else None,
        }

        # Validate and collect attributes for other_metadata
        onset = self._validate_attribute(self.tdt_struct, "onset")
        offset = self._validate_attribute(self.tdt_struct, "offset")

        self.other_metadata = {
            "name": str(self._validate_attribute(self.tdt_struct, "name")),
            "onset": str(onset.any()) if onset is not None else str(False),
            "offset": str(offset.any()) if offset is not None else str(False),
            "type": str(self._validate_attribute(self.tdt_struct, "type"))
            if self._validate_attribute(self.tdt_struct, "type") is not None
            else None,
            "type_str": self._validate_attribute(self.tdt_struct, "type_str"),
            "dform": int(self._validate_attribute(self.tdt_struct, "dform"))
            if self._validate_attribute(self.tdt_struct, "dform") is not None
            else None,
            "size": int(self._validate_attribute(self.tdt_struct, "size"))
            if self._validate_attribute(self.tdt_struct, "size") is not None
            else None,
        }

        metadata = {
            "source": type(self.tdt_struct).__name__,
            "base": self.base_metadata,
            "other": self.other_metadata,
        }

        if save and save_path:
            # Add relative save path to metadata if save is True
            relative_path = save_path.relative_to(save_path.anchor)
            metadata["save_path"] = str(relative_path)

            # Save metadata to JSON if save is True
            if save_path.exists():
                metadata_path = save_path / f"metadata_{save_path.stem}.json"
                with open(metadata_path, "w") as metadata_file:
                    json.dump(metadata, metadata_file, indent=4)
                print(f"✅ Metadata saved to {metadata_path}")
            else:
                print("❌ Save path does not exist")

        return metadata

    @field_validator("tdt_struct", mode="before")
    @classmethod
    def _check_type_str(cls, tdt_struct: Any):
        """Validates that the provided data is a TDT epoc."""
        if not isinstance(tdt_struct, tdt.StructType) or tdt_struct.type_str != "epocs":
            raise ValueError("Provided data is not a valid TDT epoc.")
        return tdt_struct

    @classmethod
    def _validate_attribute(cls, obj, name):
        """Validates if the attribute exists and is not empty."""
        attribute = getattr(obj, name, None)
        if attribute is not None:
            return attribute
        return None


@public_api
def convert_tdt_to_ant(
    tdt_block_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    stream_names: Optional[List[str]] = None,
    epoc_names: Optional[List[str]] = None,
    time_segment: Optional[Tuple[float, float]] = None,
    start: Optional[Tuple[float, float]] = 0,
    end: Optional[Tuple[float, float]] = -1,
) -> Path:
    """
    Convert TDT block data to ANT format, focusing on streams and epocs.

    Args:
        tdt_block_path: Path to TDT block directory
        output_path: Optional path for output. If None, creates a derived directory
        stream_names: List of stream names to convert. If None, converts all streams
        epoc_names: List of epoc names to convert. If None, converts all epocs
        time_segment: Optional time segment to extract (start_time, end_time) in seconds

    Returns:
        Path to the directory containing the converted data
    """
    # Convert to Path objects
    tdt_block_path = Path(tdt_block_path)
    if output_path is not None:
        output_path = Path(output_path)

    # Read the TDT block (returns a struct with data components)
    print(f"Loading TDT block from {tdt_block_path}...")
    tdt_block = tdt.read_block(str(tdt_block_path), t1=start, t2=end)

    # Create path manager for output
    working_location = drvPathCarpenter(base_path=tdt_block_path)

    # Create derived directory for output
    if output_path is not None:
        # Use provided output path
        drv_path = working_location.build_drv_directory(output_path)
    else:
        # Create default derived directory
        drv_path = working_location.build_drv_directory()

    # Get available streams and epocs
    # Just list the keys provided by the TDT API
    available_streams = (
        tdt_block.streams.keys() if hasattr(tdt_block, "streams") else []
    )
    available_epocs = tdt_block.epocs.keys() if hasattr(tdt_block, "epocs") else []

    # Determine which streams to convert
    if stream_names is None:
        stream_names = list(available_streams)
    else:
        # Check if provided stream names exist
        valid_streams = [name for name in stream_names if name in available_streams]
        invalid_streams = set(stream_names) - set(valid_streams)
        if invalid_streams:
            print(f"⚠️ These streams were not found: {', '.join(invalid_streams)}")
        stream_names = valid_streams

    # Determine which epocs to convert
    if epoc_names is None:
        epoc_names = list(available_epocs)
    else:
        # Check if provided epoc names exist
        valid_epocs = [name for name in epoc_names if name in available_epocs]
        invalid_epocs = set(epoc_names) - set(valid_epocs)
        if invalid_epocs:
            print(f"⚠️ These epocs were not found: {', '.join(invalid_epocs)}")
        epoc_names = valid_epocs

    # Process streams sequentially
    processed_streams = []
    for stream_name in stream_names:
        try:
            print(f"Converting stream: {stream_name}")
            # Create directory for this stream
            stream_dir = working_location.build_recording_directory(stream_name)

            # Access the stream
            stream_obj = tdt_block.streams[stream_name]

            # Create stream converter and save data
            stream_converter = tdtStream(tdt_struct=stream_obj)
            _, _ = stream_converter.data_to_parquet(
                save=True, save_path=stream_dir, time_segment=time_segment
            )
            processed_streams.append(stream_name)
            print(f"✅ Stream {stream_name} converted successfully")
        except Exception as e:
            print(f"❌ Error converting stream '{stream_name}': {str(e)}")

    # Process epocs sequentially
    processed_epocs = []
    for epoc_name in epoc_names:
        try:
            print(f"Converting epoc: {epoc_name}")
            # Create directory for this epoc
            epoc_dir = working_location.build_recording_directory(epoc_name)

            # Access the epoc
            epoc_obj = tdt_block.epocs[epoc_name]

            # Create epoc converter and save data
            epoc_converter = tdtEpoc(tdt_struct=epoc_obj)
            epoc_converter.data_to_parquet(save=True, save_path=epoc_dir)
            processed_epocs.append(epoc_name)
            print(f"✅ Epoc {epoc_name} converted successfully")
        except Exception as e:
            print(f"❌ Error converting epoc '{epoc_name}': {str(e)}")

    # Print summary
    print("\nConversion Summary:")
    print(f"- Processed {len(processed_streams)}/{len(stream_names)} streams")
    print(f"- Processed {len(processed_epocs)}/{len(epoc_names)} epocs")
    print(f"✅ Conversion complete. Data saved to {drv_path}")

    return drv_path
