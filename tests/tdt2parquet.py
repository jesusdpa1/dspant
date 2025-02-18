# %%
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tdt
from pydantic import BaseModel, Field, field_validator


# %%
class drvPathCarpenter(BaseModel):
    base_path: Path = Field(..., description="Base directory path for data")
    drv_path: Path = None
    drv_sub_path: Path = None

    def build_drv_directory(self, drv_base_path=None):
        """Creates a derived directory for data processing."""
        recording_name = self.base_path.name
        parent_name = self.base_path.parent.name
        print()
        # Use drv_base_path if provided, else fall back to base_path
        if drv_base_path:
            self.drv_path = drv_base_path / f"{parent_name}" / f"drv_{recording_name}"
        else:
            self.drv_path = self.base_path / f"drv_{recording_name}"

        if not self.drv_path.exists():
            self.drv_path.mkdir(
                exist_ok=False, parents=True
            )  # Will raise an error if it already exists
            print(f"✅ drv folder created at {self.drv_path}")
        else:
            print(f"✅ drv already exists at {self.drv_path}")

    def build_recording_directory(self, name: str):
        """Creates a sub-directory for a specific recording."""
        if self.drv_path is None:
            raise RuntimeError(
                "drv_path is not set. Call `build_drv_directory()` first."
            )

        self.drv_sub_path = self.drv_path / f"{name}.ant"  # Ensure assignment
        if not self.drv_sub_path.exists():
            self.drv_sub_path.mkdir(
                exist_ok=False
            )  # Will raise an error if it already exists
            print(f"✅ recording folder created at {self.drv_sub_path}")
        else:
            print(f"✅ recording folder already exists at {self.drv_sub_path}")

    @field_validator("base_path", mode="before")
    @classmethod
    def _check_path(cls, path_):
        """Validates that the base path exists."""
        path_ = Path(path_) if not isinstance(path_, Path) else path_
        if path_.exists():
            return path_
        raise ValueError(f"❌ Path does not exist: {path_}")


# %%
class tdtStream(BaseModel):
    tdt_struct: tdt.StructType = Field(..., description="TDT data structure")
    base_metadata: Optional[Dict[str, Any]] = None
    other_metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def data_to_parquet(self, save: bool = False, save_path: Path = None) -> pa.Table:
        """Converts TDT stream data into a PyArrow Table with embedded metadata."""

        metadata_dict = self.metadata_to_dict(save, save_path)

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

        data = self.tdt_struct.data  # Corrected reference
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

        if save:
            if save_path.exists():
                data_path = save_path / f"data_{save_path.name}.parquet"
                pq.write_table(data_table, data_path, compression="snappy")
                print(f"✅ Data saved to {data_path}")
            else:
                print("❌ Save path does not exist")

        return metadata_dict, data_table

    def metadata_to_dict(
        self, save: bool = False, save_path: Path = None
    ) -> Dict[str, Any]:
        """Extracts metadata from a TDT stream and returns it as a dictionary."""
        self.base_metadata = {
            "name": self.tdt_struct.name,
            "fs": float(self.tdt_struct.fs),
            "number_of_samples": self.tdt_struct.data.shape[1],
            "data_shape": self.tdt_struct.data.shape,
            "channel_numbers": len(self.tdt_struct.channel),
            "channel_names": [str(ch) for ch in range(self.tdt_struct.data.shape[0])],
            "channel_types": [
                str(self.tdt_struct.data[i, :].dtype)
                for i in range(self.tdt_struct.data.shape[0])
            ],
        }

        self.other_metadata = {
            "code": int(self.tdt_struct.code),
            "size": int(self.tdt_struct.size),
            "type": int(self.tdt_struct.type),
            "type_str": self.tdt_struct.type_str,
            "ucf": str(self.tdt_struct.ucf),
            "dform": int(self.tdt_struct.dform),
            "start_time": float(self.tdt_struct.start_time),
            "channel": [str(ch) for ch in self.tdt_struct.channel],
        }

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

        if save:
            if save_path.exists():
                metadata_path = save_path / f"metadata_{save_path.name}.json"
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


# %%
class tdtEpoc(BaseModel):
    tdt_struct: tdt.StructType = Field(..., description="TDT data structure")
    base_metadata: Optional[Dict[str, Any]] = None
    other_metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    def data_to_parquet(self) -> pa.Table:
        """Converts TDT stream data into a PyArrow Table with embedded metadata."""

        metadata_dict = self.metadata_to_dict()  # Correct function call
        base_metadata = metadata_dict["base"]
        other_metadata = metadata_dict["other"]

        # Convert metadata for embedding in Parquet
        metadata_parquet = {
            key: json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            for key, value in {**base_metadata, **other_metadata}.items()
        }

        data = self.tdt_struct.data  # Corrected reference
        onset = self.tdt_struct.onset
        offset = self.tdt_struct.offset

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
        return data_table

    def metadata_to_dict(self):
        """Returns metadata as a dictionary."""

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
            "onset": onset.any() if onset is not None else False,
            "offset": offset.any() if offset is not None else False,
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

        return {
            "source": type(self.tdt_struct).__name__,
            "base": self.base_metadata,
            "other": self.other_metadata,
        }

    @field_validator("tdt_struct", mode="before")
    @classmethod
    def _check_type_str(cls, tdt_struct: Any):
        """Validates that the provided data is a TDT stream."""
        if not isinstance(tdt_struct, tdt.StructType) or tdt_struct.type_str != "epocs":
            raise ValueError("Provided data is not a valid TDT stream.")
        return tdt_struct

    @classmethod
    def _validate_attribute(cls, obj, name):
        """Validates if the attribute exists and is not empty."""
        attribute = getattr(obj, name, None)
        if attribute is not None:
            return attribute
        return None


# %% Set up paths


def ls(directory: Path):
    entries = [entry for entry in directory.iterdir()]
    for i, entry in enumerate(entries, start=1):
        entry_type = "File" if entry.is_file() else "Directory"
        print(f"{i}. {entry.name} ({entry_type})")


home = Path(r"E:\jpenalozaa")
tank_path = home.joinpath(r"emgContusion\25-02-12_9882-1_testSubject_emgContusion")
# %%
ls(tank_path)
# %%
block_path = tank_path.joinpath("16-49-56_stim")
working_location = drvPathCarpenter(base_path=block_path)


# %%
working_location.build_drv_directory(Path("../data"))

# %% Read the block
test = tdt.read_block(str(block_path))
# %%
working_location.build_recording_directory("RawG")
a = tdtStream(tdt_struct=test.streams.RawG)
k, w = a.data_to_parquet(save=True, save_path=working_location.drv_sub_path)

# %%
b = tdtEpoc(tdt_struct=test.epocs.AmpA)
b.data_to_parquet()

# %%
# Select the stream you're interested in
tdt_type = "streams"
stream = "RawG"

# Extract relevant metadata from the stream
source = "tdt"

base_metadata = {
    "name": test[tdt_type][stream].name,
    "fs": float(
        test[tdt_type][stream].fs
    ),  # Ensure float for proper numerical representation
    "number_of_samples": test[tdt_type][stream].data.shape[1],
    "data_shape": test[tdt_type][stream].data.shape,
    "channel_numbers": len(test[tdt_type][stream].channel),
    "channel_names": [
        str(ch) for ch in np.arange(test[tdt_type][stream].data.shape[0])
    ],
    "channel_types": [
        str(test[tdt_type][stream].data[i, :].dtype)
        for i in np.arange(test[tdt_type][stream].data.shape[0])
    ],
}

other_metadata = {
    "code": int(test[tdt_type][stream].code),  # Convert np.uint32 to int
    "size": int(test[tdt_type][stream].size),
    "type": int(test[tdt_type][stream].type),
    "type_str": test[tdt_type][stream].type_str,
    "ucf": str(test[tdt_type][stream].ucf),  # Keep bool as string for consistency
    "dform": int(test[tdt_type][stream].dform),
    "start_time": float(
        test[tdt_type][stream].start_time
    ),  # Convert np.float64 to float
    "channel": [
        str(ch) for ch in test[tdt_type][stream].channel
    ],  # Convert to list of strings
}
# Combine into final JSON structure
metadata_json = {
    "source": source,
    "base": base_metadata,
    "other": other_metadata,
}

# %% Create the folder
folder_name = f"../data/{test[tdt_type][stream].name}.ant"
folder_path = Path(folder_name)
folder_path.mkdir(parents=True, exist_ok=True)

# Define file paths
metadata_path = folder_path / f"{stream}_metadata.json"
parquet_path = folder_path / f"{stream}_data.parquet"

# Save metadata to JSON file
with open(metadata_path, "w") as metadata_file:
    json.dump(metadata_json, metadata_file, indent=4)

# %% Prepare Parquet data
data = test.streams[stream].data
column_names = [str(i) for i in range(len(data))]  # Column names as string indices

# Convert metadata for embedding in Parquet
metadata_parquet = {
    key: json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    for key, value in {**base_metadata, **other_metadata}.items()
}

# Create PyArrow Table with metadata
data_table = pa.Table.from_arrays(
    data,
    names=column_names,
    metadata={key.encode(): value.encode() for key, value in metadata_parquet.items()},
)

# Write Parquet file
pq.write_table(data_table, parquet_path, compression="snappy")

print(f"✅ Metadata saved to {metadata_path}")
print(f"✅ Data saved to {parquet_path}")


# %%
