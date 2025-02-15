import json
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator
from rich import print
from rich.console import Console
from rich.table import Table


class TKEOMethod(Enum):
    """Enumeration of TKEO calculation methods"""

    LI_2007 = auto()  # Li et al. 2007 (2 samples)
    DEBURCHGRAVE_2008 = auto()  # Deburchgrave et al. 2008 (4 samples)
    ORIGINAL = auto()  # Original Teager-Kaiser method


class AntNode(BaseModel):
    """Handles data paths and validates metadata for AntNode."""

    data_path: str = Field(..., description="Parent folder for data storage")
    parquet_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Make metadata optional

    # Predefined attributes (with Optional to handle dynamic cases)
    name: Optional[str] = None
    fs: Optional[float] = None
    number_of_samples: Optional[int] = None
    data_shape: Optional[list] = None
    channel_numbers: Optional[int] = None
    channel_names: Optional[list] = None
    channel_types: Optional[list] = None

    # Define a class field to allow dynamic attributes
    class Config:
        # Allow extra attributes to be set dynamically
        extra = "allow"

    @field_validator("data_path")
    def validate_data_path(cls, value):
        """Ensures the data path has a valid `.ant` extension."""
        path_ = Path(value)
        if not path_.suffix == ".ant":
            raise ValueError("Data path must have a .ant extension")
        return value

    def validate_files(self):
        """Checks for the existence of both `.json` and `.parquet` files."""
        path_ = Path(self.data_path)
        self.parquet_path = path_.joinpath(f"{path_.stem}_data.parquet")
        self.metadata_path = path_.joinpath(f"{path_.stem}_metadata.json")

        if not self.parquet_path.exists():
            print("Warning: Parquet data file is missing")
        else:
            print("✅ Data file found")

        if not self.metadata_path.exists():
            print("Warning: JSON metadata file is missing")
        else:
            print("✅ Metadata file found")

    def load_metadata(self):
        """Loads metadata from a JSON file and validates the 'base' section."""
        self.validate_files()
        metadata_file = self.metadata_path  # Use the metadata_path attribute

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file {self.metadata_path} not found.")

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        # Dynamically set attributes from metadata
        base_metadata = metadata.get("base", {})
        for key, value in base_metadata.items():
            setattr(self, key, value)

        self.metadata = metadata  # Store full metadata
        print("✅ Metadata loaded successfully")

    def summarize(self):
        """Prints a summary of the metadata and data location."""
        console = Console()

        # Create a rich table to print the base attributes
        table = Table(title="AntNode Metadata Summary")
        table.add_column("Attribute", justify="right")
        table.add_column("Value", justify="left")

        # Add dynamically assigned attributes to the table
        for key, value in self.__dict__.items():
            if key != "metadata" and key != "data_path":  # Avoid printing these two
                table.add_row(key, str(value))

        # Print the summary table
        console.print(table)


class WorkerNode(AntNode):
    """Class used to load streams: a stream is constituted by continuous time series."""

    def __init__(self, data_path: str, chunk_size: Union[int, str] = "auto", **kwargs):
        """
        AntNode for handling large time-series datasets.
        Args:
            data_path: Path to the Parquet file.
            chunk_size: Dask array chunk size.
            **kwargs: Additional keyword arguments passed to AntNode.
        """
        super().__init__(data_path=data_path, **kwargs)
        self.chunk_size = chunk_size
        self.data = None
        self.filters: List[Callable] = []  # List to store filter functions

    def load_data(self, force_reload: bool = False):
        """Memory-map the Parquet file into a Dask array."""
        if self.data is not None and not force_reload:
            return self.data

        try:
            parquet_path_str = str(self.parquet_path)
            with pa.memory_map(parquet_path_str, "r") as mmap:
                table = pq.read_table(mmap)
                self.data = da.from_array(
                    table.to_pandas().values, chunks=self.chunk_size
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        return self.data

    def apply_filters(self):
        """
        Apply filter functions lazily using map_overlap, ensuring the correct overlap calculation.
        """
        if not hasattr(self, "fs"):
            raise AttributeError("Sampling rate (fs) must be defined in the WorkerNode")

        if not self.filters:
            return self.data  # No filters to apply

        # Simplified overlap calculation
        min_f_low = min(f.lowcut for f in self.filters if hasattr(f, "lowcut"))
        k = 5  # Number of cycles needed for stabilization
        overlap_samples = int(k * self.fs / min_f_low)  # Simplified overlap calculation

        data = self.data
        for func in self.filters:
            data = data.map_overlap(
                func,
                depth=(overlap_samples, 0),  # Apply overlap in the sample axis
                boundary="reflect",
                fs=int(self.fs),
                dtype=data.dtype,
            )
        return data

    def apply_function(self):
        """
        Apply filters (if available) and then the TKEO function.
        """
        if not hasattr(self, "fs"):
            raise AttributeError("Sampling rate (fs) must be defined in the WorkerNode")

        data = self.data
        if self.filters:
            data = self.apply_filters()

        if self.tkeo_func:
            data = data.map_overlap(
                self.tkeo_func,
                depth=(2, 0)
                if self.tkeo_func.tkeo_method == TKEOMethod.LI_2007
                else (4, 0)
                if self.tkeo_func.tkeo_method == TKEOMethod.DEBURCHGRAVE_2008
                else (1, 0),
                boundary="reflect",
            )

        return data


class CarpenterNode(BaseModel):
    # this model doesn't require .parquet to be available only json
    "Class used to load anything else that helps analyze the data: epocs, snips, scalars"

    data_path: str = Field(..., gt=0, description="data location")
    metadata_path: str = Field(default=...)
