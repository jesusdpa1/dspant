# %%
import json
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
from dask.array import Array
from metadata_classes import tdt_metadata
from pydantic import BaseModel, Field, field_validator
from rich import print
from rich.console import Console
from rich.table import Table
from scipy.signal import butter, filtfilt, sosfiltfilt


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
            **kwargs: Additional keyword arguments passed to AntNode
        """
        super().__init__(data_path=data_path, **kwargs)
        self.chunk_size = chunk_size
        self.data = None
        self.data_cache = None
        self.filters: List[Callable] = []  # List to store filter functions
        self.preprocessing_functions: List[Callable] = []
        self.postprocessing_functions: List[Callable] = []

    def load_data(self, force_reload: bool = False):
        """Memory-map the Parquet file into a Dask array."""
        if self.data is not None and not force_reload:
            return self.data

        try:
            parquet_path_str = str(self.parquet_path)
            with pa.memory_map(parquet_path_str, "r") as mmap:
                table = pq.read_table(mmap)
                self.data = da.from_array(table.to_pandas().values, chunks=100000)
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        return self.data

    def apply_functions(self, data, functions: List[Callable]):
        """Apply a sequence of functions to the data."""
        return reduce(lambda x, func: func(x), functions, data)

    def get_preprocessed(self, cache: bool = False):
        """Apply preprocessing functions and return processed data."""
        if self.data is None:
            self.load_data()

        processed = self.apply_functions(self.data, self.preprocessing_functions)
        if cache:
            self.data_cache = processed
        return processed

    def get_postprocessed(self):
        """Apply postprocessing functions and return processed data."""
        data_to_process = self.data_cache if self.data_cache else self.data
        if data_to_process is None:
            raise RuntimeError("No data available. Load or preprocess the data first.")
        return self.apply_functions(data_to_process, self.postprocessing_functions)

    def add_preprocessing(self, func: Callable):
        """Add a preprocessing function to the pipeline."""
        if not callable(func):
            raise TypeError("Preprocessing function must be callable")
        self.preprocessing_functions.append(func)

    def add_postprocessing(self, func: Callable):
        """Add a postprocessing function to the pipeline."""
        if not callable(func):
            raise TypeError("Postprocessing function must be callable")
        self.postprocessing_functions.append(func)

    def apply_filters(self):
        """
        Apply filter functions lazily using map_blocks, automatically adding fs (sampling rate) to each filter.
        """
        if not hasattr(self, "fs"):
            raise AttributeError("Sampling rate (fs) must be defined in the WorkerNode")

        # Apply each filter function lazily using map_blocks
        for func in self.filters:
            data = self.data.map_blocks(
                func,
                fs=int(self.fs),
                dtype=self.data.dtype,
            )
        return data


def bandpass_filter(data, lowcut: float, highcut: float, order: int = 4, fs=None):
    """
    Bandpass filter that dynamically accesses the sampling rate (fs).

    Args:
        data: Input data array
        lowcut (float): Lower frequency cutoff in Hz
        highcut (float): Higher frequency cutoff in Hz
        order (int): Filter order
        fs: Sampling rate (Hz)

    Returns:
        filtered_data: Filtered data array
    """
    if fs is None:
        raise ValueError("Sampling rate (fs) must be provided")

    # Compute the Nyquist frequency
    nyquist = 0.5 * fs

    # Normalize the cutoff frequencies by the Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Create the second-order sections (SOS) representation of the filter
    sos = butter(order, [low, high], btype="bandpass", output="sos")

    # Apply the filter using sosfiltfilt
    return sosfiltfilt(sos, data)


# %%

folder_name = r"../data/RawG.ant"

# Pass the folder_name as a keyword argument for 'data_path'
test = AntNode(data_path=folder_name)
test.load_metadata()
# %%
test.summarize()

# %%
data_ = WorkerNode(data_path=folder_name)
data_.load_metadata()
data_.summarize()
# %%
data_.load_data()

# %%
my_filter = partial(bandpass_filter, lowcut=10, highcut=2000)
# %%
# Add filter to filters list
data_.filters.append(my_filter)

# Apply filters lazily using map_blocks
filtered_data = data_.apply_filters()
filtered_data.compute()
# %%
