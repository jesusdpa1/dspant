import json
from functools import reduce
from typing import Callable, List, Union

import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field


class MetadataStream(BaseModel):
    """Defines and validates the structure of metadata."""

    name: str
    code: str
    size: int
    type_: str
    type_str: str
    ucf: str
    fs: float
    num_channels: int
    dform: str
    start_time: float
    channel: list
    data_shape: tuple  # (num_channels, samples)

    @classmethod
    def from_json(cls, json_path: str):
        """Load metadata from a JSON file."""
        with open(json_path, "r") as file:
            metadata = json.load(file)
        return cls(**metadata)

    @classmethod
    def from_dict(cls, metadata_dict: dict):
        """Load metadata from a dictionary."""
        return cls(**metadata_dict)


class LarveNode(BaseModel):
    """Strict data structure with validation"""

    fs: float = Field(..., gt=0, description="Sampling frequency in Hz")
    number_of_channels: int = Field(..., gt=0, description="Number of channels")
    number_of_samples: int = Field(..., gt=0, description="Number of samples")
    data_shape: tuple = Field(..., gt=0, description="Data shape (channels, samples)")


class AntNode:
    def __init__(
        self,
        name: str,
        data_path: str,
        chunk_size: Union[int, str] = "auto",  # Can handle both int and str
        metadata: Union[str, dict, None] = None,  # Optional metadata
    ):
        """
        AntNode for handling large time-series datasets.
        - `metadata`: JSON file path or dictionary (optional).
        - `data_path`: Path to the Parquet file.
        - `chunk_size`: Dask array chunk size.
        """
        # Initialize metadata from JSON file or dictionary
        self.metadata = None
        if metadata:
            self.metadata = (
                MetadataStream.from_json(metadata)
                if isinstance(metadata, str)
                else MetadataStream.from_dict(metadata)
            )

        # Initialize other attributes
        self.name = name
        self.data_ref = data_path
        self.chunk_size = chunk_size
        self.data = None  # Lazy-loaded data
        self.data_cache = None
        self.preprocessing_functions: List[Callable] = []
        self.postprocessing_functions: List[Callable] = []

    def load_data(self, force_reload: bool = False):
        """
        Memory-map the Parquet file into a Dask array.
        - `force_reload`: If True, forces reloading of the data.
        """
        if self.data is not None and not force_reload:
            return self.data  # Already loaded

        try:
            with pa.memory_map(self.data_ref, "r") as mmap:
                table = pq.read_table(mmap)
                self.data = da.from_array(
                    table.to_pandas().values, chunks=self.chunk_size
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

        return self.data

    def apply_functions(self, data, functions: List[Callable]):
        """Apply a sequence of functions to the data."""
        return reduce(lambda x, func: func(x), functions, data)

    def get_preprocessed(self, cache: bool = False):
        """
        Apply preprocessing functions and return processed data.
        - `cache`: If True, stores the result in `data_cache` for later use.
        """
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

    def summarize(self):
        """Summarize the AntNode attributes."""
        print(f"Node: {self.name}")
        if self.metadata:
            print(f"Sampling Frequency (fs): {self.metadata.fs} Hz")
            print(f"Duration: {self.metadata.fs * self.metadata.size} seconds")
        else:
            print("No metadata available.")
