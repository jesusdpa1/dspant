import json
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq
from metadata_classes import tdt_metadata
from pydantic import BaseModel, Field, field_validator
from rich import print
from rich.console import Console
from rich.table import Table


class CarpenterNode(BaseModel):
    # this model doesn't require .parquet to be available only json
    "Class used to load anything else that helps analyze the data: epocs, snips, scalars"

    data_path: str = Field(..., gt=0, description="data location")
    metadata_path: str = Field(default=...)


class WorkerNode(AntNode):
    "Class use to load streams: a stream is constitude by continuous time series"

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
