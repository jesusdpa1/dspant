# %%
import json
from enum import Enum, auto
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, field_validator
from rich import print
from rich.console import Console
from rich.table import Table
from scipy.signal import butter, sosfiltfilt

# %%


class TKEOMethod(Enum):
    """Enumeration of TKEO calculation methods"""

    LI_2007 = auto()  # Li et al. 2007 (2 samples)
    DEBURCHGRAVE_2008 = auto()  # Deburchgrave et al. 2008 (4 samples)
    ORIGINAL = auto()  # Original Teager-Kaiser method


class TKEO:
    def __init__(self, tkeo_method: TKEOMethod):
        self.tkeo_method = tkeo_method

    def __call__(self, traces):
        if self.tkeo_method == TKEOMethod.LI_2007:
            # Li et al. 2007 method (2 samples)
            return np.abs(traces[:-2] * (traces[1:-1] ** 2 - traces[:-2] * traces[2:]))

        elif self.tkeo_method == TKEOMethod.DEBURCHGRAVE_2008:
            # Deburchgrave et al. 2008 method (4 samples)
            result = np.zeros_like(traces)
            result[2:-2] = traces[2:-2] * (
                traces[3:-1] ** 2 - traces[2:-2] * traces[4:]
            )
            return np.abs(result)

        elif self.tkeo_method == TKEOMethod.ORIGINAL:
            # Original Teager-Kaiser method
            result = np.zeros_like(traces)
            result[1:-1] = traces[1:-1] ** 2 - traces[:-2] * traces[2:]
            return np.abs(result)


class BaseButterFilter:
    def __init__(self, order=4):
        self.order = order

    def __call__(self, chunk, fs):
        raise NotImplementedError("Subclass must implement __call__")


class BandPassFilter(BaseButterFilter):
    def __init__(self, lowcut, highcut, order=4):
        super().__init__(order)
        self.lowcut = lowcut
        self.highcut = highcut

    def __call__(self, chunk, fs):
        sos = butter(
            self.order,
            [self.lowcut / (0.5 * fs), self.highcut / (0.5 * fs)],
            btype="bandpass",
            analog=False,
            output="sos",
        )
        return sosfiltfilt(sos, chunk, axis=0)


class LowPassFilter(BaseButterFilter):
    def __init__(self, cutoff, order=4):
        super().__init__(order)
        self.cutoff = cutoff

    def __call__(self, chunk, fs):
        sos = butter(
            self.order,
            self.cutoff / (0.5 * fs),
            btype="lowpass",
            analog=False,
            output="sos",
        )
        return sosfiltfilt(sos, chunk, axis=0)


class HighPassFilter(BaseButterFilter):
    def __init__(self, cutoff, order=4):
        super().__init__(order)
        self.cutoff = cutoff

    def __call__(self, chunk, fs):
        sos = butter(
            self.order,
            self.cutoff / (0.5 * fs),
            btype="highpass",
            analog=False,
            output="sos",
        )
        return sosfiltfilt(sos, chunk, axis=0)


class NotchFilter(BaseButterFilter):
    def __init__(self, notch_freq, q=30, order=4):
        super().__init__(order)
        self.notch_freq = notch_freq
        self.q = q

    def __call__(self, chunk, fs):
        nyquist = 0.5 * fs
        low = (self.notch_freq - 1) / nyquist
        high = (self.notch_freq + 1) / nyquist
        sos = butter(4, [low, high], btype="bandstop", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)


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


# %%

folder_name = r"../data/RawG.ant"

# %%
data_ = WorkerNode(data_path=folder_name, chunk_size="auto")
data_.load_metadata()
data_.summarize()
# %%
data_.load_data()

# %%
# Define filter parameters
lowcut = 100.0  # Lower cutoff frequency in Hz
highcut = 2000.0  # Upper cutoff frequency in Hz
order = 4  # Filter order
notch_freq = 60  # Notch frequency in Hz

# Create filter instances
notch_filter = NotchFilter(notch_freq)
butter_filter = BandPassFilter(lowcut, highcut, order)

# Add filters to WorkerNode
data_.filters.append(notch_filter)
data_.filters.append(butter_filter)
data_filtered = data_.apply_filters()
# %%
# Define TKEO parameters
tkeo_method = TKEOMethod.DEBURCHGRAVE_2008
# Create TKEO instance
tkeo = TKEO(tkeo_method)

# Add TKEO to WorkerNode
data_.tkeo_func = tkeo

# Apply filters and TKEO
result = data_.apply_function()


# %%
import matplotlib.pyplot as plt

# %%
plt.plot(data_filtered[1000000:1100000, 1].compute())
plt.plot(result[1000000:1100000, 1].compute() * 100000000)

# %%
# filtered_data.visualize(engine="cytoscape", optimize_graph=True)

# %%
