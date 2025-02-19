# %%
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.console import Console
from rich.table import Table
from scipy.signal import butter, sosfiltfilt


# %%
class BaseNode(BaseModel):
    """Base class for handling data paths and metadata"""

    data_path: str = Field(..., description="Parent folder for data storage")
    parquet_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("data_path")
    def validate_data_path(cls, value):
        path_ = Path(value)
        if not path_.suffix == ".ant":
            raise ValueError("Data path must have a .ant extension")
        return str(path_.resolve())

    def validate_files(self):
        """Ensure required data and metadata files exist"""
        path_ = Path(self.data_path)
        self.parquet_path = path_ / f"data_{path_.stem}.parquet"
        self.metadata_path = path_ / f"metadata_{path_.stem}.json"

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.parquet_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

    def load_metadata(self):
        """Load metadata from file"""
        self.validate_files()
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        base_metadata = metadata.get("base", {})
        for key, value in base_metadata.items():
            setattr(self, key, value)

        self.metadata = metadata


class BaseStreamNode(BaseNode):
    """Base class for handling time-series data"""

    name: Optional[str] = None
    fs: Optional[float] = None
    number_of_samples: Optional[int] = None
    data_shape: Optional[List[int]] = None
    channel_numbers: Optional[int] = None
    channel_names: Optional[List[str]] = None
    channel_types: Optional[List[str]] = None


class BaseEpocNode(BaseNode):
    """Base class for handling event-based data"""

    name: Optional[str] = None


class EpocNode(BaseEpocNode):
    """Enhanced class for handling event-based epoch data"""

    def __init__(self, data_path: str, name: Optional[str] = None, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.name = name or f"epoc_node_{id(self)}"
        self.data: Optional[pl.DataFrame] = None
        self._is_active = True
        self._last_modified = datetime.now()
        self._operation_history: List[str] = []

    @property
    def is_active(self) -> bool:
        """Check if the epoc node is active"""
        return self._is_active

    @property
    def last_modified(self) -> datetime:
        """Get the last modification timestamp"""
        return self._last_modified

    def check_availability(self) -> Dict[str, Any]:
        """Check node availability and return status information"""
        status = {
            "available": self.is_active,
            "name": self.name,
            "last_modified": self.last_modified,
            "data_loaded": self.data is not None,
            "data_path": str(self.data_path),
        }

        if not self.is_active:
            status["reason"] = "Node has been deactivated"

        return status

    def load_data(self, force_reload: bool = False) -> pl.DataFrame:
        """Load epoch data from Parquet file"""
        if not self.is_active:
            raise RuntimeError(f"Epoc node '{self.name}' is not active")

        if self.data is not None and not force_reload:
            return self.data

        try:
            self.data = pl.read_parquet(str(self.parquet_path))
            self._last_modified = datetime.now()
            self._operation_history.append(f"Data loaded at {self._last_modified}")
            return self.data
        except Exception as e:
            raise RuntimeError(f"Failed to load epoch data: {e}")

    def get_data(self) -> Optional[pl.DataFrame]:
        """Get the current data frame"""
        if not self.is_active:
            raise RuntimeError(f"Epoc node '{self.name}' is not active")
        return self.data

    def deactivate(self) -> None:
        """Deactivate the epoc node"""
        self._is_active = False
        self._last_modified = datetime.now()
        self._operation_history.append(f"Node deactivated at {self._last_modified}")

    def reactivate(self) -> None:
        """Reactivate the epoc node"""
        self._is_active = True
        self._last_modified = datetime.now()
        self._operation_history.append(f"Node reactivated at {self._last_modified}")

    def get_history(self) -> List[str]:
        """Get the operation history"""
        return self._operation_history.copy()

    def summarize(self):
        """Print a comprehensive summary of the epoc node"""
        console = Console()

        table = Table(title=f"Epoc Node Summary: {self.name}")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Node status
        table.add_section()
        table.add_row("Status", "Active" if self.is_active else "Inactive")
        table.add_row("Last Modified", str(self.last_modified))
        table.add_row("Data Path", str(self.data_path))
        table.add_row("Data Loaded", "Yes" if self.data is not None else "No")

        # Data information
        if self.data is not None:
            table.add_section()
            table.add_row("Number of Events", str(len(self.data)))
            table.add_row("Columns", ", ".join(self.data.columns))
            table.add_row("Data Schema", str(self.data.schema))

        # Metadata
        if self.metadata:
            table.add_section()
            for key, value in self.metadata.get("base", {}).items():
                if key not in ["data_path", "metadata"]:
                    table.add_row(key, str(value))

        # Recent history
        if self._operation_history:
            table.add_section()
            table.add_row("Recent Operations", "\n".join(self._operation_history[-5:]))

        console.print(table)


class StreamNode(BaseStreamNode):
    """Class for loading and accessing stream data"""

    def __init__(self, data_path: str, chunk_size: Union[int, str] = "auto", **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.chunk_size = chunk_size
        self.data = None

    def load_data(self, force_reload: bool = False) -> da.Array:
        """Load data into a Dask array"""
        if self.data is not None and not force_reload:
            return self.data

        try:
            with pa.memory_map(str(self.parquet_path), "r") as mmap:
                table = pq.read_table(mmap)
                self.data = da.from_array(
                    table.to_pandas().values, chunks=self.chunk_size
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}") from e

        return self.data

    def summarize(self):
        """Print a summary of the stream node configuration and metadata"""
        console = Console()

        # Create main table
        table = Table(title="Stream Node Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row("Parquet Path", str(self.parquet_path))
        table.add_row("Metadata Path", str(self.metadata_path))
        table.add_row("Chunk Size", str(self.chunk_size))

        # Add metadata information
        if self.metadata:
            table.add_section()
            table.add_row("Name", str(self.name))
            table.add_row("Sampling Rate", f"{self.fs} Hz" if self.fs else "Not set")
            table.add_row("Number of Samples", str(self.number_of_samples))
            table.add_row("Data Shape", str(self.data_shape))
            table.add_row("Channel Numbers", str(self.channel_numbers))

            if self.channel_names:
                table.add_row("Channel Names", ", ".join(self.channel_names))
            if self.channel_types:
                table.add_row("Channel Types", ", ".join(self.channel_types))

        # Add data information if loaded
        if self.data is not None:
            table.add_section()
            table.add_row("Data Array Shape", str(self.data.shape))
            table.add_row("Data Chunks", str(self.data.chunks))
            table.add_row("Data Type", str(self.data.dtype))

        console.print(table)


class BaseProcessor(ABC):
    """Abstract base class for all processors"""

    @abstractmethod
    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """Process the input data"""
        pass

    @property
    @abstractmethod
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        pass


class ProcessingFunction(Protocol):
    """Protocol defining the interface for processing functions"""

    def __call__(
        self, data: np.ndarray, fs: Optional[float] = None, **kwargs
    ) -> np.ndarray: ...


class ProcessingPipeline:
    """Class for managing a sequence of processors"""

    def __init__(self):
        self.processors: List[BaseProcessor] = []

    def add_processor(self, processor: BaseProcessor) -> None:
        """Add a processor to the pipeline"""
        self.processors.append(processor)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """Apply all processors in sequence"""
        result = data
        for processor in self.processors:
            result = processor.process(result, fs=fs, **kwargs)
        return result


class ProcessingNode:
    """Class for applying processing to a StreamNode"""

    def __init__(self, stream_node: StreamNode, name: Optional[str] = None):
        self.stream_node = stream_node
        self.pipeline = ProcessingPipeline()
        self.name = name or f"processing_node_{id(self)}"
        self._is_active = True
        self._last_modified = datetime.now()
        self._processor_history: List[str] = []

    @property
    def is_active(self) -> bool:
        """Check if the processing node is active and available"""
        return self._is_active

    @property
    def last_modified(self) -> datetime:
        """Get the last modification timestamp"""
        return self._last_modified

    def check_availability(self) -> Dict[str, Any]:
        """
        Check if the processing node is available for use or modification
        Returns a dictionary with status information
        """
        status = {
            "available": self.is_active,
            "name": self.name,
            "last_modified": self.last_modified,
            "processor_count": len(self.pipeline.processors),
            "stream_node_path": str(self.stream_node.data_path),
        }

        if not self.is_active:
            status["reason"] = "Node has been deactivated"

        return status

    def add_processor(
        self, processor: BaseProcessor, position: Optional[int] = None
    ) -> None:
        """
        Add a processor to the pipeline at the specified position
        If position is None, append to the end
        """
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        if position is not None:
            if position < 0 or position > len(self.pipeline.processors):
                raise ValueError(f"Invalid position: {position}")
            self.pipeline.processors.insert(position, processor)
        else:
            self.pipeline.processors.append(processor)

        self._last_modified = datetime.now()
        self._processor_history.append(
            f"Added {processor.__class__.__name__} at {self._last_modified}"
        )

    def remove_processor(self, index: int) -> Optional[BaseProcessor]:
        """Remove a processor at the specified index"""
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        if 0 <= index < len(self.pipeline.processors):
            processor = self.pipeline.processors.pop(index)
            self._last_modified = datetime.now()
            self._processor_history.append(
                f"Removed {processor.__class__.__name__} at {self._last_modified}"
            )
            return processor
        return None

    def clear_processors(self) -> None:
        """Remove all processors from the pipeline"""
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        self.pipeline.processors.clear()
        self._last_modified = datetime.now()
        self._processor_history.append(
            f"Cleared all processors at {self._last_modified}"
        )

    def deactivate(self) -> None:
        """Deactivate the processing node"""
        self._is_active = False
        self._last_modified = datetime.now()
        self._processor_history.append(f"Node deactivated at {self._last_modified}")

    def reactivate(self) -> None:
        """Reactivate the processing node"""
        self._is_active = True
        self._last_modified = datetime.now()
        self._processor_history.append(f"Node reactivated at {self._last_modified}")

    def can_overwrite(self) -> Tuple[bool, str]:
        """
        Check if the processing node can be overwritten
        Returns a tuple of (can_overwrite, reason)
        """
        if not self.is_active:
            return False, f"Node '{self.name}' is not active"

        if len(self.pipeline.processors) > 0:
            return True, "Node has existing processors that will be replaced"

        return True, "Node is empty and ready for use"

    def process(self, **kwargs) -> da.Array:
        """Process the stream data through the pipeline"""
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        if not hasattr(self.stream_node, "fs") or not self.stream_node.fs:
            raise ValueError("Sampling rate (fs) must be defined in stream node")

        if not self.pipeline.processors:
            raise ValueError("No processors configured in the pipeline")

        data = self.stream_node.load_data()
        return self.pipeline.process(data, fs=self.stream_node.fs, **kwargs)

    def get_history(self) -> List[str]:
        """Get the processing history"""
        return self._processor_history.copy()

    def summarize(self):
        # TODO Need modification
        """Print a summary of the processing configuration"""
        console = Console()

        # Create main table
        table = Table(title=f"Processing Node Summary: {self.name}")

        # Add columns
        table.add_column("Component", justify="right", style="cyan")
        table.add_column("Details", justify="left")

        # Node status information
        table.add_section()
        table.add_row("Status", "Active" if self.is_active else "Inactive")
        table.add_row("Last Modified", str(self.last_modified))
        table.add_row("Stream Node", str(self.stream_node.data_path))
        table.add_row(
            "Sampling Rate",
            f"{self.stream_node.fs} Hz" if self.stream_node.fs else "Not set",
        )

        # Processing Pipeline Information
        if self.pipeline.processors:
            table.add_section()
            table.add_row("Total Processors", str(len(self.pipeline.processors)))

            for i, processor in enumerate(self.pipeline.processors, 1):
                processor_type = processor.__class__.__name__
                details = []

                # Add processor-specific details
                if isinstance(processor, FilterProcessor):
                    details.append(f"Filter function: {processor.filter_func.__name__}")
                elif isinstance(processor, TKEOProcessor):
                    details.append(f"Method: {processor.method}")
                elif isinstance(processor, NormalizationProcessor):
                    details.append(f"Method: {processor.method}")

                details.append(f"Overlap: {processor.overlap_samples} samples")
                table.add_row(
                    f"Processor {i}", f"Type: {processor_type}\n" + "\n".join(details)
                )
        else:
            table.add_row("Processors", "No processors configured")

        # Add recent history
        if self._processor_history:
            table.add_section()
            table.add_row("Recent History", "\n".join(self._processor_history[-5:]))

        console.print(table)


class TKEOProcessor(BaseProcessor):
    """TKEO processor implementation"""

    def __init__(self, method: str = "standard"):
        self.method = method
        self._overlap_samples = 2 if method == "standard" else 4

    def process(self, data: da.Array, **kwargs) -> da.Array:
        if self.method == "standard":

            def tkeo(x):
                return x[1:-1] ** 2 - x[:-2] * x[2:]
        else:

            def tkeo(x):
                return x[2:-2] ** 2 - x[:-4] * x[4:]

        return data.map_overlap(
            tkeo, depth=(self.overlap_samples, 0), boundary="reflect", dtype=data.dtype
        )

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


class FilterProcessor(BaseProcessor):
    """Filter processor implementation"""

    def __init__(self, filter_func: ProcessingFunction, overlap_samples: int):
        self.filter_func = filter_func
        self._overlap_samples = overlap_samples

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        return data.map_overlap(
            self.filter_func,
            depth=(self.overlap_samples, 0),
            boundary="reflect",
            fs=fs,
            dtype=data.dtype,
            **kwargs,
        )

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


class NormalizationProcessor(BaseProcessor):
    """Normalization processor implementation"""

    def __init__(self, method: str = "zscore"):
        self.method = method
        self._overlap_samples = 0

    def process(self, data: da.Array, **kwargs) -> da.Array:
        if self.method == "zscore":
            mean = data.mean()
            std = data.std()
            return (data - mean) / std
        elif self.method == "minmax":
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


class StftProcessor(BaseProcessor):
    """STFT processor implementation using PyTorch for large time-series Dask arrays."""

    def __init__(
        self, n_fft: int = 256, hop_length: Optional[int] = None, window: str = "hann"
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2
        self.window = window
        self._overlap_samples = n_fft - self.hop_length  # Overlap needed for continuity

    def _stft_single_channel(self, data_np: np.ndarray) -> np.ndarray:
        """Compute STFT for a single-channel NumPy array."""
        if data_np.size == 0:
            return np.empty(
                (self.n_fft // 2 + 1, 0), dtype=np.float32
            )  # Ensure non-empty output

        data_tensor = torch.tensor(data_np, dtype=torch.float32)

        # Create window function
        window_tensor = (
            torch.hann_window(self.n_fft)
            if self.window == "hann"
            else torch.ones(self.n_fft)
        )

        # Compute STFT
        stft_result = torch.stft(
            data_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window_tensor,
            return_complex=True,
            center=True,  # Ensures symmetric padding
            normalized=False,
        )

        return torch.abs(stft_result).numpy()  # Convert to magnitude spectrogram

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """Computes STFT for each channel using Dask with `map_overlap`."""

        if data.ndim != 2:
            raise ValueError(
                f"Expected input shape (number of samples, number of channels), but got {data.shape}"
            )

        # Debugging chunk sizes before and after rechunking
        print(f"Chunks before rechunk: {data.chunks}")
        if data.chunks[0][0] < self.n_fft:
            data = data.rechunk(
                {0: self.n_fft}
            )  # Ensures sufficient time samples in each block
        print(f"Chunks after rechunk: {data.chunks}")

        # Define depth for overlap (only time axis)
        depth = {0: self._overlap_samples, 1: 0}  # Overlap only along time axis

        # Apply STFT per channel
        def process_block(block):
            # Handle empty blocks by returning an empty array with proper shape
            if block.size == 0:
                return da.zeros(
                    (self.n_fft // 2 + 1, 0, block.shape[1]), dtype=np.float32
                )

            # Apply STFT for each channel
            stft_results = np.stack(
                [
                    self._stft_single_channel(block[:, ch])
                    for ch in range(block.shape[1])
                ],
                axis=-1,
            )
            return da.from_array(stft_results, chunks=-1)

        stft_results = data.map_overlap(
            process_block,
            depth=depth,
            boundary="reflect",  # Reflects edge values to avoid artifacts
            trim=False,  # Do not trim overlap to ensure correct output shape
            dtype=np.float32,
        )
        return stft_results

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


# Example filter functions that can be used with FilterProcessor
def create_bandpass_filter(
    lowcut: float, highcut: float, order: int = 4
) -> ProcessingFunction:
    def bandpass_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(
            order, [lowcut / nyquist, highcut / nyquist], btype="bandpass", output="sos"
        )
        return sosfiltfilt(sos, chunk, axis=0)

    return bandpass_filter


def create_notch_filter(
    notch_freq: float, q: float = 30, order: int = 4
) -> ProcessingFunction:
    def notch_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        low = (notch_freq - 1 / q) / nyquist
        high = (notch_freq + 1 / q) / nyquist
        sos = butter(order, [low, high], btype="bandstop", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)

    return notch_filter


def create_lowpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    def lowpass_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(order, cutoff / nyquist, btype="lowpass", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)

    return lowpass_filter


def create_highpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    def highpass_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(order, cutoff / nyquist, btype="highpass", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)

    return highpass_filter


# %%
folder_name = (
    r"../data/25-02-12_9882-1_testSubject_emgContusion/drv_16-49-56_stim/RawG.ant"
)


# %%
stream = StreamNode(folder_name)
stream.load_metadata()
stream.load_data()
# Print stream summary
stream.summarize()
# %%
# Create processing node
processor = ProcessingNode(stream)
# %%
notch = FilterProcessor(filter_func=create_notch_filter(60), overlap_samples=128)


# Add some processors
bandpass = FilterProcessor(
    create_bandpass_filter(lowcut=20, highcut=2000), overlap_samples=128
)

lowpass = FilterProcessor(filter_func=create_lowpass_filter(20), overlap_samples=128)


processor.add_processor(notch)
processor.add_processor(bandpass)
# %%
processor.summarize()
filtered_data = processor.process()
# %%

tkeo = TKEOProcessor(method="standard")
processor.add_processor(tkeo)
processor.add_processor(lowpass)
processor.summarize()
le_data = processor.process()
# %%
filter_to_plot = filtered_data[0:100000, 0].compute()
le_data_to_plot = le_data[0:100000, 0].compute()
# %%
plt.plot(filter_to_plot)
plt.plot(le_data_to_plot * 1000000)


# %%
processor.remove_processor(0)
stft_processor = StftProcessor()
processor.add_processor(stft_processor)
# %%
processed_data = processor.process()
# %%
a = processed_data[0:1000000, :].compute()

# %%
a[:]

# %%
chunk = processed_data[0:10, :].compute()  # Get a small slice
print(chunk)
# %%
channel_idx = 1

# Get the STFT data for the selected channel
stft_channel_data = a[:, :, channel_idx].compute()

# Plot the STFT data
plt.imshow(stft_channel_data, origin="lower", cmap="inferno", aspect="auto")
plt.xlabel("Time (frames)")
plt.ylabel("Frequency (bins)")
plt.title("STFT Channel {}".format(channel_idx))
plt.colorbar()
plt.show()

# %%
