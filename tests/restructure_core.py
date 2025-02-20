# %%
import json
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import dask.array as da
import librosa
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
from torchaudio import functional, transforms


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


class BaseStreamNode(BaseNode):
    """Base class for handling time-series data"""

    name: Optional[str] = None
    fs: Optional[float] = None
    number_of_samples: Optional[int] = None
    data_shape: Optional[List[int]] = None
    channel_numbers: Optional[int] = None
    channel_names: Optional[List[str]] = None
    channel_types: Optional[List[str]] = None


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
        self,
        processor: Union[BaseProcessor, List[BaseProcessor]],
        position: Optional[int] = None,
    ) -> None:
        """
        Add a processor or list of processors to the pipeline at the specified position
        If position is None, append to the end
        """
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        # Convert single processor to a list for uniform handling
        processors = processor if isinstance(processor, list) else [processor]

        for proc in processors:
            if position is not None:
                if position < 0 or position > len(self.pipeline.processors):
                    raise ValueError(f"Invalid position: {position}")
                self.pipeline.processors.insert(position, proc)
                # Only increment position if it was specified to maintain relative order
                position += 1
            else:
                self.pipeline.processors.append(proc)

            self._last_modified = datetime.now()
            self._processor_history.append(
                f"Added {proc.__class__.__name__} at {self._last_modified}"
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


class SpectrogramProcessor(BaseProcessor):
    """this function seems to only work for computing the full stft regardless of the shape given, needs to be checked later"""

    def __init__(
        self,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        pad: int = 0,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        self.spectrogram = transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs or {},
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )

        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 2

        if center:
            self._overlap_samples = n_fft
        else:
            self._overlap_samples = n_fft - self.hop_length

    def process(
        self, data: da.Array, fs: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        if data.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {data.shape}")

        def process_chunk(x: np.ndarray) -> np.ndarray:
            x_torch = torch.from_numpy(x).float().T
            spec = self.spectrogram(x_torch)
            return np.moveaxis(spec.numpy(), 0, -1)

        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},  # Using dict form like STFT
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,
        )

        return result

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


class LFCCProcessor(BaseProcessor):
    def __init__(
        self,
        n_filter: int = 128,
        n_lfcc: int = 40,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        dct_type: int = 2,
        norm: str = "ortho",
        log_lf: bool = False,
        speckwargs: Optional[dict] = None,
    ):
        # Store parameters for spectrogram computation
        self.speckwargs = speckwargs or {
            "n_fft": 400,
            "hop_length": 200,
            "power": 2.0,
        }

        # Initialize overlap samples based on spectrogram parameters
        self.n_fft = self.speckwargs.get("n_fft", 400)
        self.hop_length = self.speckwargs.get("hop_length", self.n_fft // 2)

        if self.speckwargs.get("center", True):
            self._overlap_samples = self.n_fft
        else:
            self._overlap_samples = self.n_fft - self.hop_length

        # LFCC specific parameters
        self.n_filter = n_filter
        self.n_lfcc = n_lfcc
        self.f_min = f_min
        self.f_max = f_max
        self.dct_type = dct_type
        self.norm = norm
        self.log_lf = log_lf

        # LFCC transform will be initialized in process since it needs sampling rate
        self.lfcc = None
        self.freqs = None

    def get_frequencies(self) -> Optional[np.ndarray]:
        """Return the center frequencies of the linear filterbank"""
        if self.lfcc is None:
            return None
        # The filterbank is linear, so we can compute the frequencies directly
        f_max = self.f_max if self.f_max is not None else self.lfcc.sample_rate / 2
        return np.linspace(self.f_min, f_max, self.n_filter)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        if data.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {data.shape}")

        # Initialize LFCC transform with the sampling rate
        self.lfcc = transforms.LFCC(
            sample_rate=int(fs),  # LFCC requires integer sample rate
            n_filter=self.n_filter,
            n_lfcc=self.n_lfcc,
            f_min=self.f_min,
            f_max=self.f_max if self.f_max is not None else fs / 2,
            dct_type=self.dct_type,
            norm=self.norm,
            log_lf=self.log_lf,
            speckwargs=self.speckwargs,
        )

        # Store the frequencies once LFCC is initialized
        self.freqs = self.get_frequencies()

        def process_chunk(x: np.ndarray) -> np.ndarray:
            x_torch = torch.from_numpy(x).float().T
            lfcc_features = self.lfcc(x_torch)
            return np.moveaxis(lfcc_features.numpy(), 0, -1)

        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,
        )

        if self.freqs is not None:
            result.attrs = {"frequencies": self.freqs}

        return result

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples


class MFCCProcessor(BaseProcessor):
    def __init__(
        self,
        n_mfcc: int = 40,
        dct_type: int = 2,
        norm: str = "ortho",
        log_mels: bool = False,
        melkwargs: Optional[dict] = None,
    ):
        # Store mel spectrogram parameters
        self.melkwargs = melkwargs or {
            "n_fft": 400,
            "hop_length": 200,
            "n_mels": 128,
            "power": 2.0,
        }

        # Initialize overlap samples based on mel spectrogram parameters
        self.n_fft = self.melkwargs.get("n_fft", 400)
        self.hop_length = self.melkwargs.get("hop_length", self.n_fft // 2)

        if self.melkwargs.get("center", True):
            self._overlap_samples = self.n_fft
        else:
            self._overlap_samples = self.n_fft - self.hop_length

        # MFCC specific parameters
        self.n_mfcc = n_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.log_mels = log_mels

        # MFCC transform will be initialized in process since it needs sampling rate
        self.mfcc = None
        self.mel_fbanks = None

    def get_mel_filterbanks(self, fs) -> Optional[torch.Tensor]:
        """Get mel filterbank matrix"""
        print(fs)
        if not fs:
            return None

        n_mels = self.melkwargs.get("n_mels", 128)
        f_min = self.melkwargs.get("f_min", 0.0)

        # Ensure f_max is not None by using fs/2 as default
        f_max = self.melkwargs.get("f_max") or (fs / 2)

        # Number of FFT bins
        n_freqs = self.n_fft // 2 + 1

        return functional.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=fs,
            norm=self.melkwargs.get("norm", None),
            mel_scale=self.melkwargs.get("mel_scale", "htk"),
        )

    def process(
        self, data: da.Array, fs: Optional[float] = None, **kwargs
    ) -> np.ndarray:
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        if data.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {data.shape}")

        # Initialize MFCC transform with the sampling rate
        self.mfcc = transforms.MFCC(
            sample_rate=int(fs),  # MFCC requires integer sample rate
            n_mfcc=self.n_mfcc,
            dct_type=self.dct_type,
            norm=self.norm,
            log_mels=self.log_mels,
            melkwargs=self.melkwargs,
        )

        # Get mel filterbanks
        self.mel_fbanks = self.get_mel_filterbanks(fs)

        def process_chunk(x: np.ndarray) -> np.ndarray:
            x_torch = torch.from_numpy(x).float().T
            mfcc_features = self.mfcc(x_torch)
            return np.moveaxis(mfcc_features.numpy(), 0, -1)

        result = data.map_overlap(
            process_chunk,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=np.float32,
            new_axis=-3,
        )

        # Store filterbank information as attributes
        if self.mel_fbanks is not None:
            mel_fbanks_np = self.mel_fbanks.numpy()
            result.attrs = {
                "mel_filterbanks": mel_fbanks_np,
                "freq_bins": np.linspace(0, fs / 2, mel_fbanks_np.shape[1]),
                "n_mels": mel_fbanks_np.shape[0],
            }

        return result

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
folder_name = r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion/RawG.ant"


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

spectrogram_processor = SpectrogramProcessor(
    n_fft=400,  # Adjust based on your data characteristics
    hop_length=200,  # Typically half of n_fft
    center=True,
)

processor.add_processor([notch, bandpass])
processor.summarize()
# processor.add_processor(bandpass)

# processor.add_processor(spectrogram_processor)
# %%
processed_data = processor.process()
# %%
# original = stream.data[0:1000000, 0].persist()
a = processed_data.compute()
# %%
processor.summarize()
filtered_data = processor.process()
# %%

tkeo = TKEOProcessor(method="standard")
lowpass = FilterProcessor(filter_func=create_lowpass_filter(20), overlap_samples=128)
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
processor_stft = ProcessingNode(stream)
processor_stft.summarize()
# %%
processor_stft.remove_processor(0)
mfcc_processor = MFCCProcessor(
    n_mfcc=40,
    melkwargs={
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 128,  # Reduced from 128
        "f_min": 0,
        "f_max": None,  # Will default to fs/2
    },
)

# spectogram_processor = LFCCProcessor(n_lfcc=70, n_filter=256)

processor_stft.add_processor(mfcc_processor)

# %%
processed_data = processor_stft.process()
# %%
# original = stream.data[0:1000000, 0].persist()
a = processed_data.compute()

# %%
a[:]
# %%
channel = 0
spec = a[:, 1000:2000, channel]

# Convert to dB scale
spec_db = 10 * np.log10(spec + 1e-10)

plt.figure(figsize=(12, 6))
plt.imshow(spec_db, aspect="auto", origin="lower")
plt.colorbar(label="Power (dB)")
plt.ylabel("Frequency bin")
plt.xlabel("Time frame")
plt.title(f"Spectrogram - Channel {channel}")
plt.show()


# %%


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(
        librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )


# %%

channel = 0
spec = a[:, 2000:2500, channel]

plot_spectrogram(spec)

# %%
