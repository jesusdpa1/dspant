"""
Functions to extract meps
author: Jesus Penaloza

"""

# %%

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import dask.array as da
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pendulum  # Replace datetime import
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pendulum.datetime import DateTime  # This is the correct import
from rich.console import Console
from rich.table import Table
from rich.text import Text
from scipy.signal import butter, sosfiltfilt
from torchaudio import functional, transforms

from dspant.core.nodes.data import EpocNode, StreamNode
from dspant.core.nodes.stream_processing import ProcessingNode
from dspant.processing.filters import (
    FilterProcessor,
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.processing.time_frequency import (
    LFCCProcessor,
    MFCCProcessor,
    SpectrogramProcessor,
)
from dspant.processing.transforms import TKEOProcessor

# %%

base_path = (
    r"..\data\25-02-21_9878-2_testSubject_emgContusion_Hemisection\drv_15-11-48_meps"
)

stream_path = base_path + r"\RawG.ant"
epoc_path = base_path + r"\AmpA.ant"

# %%
stream = StreamNode(stream_path)
stream.load_metadata()
stream.load_data()
# Print stream summary
stream.summarize()
# %%
epoc = EpocNode(epoc_path)
epoc.load_metadata()
epoc.load_data()
# Print stream summary
epoc.summarize()

# %%
processor = ProcessingNode(stream)
notch = FilterProcessor(filter_func=create_notch_filter(60), overlap_samples=12000)


# Add some processors
bandpass = FilterProcessor(
    create_bandpass_filter(lowcut=10, highcut=5000), overlap_samples=12000
)
processor.add_processor([notch, bandpass], group="filters")


# %%
def ms_to_sample(time_ms: float, fs: float) -> int:
    """Convert time in milliseconds to a sample index."""
    return int(round((time_ms / 1000) * fs))


def get_windows_sTime(
    data,
    fs,
    onset: list | pl.Series,
    offset: list | pl.Series = None,
    window_size: float = None,
) -> List[da.Array]:
    """
    Extracts data windows based on onset and offset times or window size.

    Args:
        onset: List or polars Series of onset times in milliseconds.
        offset: List or polars Series of offset times in milliseconds (optional).
        window_size: Window duration in milliseconds (optional, required if offset is None).

    Returns:
        List of Dask arrays containing extracted windows.
    """
    # Convert onset to polars Series if needed
    if isinstance(onset, list):
        onset = pl.Series(onset)

    # Compute offset if not provided
    if offset is None:
        if window_size is None:
            raise RuntimeError("Either 'offset' list or 'window_size' must be provided")
        window_samples = int(round((window_size / 1000) * fs))  # Fixed window length

    # Convert ms timestamps to sample indices
    onset_samples = ((onset) * fs).cast(pl.Int64)  # Ensure integer indices
    offset_samples = (onset_samples + window_samples).cast(pl.Int64)
    print(onset_samples)
    # Extract windows from Dask array
    windows = []

    for start, end in zip(onset_samples, offset_samples):
        windows.append(data[start:end])

    return windows


# %%
w = processor.process()
a = get_windows_sTime(w, processor.stream_node.fs, epoc.data["onset"], window_size=10)

# %%
aa = da.stack(a)
# %%
aa = aa.persist()
# %%
n = 100
plt.plot(aa[n, :, 0])
# %%
plt.plot(aa[n, :, 1])
# %%
