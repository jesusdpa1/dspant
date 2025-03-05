"""
Improving stft functions to allow for sequential processing of filtering to stft
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
from dspant.processor.manager.stream_processing import ProcessingNode

# %%
stream_path = (
    r"../data/25-02-12_9882-1_testSubject_emgContusion/drv_16-49-56_stim/RawG.ant"
)
epoc_path = (
    r"../data/25-02-12_9882-1_testSubject_emgContusion/drv_16-49-56_stim/AmpA.ant"
)

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
# Create processing node
processor = ProcessingNode(stream)
# %%
notch = FilterProcessor(filter_func=create_notch_filter(60), overlap_samples=128)


# Add some processors
bandpass = FilterProcessor(
    create_bandpass_filter(lowcut=20, highcut=2000), overlap_samples=128
)

processor.add_processor([notch, bandpass], group="filters")

tkeo = TKEOProcessor(method="standard")
lowpass = FilterProcessor(filter_func=create_lowpass_filter(10), overlap_samples=128)

processor.add_processor([tkeo, lowpass], group="le")
processor.summarize()
# processor.add_processor(bandpass)

#
# %%
processed_ = processor.process(["filters"])
le_ = processor.process(["filters", "le"])
# %%
# original = stream.data[0:1000000, 0].persist()
filtered_data = processed_.compute()
le_data = le_.compute()

# %%
channel = 0
start = 0
end = 100000

plt.plot(filtered_data[start:end, channel])
plt.plot(le_data[start:end, channel] * 1000000)


# %%
spectrogram_processor = SpectrogramProcessor(
    n_fft=400,  # Adjust based on your data characteristics
    hop_length=200,  # Typically half of n_fft
    center=True,
)
# mfcc_processor = MFCCProcessor(
#     n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 128, "power": 2.0}
# )
processor.add_processor([spectrogram_processor], "TF")
# %%
tf_ = processor.process(["TF"])
tf_filtered = processor.process(["filters", "TF"])
# %%
tf_data = tf_.compute()
tf_filtered_data = tf_filtered.compute()


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
start = 1000
end = 2000
spec = tf_data[:, start:end, channel]

plot_spectrogram(spec)

# %%
spec = tf_filtered_data[:, start:end, channel]

plot_spectrogram(spec)

# %%
