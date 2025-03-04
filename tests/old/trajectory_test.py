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

# import librosa
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
base_path = (
    r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"
)

stream_path = base_path + r"/RawG.ant"
epoc_path = base_path + r"/AmpA.ant"

# %%
stream = StreamNode(stream_path)
stream.load_metadata()
stream.load_data()
# Print stream summary
stream.summarize()
# %%
# Create processing node
processor = ProcessingNode(stream)
# %%
notch = FilterProcessor(filter_func=create_notch_filter(60), overlap_samples=1024)


# Add some processors
bandpass = FilterProcessor(
    create_bandpass_filter(lowcut=20, highcut=2000), overlap_samples=1024
)

processor.add_processor([notch, bandpass], group="filters")

# %%
data_filtered = processor.process()
# %%
data_filtered

# %%
window_ = int(stream.fs * (5 * 60))

data_segment = data_filtered[int(stream.fs) : window_, 0]
# %%
data_segment
# %%
plt.plot(data_segment[0:100000])
# %%
from ssqueezepy import Wavelet, cwt, imshow, stft
from ssqueezepy.experimental import scale_to_freq

fs = stream.fs
x = data_segment[0:1000000]
# %%
N = len(x)
t = np.linspace(0, N / fs, N)
wavelet = Wavelet()
Wx, scales = cwt(x, wavelet)
# %%
freqs_cwt = scale_to_freq(scales, wavelet, len(x), fs=fs)

ikw = dict(abs=1, xticks=t, xlabel="Time [sec]", ylabel="Frequency [Hz]")
imshow(Wx, **ikw, yticks=freqs_cwt)

# %%
Wx
# %%
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import IncrementalPCA

# 1. Load the CWT Data (Dask Array) -----------------------------
cwt_data = da.from_array(abs(Wx), chunks=(317, 10_000))  # Assume Wx is already defined

# Transpose data for PCA: shape (samples, frequency bands)
reshaped_data = cwt_data.T  # Shape: (time_samples, freq_bands)

# %%
# Apply Incremental PCA
n_components = 3
pca = IncrementalPCA(n_components=n_components)


# Parallelizing PCA fitting using Dask
def batch_pca_fit(chunk):
    # No need to compute since chunk is already a NumPy array
    pca.partial_fit(chunk)
    return chunk


# Apply the batch PCA fitting across chunks
low_dim_data = reshaped_data.map_blocks(
    batch_pca_fit, dtype=float, chunks=(10_000, n_components)
)

# %%
# Visualization
low_dim_data_np = low_dim_data.compute()


# %%
# Plot the 3 principal components over time
plt.figure(figsize=(10, 6))
time_axis = np.linspace(0, len(low_dim_data_np) / fs, len(low_dim_data_np))

plt.subplot(3, 1, 1)
plt.plot(time_axis, low_dim_data_np[:, 0])
plt.title("PC1 over time")

plt.subplot(3, 1, 2)
plt.plot(time_axis, low_dim_data_np[:, 1])
plt.title("PC2 over time")

plt.subplot(3, 1, 3)
plt.plot(time_axis, low_dim_data_np[:, 2])
plt.title("PC3 over time")

plt.tight_layout()
plt.show()
# %%
# 3D trajectory visualization
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.plot(low_dim_data_np[:, 0], low_dim_data_np[:, 1], low_dim_data_np[:, 2])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("EMG Signal Trajectory in PCA Space")

plt.show()
# %%
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.sum(pca.explained_variance_ratio_))
# %%
from sklearn.decomposition import NMF

# Apply NMF instead of PCA (requires non-negative data)
nmf = NMF(n_components=4, max_iter=200, random_state=42)
nmf_components = nmf.fit_transform(abs(Wx.T))

# Visualize NMF trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(nmf_components[:, 0], nmf_components[:, 1], nmf_components[:, 2])
ax.set_title("EMG Signal Trajectory in NMF Space")
plt.show()
# %%
from sklearn.cluster import DBSCAN

# Apply clustering to identify states in the trajectory
clustering = DBSCAN(eps=0.00005, min_samples=20).fit(nmf_components)
labels = clustering.labels_

# Visualize the clusters
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(
    nmf_components[:, 0],
    nmf_components[:, 1],
    nmf_components[:, 2],
    c=labels,
    cmap="viridis",
)
ax.set_title("Clustered EMG Trajectory in NMF Space")
plt.colorbar(scatter, ax=ax, label="Cluster ID")
plt.show()
# %%
# Calculate velocity along the trajectory
velocity = np.sqrt(np.sum(np.diff(nmf_components, axis=0) ** 2, axis=1))

# Plot velocity over time
plt.figure(figsize=(12, 6))
time_points = np.linspace(0, len(velocity) / fs, len(velocity))
plt.plot(time_points, velocity)
plt.xlabel("Time (s)")
plt.ylabel("Trajectory Velocity")
plt.title("Velocity of Movement in NMF Space")
plt.grid(True)
plt.show()
