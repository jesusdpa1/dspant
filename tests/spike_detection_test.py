"""
Functions to extract test peak detection spiking activity
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)
from dspant.processor.spatial import create_cmr_processor, create_whitening_processor

sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_stream_path = base_path + r"/HDEG.ant"
# %%

# Load EMG data
stream_emg = StreamNode(emg_stream_path)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

# %%

# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = ButterFilter("bandpass", (300, 6000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)
# %%

# Create processing node with filters
processor_hd = create_processing_node(stream_emg)
# %%

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
# %%

bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%

cmr_processor = create_cmr_processor()
whiten_processor = create_whitening_processor(eps=1e-6)
# %%

# Add processors to the processing node
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
processor_hd.add_processor([cmr_processor, whiten_processor], group="spatial")

# %%

# View summary of the processing node
processor_hd.summarize()

# %%

whiten_data = processor_hd.process().persist()
# %%

plt.plot(whiten_data[0:50000, 2])

# %%

"""
Function to perform peak detection one channel at a time to avoid memory issues
"""

import numpy as np
import polars as pl
from tqdm.notebook import tqdm

from dspant.neuroproc.detection import create_negative_peak_detector

# %%
# Example usage
# note adjust rechuncking --> needs to be able to be caculated automatically
fs = stream_emg.fs
data_to_process = whiten_data[:, :].rechunk((int(fs) * 580, -1))
threshold = 10
refractory_period = 0.002
detector = create_negative_peak_detector(
    threshold=threshold, refractory_period=refractory_period
)
spike_df = detector.detect(data_to_process, fs=fs)
# %%
from dspant.neuroproc.extraction import extract_spike_waveforms

# %%

channel_idx = 2
channel_to_process = data_to_process[:, channel_idx]
data_length = data_to_process.shape[0]
pre_samples = 10
post_samples = 40

# Filter spikes for the specific channel and within valid boundaries
valid_spikes = spike_df.filter(
    (pl.col("channel") == channel_idx)
    & (pl.col("index") >= pre_samples)
    & (pl.col("index") < data_length - post_samples)
)

print(f"Total spikes: {len(spike_df)}")
print(f"Valid spikes for channel {channel_idx}: {len(valid_spikes)}")
print(
    f"Removed {len(spike_df) - len(valid_spikes)} spikes outside valid boundaries or not in this channel"
)
# %%
# Now extract waveforms using only valid spikes
waveforms, spike_times, metadata = extract_spike_waveforms(
    channel_to_process,
    valid_spikes,
    pre_samples=pre_samples,
    post_samples=post_samples,
    align_to_min=False,
    use_numba=True,
    compute_result=True,
)

print(f"Successfully extracted {len(waveforms)} waveforms")


# %%


import dask.array as da
import matplotlib.pyplot as plt
import numpy as np

from dspant.neuroproc.clustering import create_pca_kmeans

# %%
# Assuming waveforms is already defined
waveforms_da = waveforms.rechunk((1000, -1, -1))

# Create a PCA-KMeans processor with desired parameters
processor = create_pca_kmeans(
    n_clusters=5,  # Number of clusters to find
    n_components=10,  # Number of PCA components to use
    normalize=True,  # Whether to normalize waveforms
)
# %%
# Run clustering
cluster_labels = processor.process(waveforms_da)

# Compute the labels (converts from dask to numpy)
labels = cluster_labels.compute()
# %%
# Extract PCA components (this is automatically stored during processing)
# Extract PCA components
pca_components = processor._pca_components

# Create a figure with subplots - PCA scatter and average waveforms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Unique cluster labels
unique_labels = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

# 1. Plot PCA scatter on the first subplot
for i, label in enumerate(unique_labels):
    mask = labels == label
    ax1.scatter(
        pca_components[mask, 0],  # First PCA component
        pca_components[mask, 1],  # Second PCA component
        label=f"Cluster {label}",
        alpha=0.7,
        s=30,
        color=colors[i],
    )

ax1.set_title("PCA Clustering Results")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.7)

# 2. Plot average waveforms for each cluster on the second subplot
number_of_samples = 100
for i, label in enumerate(unique_labels):
    mask = labels == label
    # Get waveforms for this cluster
    cluster_waveforms = waveforms[mask]

    # Randomly sample waveforms if cluster is larger than number_of_samples
    if cluster_waveforms.shape[0] > number_of_samples:
        sample_indices = np.random.choice(
            cluster_waveforms.shape[0], size=number_of_samples, replace=False
        )
        cluster_waveforms = cluster_waveforms[sample_indices]

    # Calculate mean waveform from sampled waveforms (assuming first channel for simplicity)
    mean_waveform = np.mean(cluster_waveforms, axis=0)

    # Plot mean waveform
    time_points = np.arange(mean_waveform.shape[0]) / fs
    ax2.plot(
        time_points,
        mean_waveform[:, 0],
        label=f"Cluster {label}",
        color=colors[i],
        linewidth=2,
    )

ax2.set_title("Average Waveforms by Cluster")
ax2.set_xlabel("Sample")
ax2.set_ylabel("Amplitude")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# %%
