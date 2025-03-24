"""
Example script for exporting spike sorting results to Phy format
Based on your existing workflow with dspant
"""

# %%
import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from tqdm import tqdm

from dspant.engine import create_processing_node
from dspant.neuroproc.detection import create_negative_peak_detector
from dspant.neuroproc.extraction import extract_spike_waveforms
from dspant.neuroproc.sorters.pca_kmeans_numba import create_numba_pca_kmeans

# Import the template utility functions
from dspant.neuroproc.utils.template_utils import (
    align_template,
    compute_template_metrics,
    compute_template_pca,
    compute_template_similarity,
    compute_templates,
)
from dspant.neuroproc.vizualization import (
    plot_multi_channel_data,
    plot_spike_events,
    plot_spike_raster,
)
from dspant.nodes import StreamNode
from dspant.processors.filters import ButterFilter, FilterProcessor
from dspant.processors.spatial import create_cmr_processor, create_whitening_processor

# Set up styles
sns.set_theme(style="darkgrid")
# %%
# ---- STEP 1: Load and preprocess data (using your exact code) ----

home_path = Path(r"E:\jpenalozaa")
base_path = home_path.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)

# home_path = Path.home()
# base_path = home_path.joinpath(
#     r"data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
# )
emg_stream_path = base_path.joinpath(r"HDEG.ant")
# %%
# Load EMG data
stream_emg = StreamNode(str(emg_stream_path), chunk_size=100000)
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

# Create processing node with filters
processor_hd = create_processing_node(stream_emg)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
cmr_processor = create_cmr_processor()
whiten_processor = create_whitening_processor(eps=1e-6)

# Add processors to the processing node
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
processor_hd.add_processor([cmr_processor, whiten_processor], group="spatial")

# %%
# Process data
whiten_data = processor_hd.process().persist()
# %%
# ---- STEP 2: Detect spikes (using your code) ----
# Rechunk data for processing
data_to_process = whiten_data[:, :].rechunk((int(fs) * 10000, -1))
# %%
threshold = 15
refractory_period = 0.002
detector = create_negative_peak_detector(
    threshold=threshold, refractory_period=refractory_period
)
spike_df = detector.detect(data_to_process, fs=fs)

print(f"Detected {len(spike_df)} spikes across all channels")

# ---- STEP 3: Extract waveforms (using your code) ----
channel_idx = 16  # We'll use your channel 16 as in your example
channel_to_process = data_to_process[:, channel_idx]
data_length = data_to_process.shape[0]
pre_samples = 30
post_samples = 80
# %%
# Filter spikes for the specific channel and within valid boundaries
valid_spikes = spike_df.filter(
    (pl.col("channel") == channel_idx)
    & (pl.col("index") >= pre_samples)
    & (pl.col("index") < data_length - post_samples)
)

print(f"Valid spikes for channel {channel_idx}: {len(valid_spikes)}")

# %%
plot_spike_events(
    np.array(data_to_process),
    spike_df=spike_df,
    fs=fs,
    channels=[0, 1, 2, 3],
    time_window=[0.2, 0.8],
    sort_spikes="time",
    sort_channels=True,
    sort_order="descending",
)
# %%
a = plot_spike_raster(
    spike_df=spike_df,
    channels=np.arange(0, 4, 1),
    time_window=[0.2, 0.8],
    sort_spikes="time",
    color="black",
    color_mode="single",
    sort_channels=True,
    figsize=(10, 10),
    marker_width=2,
    marker_size=50,
)
# %%
a = plot_multi_channel_data(
    np.array(data_to_process),
    channels=np.arange(0, 32, 1),
    time_window=[0.2, 0.8],
    fs=fs,
    figsize=(10, 10),
    color="black",
    color_mode="single",
    grid=True,
)

# %%
# Extract waveforms using your settings
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
# ---- STEP 4: Run spike sorting (using your code) ----
# Create the processor
sorter = create_numba_pca_kmeans(n_clusters=3, n_components=10)

start_time = time.time()
# Process spike waveforms
cluster_labels = sorter.process(waveforms, compute_now=True)
elapsed_time = time.time() - start_time
print(f"Sorting completed in {elapsed_time:.2f} seconds")


# %%
# Get unique clusters
unique_clusters = np.unique(np.array(cluster_labels))
n_clusters = len(unique_clusters)
print(f"Found {n_clusters} clusters")

# ---- STEP 5: Compute templates using the new functions ----
print("Computing templates...")
templates = compute_templates(
    np.array(waveforms), np.array(cluster_labels), unique_clusters
)

# %%
# Visualize templates
plt.figure(figsize=(12, 8))
for i, cluster_id in enumerate(unique_clusters):
    plt.subplot(n_clusters, 1, i + 1)
    plt.plot(templates[i])
    plt.title(f"Cluster {cluster_id} Template")
plt.tight_layout()
plt.savefig("cluster_templates.png")
# %%
