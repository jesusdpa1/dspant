"""
Functions to extract test peak detection spiking activity
Author: Jesus Penaloza ( PCA-KMeans sorting test with refactor code :))
"""

# %%
# %%
# emg to hd
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from tqdm.notebook import tqdm

from dspant.engine import create_processing_node
from dspant.neuroproc.detection import create_negative_peak_detector
from dspant.neuroproc.extraction import extract_spike_waveforms
from dspant.neuroproc.sorters import (
    create_pca_kmeans,
)
from dspant.neuroproc.vizualization import plot_multi_channel_data, plot_spike_events
from dspant.nodes import StreamNode
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)
from dspant.processor.spatial import create_cmr_processor, create_whitening_processor

sns.set_theme(style="darkgrid")
# %%

# base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
home = Path(r"E:\jpenalozaa")  # Path().home()  #
base_path = home.joinpath(r"camber_presentation\drv\drv_02_hemisection")
# emg_stream_path = base_path.joinpath(r"RawG.ant")
hd_stream_path = base_path.joinpath(r"EMGM.ant")

# %%

# Load EMG data
stream_emg = StreamNode(str(emg_stream_path))
stream_emg.load_metadata()
stream_emg.load_data()
print("EMG data loaded successfully")
stream_emg.summarize()
processor_emg = create_processing_node(stream_emg)
# %%
# Load HD data

stream_hd = StreamNode(str(hd_stream_path), chunk_size=100000)
stream_hd.load_metadata()
stream_hd.load_data()
print("EMG data loaded successfully")
stream_hd.summarize()
processor_hd = create_processing_node(stream_hd)
# %%
# Get sampling rate from the stream node
fs = stream_hd.fs
# Create processing node with filters

# Create processing node with filters

# %%

# HD processor
# Create filters with improved visualization
bandpass_filter = ButterFilter("bandpass", (300, 6000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)


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
processor_hd.summarize()
# %%
# EMG processor

# Create filters for preprocessing
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)


# Create processors for filtering
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Add filters to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

processor_emg.summarize()
# %%

emg_filtered = processor_emg.process().persist()
# %%
hd_filtered = processor_hd.process().persist()

# %%
start = int(fs * 0)
end = int(fs * 2)
# %%
plt.plot(emg_filtered[start:end, 1])
# %%

plt.plot(hd_filtered[start:end, 2])

# %%

# hd_channel_list = np.arange(start=16, stop=32, step=1)
# %%
a = plot_multi_channel_data(
    emg_filtered, time_window=[0.1, 10], fs=fs, colormap="turbo"
)
# %%
hd_channel_list = np.arange(start=16, stop=32, step=1)
b = plot_multi_channel_data(
    hd_filtered,
    time_window=[750, 760],
    figsize=(15, 15),
    fs=fs,
    color="black",
    color_mode="single",
    channels=list(hd_channel_list),
)
# %%
# Example usage
# note adjust rechuncking --> needs to be able to be caculated automatically
fs = stream_emg.fs
data_to_process = whiten_data[:, :].rechunk((int(fs) * 10000, -1))
threshold = 10
refractory_period = 0.002
detector = create_negative_peak_detector(
    threshold=threshold, refractory_period=refractory_period
)
spike_df = detector.detect(data_to_process, fs=fs)
# %%

plot_spike_events(
    data_to_process[: int(2 * fs), :].compute(),
    spike_df,
    channels=[
        1,
        2,
    ],
    fs=fs,
    time_window=(0, 1),
)

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

# Convert extracted waveforms to dask array with appropriate chunking
waveforms_da = waveforms.rechunk((1000, -1, -1))

# %%
# Create a PCA-KMeans processor with desired parameters
pca_kmeans = create_pca_kmeans(
    n_clusters=5,  # Number of clusters to find
    n_components=10,  # Number of PCA components to use
    normalize=True,  # Whether to normalize waveforms
    random_state=42,  # For reproducibility
)

# %%

# Run clustering (compute_now=True will compute the dask array)
cluster_labels = pca_kmeans.process(waveforms_da[:, :, 0], compute_now=True)
# %%
# Visualize clustering results
fig = pca_kmeans.plot_clusters(
    plot_waveforms=True,
    include_silhouette=True,
    title="EMG Spike Clustering",
    figsize=(16, 12),
    waveforms=da.from_array(np.array(waveforms_da[:, :, 0])),
)

plt.tight_layout()
plt.show()

# %%
