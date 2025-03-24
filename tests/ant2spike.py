"""
Functions to extract test peak detection spiking activity
Author: Jesus Penaloza ( PCA-KMeans sorting test with refactor code :))
"""

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

from typing import Dict, List, Optional, Union

import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def export_data_to_parquet(
    data: da.Array,
    filename: str,
    fs: Optional[float] = None,
    chunk_size: int = 1000000,
    channel_names: Optional[List[str]] = None,
    include_timestamp: bool = False,
) -> None:
    """
    Export any timeseries data to a Parquet file by processing in chunks.
    Leverages PyArrow's automatic type inference.

    Args:
        data: Timeseries data array (samples × channels)
        filename: Path to save the Parquet file
        fs: Sampling frequency
        chunk_size: Number of samples to process in each chunk
        channel_names: Custom names for the channels. If None, uses "channel_0", "channel_1", etc.
        include_timestamp: Whether to include timestamps in the export
    """
    # Get dimensions
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples, n_channels = data.shape

    # Determine channel names
    if channel_names is None:
        channel_names = [f"channel_{c}" for c in range(n_channels)]
    elif len(channel_names) != n_channels:
        raise ValueError(
            f"Expected {n_channels} channel names, got {len(channel_names)}"
        )

    # Process first chunk to determine schema
    first_chunk_size = min(chunk_size, n_samples)
    first_chunk = data[:first_chunk_size].compute()

    # Create first chunk DataFrame with exact length
    sample_indices = np.arange(first_chunk_size)
    first_df = pd.DataFrame({"sample_idx": sample_indices})

    if include_timestamp and fs is not None:
        first_df["timestamp"] = sample_indices / fs

    for i, name in enumerate(channel_names):
        first_df[name] = first_chunk[:, i]

    # Let PyArrow infer schema from the first chunk
    table = pa.Table.from_pandas(first_df)
    schema = table.schema

    # Create PyArrow writer with metadata
    writer = pq.ParquetWriter(filename, schema)

    # Write first chunk
    writer.write_table(table)

    # Process remaining chunks
    current_idx = first_chunk_size
    while current_idx < n_samples:
        # Calculate exact chunk boundaries
        start_idx = current_idx
        end_idx = min(start_idx + chunk_size, n_samples)

        # Validate that we're not requesting out of bounds
        if start_idx >= n_samples:
            print(
                f"Warning: Skipping processing at index {start_idx} which exceeds data length {n_samples}"
            )
            break

        # Calculate exact chunk size for this iteration
        current_chunk_size = end_idx - start_idx

        # Create exact sample indices array for this chunk
        sample_indices = np.arange(start_idx, end_idx)

        # Extract chunk with validation
        try:
            chunk = data[start_idx:end_idx].compute()

            # Validate chunk dimensions
            if len(chunk) != current_chunk_size:
                print(f"Warning: Chunk size mismatch at {start_idx}-{end_idx}")
                print(f"Expected {current_chunk_size} samples but got {len(chunk)}")

                # Adjust sample indices to match the actual chunk size
                sample_indices = np.arange(start_idx, start_idx + len(chunk))
        except Exception as e:
            print(f"Error processing chunk {start_idx}-{end_idx}: {str(e)}")
            # Move to next chunk
            current_idx = end_idx
            continue

        # Create DataFrame with exactly matching dimensions
        chunk_df = pd.DataFrame({"sample_idx": sample_indices})

        if include_timestamp and fs is not None:
            chunk_df["timestamp"] = sample_indices / fs

        # Add channel data with dimension validation
        for j, name in enumerate(channel_names):
            channel_data = chunk[:, j]

            # Ensure data length matches DataFrame length
            if len(channel_data) != len(sample_indices):
                print(
                    f"Warning: Channel {name} length ({len(channel_data)}) doesn't match index length ({len(sample_indices)})"
                )

                if len(channel_data) > len(sample_indices):
                    # Truncate data if too long
                    channel_data = channel_data[: len(sample_indices)]
                else:
                    # Pad data if too short
                    padding = np.zeros(
                        len(sample_indices) - len(channel_data),
                        dtype=channel_data.dtype,
                    )
                    if len(channel_data) > 0:  # Use last value for padding if available
                        padding.fill(channel_data[-1])
                    channel_data = np.concatenate([channel_data, padding])

            chunk_df[name] = channel_data

        # Convert to PyArrow table and write
        chunk_table = pa.Table.from_pandas(chunk_df, schema=schema)
        writer.write_table(chunk_table)

        print(
            f"Processed samples {start_idx:,} to {end_idx:,} ({(end_idx / n_samples) * 100:.1f}%)"
        )

        # Update current index for next iteration
        current_idx = end_idx

    # Close writer
    writer.close()
    print(f"Data successfully exported to {filename}")
    print(f"Total samples: {n_samples:,}, Channels: {n_channels}")


# %%

# base_path = r"E:\jpenalozaa\"
home = Path(r"E:\jpenalozaa")  # Path().home()  #
base_path = home.joinpath(
    r"topoMapping\25-03-22_4896-2_testSubject_topoMapping\17-02-16_meps\drv_17-02-16_meps"
)
# emg_stream_path = base_path.joinpath(r"RawG.ant")
hd_stream_path = base_path.joinpath(r"HDEG.ant")

# %%
home = Path(r"E:\jpenalozaa")  # Path().home()  #
base_path = home.joinpath(
    r"topoMapping\25-03-22_4896-2_testSubject_topoMapping\17-02-16_meps\drv_17-02-16_meps"
)
# emg_stream_path = base_path.joinpath(r"RawG.ant")
hd_stream_path = base_path.joinpath(r"HDEG.ant")

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
# # EMG processor

# # Create filters for preprocessing
# bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
# notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)


# # Create processors for filtering
# notch_processor = FilterProcessor(
#     filter_func=notch_filter.get_filter_function(), overlap_samples=40
# )
# bandpass_processor = FilterProcessor(
#     filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
# )

# # Add filters to the processing node
# processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# processor_emg.summarize()
# %%

# emg_filtered = processor_emg.process().persist()
# %%
hd_filtered = processor_hd.process().persist()

# %%
start = int(fs * 0)
end = int(fs * 2)
# %%
# plt.plot(emg_filtered[start:end, 1])
# %%

plt.plot(hd_filtered[start:end, 2])

# %%

# hd_channel_list = np.arange(start=16, stop=32, step=1)
# %%
a = plot_multi_channel_data(
    emg_filtered, time_window=[0.1, 10], fs=fs, colormap="turbo"
)
# %%
hd_channel_list = np.arange(start=0, stop=16, step=1)
b = plot_multi_channel_data(
    hd_filtered,
    time_window=[758.80, 758.82],
    figsize=(5, 15),
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


# %%
export_data_to_parquet(
    hd_filtered, "./hd_filtered.parquet", fs, include_timestamp=False
)

# %%
