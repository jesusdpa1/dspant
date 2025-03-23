"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from dspant._rs import compute_tkeo
from dspant.emgproc.activity_detection import (
    EMGOnsetDetector,
    create_absolute_threshold_detector,
)
from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.basic import create_normalizer, create_tkeo, create_tkeo_envelope

# Import our Rust-accelerated version for comparison
from dspant.processor.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)

sns.set_theme(style="darkgrid")
# %%
# /home/jesusdpa1/data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/
#
base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_stream_path = base_path + r"/RawG.ant"
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
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
# %%
fig_bp = bandpass_filter.plot_frequency_response(
    show_phase=True, cutoff_lines=True, freq_scale="log", y_min=-80
)
# plt.show()  # This displays and clears the current figure
# plt.savefig("bandpass_filter.png", dpi=300, bbox_inches='tight')
# %%
notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)
fig_notch = notch_filter.plot_frequency_response(
    title="60 Hz Notch Filter", cutoff_lines=True, freq_scale="log", y_min=-80
)
plt.show()  # This displays and clears the current figure
# plt.savefig("notch_filter.png", dpi=300, bbox_inches='tight')
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
# Add processors to the processing node
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")
# %%
# View summary of the processing node
processor_hd.summarize()

# %%
# Apply filters and plot results
filter_data = processor_hd.process(group=["filters"]).persist()
# %%
# ======= ENVELOPE DETECTION TESTING =======
# Create and apply multiple envelope detection methods

# First, get the filtered data as our base
base_data = filter_data

# %%
# Create Rust-accelerated TKEO envelope pipeline
tkeo_pipeline_rs = create_tkeo_envelope_rs(method="modified")
# Add to processing node
processor_tkeo_rs = create_processing_node(stream_emg, name="TKEO Rust")
# First add basic filtering
zscore_normalizer = create_normalizer()
processor_tkeo_rs.add_processor(
    [notch_processor, bandpass_processor], group="preprocess"
)
# Add Rust-accelerated TKEO envelope processors
processor_tkeo_rs.add_processor(tkeo_pipeline_rs, group="envelope")
processor_tkeo_rs.add_processor(zscore_normalizer, group="normalizer")
# View processor summary
processor_tkeo_rs.summarize()

# %%

tkeo_envelope_rs = processor_tkeo_rs.process().persist()
# %%
import dask.array as da

# Compute the TKEO envelope first

# Create a new Dask array with more reasonable chunk sizes
# For a ~51M x 2 array, let's use chunks of about 1 second of data
chunk_samples = int(fs)  # One second of data
tkeo_envelope_dask = da.from_array(
    tkeo_envelope_rs,
    chunks=(chunk_samples, 2),  # ~24k samples per chunk, all channels in one chunk
)

print(
    f"Created Dask array with shape {tkeo_envelope_dask.shape} and chunks {tkeo_envelope_dask.chunks}"
)

# %%
plt.plot(tkeo_envelope_rs[:100000, 0])
# %%
# Test onset detection with both implementations

# Create onset detector
onset_detector = create_absolute_threshold_detector(
    threshold=0,
    min_duration=0.0002,
)


# Detect onsets in Rust envelope
tkeo_activation_mask = onset_detector.process(tkeo_envelope_rs, fs=fs)
tkeo_onsets = onset_detector.extract_events_from_mask(
    tkeo_activation_mask, tkeo_envelope_rs, fs
)


# %%
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
export_data_to_parquet(
    tkeo_activation_mask, "./data/mask02.parquet", fs, include_timestamp=False
)

# %%
export_data_to_parquet(
    stream_emg.data, "./data/data.parquet", fs, include_timestamp=False
)

# %%
