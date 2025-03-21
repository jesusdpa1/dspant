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

base_path = r"/home/jesusdpa1/data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
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
rust_onsets = onset_detector.process(tkeo_envelope_rs, fs=fs)
# %%
rust_onsets_df = onset_detector.to_dataframe(rust_onsets)
print(f"Rust TKEO detected {len(rust_onsets_df)} onsets")

# %%
a = onset_detector.create_binary_mask(tkeo_envelope_rs, fs=fs).persist()
# %%
plt.plot(tkeo_envelope_rs[:100000, 0])
plt.plot(a[:100000, 0])
# %%
