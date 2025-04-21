"""
Script to showcase CMR
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.io.exporters.array2ant import save_to_ant
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.vizualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")
load_dotenv()
# %%

data_dir = Path(os.getenv("DATA_DIR"))

base_path = data_dir.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

hd_stream_path = base_path.joinpath("HDEG.ant")

# %%
# Load EMG data
stream_hd = StreamNode(str(hd_stream_path))
stream_hd.load_metadata()
stream_hd.load_data()
# Print stream_emg summary
stream_hd.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_hd.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=30, fs=fs)
# %%
bandpass_plot = bandpass_filter.plot_frequency_response()
notch_plot = notch_filter.plot_frequency_response()
# %%
# Create processing node with filters
processor_hd = create_processing_node(stream_hd)
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
cmr_processor = create_cmr_processor_rs()
cmr_data = cmr_processor.process(filter_data, fs).persist()
cmr_reference = cmr_processor.get_reference(filter_data)

# %%
a = plot_multi_channel_data(
    filter_data, channels=[1, 2, 3, 4], fs=fs, time_window=[0, 5]
)

# %%


# Save to current directory
output_dir = save_to_ant(
    output_path=r"X:\data\becca\drv_00_baseline_25-02-26_9881-2_testSubject_topoMapping",
    data=cmr_reference,
    metadata=stream_hd.metadata,
    fs=fs,  # 1kHz sampling rate
    name="referenceChannel",
)

print(f"Data saved to {output_dir}")

# %%
