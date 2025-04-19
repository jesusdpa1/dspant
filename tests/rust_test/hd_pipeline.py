"""
Functions to test whitening and CMR Rust implementations versus Python versions
Author: Jesus Penaloza (based on original code)
"""

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode

# Import Rust implementations
try:
    from dspant.processors.filters.iir_filters import (
        create_filter_processor,
    )
    from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
    from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs

    HAS_RUST = True
except ImportError:
    print(
        "Warning: Rust extensions not available. Only testing Python implementations."
    )
    HAS_RUST = False

from dspant.vizualization import plot_multi_channel_data

sns.set_theme(style="darkgrid")

# %%
# Configure paths
# home = Path().home()  # Change to your path
# /home/jesusdpa1/data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline
# E:\jpenalozaa\topoMapping\25-03-22_4896-2_testSubject_topoMapping\drv\drv_17-02-16_meps
home = Path(r"E:\jpenalozaa")
base_path = home.joinpath(
    r"topoMapping\25-03-22_4896-2_testSubject_topoMapping\drv\drv_17-02-16_meps"
)
hd_stream_path = base_path.joinpath(r"HDEG.ant")

# %%
# Load HD data
print("Loading data...")
stream_hd = StreamNode(str(hd_stream_path), chunk_size=500000)
stream_hd.load_metadata()
stream_hd.load_data()
print("EMG data loaded successfully")
stream_hd.summarize()

# %%
# Get sampling rate from the stream node
fs = stream_hd.fs
print(f"Sampling rate: {fs} Hz")

# %%
rust_processor = create_processing_node(stream_hd, name="Rust Butterworth")

# Create IIR filters with Rust acceleration
bandpass_rust = create_filter_processor(
    filter_type="butter",
    btype="bandpass",
    cutoff=(300, 6000),
    order=4,
    filtfilt=True,
    use_rust=True,
)

notch_rust = create_filter_processor(
    filter_type="butter",
    btype="bandstop",
    cutoff=(59, 61),
    order=4,
    filtfilt=True,
    use_rust=True,
)
# %%
# spatial processors
cmr_processor_rs = create_cmr_processor_rs()
whitening_processor_rs = create_whitening_processor_rs(eps=1e-6)
# %%
# Setup and test with different processor configurations

# 1. Setup Python version of CMR
hd_processor = create_processing_node(stream_hd)
hd_processor.add_processor([notch_rust, bandpass_rust], group="filters")
hd_processor.add_processor([cmr_processor_rs, whitening_processor_rs], group="spatial")

# %%
hd_process = hd_processor.process().persist()

# %%

channel_list = np.arange(start=0, stop=32, step=1)

a = plot_multi_channel_data(
    hd_process,
    time_window=[100, 103],
    figsize=(10, 15),
    fs=fs,
    color="black",
    color_mode="single",
    channels=list(channel_list),
)

# %%
