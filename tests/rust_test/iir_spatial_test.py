"""
Functions to test Butterworth filter Rust implementation versus Python version
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

# Import Python implementations
from dspant.processors.spatial import create_whitening_processor
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.vizualization import plot_multi_channel_data

# Import Rust implementation of IIR filters
try:
    from dspant.processors.filters.iir_filters import (
        create_filter_processor,
    )

    HAS_RUST = True
except ImportError:
    print(
        "Warning: Rust IIR filter extensions not available. Only testing Python implementations."
    )
    HAS_RUST = False

sns.set_theme(style="darkgrid")

# %%
# Configure paths
home = Path(r"E:\jpenalozaa")  # Change to your path
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
# Create processing nodes for comparison tests
print("Creating processing nodes...")

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

rust_processor.add_processor([notch_rust, bandpass_rust], group="filters")
rust_processor.summarize()

# %%
start_time = time.time()
rust_result = rust_processor.process(group=["filters"]).persist()
rust_time = time.time() - start_time
print(f"Rust implementation time: {rust_time:.4f} seconds")
# %%
cmr_processor_rs = create_cmr_processor_rs()
whitening_processor_py = create_whitening_processor(eps=1e-6)
# %%
rust_processor.add_processor(
    [cmr_processor_rs, whitening_processor_py], group="spatial"
)
rust_processor.summarize()
# %%
data_whiten = rust_processor.process().persist()
# %%
channel_list = np.arange(start=0, stop=32, step=1)
a = plot_multi_channel_data(
    data_whiten,
    time_window=[0, 3],
    figsize=(10, 15),
    fs=fs,
    color="black",
    color_mode="single",
    channels=list(channel_list),
)

# %%
