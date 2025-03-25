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
from dspant.processors.filters import (
    ButterFilter,
    FilterProcessor,
)

# Import standard Python Butterworth filter
from dspant.processors.filters.butter_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
)

# Import Rust implementation of IIR filters
try:
    from dspant.processors.filters.iir_filters import (
        IIRFilterProcessor,
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
base_path = home.joinpath(r"camber_presentation\drv\drv_02_hemisection")
hd_stream_path = base_path.joinpath(r"EMGM.ant")

# %%
# Load HD data
print("Loading data...")
stream_hd = StreamNode(str(hd_stream_path), chunk_size=100000)
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

# 1. Original Python Butterworth implementation
python_processor = create_processing_node(stream_hd, name="Python Butterworth")
bandpass_py = FilterProcessor(
    filter_func=create_bandpass_filter(300, 6000, order=4),
    overlap_samples=40,
    parallel=True,
)
notch_py = FilterProcessor(
    filter_func=create_bandpass_filter(59, 61, order=4),
    overlap_samples=40,
    parallel=True,
)
python_processor.add_processor([notch_py, bandpass_py], group="filters")

# 2. Rust Butterworth implementation (if available)
if HAS_RUST:
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
# Performance comparison
print("Running performance tests...")

# Time Python implementation
start_time = time.time()
py_result = python_processor.process(group=["filters"]).compute()
py_time = time.time() - start_time
print(f"Python implementation time: {py_time:.4f} seconds")

# Time Rust implementation if available
if HAS_RUST:
    start_time = time.time()
    rust_result = rust_processor.process(group=["filters"]).compute()
    rust_time = time.time() - start_time
    print(f"Rust implementation time: {rust_time:.4f} seconds")
    print(f"Speedup: {py_time / rust_time:.2f}x")

    # Verify results are similar
    difference = np.abs(py_result - rust_result)
    mean_diff = np.mean(difference)
    max_diff = np.max(difference)
    print(f"Mean difference: {mean_diff:.8f}")
    print(f"Max difference: {max_diff:.8f}")

# %%
# Plot comparison of the first 10,000 samples for a few channels
plt.figure(figsize=(16, 10))

# Plot Python result
plt.subplot(2, 1, 1)
plt.title("Python Butterworth Filter")
plt.plot(py_result[:100000, 0:4])
plt.grid(True)

# Plot Rust result if available
if HAS_RUST:
    plt.subplot(2, 1, 2)
    plt.title("Rust Butterworth Filter")
    plt.plot(rust_result[:100000, 0:4])
    plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Channel-by-channel comparison on a single sample
if HAS_RUST:
    channel_to_plot = 0
    sample_length = 5000

    plt.figure(figsize=(16, 10))

    plt.subplot(3, 1, 1)
    plt.title("Python Implementation")
    plt.plot(py_result[:sample_length, channel_to_plot])

    plt.subplot(3, 1, 2)
    plt.title("Rust Implementation")
    plt.plot(rust_result[:sample_length, channel_to_plot])

    plt.subplot(3, 1, 3)
    plt.title("Difference (Python - Rust)")
    plt.plot(
        py_result[:sample_length, channel_to_plot]
        - rust_result[:sample_length, channel_to_plot]
    )

    plt.tight_layout()
    plt.show()
# %%
