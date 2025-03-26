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
from dspant.processors.filters import (
    ButterFilter,
    FilterProcessor,
)

# Import Python implementations
from dspant.processors.spatial import create_cmr_processor, create_whitening_processor

# Import Rust implementation of IIR filters

# Import Rust implementations
try:
    from dspant.processors.filters.iir_filters import (
        IIRFilterProcessor,
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
# /home/jesusdpa1/data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline
home = Path().home()  # Change to your path
base_path = home.joinpath(
    r"data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
hd_stream_path = base_path.joinpath(r"HDEG.ant")

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
# Setup and test with different processor configurations

# 1. Setup Python version of CMR
filter_processor = create_processing_node(stream_hd)
filter_processor.add_processor([notch_rust, bandpass_rust], group="filters")

# %%
test = filter_processor.process().persist()

# %%

cmr_processor_py = create_cmr_processor()
cmr_processor_rs = create_cmr_processor_rs()

# %%%
start = time.time()
cmr_data = cmr_processor_py.process(test, fs).persist()
end = time.time()
print(f"Processing time Python: {end - start:.4f} seconds")

# %%
start = time.time()
cmr_data_rs = cmr_processor_rs.process(test, fs).persist()
end = time.time()
print(f"Processing time Rust: {end - start:.4f} seconds")

# Calculate speedup
python_time = end - start
rust_time = end - start  # This is wrong, you need to fix the variables
speedup = python_time / rust_time
print(f"Speedup: {speedup:.2f}x")

# %%

# Timing comparison for whitening processors
whitening_processor_py = create_whitening_processor(eps=1e-6)
start_py = time.time()
whitened_data = whitening_processor_py.process(test, fs).persist()
end_py = time.time()
python_time = end_py - start_py
print(f"Whitening processing time Python: {python_time:.4f} seconds")
# %%
whitening_processor_rs = create_whitening_processor_rs(eps=1e-6, use_parallel=False)
start_rs = time.time()
whitened_data_rs = whitening_processor_rs.process(test, fs).persist()
end_rs = time.time()
rust_time = end_rs - start_rs
print(f"Whitening processing time Rust: {rust_time:.4f} seconds")

# Calculate speedup
speedup = python_time / rust_time
print(f"Whitening speedup: {speedup:.2f}x")
# %%
# Optional: Verify results are similar
if whitened_data is not None and whitened_data_rs is not None:
    difference = np.abs(whitened_data - whitened_data_rs)
    mean_diff = np.mean(difference)
    max_diff = np.max(difference)
    print(f"Mean difference: {mean_diff.compute():.8f}")
    print(f"Max difference: {max_diff.compute():.8f}")
# %%
# 2. Setup Python version of Whitening

whitening_processor_py = create_whitening_processor(eps=1e-6)
whiten_data = whitening_processor_py.process(cmr_data, fs).persist()


# 3. Setup Rust version of CMR (if available)
if HAS_RUST:
    processor_cmr_rs = create_processing_node(stream_hd, name="Rust CMR")
    processor_cmr_rs.add_processor(
        [notch_processor, bandpass_processor], group="filters"
    )

    processor_cmr_rs.add_processor(cmr_processor_rs, group="spatial")
    processor_cmr_rs.summarize()

# 4. Setup Rust version of Whitening (if available)
if HAS_RUST:
    processor_white_rs = create_processing_node(stream_hd, name="Rust Whitening")
    processor_white_rs.add_processor(
        [notch_processor, bandpass_processor], group="filters"
    )

    processor_white_rs.add_processor(whitening_processor_rs, group="spatial")
    processor_white_rs.summarize()

# %%
# Process data with all configurations and measure timing
print("Processing with different configurations...")

# Define a small subset of data for initial testing
test_duration_sec = 10
test_samples = int(fs * test_duration_sec)
test_end = min(test_samples, stream_hd.data.shape[0])

# Use smaller chunk of data for testing
print(f"Testing with {test_duration_sec} seconds of data ({test_end} samples)")

# 1. Process with Python CMR
start_time = time.time()
data_cmr_py = processor_cmr_py.process().persist()
py_cmr_time = time.time() - start_time
print(f"Python CMR processing time: {py_cmr_time:.4f} seconds")
# %%
# 2. Process with Python Whitening
start_time = time.time()
data_white_py = processor_white_py.process().persist()
py_white_time = time.time() - start_time
print(f"Python Whitening processing time: {py_white_time:.4f} seconds")

# # 3. Process with Rust CMR (if available)
# if HAS_RUST:
#     start_time = time.time()
#     data_cmr_rs = processor_cmr_rs.process(end=test_end).compute()
#     rs_cmr_time = time.time() - start_time
#     print(f"Rust CMR processing time: {rs_cmr_time:.4f} seconds")
#     print(f"CMR Speedup: {py_cmr_time / rs_cmr_time:.2f}x")

# # 4. Process with Rust Whitening (if available)
# if HAS_RUST:
#     start_time = time.time()
#     data_white_rs = processor_white_rs.process(end=test_end).compute()
#     rs_white_time = time.time() - start_time
#     print(f"Rust Whitening processing time: {rs_white_time:.4f} seconds")
#     print(f"Whitening Speedup: {py_white_time / rs_white_time:.2f}x")

# %%
# Verify that results are similar if Rust is available
if HAS_RUST:
    # Compare CMR implementations
    cmr_difference = np.abs(data_cmr_py - data_cmr_rs)
    cmr_mean_diff = np.mean(cmr_difference)
    cmr_max_diff = np.max(cmr_difference)
    print(f"CMR - Mean difference: {cmr_mean_diff:.8f}")
    print(f"CMR - Max difference: {cmr_max_diff:.8f}")

    # Compare Whitening implementations
    white_difference = np.abs(data_white_py - data_white_rs)
    white_mean_diff = np.mean(white_difference)
    white_max_diff = np.max(white_difference)
    print(f"Whitening - Mean difference: {white_mean_diff:.8f}")
    print(f"Whitening - Max difference: {white_max_diff:.8f}")

# %%
# Plot the outputs to visually compare results
print("Generating comparison plots...")

# Set up the plot parameters
plt.figure(figsize=(15, 10))
sample_start = 0
sample_length = min(10000, test_end)  # 10 seconds or as much as we have
channel_to_plot = 5  # Choose a representative channel

# Define font sizes with appropriate scaling
TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12

# 1. Plot filtered data
plt.subplot(5, 1, 1)
plt.title("Filtered Data (Pre-processed)", fontsize=TITLE_SIZE)
plt.plot(
    data_cmr_py[sample_start : sample_start + sample_length, channel_to_plot],
    color="black",
    label="Filtered",
)
plt.ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
plt.legend()
plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
plt.grid(True, alpha=0.3)

# 2. Plot Python CMR
plt.subplot(5, 1, 2)
plt.title("Python CMR", fontsize=TITLE_SIZE)
plt.plot(
    data_cmr_py[sample_start : sample_start + sample_length, channel_to_plot],
    color="blue",
    label="Python CMR",
)
plt.ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
plt.legend()
plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
plt.grid(True, alpha=0.3)

# 3. Plot Rust CMR (if available)
if HAS_RUST:
    plt.subplot(5, 1, 3)
    plt.title("Rust CMR", fontsize=TITLE_SIZE)
    plt.plot(
        data_cmr_rs[sample_start : sample_start + sample_length, channel_to_plot],
        color="green",
        label="Rust CMR",
    )
    plt.ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
    plt.legend()
    plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    plt.grid(True, alpha=0.3)

# 4. Plot Python Whitening
plt.subplot(5, 1, 4)
plt.title("Python Whitening", fontsize=TITLE_SIZE)
plt.plot(
    data_white_py[sample_start : sample_start + sample_length, channel_to_plot],
    color="purple",
    label="Python Whitening",
)
plt.ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
plt.legend()
plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
plt.grid(True, alpha=0.3)

# 5. Plot Rust Whitening (if available)
if HAS_RUST:
    plt.subplot(5, 1, 5)
    plt.title("Rust Whitening", fontsize=TITLE_SIZE)
    plt.plot(
        data_white_rs[sample_start : sample_start + sample_length, channel_to_plot],
        color="orange",
        label="Rust Whitening",
    )
    plt.ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
    plt.xlabel("Samples", fontsize=AXIS_LABEL_SIZE)
    plt.legend()
    plt.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# Plot multi-channel data for better comparison
channel_list = np.arange(start=16, stop=32, step=1)

if HAS_RUST:
    # Compare Python vs Rust CMR using multi-channel plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))

    # Plot Python CMR
    plt.sca(axes[0])
    plot_multi_channel_data(
        data_cmr_py,
        time_window=[2, 3],
        figsize=(10, 15),
        fs=fs,
        color="blue",
        color_mode="single",
        channels=list(channel_list),
    )
    plt.title("Python CMR", fontsize=TITLE_SIZE + 2)

    # Plot Rust CMR
    plt.sca(axes[1])
    plot_multi_channel_data(
        data_cmr_rs,
        time_window=[2, 3],
        figsize=(10, 15),
        fs=fs,
        color="green",
        color_mode="single",
        channels=list(channel_list),
    )
    plt.title("Rust CMR", fontsize=TITLE_SIZE + 2)

    plt.tight_layout()
    plt.savefig("multichannel_cmr_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Compare Python vs Rust Whitening using multi-channel plotting
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))

    # Plot Python Whitening
    plt.sca(axes[0])
    plot_multi_channel_data(
        data_white_py,
        time_window=[2, 3],
        figsize=(10, 15),
        fs=fs,
        color="purple",
        color_mode="single",
        channels=list(channel_list),
    )
    plt.title("Python Whitening", fontsize=TITLE_SIZE + 2)

    # Plot Rust Whitening
    plt.sca(axes[1])
    plot_multi_channel_data(
        data_white_rs,
        time_window=[2, 3],
        figsize=(10, 15),
        fs=fs,
        color="orange",
        color_mode="single",
        channels=list(channel_list),
    )
    plt.title("Rust Whitening", fontsize=TITLE_SIZE + 2)

    plt.tight_layout()
    plt.savefig("multichannel_whitening_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
# Run a scale test to measure performance at different data sizes
if HAS_RUST:
    print("Running scale test...")

    # Define different test durations
    durations = [5, 10, 30, 60]  # seconds

    # Initialize result lists
    py_cmr_times = []
    rs_cmr_times = []
    py_white_times = []
    rs_white_times = []

    for duration in durations:
        samples = int(fs * duration)
        end_sample = min(samples, stream_hd.data.shape[0])
        print(f"Testing with {duration} seconds of data ({end_sample} samples)")

        # Python CMR
        start_time = time.time()
        _ = processor_cmr_py.process(end=end_sample).compute()
        py_cmr_times.append(time.time() - start_time)

        # Rust CMR
        start_time = time.time()
        _ = processor_cmr_rs.process(end=end_sample).compute()
        rs_cmr_times.append(time.time() - start_time)

        # Python Whitening
        start_time = time.time()
        _ = processor_white_py.process(end=end_sample).compute()
        py_white_times.append(time.time() - start_time)

        # Rust Whitening
        start_time = time.time()
        _ = processor_white_rs.process(end=end_sample).compute()
        rs_white_times.append(time.time() - start_time)

    # Plot scaling results
    plt.figure(figsize=(15, 8))

    # CMR scaling
    plt.subplot(1, 2, 1)
    plt.plot(durations, py_cmr_times, "o-", color="blue", label="Python CMR")
    plt.plot(durations, rs_cmr_times, "o-", color="green", label="Rust CMR")
    plt.title("CMR Processing Time Scaling", fontsize=TITLE_SIZE)
    plt.xlabel("Duration (seconds)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("Processing Time (seconds)", fontsize=AXIS_LABEL_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Whitening scaling
    plt.subplot(1, 2, 2)
    plt.plot(durations, py_white_times, "o-", color="purple", label="Python Whitening")
    plt.plot(durations, rs_white_times, "o-", color="orange", label="Rust Whitening")
    plt.title("Whitening Processing Time Scaling", fontsize=TITLE_SIZE)
    plt.xlabel("Duration (seconds)", fontsize=AXIS_LABEL_SIZE)
    plt.ylabel("Processing Time (seconds)", fontsize=AXIS_LABEL_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("performance_scaling.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print speedup factors for each duration
    print("\nCMR Speedup Factors:")
    for i, duration in enumerate(durations):
        speedup = py_cmr_times[i] / rs_cmr_times[i]
        print(f"{duration} seconds: {speedup:.2f}x")

    print("\nWhitening Speedup Factors:")
    for i, duration in enumerate(durations):
        speedup = py_white_times[i] / rs_white_times[i]
        print(f"{duration} seconds: {speedup:.2f}x")

print("All tests completed!")
