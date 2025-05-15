"""
Create visualization showing neural activity aligned to EMG contractions
Author: Jesus Penaloza
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
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.density_estimation import (
    SpikeDensityEstimator,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()

# %%
# Load data (using your existing code)
data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/RawG.ant")
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)

# Load EMG and spike data
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
sorter_data = load_kilosort(sorter_path)
fs = stream_emg.fs

# %%
# Process EMG data
processor_emg = create_processing_node(stream_emg)

# Create filters
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Filter EMG data
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
filtered_emg = processor_emg.process(group=["filters"]).persist()

# %%
# Apply TKEO
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=15)
tkeo_data = tkeo_processor.process(filtered_emg[0:1000000, :], fs=fs).persist()

# Normalize TKEO
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()

# %%
# Create 3x1 grid visualization with debugging
# Set time window for visualization
start_time = 0.0  # start time in seconds
end_time = 4  # end time in seconds

# Create spike density estimator
bin_size_ms = 2.0
sigma_ms = 4.0
estimator = SpikeDensityEstimator(bin_size_ms=bin_size_ms, sigma_ms=sigma_ms)

# Let's debug the issue
print(f"Requested time range: {start_time}s to {end_time}s")

# Get spike density
time_bins, spike_density, used_unit_ids = estimator.estimate(
    sorter=sorter_data, start_time_s=start_time, end_time_s=end_time
)

print(f"Actual time bins range: {time_bins[0]}s to {time_bins[-1]}s")
print(f"Number of bins: {len(time_bins)}")
print(f"Expected duration: {end_time - start_time}s")
print(f"Actual duration: {time_bins[-1] - time_bins[0]}s")

# Convert times to samples
start_sample = int(start_time * fs)
end_sample = int(end_time * fs)

# Create figure with subplots
figsize = (14, 12)
fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

# First row: EMG with envelope
ax_emg = axes[0]

# Extract EMG data window
emg_time = np.arange(start_sample, end_sample) / fs
emg_signal = filtered_emg[start_sample:end_sample, 0].compute()
tkeo_signal = zscore_tkeo[start_sample:end_sample, 0].compute()

# Plot EMG and envelope
ax_emg.plot(emg_time, emg_signal, color="gray", alpha=0.8, label="EMG")
ax_emg.plot(
    emg_time,
    tkeo_signal * np.max(np.abs(emg_signal)),
    color="red",
    linewidth=2,
    label="TKEO Envelope",
)
ax_emg.set_ylabel("EMG Signal")
ax_emg.set_title("EMG Signal with TKEO Envelope")
ax_emg.legend(loc="upper right")
ax_emg.grid(True, alpha=0.3)

# Second row: Continuous raster plot
ax_raster = axes[1]

# Get spike times for all units in the time window
for unit_idx, unit_id in enumerate(used_unit_ids):
    # Get spike times for this unit
    spike_times = sorter_data.get_unit_spike_train(unit_id)
    spike_times_sec = spike_times / fs

    # Filter spikes within time window
    mask = (spike_times_sec >= start_time) & (spike_times_sec < end_time)
    selected_spikes = spike_times_sec[mask]

    # Plot spikes
    if len(selected_spikes) > 0:
        ax_raster.vlines(
            selected_spikes, unit_idx - 0.4, unit_idx + 0.4, color="black", linewidth=1
        )

ax_raster.set_ylabel("Unit ID")
ax_raster.set_title("Continuous Raster Plot")
ax_raster.set_yticks(range(len(used_unit_ids)))
ax_raster.set_yticklabels(used_unit_ids)
ax_raster.set_ylim(-0.5, len(used_unit_ids) - 0.5)
ax_raster.grid(True, alpha=0.3)

# Third row: Spike density heatmap
ax_density = axes[2]

# Create density heatmap
im = ax_density.imshow(
    spike_density.T,
    aspect="auto",
    origin="lower",
    interpolation="none",
    cmap="viridis",
    extent=[time_bins[0], time_bins[-1], -0.5, len(used_unit_ids) - 0.5],
)

ax_density.set_xlabel("Time (s)")
ax_density.set_ylabel("Unit ID")
ax_density.set_title("Spike Density Heatmap")
ax_density.set_yticks(range(len(used_unit_ids)))
ax_density.set_yticklabels(used_unit_ids)

# Set common x-axis properties - force to requested range
for ax in axes:
    ax.set_xlim(start_time, end_time)  # Use the requested time range
    ax.grid(True, alpha=0.3)
ax_density.grid(False)
# Adjust layout
plt.tight_layout()
plt.show()
# %%
