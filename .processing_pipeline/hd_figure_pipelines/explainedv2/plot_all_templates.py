# %%

import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

from dspant.engine import create_processing_node
from dspant.io.loaders.parquet_reader import ParquetReader
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.io.loaders.zarr_loader import ZarrReader
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.extractors.template_extractor import TemplateExtractor
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()
# %%
# Data loading configuration
BASE_DIR = Path(os.getenv("BASE_DIR"))
DATA_DIR = BASE_DIR.joinpath(
    r"hd_paper/large_contacts/24-09-06_5042-2_testSubject_DST-and-contusion"
)

ZARR_PATH = DATA_DIR.joinpath(r"drv_zarr/drv_sliced_5min_00_baseline.zarr")

SORTER_PATH = DATA_DIR.joinpath(
    r"drv_ks4/drv_sliced_5min_00_baseline/ks4_output_2025-06-16_14-21/sorter_output"
)

# %%
# Load data
# HD DATA
stream_hd = ZarrReader(str(ZARR_PATH))
stream_hd.load_metadata()
stream_hd.load_data()
FS = stream_hd.sampling_frequency
# %%
# SORTER DATA
sorter_data = load_kilosort(SORTER_PATH)
sorter_data.load_data(load_templates=True, force_reload=True)
sorter_data.summarize()
# %%
sorter_data.get_channel_summary()
# %%
# PREPROCESSING
notch_filter = create_notch_filter(60, q=10, fs=FS)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=4)

# Create pre-processing functions
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)

cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()
# %%
notch_data = notch_processor.process(stream_hd.data, FS)
bandpassed_data = bandpass_processor_hd.process(notch_data, FS)
# %%
cmr_data = cmr_processor.process(bandpassed_data, FS).persist()
# %%
whitened_data = whiten_processor.process(cmr_data, FS).persist()

# %%
a = plot_multi_channel_data(
    whitened_data,
    fs=FS,
    channels=range(10),
    time_window=(0, 1),
    figsize=(15, 10),
)

# %%
# Plot 1 channel for 1 second with overlaid spike waveforms
# Correctly identifying which units belong to which channel

# Configuration
channel_to_plot = 0  # Change this to plot different channels
time_start = 0  # Start time in seconds
time_duration = 1  # Duration in seconds
waveform_window_ms = 2  # Window around each spike in milliseconds

# Convert time to samples
start_sample = int(time_start * FS)
end_sample = int((time_start + time_duration) * FS)
waveform_samples = int(waveform_window_ms * FS / 1000)  # Convert ms to samples

# Extract the data for the specified channel and time window
channel_data = whitened_data[start_sample:end_sample, channel_to_plot].compute()
time_axis = np.arange(len(channel_data)) / FS + time_start

# Find which units have their primary channel on the channel we're plotting
templates = sorter_data.templates_data[
    "templates"
]  # Shape: (n_units, n_timepoints, n_channels)
channel_map = sorter_data.templates_data.get(
    "channel_map", np.arange(templates.shape[2])
)

# For each unit, find the channel with maximum template amplitude
units_on_channel = []
unit_primary_channels = {}

for unit_idx, unit_id in enumerate(sorter_data.unit_ids):
    if unit_idx < templates.shape[0]:  # Make sure template exists
        # Get template for this unit across all channels
        unit_template = templates[unit_idx]  # Shape: (n_timepoints, n_channels)

        # Find channel with maximum absolute amplitude
        max_amp_per_channel = np.max(np.abs(unit_template), axis=0)
        primary_channel_idx = np.argmax(max_amp_per_channel)

        # Map to actual channel number
        primary_channel = (
            channel_map[primary_channel_idx]
            if len(channel_map) > primary_channel_idx
            else primary_channel_idx
        )
        unit_primary_channels[unit_id] = int(primary_channel)

        # Check if this unit belongs to the channel we're plotting
        if int(primary_channel) == channel_to_plot:
            units_on_channel.append(unit_id)

print(f"Units with primary channel on channel {channel_to_plot}: {units_on_channel}")
print(f"Unit primary channel mapping: {unit_primary_channels}")

# Get spike times in the time window for units on this channel
all_spike_times_s = sorter_data.spike_times / FS
spike_mask = (all_spike_times_s >= time_start) & (
    all_spike_times_s < time_start + time_duration
)
spikes_in_window = sorter_data.spike_times[spike_mask]
spike_clusters_in_window = sorter_data.spike_clusters[spike_mask]

# Filter spikes to only include units that belong to this channel
channel_spike_mask = np.isin(spike_clusters_in_window, units_on_channel)
spikes_on_channel = spikes_in_window[channel_spike_mask]
spike_clusters_on_channel = spike_clusters_in_window[channel_spike_mask]

# Create the plot
fig, ax = plt.subplots(1, 1, figsize=(15, 8))

# Plot the continuous data
ax.plot(
    time_axis, channel_data, "k-", linewidth=0.8, alpha=0.7, label="Continuous data"
)

# Get unique units and assign colors (avoiding grey/black)
unique_units = np.unique(spike_clusters_on_channel)
# Use a color palette with vibrant, distinct colors
color_palette = [
    "#e41a1c",
    "#ff7f00",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#00ced1",
    "#ff1493",
    "#32cd32",
    "#8b0000",
    "#00008b",
    "#ff8c00",
    "#9400d3",
]
# Cycle through colors if we have more units than colors
unit_colors = {
    unit_id: color_palette[i % len(color_palette)]
    for i, unit_id in enumerate(unique_units)
}

# Extract and plot waveforms around each spike
waveforms_plotted = {}  # Track which units have been labeled

for spike_idx, (spike_time_samples, unit_id) in enumerate(
    zip(spikes_on_channel, spike_clusters_on_channel)
):
    # Calculate waveform extraction window
    spike_sample_in_chunk = spike_time_samples - start_sample
    waveform_start = max(0, spike_sample_in_chunk - waveform_samples // 2)
    waveform_end = min(len(channel_data), spike_sample_in_chunk + waveform_samples // 2)

    # Skip if spike is too close to edges
    if waveform_end - waveform_start < waveform_samples // 2:
        continue

    # Extract waveform
    waveform = channel_data[waveform_start:waveform_end]
    waveform_time = time_axis[waveform_start:waveform_end]

    # Plot waveform
    color = unit_colors[unit_id]
    alpha = 0.7 if unit_id in waveforms_plotted else 0.9  # First waveform more opaque
    label = f"Unit {unit_id}" if unit_id not in waveforms_plotted else None

    ax.plot(
        waveform_time, waveform, color=color, linewidth=2.5, alpha=alpha, label=label
    )

    # Mark spike time
    spike_time_s = spike_time_samples / FS
    if time_start <= spike_time_s < time_start + time_duration:
        ax.axvline(spike_time_s, color=color, linestyle="--", alpha=0.4, linewidth=1)

    # Mark this unit as having been labeled
    waveforms_plotted[unit_id] = True

# Formatting
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Amplitude (μV)", fontsize=12)
ax.set_title(
    f"Channel {channel_to_plot} - {time_duration}s window with spike waveforms\n"
    f"({len(spikes_on_channel)} spikes from {len(unique_units)} units with primary channel {channel_to_plot})",
    fontsize=14,
)

if len(unique_units) > 0:
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
ax.grid(True, alpha=0.3)

# Set x-axis limits
ax.set_xlim(time_start, time_start + time_duration)

plt.tight_layout()
plt.show()

# Print detailed summary
print(f"\nDetailed Summary:")
print(f"Channel: {channel_to_plot}")
print(f"Time window: {time_start}-{time_start + time_duration}s")
print(f"Total spikes in time window (all channels): {len(spikes_in_window)}")
print(
    f"Spikes from units with primary channel {channel_to_plot}: {len(spikes_on_channel)}"
)
print(
    f"Units with primary channel {channel_to_plot}: {sorted(unique_units) if len(unique_units) > 0 else 'None'}"
)

if len(unique_units) > 0:
    for unit_id in unique_units:
        unit_spike_count = np.sum(spike_clusters_on_channel == unit_id)
        print(f"  Unit {unit_id}: {unit_spike_count} spikes (all waveforms plotted)")
else:
    print("No units have their primary channel on the selected channel.")
    print("Try a different channel or check the channel mapping.")

# %%
