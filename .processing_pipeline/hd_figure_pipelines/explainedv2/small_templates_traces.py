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
    r"hd_paper/small_contacts/25-03-26_4902-1_testSubject_topoMapping"
)

ZARR_PATH = DATA_DIR.joinpath(r"drv_zarr/drv_sliced_5min_00_baseline.zarr")

SORTER_PATH = DATA_DIR.joinpath(
    r"drv_ks4/drv_sliced_5min_00_baseline/ks4_output_2025-06-17_13-43/sorter_output"
)

# %%
# Load data
# HD DATA
stream_hd = ZarrReader(str(ZARR_PATH))
stream_hd.load_metadata()
stream_hd.load_data()
FS = stream_hd.sampling_frequency

# SORTER DATA
sorter_data = load_kilosort(SORTER_PATH)

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
    channels=range(32),
    time_window=(0, 1),
    figsize=(15, 20),
)

# %%

CHANNELS = [1, 5, 7, 9, 13, 15, 17, 24]  # 8 channels to plot
RECORDING_START_TIME = 9.9  # Start time in seconds
WINDOW_DURATION = 0.35  # Window duration in seconds
waveform_window_ms = 2  # Window around each spike in milliseconds

# Layout configuration
ROW_SPACING = 35.0  # Vertical spacing between channels
FIGURE_SIZE = (15, 12)
AMPLIFICATION = 2.0  # Trace amplitude scaling factor

# Use the new variable names
channels_to_plot = CHANNELS
time_start = RECORDING_START_TIME
time_duration = WINDOW_DURATION
TRANSPARENT_BACKGROUND = True

# Font sizes
FONT_SIZE = 35
AXIS_LABEL_SIZE = int(FONT_SIZE * 1.0)
TICK_SIZE = int(FONT_SIZE * 0.9)
LABEL_SIZE = int(FONT_SIZE * 0.9)

# MANUAL UNIT ASSIGNMENT - Modify this dictionary to assign units to channels
# Format: {channel_number: [list_of_unit_ids]}
manual_units_per_channel = {
    0: [12],  #
    1: [13],  #
    2: [15],  #
    4: [10],  #
    6: [21],  #
    18: [18],  #
    22: [16],  #
    25: [19],  #
}

# Alternative: Use automatic assignment if you want to see what the algorithm would choose
use_manual_assignment = True  # Set to False to use automatic assignment

# Convert time to samples
start_sample = int(time_start * FS)
end_sample = int((time_start + time_duration) * FS)
waveform_samples = int(waveform_window_ms * FS / 1000)  # Convert ms to samples

# Extract the data for the specified channels and time window
channel_data = whitened_data[start_sample:end_sample, channels_to_plot].compute()
time_axis = np.arange(channel_data.shape[0]) / FS + time_start

# Determine unit assignment method
if use_manual_assignment:
    print("Using MANUAL unit assignment")
    units_per_channel = {}

    # Initialize with empty lists for all channels
    for ch in channels_to_plot:
        units_per_channel[ch] = []

    # Apply manual assignments only for channels that exist in both lists
    for ch in channels_to_plot:
        if ch in manual_units_per_channel:
            # Filter to only include units that actually exist in the data
            available_units = set(sorter_data.unit_ids)
            valid_units = [
                u for u in manual_units_per_channel[ch] if u in available_units
            ]
            units_per_channel[ch] = valid_units

            # Warn about units that don't exist
            invalid_units = [
                u for u in manual_units_per_channel[ch] if u not in available_units
            ]
            if invalid_units:
                print(
                    f"Warning: Channel {ch} - Units {invalid_units} not found in data"
                )
else:
    print("Using AUTOMATIC unit assignment based on template amplitude")
    # Find which units have their primary channel on each channel we're plotting
    templates = sorter_data.templates_data[
        "templates"
    ]  # Shape: (n_units, n_timepoints, n_channels)
    channel_map = sorter_data.templates_data.get(
        "channel_map", np.arange(templates.shape[2])
    )

    # For each unit, find the channel with maximum template amplitude
    units_per_channel = {ch: [] for ch in channels_to_plot}
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

            # Check if this unit belongs to any of the channels we're plotting
            if int(primary_channel) in channels_to_plot:
                units_per_channel[int(primary_channel)].append(unit_id)

# Print summary
print("Units per channel:")
for ch in channels_to_plot:
    print(f"  Channel {ch}: {units_per_channel[ch]}")

# Get spike times in the time window
all_spike_times_s = sorter_data.spike_times / FS
spike_mask = (all_spike_times_s >= time_start) & (
    all_spike_times_s < time_start + time_duration
)
spikes_in_window = sorter_data.spike_times[spike_mask]
spike_clusters_in_window = sorter_data.spike_clusters[spike_mask]

# Create color palette for units (consistent across channels)
all_assigned_units = []
for ch_units in units_per_channel.values():
    all_assigned_units.extend(ch_units)
all_assigned_units = sorted(set(all_assigned_units))

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
    "#ff69b4",
    "#00ff7f",
    "#dc143c",
    "#00bfff",
    "#ffd700",
]
unit_colors = {
    unit_id: color_palette[i % len(color_palette)]
    for i, unit_id in enumerate(all_assigned_units)
}

# Normalize each channel for consistent visualization
channel_data_normalized = np.zeros_like(channel_data, dtype=float)
for i in range(len(channels_to_plot)):
    ch_data = channel_data[:, i]
    # Z-score normalization
    ch_mean = np.mean(ch_data)
    ch_std = np.std(ch_data)
    if ch_std > 0:
        channel_data_normalized[:, i] = (ch_data - ch_mean) / ch_std
    else:
        channel_data_normalized[:, i] = ch_data - ch_mean

# Create the compact multi-channel plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

total_channels = len(channels_to_plot)

# Plot each channel
for plot_idx, channel_idx in enumerate(channels_to_plot):
    # Calculate y position (top to bottom)
    y_position = (total_channels - 1 - plot_idx) * ROW_SPACING

    # Get normalized channel signal
    channel_signal = channel_data_normalized[:, plot_idx]

    # Plot the continuous data as baseline in dark color
    baseline_trace = (channel_signal * AMPLIFICATION) + y_position
    ax.plot(
        time_axis,
        baseline_trace,
        color="#2C3E50",  # Dark blue-grey
        linewidth=1.0,
        alpha=0.7,
        zorder=1,
    )

    # Add channel label
    ax.text(
        -0.02,
        y_position,
        f"Ch {channel_idx}",
        ha="right",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        color="#2C3E50",
        transform=ax.get_yaxis_transform(),
    )

    # Filter spikes to only include units that belong to this channel
    units_on_channel = units_per_channel[channel_idx]
    if len(units_on_channel) > 0:
        channel_spike_mask = np.isin(spike_clusters_in_window, units_on_channel)
        spikes_on_channel = spikes_in_window[channel_spike_mask]
        spike_clusters_on_channel = spike_clusters_in_window[channel_spike_mask]

        # Track which units have been labeled for legend
        waveforms_plotted = {}

        # Extract and plot waveforms around each spike
        for spike_idx, (spike_time_samples, unit_id) in enumerate(
            zip(spikes_on_channel, spike_clusters_on_channel)
        ):
            # Calculate waveform extraction window
            spike_sample_in_chunk = spike_time_samples - start_sample
            waveform_start = max(0, spike_sample_in_chunk - waveform_samples // 2)
            waveform_end = min(
                len(channel_signal), spike_sample_in_chunk + waveform_samples // 2
            )

            # Skip if spike is too close to edges
            if waveform_end - waveform_start < waveform_samples // 2:
                continue

            # Extract waveform
            waveform = channel_signal[waveform_start:waveform_end]
            waveform_time = time_axis[waveform_start:waveform_end]

            # Apply y position offset
            waveform_positioned = (waveform * AMPLIFICATION) + y_position

            # Plot waveform
            color = unit_colors.get(
                unit_id, "#808080"
            )  # Default to gray if unit not in color map
            alpha = 0.8 if unit_id in waveforms_plotted else 0.9
            label = f"Unit {unit_id}" if unit_id not in waveforms_plotted else None

            ax.plot(
                waveform_time,
                waveform_positioned,
                color=color,
                linewidth=2.0,
                alpha=alpha,
                label=label,
                zorder=2,
            )

            # Mark this unit as having been labeled
            waveforms_plotted[unit_id] = True

    # Add statistics text box for each channel
    n_spikes = (
        len(
            [
                s
                for s, c in zip(spikes_on_channel, spike_clusters_on_channel)
                if c in units_on_channel
            ]
        )
        if len(units_on_channel) > 0
        else 0
    )

# Set compact axis limits
y_bottom = -ROW_SPACING * 0.8  # Small margin below last channel
y_top = (
    total_channels - 1
) * ROW_SPACING + ROW_SPACING * 0.5  # Minimal margin above first channel

# Format axis
ax.set_xlim(time_start, time_start + time_duration)
ax.set_ylim(y_bottom, y_top)
ax.set_xlabel("Time (s)", fontsize=TICK_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# Remove y-axis ticks and spines
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)


plt.tight_layout()
plt.subplots_adjust(top=0.88)

plt.show()

# %%
# WIDGET
# Simple Interactive Channel and Unit Spike Visualization
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np

# Configuration
RECORDING_START_TIME = 9.75  # Start time in seconds
WINDOW_DURATION = 0.35  # Window duration in seconds
waveform_window_ms = 4  # Window around each spike in milliseconds
AMPLIFICATION = 2.0  # Trace amplitude scaling factor

# Available channels and units
AVAILABLE_CHANNELS = [1, 5, 7, 9, 13, 15, 17, 24]
available_units = sorted(list(sorter_data.unit_ids))

# Time setup
time_start = RECORDING_START_TIME
time_duration = WINDOW_DURATION
start_sample = int(time_start * FS)
end_sample = int((time_start + time_duration) * FS)
waveform_samples = int(waveform_window_ms * FS / 1000)

# Get spike data for the time window
all_spike_times_s = sorter_data.spike_times / FS
spike_mask = (all_spike_times_s >= time_start) & (
    all_spike_times_s < time_start + time_duration
)
spikes_in_window = sorter_data.spike_times[spike_mask]
spike_clusters_in_window = sorter_data.spike_clusters[spike_mask]

# Color palette
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
    "#ff69b4",
    "#00ff7f",
    "#dc143c",
    "#00bfff",
    "#ffd700",
]


def plot_channel_unit(channel_idx=6, unit_id=0):
    """Plot function that gets called when sliders change"""

    # Clear any existing plots
    plt.clf()

    # Load and normalize channel data
    channel_data = whitened_data[start_sample:end_sample, channel_idx].compute()
    time_axis = np.arange(channel_data.shape[0]) / FS + time_start

    # Normalize
    ch_mean = np.mean(channel_data)
    ch_std = np.std(channel_data)
    if ch_std > 0:
        channel_data_normalized = (channel_data - ch_mean) / ch_std
    else:
        channel_data_normalized = channel_data - ch_mean

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Plot baseline continuous data
    plt.plot(
        time_axis,
        channel_data_normalized * AMPLIFICATION,
        color="#2C3E50",
        linewidth=1.2,
        alpha=0.8,
        label="Continuous data",
        zorder=1,
    )

    # Filter spikes for the selected unit
    unit_spike_mask = spike_clusters_in_window == unit_id
    unit_spikes = spikes_in_window[unit_spike_mask]
    n_spikes = len(unit_spikes)

    # Plot spike waveforms if any exist
    if n_spikes > 0:
        unit_color = color_palette[unit_id % len(color_palette)]

        # Track if we've added the unit to legend yet
        unit_labeled = False

        for spike_time_samples in unit_spikes:
            # Calculate waveform extraction window
            spike_sample_in_chunk = spike_time_samples - start_sample
            waveform_start = max(0, spike_sample_in_chunk - waveform_samples // 2)
            waveform_end = min(
                len(channel_data_normalized),
                spike_sample_in_chunk + waveform_samples // 2,
            )

            # Skip if spike is too close to edges
            if waveform_end - waveform_start < waveform_samples // 2:
                continue

            # Extract waveform
            waveform = channel_data_normalized[waveform_start:waveform_end]
            waveform_time = time_axis[waveform_start:waveform_end]

            # Plot waveform
            plt.plot(
                waveform_time,
                waveform * AMPLIFICATION,
                color=unit_color,
                linewidth=2.5,
                alpha=0.8,
                label=f"Unit {unit_id}" if not unit_labeled else "",
                zorder=2,
            )
            unit_labeled = True

    # Formatting
    plt.xlim(time_start, time_start + time_duration)
    plt.xlabel("Time (s)", fontsize=TICK_SIZE)
    plt.ylabel("Normalized Amplitude", fontsize=12)
    plt.title(
        f"Channel {channel_idx} - Unit {unit_id} ({n_spikes} spikes)\n"
        f"Time: {time_start:.2f}-{time_start + time_duration:.2f}s",
        fontsize=14,
        pad=20,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    # Add statistics text
    plt.text(
        0.02,
        0.98,
        f"Ch {channel_idx}, Unit {unit_id}: {n_spikes} spikes\nWaveform window: ±{waveform_window_ms}ms",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    plt.show()


# Create sliders
channel_slider = widgets.Dropdown(
    options=AVAILABLE_CHANNELS,
    value=24,  # Default channel
    description="Channel:",
)

unit_slider = widgets.IntSlider(
    value=available_units[0] if available_units else 0,
    min=min(available_units) if available_units else 0,
    max=max(available_units) if available_units else 0,
    step=1,
    description="Unit ID:",
)

# Create the interactive widget - this is the key simple part!
interactive_plot = widgets.interactive(
    plot_channel_unit, channel_idx=channel_slider, unit_id=unit_slider
)

# Display it
interactive_plot

# %%

CHANNELS = [1, 5, 7, 9, 13, 15, 17, 24]  # 8 channels to plot
RECORDING_START_TIME = 9.75  # Start time in seconds
WINDOW_DURATION = 0.35  # Window duration in seconds
waveform_window_ms = 2  # Window around each spike in milliseconds

# Layout configuration
ROW_SPACING = 35.0  # Vertical spacing between channels
FIGURE_SIZE = (20, 12)  # Wider to accommodate templates
AMPLIFICATION = 2.0  # Trace amplitude scaling factor

# Template extraction parameters
N_WAVEFORMS_EXTRACT = 500  # Number of waveforms to extract for templates
TEMPLATE_WINDOW_MS = 4.0  # Template window in milliseconds (wider than spike window)

# Use the new variable names
channels_to_plot = CHANNELS
time_start = RECORDING_START_TIME
time_duration = WINDOW_DURATION

# Font sizes
FONT_SIZE = 35  # Reduced from 35 to fit better
AXIS_LABEL_SIZE = int(FONT_SIZE * 1.0)
TICK_SIZE = int(FONT_SIZE * 0.6)
LABEL_SIZE = int(FONT_SIZE * 0.9)

# MANUAL UNIT ASSIGNMENT - Modify this dictionary to assign units to channels
# Format: {channel_number: [list_of_unit_ids]}
manual_units_per_channel = {
    1: [49],  #
    5: [48],  #
    7: [38],  #
    9: [10],  #
    13: [18],  #
    15: [29],  #
    17: [14],  #
    24: [19],  #
}

# Alternative: Use automatic assignment if you want to see what the algorithm would choose
use_manual_assignment = True  # Set to False to use automatic assignment

# Convert time to samples
start_sample = int(time_start * FS)
end_sample = int((time_start + time_duration) * FS)
waveform_samples = int(waveform_window_ms * FS / 1000)  # Convert ms to samples
template_samples = int(TEMPLATE_WINDOW_MS * FS / 1000)  # Template window samples

# Extract the data for the specified channels and time window
channel_data = whitened_data[start_sample:end_sample, channels_to_plot].compute()
time_axis = np.arange(channel_data.shape[0]) / FS + time_start

# Determine unit assignment method
if use_manual_assignment:
    print("Using MANUAL unit assignment")
    units_per_channel = {}

    # Initialize with empty lists for all channels
    for ch in channels_to_plot:
        units_per_channel[ch] = []

    # Apply manual assignments only for channels that exist in both lists
    for ch in channels_to_plot:
        if ch in manual_units_per_channel:
            # Filter to only include units that actually exist in the data
            available_units = set(sorter_data.unit_ids)
            valid_units = [
                u for u in manual_units_per_channel[ch] if u in available_units
            ]
            units_per_channel[ch] = valid_units

            # Warn about units that don't exist
            invalid_units = [
                u for u in manual_units_per_channel[ch] if u not in available_units
            ]
            if invalid_units:
                print(
                    f"Warning: Channel {ch} - Units {invalid_units} not found in data"
                )
else:
    print("Using AUTOMATIC unit assignment based on template amplitude")
    # Find which units have their primary channel on each channel we're plotting
    templates = sorter_data.templates_data[
        "templates"
    ]  # Shape: (n_units, n_timepoints, n_channels)
    channel_map = sorter_data.templates_data.get(
        "channel_map", np.arange(templates.shape[2])
    )

    # For each unit, find the channel with maximum template amplitude
    units_per_channel = {ch: [] for ch in channels_to_plot}
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

            # Check if this unit belongs to any of the channels we're plotting
            if int(primary_channel) in channels_to_plot:
                units_per_channel[int(primary_channel)].append(unit_id)

# Print summary
print("Units per channel:")
for ch in channels_to_plot:
    print(f"  Channel {ch}: {units_per_channel[ch]}")

# Get spike times in the time window
all_spike_times_s = sorter_data.spike_times / FS
spike_mask = (all_spike_times_s >= time_start) & (
    all_spike_times_s < time_start + time_duration
)
spikes_in_window = sorter_data.spike_times[spike_mask]
spike_clusters_in_window = sorter_data.spike_clusters[spike_mask]

# Create color palette for units (consistent across channels)
all_assigned_units = []
for ch_units in units_per_channel.values():
    all_assigned_units.extend(ch_units)
all_assigned_units = sorted(set(all_assigned_units))

color_palette = [
    "#e41a1c",
    "#ff7f00",
    "#377eb8",
    "#4daf4a",
    "#670077",
    "#001ba1",
    "#a65628",
    "#9b4600",
    "#00ced1",
    "#ff1493",
    "#32cd32",
    "#8b0000",
    "#00008b",
    "#ff8c00",
    "#6A0097",
    "#ff69b4",
    "#00ff7f",
    "#dc143c",
    "#00bfff",
    "#ffd700",
]
unit_colors = {
    unit_id: color_palette[i % len(color_palette)]
    for i, unit_id in enumerate(all_assigned_units)
}

# Normalize each channel for consistent visualization
channel_data_normalized = np.zeros_like(channel_data, dtype=float)
for i in range(len(channels_to_plot)):
    ch_data = channel_data[:, i]
    # Z-score normalization
    ch_mean = np.mean(ch_data)
    ch_std = np.std(ch_data)
    if ch_std > 0:
        channel_data_normalized[:, i] = (ch_data - ch_mean) / ch_std
    else:
        channel_data_normalized[:, i] = ch_data - ch_mean


# Function to extract waveforms for template analysis
def extract_unit_waveforms(unit_id, channel_idx, n_waveforms=N_WAVEFORMS_EXTRACT):
    """Extract waveforms for a specific unit from the entire recording"""

    # Get all spikes for this unit (not just in time window)
    unit_spike_mask = sorter_data.spike_clusters == unit_id
    unit_spike_times = sorter_data.spike_times[unit_spike_mask]

    # Limit to requested number of waveforms
    if len(unit_spike_times) > n_waveforms:
        # Take evenly distributed samples
        indices = np.linspace(0, len(unit_spike_times) - 1, n_waveforms, dtype=int)
        unit_spike_times = unit_spike_times[indices]

    # Pre-allocate array for consistent dimensions
    pre_samples = template_samples // 2
    post_samples = template_samples - pre_samples

    waveforms = []

    for spike_time in unit_spike_times:
        # Calculate extraction window with exact pre/post samples
        waveform_start = spike_time - pre_samples
        waveform_end = spike_time + post_samples

        # Check bounds
        if waveform_start >= 0 and waveform_end < whitened_data.shape[0]:
            # Extract waveform for this channel
            waveform = whitened_data[waveform_start:waveform_end, channel_idx].compute()

            # Ensure exact length
            if len(waveform) == template_samples:
                waveforms.append(waveform)

    if len(waveforms) > 0:
        # Convert to 3D array: (n_waveforms, n_timepoints, 1)
        waveforms_array = np.array(waveforms)

        # Normalize each waveform individually
        normalized_waveforms = np.zeros_like(waveforms_array)
        for i, wf in enumerate(waveforms_array):
            wf_mean = np.mean(wf)
            wf_std = np.std(wf)
            if wf_std > 0:
                normalized_waveforms[i] = (wf - wf_mean) / wf_std
            else:
                normalized_waveforms[i] = wf - wf_mean

        return normalized_waveforms
    else:
        return np.array([])


mpu.set_publication_style()
# Create the 2-column grid
fig = plt.figure(figsize=FIGURE_SIZE)
gs = fig.add_gridspec(8, 2, width_ratios=[4, 0.8], hspace=0.1, wspace=0.1)

# LEFT COLUMN: Multi-channel plot
ax_main = fig.add_subplot(gs[:, 0])

total_channels = len(channels_to_plot)

# Plot each channel in the left column
for plot_idx, channel_idx in enumerate(channels_to_plot):
    # Calculate y position (top to bottom)
    y_position = (total_channels - 1 - plot_idx) * ROW_SPACING

    # Get normalized channel signal
    channel_signal = channel_data_normalized[:, plot_idx]

    # Plot the continuous data as baseline in dark color
    baseline_trace = (channel_signal * AMPLIFICATION) + y_position
    ax_main.plot(
        time_axis,
        baseline_trace,
        color="#2C3E50",  # Dark blue-grey
        linewidth=1.0,
        alpha=0.7,
        zorder=1,
    )

    # Add channel label
    ax_main.text(
        -0.02,
        y_position,
        f"Ch {plot_idx + 1}",
        ha="right",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        color="#2C3E50",
        transform=ax_main.get_yaxis_transform(),
    )

    # Filter spikes to only include units that belong to this channel
    units_on_channel = units_per_channel[channel_idx]
    if len(units_on_channel) > 0:
        channel_spike_mask = np.isin(spike_clusters_in_window, units_on_channel)
        spikes_on_channel = spikes_in_window[channel_spike_mask]
        spike_clusters_on_channel = spike_clusters_in_window[channel_spike_mask]

        # Track which units have been labeled for legend
        waveforms_plotted = {}

        # Extract and plot waveforms around each spike
        for spike_idx, (spike_time_samples, unit_id) in enumerate(
            zip(spikes_on_channel, spike_clusters_on_channel)
        ):
            # Calculate waveform extraction window
            spike_sample_in_chunk = spike_time_samples - start_sample
            waveform_start = max(0, spike_sample_in_chunk - waveform_samples // 2)
            waveform_end = min(
                len(channel_signal), spike_sample_in_chunk + waveform_samples // 2
            )

            # Skip if spike is too close to edges
            if waveform_end - waveform_start < waveform_samples // 2:
                continue

            # Extract waveform
            waveform = channel_signal[waveform_start:waveform_end]
            waveform_time = time_axis[waveform_start:waveform_end]

            # Apply y position offset
            waveform_positioned = (waveform * AMPLIFICATION) + y_position

            # Plot waveform
            color = unit_colors.get(
                unit_id, "#808080"
            )  # Default to gray if unit not in color map
            alpha = 0.8 if unit_id in waveforms_plotted else 0.9
            label = f"Unit {unit_id}" if unit_id not in waveforms_plotted else None

            ax_main.plot(
                waveform_time,
                waveform_positioned,
                color=color,
                linewidth=2.0,
                alpha=alpha,
                label=label,
                zorder=2,
            )

            # Mark this unit as having been labeled
            waveforms_plotted[unit_id] = True

# Set compact axis limits for main plot
y_bottom = -ROW_SPACING * 0.8
y_top = (total_channels - 1) * ROW_SPACING + ROW_SPACING * 0.5

# Format main axis
ax_main.set_xlim(time_start, time_start + time_duration)
ax_main.set_ylim(y_bottom, y_top)
ax_main.set_xlabel("Time (s)", fontsize=TICK_SIZE)
ax_main.tick_params(labelsize=TICK_SIZE)
ax_main.set_yticks([])
ax_main.spines["top"].set_visible(False)
ax_main.spines["right"].set_visible(False)
ax_main.spines["left"].set_visible(False)

# RIGHT COLUMN: Template waveforms for each channel
pre_samples = template_samples // 2
post_samples = template_samples - pre_samples
template_time_axis = (
    (np.arange(template_samples) - pre_samples) / FS * 1000
)  # in ms, centered at 0

for plot_idx, channel_idx in enumerate(channels_to_plot):
    ax_template = fig.add_subplot(gs[plot_idx, 1])

    # Get units for this channel
    units_on_channel = units_per_channel[channel_idx]

    if len(units_on_channel) > 0:
        for unit_id in units_on_channel:
            # Extract waveforms for this unit on this channel
            waveforms = extract_unit_waveforms(unit_id, channel_idx)

            if len(waveforms) > 0:
                # Get unit color
                color = unit_colors.get(unit_id, "#808080")

                # Plot individual waveforms (lighter)
                for i, wf in enumerate(waveforms):
                    ax_template.plot(
                        template_time_axis,
                        wf,
                        color=color,
                        alpha=0.05,
                        linewidth=0.3,
                        zorder=1,
                    )

                # Calculate mean and SEM from the 3D array
                mean_waveform = np.mean(waveforms, axis=0)
                sem_waveform = np.std(waveforms, axis=0) / np.sqrt(waveforms.shape[0])

                # Plot mean waveform (thick)
                ax_template.plot(
                    template_time_axis,
                    mean_waveform,
                    color=color,
                    linewidth=3,
                    alpha=0.9,
                    label=f"Unit {unit_id} (n={len(waveforms)})",
                    zorder=3,
                )

                # Plot SEM as shaded area
                ax_template.fill_between(
                    template_time_axis,
                    mean_waveform - sem_waveform,
                    mean_waveform + sem_waveform,
                    color=color,
                    alpha=0.2,
                    zorder=2,
                )
                ax_template.yaxis.set_visible(False)

    ax_template.tick_params(labelsize=TICK_SIZE)
    ax_template.grid(True, alpha=0.3)
    ax_template.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)

    # Only show x-label on bottom plot
    if plot_idx == len(channels_to_plot) - 1:
        ax_template.set_xlabel("Time (ms)", fontsize=TICK_SIZE)
    else:
        ax_template.set_xticklabels([])


plt.tight_layout()
plt.show()

# %%

mpu.save_figure(fig, "traces+tempalte.png", dpi=600)

# %%
