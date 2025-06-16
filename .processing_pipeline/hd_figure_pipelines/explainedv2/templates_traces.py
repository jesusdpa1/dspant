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
    channels=range(10),
    time_window=(0, 1),
    figsize=(15, 10),
)

# %%

# %%
# HD Multi-Channel Recording with Template Overlays
# Note: Assumes data is already loaded and preprocessed:
# - whitened_data: final processed data array
# - sorter_data: loaded kilosort data
# - FS: sampling frequency

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np

# Set publication style
mpu.set_publication_style()

# Configuration
CHANNELS = [0, 1, 2, 4, 6, 18, 22, 25]  # 8 channels to plot
RECORDING_START_TIME = 1.0  # Start time in seconds (adjusted for 5min data)
WINDOW_DURATION = 0.35  # Window duration in seconds
ROW_SPACING = 35.0  # Vertical spacing between channels
FIGURE_SIZE = (15, 12)
AMPLIFICATION = 2.0  # Trace amplitude scaling factor

# Template parameters
NUM_SPIKES_TO_TEMPLATE = 100
PRE_SAMPLES_MS = 1.0  # ms before spike
POST_SAMPLES_MS = 2.0  # ms after spike
MIN_SPIKES_THRESHOLD = 50
TEMPLATE_ALPHA = 0.8
TEMPLATE_LINEWIDTH = 2.5

# Colors
DARK_NAVY = mpu.PRIMARY_COLOR  # Raw traces
TEMPLATE_COLORS = [
    "#E0A500",  # Gold
    "#4512FA",  # Purple
    "#A43E2C",  # Red
    "#0B7A57",  # Green
    "#FF6B6B",  # Light Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Light Green
]

# Font sizes
FONT_SIZE = 35
TITLE_SIZE = int(FONT_SIZE * 1.0)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.8)
TICK_SIZE = int(FONT_SIZE * 0.8)
LABEL_SIZE = int(FONT_SIZE * 0.8)

print("Starting 8-channel template overlay visualization...")
print(f"Using data shape: {whitened_data.shape}")
print(f"Sampling frequency: {FS} Hz")


# %%
# STEP 1: Select best unit for each channel
def get_best_unit_for_channel(channel_id, sorter_data):
    """Select the best unit for a given channel based on quality and spike count"""

    # Get all unique units
    unique_units = np.unique(sorter_data.spike_clusters)
    channel_units = []

    for unit_id in unique_units:
        # Check if templates are available to find unit's primary channel
        if (
            hasattr(sorter_data, "templates_data")
            and sorter_data.templates_data is not None
        ):
            templates = sorter_data.templates_data["templates"]
            if unit_id < len(templates):
                unit_template = templates[unit_id]
                max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
                unit_channel = max_channel - 1  # Convert to 0-indexed

                if unit_channel == channel_id:
                    # Get spike count
                    unit_mask = sorter_data.spike_clusters == unit_id
                    n_spikes = np.sum(unit_mask)

                    if n_spikes >= MIN_SPIKES_THRESHOLD:
                        # Get quality and amplitude
                        quality = "unknown"
                        amplitude = 0
                        if hasattr(sorter_data, "unit_properties"):
                            quality = sorter_data.unit_properties.get(
                                "KSLabel", {}
                            ).get(unit_id, "unknown")
                            amplitude = sorter_data.unit_properties.get(
                                "Amplitude", {}
                            ).get(unit_id, 0)

                        channel_units.append(
                            {
                                "unit_id": unit_id,
                                "n_spikes": n_spikes,
                                "quality": quality,
                                "amplitude": amplitude,
                            }
                        )

    if not channel_units:
        return None

    # Sort by quality (good > mua) then by spike count
    def unit_score(unit):
        quality_score = (
            2 if unit["quality"] == "good" else 1 if unit["quality"] == "mua" else 0
        )
        return (quality_score, unit["n_spikes"], unit["amplitude"])

    best_unit = max(channel_units, key=unit_score)
    print(
        f"Channel {channel_id}: Selected Unit {best_unit['unit_id']} ({best_unit['quality']}, {best_unit['n_spikes']} spikes)"
    )
    return best_unit["unit_id"]


# %%
# STEP 2: Extract template for a unit
def extract_unit_template(unit_id, channel_id, whitened_data, sorter_data, fs):
    """Extract normalized template for a unit"""

    # Get spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    if len(all_spike_times) == 0:
        return None

    # Subsample spikes if necessary
    if len(all_spike_times) > NUM_SPIKES_TO_TEMPLATE:
        np.random.seed(42 + unit_id)
        sampled_indices = np.random.choice(
            len(all_spike_times), NUM_SPIKES_TO_TEMPLATE, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
    else:
        sampled_spikes = all_spike_times

    # Extract waveforms
    pre_samples = int(PRE_SAMPLES_MS * fs / 1000)
    post_samples = int(POST_SAMPLES_MS * fs / 1000)

    waveforms = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whitened_data.shape[0]:
            try:
                waveform = whitened_data[start_idx:end_idx, channel_id].compute()
                waveforms.append(waveform)
            except:
                continue

    if len(waveforms) < 10:
        return None

    # Compute template
    waveforms_array = np.array(waveforms)
    template_mean = np.mean(waveforms_array, axis=0)

    # Z-score normalization
    if np.std(template_mean) > 0:
        template_normalized = (template_mean - np.mean(template_mean)) / np.std(
            template_mean
        )
    else:
        template_normalized = template_mean - np.mean(template_mean)

    # Create time array
    template_time = np.arange(len(template_normalized)) / fs

    return {
        "template": template_normalized,
        "template_time": template_time,
        "spike_times": sampled_spikes,
        "n_waveforms": len(waveforms),
    }


# %%
# STEP 3: Find best units for each channel
print("Finding best units for each channel...")
channel_units = {}
channel_templates = {}

for ch_idx, channel in enumerate(CHANNELS):
    # Find best unit for this channel
    best_unit = get_best_unit_for_channel(channel, sorter_data)

    if best_unit is not None:
        # Extract template
        template_info = extract_unit_template(
            best_unit, channel, whitened_data, sorter_data, FS
        )

        if template_info is not None:
            channel_units[channel] = best_unit
            channel_templates[channel] = template_info
            print(
                f"✓ Channel {channel}: Unit {best_unit}, {template_info['n_waveforms']} waveforms"
            )
        else:
            print(f"✗ Channel {channel}: Unit {best_unit} template extraction failed")
    else:
        print(f"✗ Channel {channel}: No suitable units found")

print(f"\nSuccessfully extracted templates for {len(channel_templates)} channels")

# %%
# STEP 4: Create the plot with template overlays
# Extract HD data for the specified time window
start_sample = int(RECORDING_START_TIME * FS)
end_sample = int((RECORDING_START_TIME + WINDOW_DURATION) * FS)

# Get HD data for selected channels
hd_data = whitened_data[start_sample:end_sample, CHANNELS].compute()
hd_time = np.arange(hd_data.shape[0]) / FS

# Normalize each channel
hd_normalized = np.zeros_like(hd_data, dtype=float)
for i in range(len(CHANNELS)):
    channel_data = hd_data[:, i]
    # Z-score normalization
    channel_mean = np.mean(channel_data)
    channel_std = np.std(channel_data)
    if channel_std > 0:
        hd_normalized[:, i] = (channel_data - channel_mean) / channel_std
    else:
        hd_normalized[:, i] = channel_data - channel_mean

# Create plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
total_channels = len(CHANNELS)

# Plot each HD channel with template overlays
for ch_idx, channel in enumerate(CHANNELS):
    # Calculate y position (top to bottom)
    y_position = (total_channels - 1 - ch_idx) * ROW_SPACING

    # Plot raw channel data in dark navy
    ax.plot(
        hd_time,
        (hd_normalized[:, ch_idx] * AMPLIFICATION) + y_position,
        color=DARK_NAVY,
        linewidth=1.5,
        alpha=0.6,  # Slightly more transparent to show templates
        label=f"Ch {channel}" if ch_idx == 0 else None,
    )

    # Overlay template if available
    if channel in channel_templates:
        template_info = channel_templates[channel]
        unit_id = channel_units[channel]
        template_color = TEMPLATE_COLORS[ch_idx % len(TEMPLATE_COLORS)]

        # Find spikes in the current time window
        window_start_sample = start_sample
        window_end_sample = end_sample

        # Get spikes in time window
        spike_mask = (template_info["spike_times"] >= window_start_sample) & (
            template_info["spike_times"] <= window_end_sample
        )
        window_spikes = template_info["spike_times"][spike_mask]

        # Overlay templates at spike locations
        template_duration = len(template_info["template"]) / FS

        for spike_idx, spike_time in enumerate(
            window_spikes[:10]
        ):  # Limit to first 10 spikes for clarity
            # Convert spike time to relative time in the window
            spike_time_rel = (spike_time - start_sample) / FS

            # Check if template fits in window
            if (
                spike_time_rel >= 0
                and spike_time_rel + template_duration <= WINDOW_DURATION
            ):
                template_time_shifted = template_info["template_time"] + spike_time_rel
                template_scaled = (
                    template_info["template"] * AMPLIFICATION
                ) + y_position

                # Plot template overlay
                ax.plot(
                    template_time_shifted,
                    template_scaled,
                    color=template_color,
                    linewidth=TEMPLATE_LINEWIDTH,
                    alpha=TEMPLATE_ALPHA,
                    label=f"Unit {unit_id}" if spike_idx == 0 else None,
                )

    # Add channel label
    ax.text(
        -0.02,
        y_position,
        f"Ch {channel}",
        ha="right",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        color=DARK_NAVY,
        transform=ax.get_yaxis_transform(),
    )

# Set axis limits
y_bottom = -ROW_SPACING * 0.8
y_top = (total_channels - 1) * ROW_SPACING + ROW_SPACING * 0.5

# Format axis
ax.set_xlim(0, WINDOW_DURATION)
ax.set_ylim(y_bottom, y_top)
ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# Remove y-axis ticks and spines
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Add title
ax.set_title("HD Channels with Spike Template Overlays", fontsize=TITLE_SIZE, pad=20)

# Add legend for templates
template_handles = []
template_labels = []
for ch_idx, channel in enumerate(CHANNELS):
    if channel in channel_templates:
        unit_id = channel_units[channel]
        template_color = TEMPLATE_COLORS[ch_idx % len(TEMPLATE_COLORS)]
        template_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=template_color,
                linewidth=TEMPLATE_LINEWIDTH,
                alpha=TEMPLATE_ALPHA,
            )
        )
        template_labels.append(f"Ch {channel} - Unit {unit_id}")

if template_handles:
    ax.legend(
        template_handles,
        template_labels,
        loc="upper right",
        fontsize=TICK_SIZE * 0.8,
        framealpha=0.9,
    )

plt.tight_layout()

# Save figure
# FIGURE_DIR = Path(os.getenv("FIGURE_DIR_HD", "."))
# FIGURE_PATH = FIGURE_DIR.joinpath("hd_channels_with_templates.png")
# mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

# %%
# Print summary
print(f"\n=== HD Recording with Template Overlays Summary ===")
print(f"Channels plotted: {CHANNELS}")
print(
    f"Time window: {RECORDING_START_TIME}s - {RECORDING_START_TIME + WINDOW_DURATION}s"
)
print(f"Templates extracted for {len(channel_templates)} channels:")

for channel in CHANNELS:
    if channel in channel_templates:
        unit_id = channel_units[channel]
        n_waveforms = channel_templates[channel]["n_waveforms"]
        print(f"  Channel {channel}: Unit {unit_id} ({n_waveforms} waveforms)")
    else:
        print(f"  Channel {channel}: No template available")

print(f"\nTemplate parameters:")
print(f"  Window: {PRE_SAMPLES_MS}ms pre + {POST_SAMPLES_MS}ms post")
print(f"  Max spikes per template: {NUM_SPIKES_TO_TEMPLATE}")
print(f"  Min spikes threshold: {MIN_SPIKES_THRESHOLD}")
print(f"  Template alpha: {TEMPLATE_ALPHA}")

# %%
