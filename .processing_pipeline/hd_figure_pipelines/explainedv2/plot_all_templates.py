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
sorter_data = PhyKilosortLoader(SORTER_PATH)
# %%
sorter_data.get
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
# ALL UNITS TEMPLATE OVERVIEW
# This creates a comprehensive plot showing all available unit templates
# to help with manual selection for the 8-channel visualization

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np

# Set publication style
mpu.set_publication_style()

# Template extraction parameters
NUM_SPIKES_TO_TEMPLATE = 100
PRE_SAMPLES_MS = 1.0  # ms before spike
POST_SAMPLES_MS = 2.0  # ms after spike
MIN_SPIKES_THRESHOLD = 20  # Lower threshold to see more units

# Visualization parameters
TEMPLATES_PER_ROW = 6
X_SPACING = 1.2
Y_SPACING = 6.0
TIME_SCALE = 0.15
FIGURE_SIZE = (20, 15)

# Colors by quality
QUALITY_COLORS = {
    "good": "#00833B",  # Green
    "mua": "#835100",  # Orange
    "unknown": "#666666",  # Gray
}

print("Extracting templates for all units...")
print(
    f"Total unique units in sorter data: {len(np.unique(sorter_data.spike_clusters))}"
)


# %%
def extract_template_for_unit(unit_id, whitened_data, sorter_data, fs):
    """Extract template for any unit, finding its best channel automatically"""

    # Get spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    if len(all_spike_times) < MIN_SPIKES_THRESHOLD:
        return None

    # Find best channel for this unit using templates if available
    best_channel = 0
    if (
        hasattr(sorter_data, "templates_data")
        and sorter_data.templates_data is not None
    ):
        templates = sorter_data.templates_data["templates"]
        if unit_id < len(templates):
            unit_template = templates[unit_id]
            max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
            best_channel = max_channel - 1  # Convert to 0-indexed

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
                waveform = whitened_data[start_idx:end_idx, best_channel].compute()
                waveforms.append(waveform)
            except:
                continue

    if len(waveforms) < 5:
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

    # Create time array in milliseconds
    time_ms = np.arange(len(template_normalized)) / fs * 1000 - PRE_SAMPLES_MS

    # Get unit properties
    quality = "unknown"
    amplitude = 0
    if hasattr(sorter_data, "unit_properties"):
        quality = sorter_data.unit_properties.get("KSLabel", {}).get(unit_id, "unknown")
        amplitude = sorter_data.unit_properties.get("Amplitude", {}).get(unit_id, 0)

    return {
        "template": template_normalized,
        "time_ms": time_ms,
        "n_waveforms": len(waveforms),
        "best_channel": best_channel,
        "total_spikes": len(all_spike_times),
        "quality": quality,
        "amplitude": amplitude,
    }


# %%
# Extract templates for all units
print("Extracting templates for all units...")
all_templates = {}
failed_units = []

unique_units = np.unique(sorter_data.spike_clusters)
print(f"Processing {len(unique_units)} units...")

for unit_id in unique_units:
    template_info = extract_template_for_unit(unit_id, whitened_data, sorter_data, FS)
    if template_info:
        all_templates[unit_id] = template_info
        print(
            f"✓ Unit {unit_id}: {template_info['quality']}, Ch {template_info['best_channel']}, "
            f"{template_info['n_waveforms']}/{template_info['total_spikes']} spikes, "
            f"amp={template_info['amplitude']:.1f}"
        )
    else:
        failed_units.append(unit_id)

print(f"\nSuccessfully extracted {len(all_templates)} templates")
if failed_units:
    print(f"Failed units: {failed_units}")

# %%
# Create comprehensive template overview plot
if all_templates:
    # Sort units by quality and channel for better organization
    def sort_key(item):
        unit_id, info = item
        quality_priority = {"good": 0, "mua": 1, "unknown": 2}
        return (
            quality_priority.get(info["quality"], 2),
            info["best_channel"],
            -info["total_spikes"],
        )

    sorted_templates = dict(sorted(all_templates.items(), key=sort_key))

    # Calculate grid dimensions
    n_units = len(sorted_templates)
    n_rows = (n_units + TEMPLATES_PER_ROW - 1) // TEMPLATES_PER_ROW

    # Adjust figure size based on number of units
    fig_height = max(FIGURE_SIZE[1], n_rows * 3)
    fig, ax = plt.subplots(figsize=(FIGURE_SIZE[0], fig_height))

    # Plot all templates
    unit_list = list(sorted_templates.keys())
    for i, unit_id in enumerate(unit_list):
        # Calculate grid position
        row = i // TEMPLATES_PER_ROW
        col = i % TEMPLATES_PER_ROW

        # Grid coordinates
        x_pos = col * X_SPACING
        y_pos = -row * Y_SPACING

        # Get template data
        template_info = sorted_templates[unit_id]

        # Get color based on quality
        color = QUALITY_COLORS.get(template_info["quality"], QUALITY_COLORS["unknown"])

        # Plot template
        ax.plot(
            template_info["time_ms"] * TIME_SCALE + x_pos,
            template_info["template"] + y_pos,
            color=color,
            linewidth=2,
            alpha=0.8,
        )

        # Add unit ID label (prominent)
        ax.text(
            x_pos - 0.15,
            y_pos + 1.5,
            f"U{unit_id}",
            fontsize=14,
            ha="center",
            va="bottom",
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor=color
            ),
        )

        # Add channel info
        ax.text(
            x_pos - 0.15,
            y_pos - 1.2,
            f"Ch{template_info['best_channel']}",
            fontsize=10,
            ha="center",
            va="top",
            color=color,
            alpha=0.9,
        )

        # Add spike count
        ax.text(
            x_pos - 0.15,
            y_pos - 1.8,
            f"n={template_info['total_spikes']}",
            fontsize=9,
            ha="center",
            va="top",
            color=color,
            alpha=0.7,
        )

        # Add quality indicator
        ax.text(
            x_pos + 0.15,
            y_pos + 1.5,
            template_info["quality"],
            fontsize=9,
            ha="center",
            va="bottom",
            color=color,
            alpha=0.8,
            style="italic",
        )

    # Format plot
    ax.set_title(
        f"All Unit Templates Overview ({len(sorted_templates)} units)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=color, linewidth=3, label=f"{quality} units")
        for quality, color in QUALITY_COLORS.items()
        if any(info["quality"] == quality for info in sorted_templates.values())
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # Add instructions
    ax.text(
        0.02,
        0.98,
        "Unit Selection Guide:\n"
        + "• Unit ID shown in bold box\n"
        + "• Channel (Ch#) and spike count (n=) below\n"
        + "• Quality indicated by color and text\n"
        + "• Templates sorted by quality → channel → spike count",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

    # Print summary by channel for easy reference
    print(f"\n=== UNIT SUMMARY BY CHANNEL ===")
    channel_units = {}
    for unit_id, info in sorted_templates.items():
        ch = info["best_channel"]
        if ch not in channel_units:
            channel_units[ch] = []
        channel_units[ch].append(
            {
                "unit_id": unit_id,
                "quality": info["quality"],
                "spikes": info["total_spikes"],
                "amplitude": info["amplitude"],
            }
        )

    # Focus on the 8 channels we want to plot
    target_channels = [0, 1, 2, 4, 6, 18, 22, 25]
    print("Units available for target channels:")
    for ch in target_channels:
        if ch in channel_units:
            print(f"\nChannel {ch}:")
            for unit in sorted(
                channel_units[ch],
                key=lambda x: (-1 if x["quality"] == "good" else 0, -x["spikes"]),
            ):
                print(
                    f"  Unit {unit['unit_id']}: {unit['quality']}, {unit['spikes']} spikes, amp={unit['amplitude']:.1f}"
                )
        else:
            print(f"\nChannel {ch}: No units found")

    print(f"\n=== RECOMMENDED UNITS FOR 8-CHANNEL PLOT ===")
    recommended = {}
    for ch in target_channels:
        if ch in channel_units:
            # Prefer good quality, then highest spike count
            best_unit = max(
                channel_units[ch],
                key=lambda x: (
                    1 if x["quality"] == "good" else 0,
                    x["spikes"],
                    x["amplitude"],
                ),
            )
            recommended[ch] = best_unit["unit_id"]
            print(
                f"Channel {ch}: Unit {best_unit['unit_id']} ({best_unit['quality']}, {best_unit['spikes']} spikes)"
            )
        else:
            print(f"Channel {ch}: No units available")

    print(f"\nSuggested MANUAL_UNIT_SELECTION = {recommended}")

else:
    print("No templates could be extracted!")

# %%
