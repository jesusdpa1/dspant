"""
spikes overlay
Author: jpenalozaa
Description: Creates an upper triangular grid of cross-correlograms for spike units
"""

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
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
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
data_dir = Path(os.getenv("DATA_DIR"))
hd_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/HDEG.ant")
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/RawG.ant")

sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)


stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()

stream_hd = StreamNode(str(hd_path))
stream_hd.load_metadata()
stream_hd.load_data()
# %%

sorter_data = load_kilosort(sorter_path, load_templates=True)

# %%
# Get sampling rate
FS = stream_hd.fs

# %%
notch_filter = create_notch_filter(60, q=10, fs=FS)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=4)
bandpass_filter_emg = create_bandpass_filter(20, 2000, fs=FS, order=4)
# Create pre-processing functions
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)

bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)

bandpass_processor_emg = FilterProcessor(
    filter_func=bandpass_filter_emg.get_filter_function(), overlap_samples=40
)

cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()
# %%
# Create and apply filters for HD data
processor_hd = create_processing_node(stream_hd)
processor_emg = create_processing_node(stream_emg)
# Add processors
processor_hd.add_processor(
    [notch_processor, bandpass_processor_hd],
    group="filters",
)

processor_emg.add_processor(
    [notch_processor, bandpass_processor_emg],
    group="filters",
)

# %%
# Slice recordings 5min
START_TIME = int(0.0 * FS)  # Start time in seconds
END_TIME = int(5.0 * 60 * FS)  # End time in seconds
# Create time range for processing
time_range = slice(START_TIME, END_TIME)

filtered_hd = processor_hd.process(group=["filters"]).persist()[START_TIME:END_TIME, :]

filtered_emg = processor_emg.process(group=["filters"]).persist()[
    START_TIME:END_TIME, :
]

# %%


cmr_hd_small = cmr_processor.process(
    filtered_hd,
    fs=FS,
).persist()
# %%
whiten_hd_small = whiten_processor.process(cmr_hd_small, fs=FS).persist()

# %%

# Template grouping and visualization
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Define groups (assuming correction for GROUP_03 and GROUP_04)
GROUP_01 = np.arange(0, 16, 2)  # [0, 2, 4, 6, 8, 10, 12, 14]
GROUP_01_COLORS = ["#1A0083", "#4512FA"]
GROUP_02 = np.arange(1, 17, 2)  # [1, 3, 5, 7, 9, 11, 13, 15]
GROUP_02_COLORS = ["#00833B", "#3CFE53"]
GROUP_03 = np.arange(18, 32, 2)  # [18, 20, 22, 24, 26, 28, 30]
GROUP_03_COLORS = ["#830000", "#FE4343"]
GROUP_04 = np.arange(19, 33, 2)  # [19, 21, 23, 25, 27, 29, 31]
GROUP_04_COLORS = ["#835100", "#FAC812"]


GROUP_ID = "19-33-2"
TEMPLATE_CHANNELS = GROUP_04.tolist()  # Example channel group
COLOR_WAVEFORM = GROUP_04_COLORS
NUM_SPIKES_TO_TEMPLATE = 500
PRE_SAMPLES_MS = 2.0  # 1ms before spike
POST_SAMPLES_MS = 2.0  # 2ms after spike

# Grid layout constants
TEMPLATES_PER_ROW = 4
X_SPACING = 1.0
Y_SPACING = 7.0
Y_COLUMN_OFFSET = 3.5  # Vertical offset for alternating columns (staggered pattern)
TIME_SCALE = 0.2
FIGURE_SIZE = (8, 10)

# Color constants


# Quality control constants
MIN_SPIKES_THRESHOLD = 50

# Define channels to analyze


def get_units_for_channels(channel_list):
    """Get all good units that belong to specified channels"""
    units_in_channels = []
    templates = sorter_data.templates_data["templates"]

    for unit_id in sorter_data.unit_ids:
        # Find the channel with maximum amplitude for this unit
        unit_template = templates[unit_id]
        max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
        unit_channel = max_channel - 1

        # Check if unit belongs to our channel group and has enough spikes
        if unit_channel in channel_list:
            unit_mask = sorter_data.spike_clusters == unit_id
            n_spikes = np.sum(unit_mask)
            if n_spikes >= MIN_SPIKES_THRESHOLD:  # Quality threshold
                units_in_channels.append(unit_id)

    return units_in_channels


def extract_normalized_template(unit_id, num_spikes=NUM_SPIKES_TO_TEMPLATE):
    """Extract and normalize template for a single unit"""
    # Get unit's primary channel
    templates = sorter_data.templates_data["templates"]
    unit_template = templates[unit_id]
    max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
    unit_channel = max_channel - 1

    # Get spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    # Subsample spikes if necessary
    if len(all_spike_times) > num_spikes:
        np.random.seed(42)
        sampled_indices = np.random.choice(
            len(all_spike_times), num_spikes, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
    else:
        sampled_spikes = all_spike_times

    # Extract waveforms around spike times using constants
    pre_samples = int(PRE_SAMPLES_MS * FS / 1000)
    post_samples = int(POST_SAMPLES_MS * FS / 1000)

    waveforms = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whiten_hd_small.shape[0]:
            waveform = whiten_hd_small[start_idx:end_idx, unit_channel].compute()
            waveforms.append(waveform)

    if waveforms:
        waveforms_array = np.array(waveforms)
        template_mean = np.mean(waveforms_array, axis=0)

        # Z-score normalization for this unit
        template_normalized = (template_mean - np.mean(template_mean)) / np.std(
            template_mean
        )

        # Create time array in milliseconds
        time_ms = np.arange(pre_samples + post_samples + 1) / FS * 1000 - (
            PRE_SAMPLES_MS
        )

        return {
            "template": template_normalized,
            "time_ms": time_ms,
            "n_waveforms": len(waveforms),
            "channel": unit_channel,
        }

    return None


# Get units and extract templates
units_to_plot = get_units_for_channels(TEMPLATE_CHANNELS)
print(f"Found {len(units_to_plot)} units in channels {TEMPLATE_CHANNELS}")

templates_data = {}
for unit_id in units_to_plot:
    template_info = extract_normalized_template(unit_id)
    if template_info:
        templates_data[unit_id] = template_info
        print(f"Unit {unit_id}: Template from {template_info['n_waveforms']} spikes")

print(f"Successfully extracted {len(templates_data)} templates")

# Create clean grid plot with transparent background
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
fig.patch.set_alpha(0.0)  # Transparent figure background
ax.patch.set_alpha(0.0)  # Transparent axes background

# Plot templates in staggered grid using constants
unit_list = list(templates_data.keys())
for i, unit_id in enumerate(unit_list):
    # Calculate grid position
    row = i // TEMPLATES_PER_ROW
    col = i % TEMPLATES_PER_ROW

    # Grid coordinates with staggered column offset
    x_pos = col * X_SPACING
    y_pos = -row * Y_SPACING

    # Add vertical offset for alternating columns (brick/staggered pattern)
    if col % 2 == 1:  # Odd columns (1, 3, 5...) get shifted down
        y_pos -= Y_COLUMN_OFFSET

    # Get template data
    template_info = templates_data[unit_id]

    # Get alternating color
    color = COLOR_WAVEFORM[i % len(COLOR_WAVEFORM)]

    # Plot normalized template
    ax.plot(
        template_info["time_ms"] * TIME_SCALE + x_pos,
        template_info["template"] + y_pos,
        color=color,
        linewidth=2,
        alpha=0.8,
    )

    # Add unit label at top left corner of template
    ax.text(
        x_pos - 0.6,  # Left of template
        y_pos + 1.2,  # Above template
        f"U{unit_id}",
        fontsize=9,
        ha="left",
        va="bottom",
        fontweight="bold",
        color=color,
    )

# Remove all visual elements except the waveforms
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()

plt.tight_layout()

# Save the figure
FIGURE_TITLE = f"{GROUP_ID}_templates"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR_HD"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

mpu.save_figure(fig, FIGURE_PATH, dpi=600)

# Show the plot
plt.show()
# %%

# Single row template visualization
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Template extraction constants
NUM_SPIKES_TO_TEMPLATE = 500
PRE_SAMPLES_MS = 2.0  # ms before spike
POST_SAMPLES_MS = 2.0  # ms after spike

# Grid layout constants - SINGLE COLUMN
TEMPLATES_PER_COLUMN = 5  # Single column of 5 units
X_SPACING = 1.0
Y_SPACING = 5.0  # Reduced spacing for vertical layout
TIME_SCALE = 0.2
FIGURE_SIZE = (4, 12)  # Taller, narrower for single column

# Define specific units and their colors
WAVEFORM_UNITS = [21, 20, 47, 22, 6]  # Specify exactly which units to plot
WAVEFORM_COLORS = [
    "#00833B",
    "#00833B",
    "#1A0083",
    "#830000",
    "#835100",
]  # 5 different colors

# Quality control constants
MIN_SPIKES_THRESHOLD = 50

# Save constants
GROUP_ID = "single_column"  # Identifier for saving


def extract_normalized_template(unit_id, num_spikes=NUM_SPIKES_TO_TEMPLATE):
    """Extract and normalize template for a single unit"""
    # Get unit's primary channel
    templates = sorter_data.templates_data["templates"]
    unit_template = templates[unit_id]
    max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
    unit_channel = max_channel - 1

    # Get spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    # Check if unit has enough spikes
    if len(all_spike_times) < MIN_SPIKES_THRESHOLD:
        print(f"Unit {unit_id}: Only {len(all_spike_times)} spikes, below threshold")
        return None

    # Subsample spikes if necessary
    if len(all_spike_times) > num_spikes:
        np.random.seed(42)
        sampled_indices = np.random.choice(
            len(all_spike_times), num_spikes, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
    else:
        sampled_spikes = all_spike_times

    # Extract waveforms around spike times using constants
    pre_samples = int(PRE_SAMPLES_MS * FS / 1000)
    post_samples = int(POST_SAMPLES_MS * FS / 1000)

    waveforms = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whiten_hd_small.shape[0]:
            waveform = whiten_hd_small[start_idx:end_idx, unit_channel].compute()
            waveforms.append(waveform)

    if waveforms:
        waveforms_array = np.array(waveforms)
        template_mean = np.mean(waveforms_array, axis=0)

        # Z-score normalization for this unit
        template_normalized = (template_mean - np.mean(template_mean)) / np.std(
            template_mean
        )

        # Create time array in milliseconds
        time_ms = np.arange(pre_samples + post_samples + 1) / FS * 1000 - (
            PRE_SAMPLES_MS
        )

        return {
            "template": template_normalized,
            "time_ms": time_ms,
            "n_waveforms": len(waveforms),
            "channel": unit_channel,
        }

    return None


# Extract templates for specified units
print(f"Extracting templates for units: {WAVEFORM_UNITS}")

templates_data = {}
for unit_id in WAVEFORM_UNITS:
    template_info = extract_normalized_template(unit_id)
    if template_info:
        templates_data[unit_id] = template_info
        print(
            f"Unit {unit_id}: Template from {template_info['n_waveforms']} spikes on channel {template_info['channel']}"
        )
    else:
        print(f"Unit {unit_id}: Failed to extract template")

print(f"Successfully extracted {len(templates_data)} templates")

# Only proceed if we have templates
if templates_data:
    # Create clean grid plot with transparent background
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_alpha(0.0)  # Transparent figure background
    ax.patch.set_alpha(0.0)  # Transparent axes background

    # Plot templates in single column
    for i, unit_id in enumerate(WAVEFORM_UNITS):
        if unit_id not in templates_data:
            continue  # Skip units that failed extraction

        # Calculate position (single column, so col=0)
        row = i
        x_pos = 0  # Single column at x=0
        y_pos = -row * Y_SPACING  # Stack vertically downward

        # Get template data
        template_info = templates_data[unit_id]

        # Get specific color for this unit
        color = WAVEFORM_COLORS[i % len(WAVEFORM_COLORS)]

        # Plot normalized template
        ax.plot(
            template_info["time_ms"] * TIME_SCALE + x_pos,
            template_info["template"] + y_pos,
            color=color,
            linewidth=2,
            alpha=0.8,
        )

        # Add unit label at top left corner of template
        ax.text(
            x_pos - 0.6,  # Left of template
            y_pos + 1.2,  # Above template
            f"U{unit_id}",
            fontsize=9,
            ha="left",
            va="bottom",
            fontweight="bold",
            color=color,
        )

    # Remove all visual elements except the waveforms
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()

    # Save the figure
    FIGURE_TITLE = f"{GROUP_ID}_templates"
    FIGURE_DIR = Path(os.getenv("FIGURE_DIR_HD"))
    FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

    mpu.save_figure(fig, FIGURE_PATH, dpi=600)

    # Show the plot
    plt.show()

    print(f"\nSingle Column Summary:")
    print(f"- {len(templates_data)} templates plotted")
    print(f"- Template window: {PRE_SAMPLES_MS}ms pre + {POST_SAMPLES_MS}ms post")
    print(f"- Subsampled to {NUM_SPIKES_TO_TEMPLATE} spikes per unit")
    print(f"- Units plotted: {list(templates_data.keys())}")
    print(f"- Colors used: {WAVEFORM_COLORS[: len(templates_data)]}")

else:
    print("No valid templates extracted. Check unit IDs and data availability.")

# %%
