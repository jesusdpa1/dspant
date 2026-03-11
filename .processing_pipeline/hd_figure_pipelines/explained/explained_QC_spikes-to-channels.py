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
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)

# Load EMG and spike data
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
# Create and apply filters for HD data
processor_hd = create_processing_node(stream_hd)

# Add processors
processor_hd.add_processor(
    [notch_processor, bandpass_processor_hd],
    group="filters",
)

# %%
# Apply filters to HD data
filtered_hd_small = processor_hd.process(group=["filters"]).persist()
# %%
cmr_hd_small = cmr_processor.process(filtered_hd_small, fs=FS).persist()
# %%
whiten_hd_small = whiten_processor.process(cmr_hd_small, fs=FS).persist()

# %%
from matplotlib.ticker import FormatStrFormatter

NAVY_BLUE = "#2D3142"
WAVEFORM_COLORS = ["#E0A500", "#4512FA", "#A43E2C", "#0B7A57"]
WAVEFORM_IDS = [31, 32]
NUM_SPIKES_TO_TEMPLATE = 100

# Define font sizes
FONT_SIZE = 30
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)

# Correlogram parameters
BIN_SIZE = 1.0  # ms
MAX_LAG = 100.0  # ms

# ISI parameters
ISI_BIN_SIZE = 1.0  # ms
ISI_MAX = 100.0  # ms

# STEP 1: Get channel for each unit
unit_channels = {}
templates = sorter_data.templates_data["templates"]
for unit_id in WAVEFORM_IDS:
    unit_template = templates[unit_id]
    max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
    unit_channels[unit_id] = max_channel - 1
    print(f"Unit {unit_id} -> Channel {max_channel - 1}")

# STEP 2: Extract waveforms from subsample of spike times
unit_waveforms = {}
unit_sampled_spikes = {}

for unit_id in WAVEFORM_IDS:
    unit_channel = unit_channels[unit_id]

    # Get all spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    # Subsample to NUM_SPIKES_TO_TEMPLATE spikes
    if len(all_spike_times) > NUM_SPIKES_TO_TEMPLATE:
        np.random.seed(42)
        sampled_indices = np.random.choice(
            len(all_spike_times), NUM_SPIKES_TO_TEMPLATE, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
        print(
            f"Unit {unit_id}: Sampled {NUM_SPIKES_TO_TEMPLATE} out of {len(all_spike_times)} spikes"
        )
    else:
        sampled_spikes = all_spike_times
        print(f"Unit {unit_id}: Using all {len(all_spike_times)} spikes")

    unit_sampled_spikes[unit_id] = sampled_spikes

    # Extract waveforms manually
    pre_samples = int(1 * FS / 1000)  # 1ms
    post_samples = int(2 * FS / 1000)  # 2ms

    waveforms_list = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whiten_hd_small.shape[0]:
            waveform = whiten_hd_small[start_idx:end_idx, unit_channel].compute()
            waveforms_list.append(waveform)

    if waveforms_list:
        unit_waveforms[unit_id] = np.array(waveforms_list)
        print(f"Unit {unit_id}: Extracted {len(waveforms_list)} waveforms")
    else:
        unit_waveforms[unit_id] = None
        print(f"Unit {unit_id}: No valid waveforms extracted")

# STEP 3: Compute templates using TemplateExtractor
unit_templates = {}
unit_template_stats = {}
template_time_ms = np.arange(pre_samples + post_samples + 1) / FS * 1000 - (
    pre_samples / FS * 1000
)

for unit_id in WAVEFORM_IDS:
    if unit_waveforms[unit_id] is not None:
        waveforms_array = da.from_array(unit_waveforms[unit_id][:, :, np.newaxis])
        template_stats = TemplateExtractor.extract_template_distributions(
            waveforms_array, normalization=None
        )
        unit_templates[unit_id] = template_stats["template_mean"][:, 0]
        unit_template_stats[unit_id] = template_stats
        print(
            f"Unit {unit_id}: Template with statistics computed from {template_stats['n_waveforms']} waveforms"
        )
    else:
        unit_templates[unit_id] = None
        unit_template_stats[unit_id] = None


# STEP 4: Helper functions for correlograms and ISI
def get_spike_times_for_unit(sorter_node, unit_id):
    """Extract spike times for a specific unit and convert to milliseconds."""
    unit_mask = sorter_node.spike_clusters == unit_id
    unit_spike_samples = sorter_node.spike_times[unit_mask]
    spike_times_ms = (unit_spike_samples / sorter_node.sampling_frequency) * 1000
    return spike_times_ms


def compute_correlogram(
    spike_times_1, spike_times_2, bin_size=BIN_SIZE, max_lag=MAX_LAG
):
    """Compute cross-correlogram between two spike trains."""
    n_bins = int(2 * max_lag / bin_size)
    bin_edges = np.linspace(-max_lag, max_lag, n_bins + 1)
    lag_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    correlogram = np.zeros(len(lag_centers))

    is_autocorr = np.array_equal(spike_times_1, spike_times_2)

    all_diffs = []
    for spike_1 in spike_times_1:
        time_diffs = spike_times_2 - spike_1
        valid_diffs = time_diffs[(time_diffs >= -max_lag) & (time_diffs <= max_lag)]

        if is_autocorr:
            valid_diffs = valid_diffs[np.abs(valid_diffs) > (bin_size / 2)]

        all_diffs.extend(valid_diffs)

    if len(all_diffs) > 0:
        correlogram, _ = np.histogram(all_diffs, bins=bin_edges)

    return lag_centers, correlogram


def compute_isi(spike_times_ms, bin_size=ISI_BIN_SIZE, max_isi=ISI_MAX):
    """Compute Inter-Spike Interval histogram."""
    if len(spike_times_ms) < 2:
        return np.array([]), np.array([])

    # Calculate ISIs
    isis = np.diff(spike_times_ms)

    # Filter ISIs within range
    isis = isis[isis <= max_isi]

    # Create histogram
    n_bins = int(max_isi / bin_size)
    bin_edges = np.linspace(0, max_isi, n_bins + 1)
    isi_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    isi_hist, _ = np.histogram(isis, bins=bin_edges)

    return isi_centers, isi_hist


# STEP 5: Set up plotting data
CHANNELS = list(unit_channels.values())
START_SAMPLE = int(1.2 * FS)
END_SAMPLE = int(1.5 * FS)
time_array = np.arange(END_SAMPLE - START_SAMPLE) / FS
data_computed = whiten_hd_small[START_SAMPLE:END_SAMPLE, :].compute()

# Set publication style
mpu.set_publication_style()

# STEP 6: Create 2x2 figure with custom proportions
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(2, 2, width_ratios=[7, 3], height_ratios=[1, 1], hspace=0.5, wspace=0.2)

# TOP LEFT: Time series with spikes (70% width)
ax1 = fig.add_subplot(gs[0, 0])

# Plot background channels
for idx, channel in enumerate(CHANNELS):
    offset = idx * 4
    signal = data_computed[:, channel] + offset
    ax1.plot(time_array, signal, color=NAVY_BLUE, alpha=0.5)

# Plot spikes
spike_window_ms = 2
spike_window_samples = int(spike_window_ms / 1000 * FS)
half_window = spike_window_samples // 2

for unit_idx, unit_id in enumerate(WAVEFORM_IDS):
    unit_channel = unit_channels[unit_id]
    channel_idx = CHANNELS.index(unit_channel)
    offset = unit_idx * 4
    color = WAVEFORM_COLORS[unit_idx]

    # Get spikes in time window
    unit_mask = sorter_data.spike_clusters == unit_id
    unit_spike_times = sorter_data.spike_times[unit_mask]
    time_mask = (unit_spike_times >= START_SAMPLE) & (unit_spike_times <= END_SAMPLE)
    filtered_spikes = unit_spike_times[time_mask]

    # Plot each spike
    for spike_idx, spike_time in enumerate(filtered_spikes):
        spike_start_rel = spike_time - START_SAMPLE - half_window
        spike_end_rel = spike_time - START_SAMPLE + half_window

        if spike_start_rel >= 0 and spike_end_rel < len(time_array):
            spike_data = data_computed[spike_start_rel:spike_end_rel, unit_channel]
            spike_time_array = time_array[spike_start_rel:spike_end_rel]

            label = f"Unit {unit_id}" if spike_idx == 0 else None
            ax1.plot(
                spike_time_array,
                spike_data + offset,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=label,
            )

# Format left panel
left_y_positions = []
left_y_labels = []
for idx, channel in enumerate(CHANNELS):
    offset = idx * 4
    left_y_positions.append(offset)
    left_y_labels.append(f"Ch {channel}")

ax1.set_yticks(left_y_positions)
ax1.set_yticklabels(left_y_labels)
ax1.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE)
ax1.set_ylabel("Channels", fontsize=AXIS_LABEL_SIZE)
ax1.set_title("Raw Signal with Spike Waveforms", fontsize=SUBTITLE_SIZE)
ax1.legend(fontsize=TICK_SIZE)
ax1.tick_params(labelsize=TICK_SIZE)

# TOP RIGHT: Templates (30% width)
ax2 = fig.add_subplot(gs[0, 1])

for unit_idx, unit_id in enumerate(WAVEFORM_IDS):
    if unit_templates[unit_id] is not None and unit_template_stats[unit_id] is not None:
        offset = unit_idx * 4
        color = WAVEFORM_COLORS[unit_idx]

        template_mean = unit_templates[unit_id]
        template_std = unit_template_stats[unit_id]["template_std"][:, 0]
        n_waveforms = unit_template_stats[unit_id]["n_waveforms"]
        template_sem = template_std / np.sqrt(n_waveforms)
        sem_scale = 3
        template_sem_scaled = template_sem * sem_scale

        ax2.fill_between(
            template_time_ms,
            (template_mean - template_sem_scaled) + offset,
            (template_mean + template_sem_scaled) + offset,
            color=color,
            alpha=0.4,
            label=f"Unit {unit_id} SEM (×{sem_scale})",
        )

        ax2.plot(
            template_time_ms,
            template_mean + offset,
            color=color,
            linewidth=3,
            alpha=0.9,
            label=f"Unit {unit_id} Template",
        )

# Format right panel
right_y_positions = []
right_y_labels = []
for unit_idx, unit_id in enumerate(WAVEFORM_IDS):
    offset = unit_idx * 4
    right_y_positions.append(offset)
    right_y_labels.append(f"Unit {unit_id}")

ax2.set_yticks(right_y_positions)
ax2.set_yticklabels(right_y_labels)
ax2.set_xlabel("Time [ms]", fontsize=AXIS_LABEL_SIZE)
ax2.set_ylabel("Units", fontsize=AXIS_LABEL_SIZE)
ax2.set_title("Spike Templates", fontsize=SUBTITLE_SIZE)
ax2.legend(fontsize=TICK_SIZE)
ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax2.tick_params(labelsize=TICK_SIZE)

# BOTTOM LEFT: Correlogram matrix (2x2 grid)
ax3 = fig.add_subplot(gs[1, 0])

# Get spike times for both units
spike_times_1 = get_spike_times_for_unit(sorter_data, WAVEFORM_IDS[0])
spike_times_2 = get_spike_times_for_unit(sorter_data, WAVEFORM_IDS[1])

# Create sub-grid within the left portion for 2x2 correlogram matrix
ax3.set_visible(False)  # Hide the main axis
gs_sub = GridSpec(2, 2, right=0.62, bottom=0.1, top=0.45, hspace=0.2, wspace=0.2)


# Function to plot single correlogram
def plot_single_correlogram(ax_sub, lags, correlogram, title, color):
    ax_sub.bar(
        lags,
        correlogram,
        width=BIN_SIZE * 0.8,
        color=color,
        alpha=0.7,
        edgecolor="none",
    )
    ax_sub.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_sub.set_xlim(-MAX_LAG, MAX_LAG)
    ax_sub.set_title(title, fontsize=TICK_SIZE)
    ax_sub.tick_params(labelsize=TICK_SIZE * 0.8)

    # Reduce number of ticks for clarity
    ax_sub.set_xticks([-MAX_LAG, 0, MAX_LAG])
    if correlogram.max() > 0:
        ax_sub.set_yticks([0, int(correlogram.max())])


# Top-left: Unit 1 auto-correlogram
ax_sub_00 = fig.add_subplot(gs_sub[0, 0])
lags_auto1, correlogram_auto1 = compute_correlogram(spike_times_1, spike_times_1)
plot_single_correlogram(
    ax_sub_00,
    lags_auto1,
    correlogram_auto1,
    f"{WAVEFORM_IDS[0]}×{WAVEFORM_IDS[0]}",
    WAVEFORM_COLORS[0],
)

# Top-right: Cross-correlogram (1 vs 2)
ax_sub_01 = fig.add_subplot(gs_sub[0, 1])
lags_cross12, correlogram_cross12 = compute_correlogram(spike_times_1, spike_times_2)
plot_single_correlogram(
    ax_sub_01,
    lags_cross12,
    correlogram_cross12,
    f"{WAVEFORM_IDS[0]}×{WAVEFORM_IDS[1]}",
    NAVY_BLUE,
)

# Bottom-left: Cross-correlogram (2 vs 1)
ax_sub_10 = fig.add_subplot(gs_sub[1, 0])
lags_cross21, correlogram_cross21 = compute_correlogram(spike_times_2, spike_times_1)
plot_single_correlogram(
    ax_sub_10,
    lags_cross21,
    correlogram_cross21,
    f"{WAVEFORM_IDS[1]}×{WAVEFORM_IDS[0]}",
    NAVY_BLUE,
)

# Bottom-right: Unit 2 auto-correlogram
ax_sub_11 = fig.add_subplot(gs_sub[1, 1])
lags_auto2, correlogram_auto2 = compute_correlogram(spike_times_2, spike_times_2)
plot_single_correlogram(
    ax_sub_11,
    lags_auto2,
    correlogram_auto2,
    f"{WAVEFORM_IDS[1]}×{WAVEFORM_IDS[1]}",
    WAVEFORM_COLORS[1],
)

# Add common labels for correlogram matrix
fig.text(
    0.38,
    0.05,
    "Lag [ms]",
    ha="center",
    va="center",
    fontsize=AXIS_LABEL_SIZE,
    fontweight="bold",
)
fig.text(
    0.09,
    0.275,
    "Count",
    ha="center",
    va="center",
    rotation=90,
    fontsize=AXIS_LABEL_SIZE,
    fontweight="bold",
)
fig.text(
    0.38,
    0.5,
    "Correlogram Matrix",
    ha="center",
    va="center",
    fontsize=SUBTITLE_SIZE,
    fontweight="bold",
)

# BOTTOM RIGHT: ISI plots - create 1x2 sub-grid
ax4 = fig.add_subplot(gs[1, 1])

# Create sub-grid within the right portion for 1x2 ISI plots
ax4.set_visible(False)  # Hide the main axis
gs_isi = GridSpec(
    1,
    2,
    left=0.69,
    right=0.9,
    bottom=0.1,
    top=0.45,
    hspace=0.3,
    wspace=0.3,
)


# Function to plot single ISI histogram
def plot_single_isi(ax_sub, isi_centers, isi_hist, title, color):
    if len(isi_centers) > 0:
        ax_sub.bar(
            isi_centers,
            isi_hist,
            width=ISI_BIN_SIZE * 0.8,
            color=color,
            alpha=0.7,
            edgecolor="none",
        )
        ax_sub.set_xlim(0, ISI_MAX)
        ax_sub.set_title(title, fontsize=TICK_SIZE)
        ax_sub.tick_params(labelsize=TICK_SIZE * 0.8)

        # Reduce number of ticks for clarity
        ax_sub.set_xticks([0, ISI_MAX // 2, ISI_MAX])
        if isi_hist.max() > 0:
            ax_sub.set_yticks([0, int(isi_hist.max())])


# Left: Unit 1 ISI
ax_isi_0 = fig.add_subplot(gs_isi[0, 0])
spike_times_ms_1 = get_spike_times_for_unit(sorter_data, WAVEFORM_IDS[0])
isi_centers_1, isi_hist_1 = compute_isi(spike_times_ms_1)
plot_single_isi(
    ax_isi_0, isi_centers_1, isi_hist_1, f"Unit {WAVEFORM_IDS[0]}", WAVEFORM_COLORS[0]
)

# Right: Unit 2 ISI
ax_isi_1 = fig.add_subplot(gs_isi[0, 1])
spike_times_ms_2 = get_spike_times_for_unit(sorter_data, WAVEFORM_IDS[1])
isi_centers_2, isi_hist_2 = compute_isi(spike_times_ms_2)
plot_single_isi(
    ax_isi_1, isi_centers_2, isi_hist_2, f"Unit {WAVEFORM_IDS[1]}", WAVEFORM_COLORS[1]
)

# Add common labels for ISI plots
fig.text(
    0.8,
    0.05,
    "ISI [ms]",
    ha="center",
    va="center",
    fontsize=AXIS_LABEL_SIZE,
    fontweight="bold",
)
fig.text(
    0.75,
    0.275,
    "Count",
    ha="center",
    va="center",
    rotation=90,
    fontsize=AXIS_LABEL_SIZE,
    fontweight="bold",
)
fig.text(
    0.8,
    0.5,
    "Inter-Spike Interval Distributions",
    ha="center",
    va="center",
    fontsize=SUBTITLE_SIZE,
    fontweight="bold",
)

# Add panel labels
mpu.add_panel_label(
    ax1,
    "A",
    x_offset_factor=0.04,
    y_offset_factor=0.04,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2,
    "B",
    x_offset_factor=0.04,
    y_offset_factor=0.04,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3,
    "C",
    x_offset_factor=0.04,
    y_offset_factor=0.1,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax4,
    "D",
    x_offset_factor=0.04,
    y_offset_factor=0.1,
    fontsize=SUBTITLE_SIZE,
)
plt.tight_layout()

# Save the figure
FIGURE_TITLE = "visualization_QC"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR_HD"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

mpu.save_figure(fig, FIGURE_PATH, dpi=600)

# Show the plot
plt.show()
# %%
