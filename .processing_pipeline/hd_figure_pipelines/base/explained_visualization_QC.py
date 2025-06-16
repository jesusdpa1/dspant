"""
Correlogram Grid Plot for Spike Sorting Data
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
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()

# Data loading configuration
data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/HDEG.ant")
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)

# Load EMG and spike data
stream_hd = StreamNode(str(emg_path))
stream_hd.load_metadata()
stream_hd.load_data()
sorter_data = load_kilosort(sorter_path)
# %%

# Get sampling rate
FS = stream_hd.fs
# %%
notch_filter = create_notch_filter(60, q=30, fs=FS)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=5)

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
# Define font sizes with appropriate scaling
FONT_SIZE = 35
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)

# Define colors for waveforms and plots
WAVEFORM_01_COLOR = mpu.COLORS["blue"]
WAVEFORM_02_COLOR = mpu.COLORS["orange"]
WAVEFORM_03_COLOR = mpu.COLORS["green"]
WAVEFORM_04_COLOR = mpu.COLORS["red"]
CORRELOGRAM_COLOR = mpu.COLORS["purple"]
BACKGROUND_COLOR = "#f8f9fa"

# Correlogram parameters
BIN_SIZE = 1.0  # ms
MAX_LAG = 100.0  # ms (200ms window total: -100 to +100)

# Filter for good quality units only
GOOD_UNITS = [
    uid
    for uid in sorter_data.unit_ids
    if sorter_data.unit_properties["KSLabel"][uid] == "good"
]
WAVEFORMS_ID = [10, 31]  # Take first 5 good units for visualization

# Calculate grid dimensions
N_WAVEFORMS = len(WAVEFORMS_ID)


def get_spike_times_for_unit(sorter_node, unit_id):
    """
    Extract spike times for a specific unit and convert to milliseconds.

    Args:
        sorter_node: SorterNode containing spike data
        unit_id: Unit ID to extract spikes for

    Returns:
        Array of spike times in milliseconds
    """
    # Get indices for this unit
    unit_mask = sorter_node.spike_clusters == unit_id
    unit_spike_samples = sorter_node.spike_times[unit_mask]

    # Convert from samples to milliseconds
    spike_times_ms = (unit_spike_samples / sorter_node.sampling_frequency) * 1000

    return spike_times_ms


def compute_correlogram(
    spike_times_1, spike_times_2, bin_size=BIN_SIZE, max_lag=MAX_LAG
):
    """
    Compute cross-correlogram between two spike trains.

    Args:
        spike_times_1: Array of spike times for first unit (in ms)
        spike_times_2: Array of spike times for second unit (in ms)
        bin_size: Bin size in ms
        max_lag: Maximum lag in ms

    Returns:
        lags: Array of lag values (bin centers)
        correlogram: Array of correlogram counts
    """
    # Create lag bins
    n_bins = int(2 * max_lag / bin_size)
    bin_edges = np.linspace(-max_lag, max_lag, n_bins + 1)
    lag_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initialize correlogram
    correlogram = np.zeros(len(lag_centers))

    # Check if this is an autocorrelogram (same unit)
    is_autocorr = np.array_equal(spike_times_1, spike_times_2)

    # For each spike in the first train, find time differences with second train
    all_diffs = []
    for spike_1 in spike_times_1:
        # Calculate all time differences: spike_times_2 - spike_1
        time_diffs = spike_times_2 - spike_1
        # Keep only differences within the lag window
        valid_diffs = time_diffs[(time_diffs >= -max_lag) & (time_diffs <= max_lag)]

        # For autocorrelograms, exclude zero-lag (spike with itself)
        if is_autocorr:
            valid_diffs = valid_diffs[np.abs(valid_diffs) > (bin_size / 2)]

        all_diffs.extend(valid_diffs)

    # Bin the time differences
    if len(all_diffs) > 0:
        correlogram, _ = np.histogram(all_diffs, bins=bin_edges)

    return lag_centers, correlogram


def plot_correlogram(ax, lags, correlogram, title=None):
    """
    Plot a single correlogram.

    Args:
        ax: Matplotlib axis
        lags: Array of lag values
        correlogram: Array of correlogram counts
        title: Optional title for the plot
    """
    ax.bar(
        lags,
        correlogram,
        width=BIN_SIZE * 0.8,
        color=CORRELOGRAM_COLOR,
        alpha=0.7,
        edgecolor="none",
    )

    # Add vertical line at zero lag
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # Format axis
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Lag [ms]" if title else None,
        ylabel="Count" if title else None,
        xlim=(-MAX_LAG, MAX_LAG),
        title_fontsize=TICK_SIZE,
        label_fontsize=TICK_SIZE,
        tick_fontsize=int(TICK_SIZE * 0.8),
    )

    # Reduce number of ticks for clarity
    ax.set_xticks([-MAX_LAG, 0, MAX_LAG])
    ax.set_yticks([0, ax.get_ylim()[1]])


# Set publication style
mpu.set_publication_style()

print(f"Total units available: {len(sorter_data.unit_ids)}")
print(f"Good quality units: {len(GOOD_UNITS)}")
print(f"Selected units for visualization: {WAVEFORMS_ID}")
print(f"Sampling frequency: {sorter_data.sampling_frequency} Hz")

# Create figure with NxN grid
fig = plt.figure(figsize=(16, 16))

# Create grid layout - upper triangular part + diagonal
for i in range(N_WAVEFORMS):
    for j in range(N_WAVEFORMS):
        # Plot upper triangular matrix (including diagonal)
        if j >= i:
            ax = plt.subplot(N_WAVEFORMS, N_WAVEFORMS, i * N_WAVEFORMS + j + 1)

            # Get real spike data for this pair
            spike_times_1 = get_spike_times_for_unit(sorter_data, WAVEFORMS_ID[i])
            spike_times_2 = get_spike_times_for_unit(sorter_data, WAVEFORMS_ID[j])

            # Compute correlogram
            lags, correlogram = compute_correlogram(spike_times_1, spike_times_2)

            # Create title for this subplot
            if i == j:
                # Diagonal - autocorrelogram
                title = f"{WAVEFORMS_ID[i]}×{WAVEFORMS_ID[j]}"
            else:
                # Off-diagonal - cross-correlogram
                title = f"{WAVEFORMS_ID[i]}×{WAVEFORMS_ID[j]}"

            # Plot correlogram
            plot_correlogram(ax, lags, correlogram, title)

        else:
            # Hide lower triangular subplots
            ax = plt.subplot(N_WAVEFORMS, N_WAVEFORMS, i * N_WAVEFORMS + j + 1)
            ax.set_visible(False)

# Finalize the figure
mpu.finalize_figure(
    fig,
    title=f"Cross-Correlogram Matrix for {N_WAVEFORMS} Spike Units",
    title_y=0.95,
    title_fontsize=TITLE_SIZE,
    hspace=0.3,
    wspace=0.3,
)

# Apply tight layout
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Format all visible axes to show 1 decimal place
for ax in fig.get_axes():
    if ax.get_visible():
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.0f")
        )  # Integer counts for y-axis

# Figure output path
FIGURE_TITLE = "correlogram_matrix"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

# Save figure
# mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print(
    f"Created {N_WAVEFORMS}x{N_WAVEFORMS} correlogram matrix with upper triangular display"
)
print(f"Units analyzed: {WAVEFORMS_ID}")
print(f"Bin size: {BIN_SIZE} ms, Max lag: {MAX_LAG} ms")
print(f"Total spikes analyzed: {len(sorter_data.spike_times)}")

# %%
