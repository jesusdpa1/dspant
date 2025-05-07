"""
Create Joint PSTH visualization showing correlation between two neural units aligned to EMG contractions
Author: Jesus Penaloza
"""

import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
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
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()

# Data loading configuration
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

# Process EMG data to find contraction onsets
processor_emg = create_processing_node(stream_emg)

# Create filters
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)

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

# Apply TKEO
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=15)
tkeo_data = tkeo_processor.process(filtered_emg[0:1000000, :], fs=fs).persist()

# Normalize TKEO
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()

# Detect EMG onsets
st_tkeo_processor = create_single_threshold_detector(
    threshold_method="absolute",
    threshold_value=0.1,
    refractory_period=0.01,
    min_contraction_duration=0.01,
)
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)

# Convert onsets to seconds
emg_onsets = tkeo_epochs["onset_idx"].to_numpy() / fs

# Define time window for analysis
PRE_EVENT = 0.5  # seconds before event
POST_EVENT = 1.0  # seconds after event

# Create PSTH analyzers for both units
psth_analyzer = PSTHAnalyzer(
    bin_size_ms=10.0,  # 10ms bins for good temporal resolution
    window_size_ms=(PRE_EVENT + POST_EVENT) * 1000,
    sigma_ms=0,  # No smoothing for raw data
    baseline_window_ms=None,
)

# Compute PSTH aligned to EMG onsets
psth_result = psth_analyzer.transform(
    sorter=sorter_data,
    events=emg_onsets,
    pre_time_ms=PRE_EVENT * 1000,
    post_time_ms=POST_EVENT * 1000,
    include_raster=True,
)


def plot_joint_psth(unit1_index=26, unit2_index=27, normalize=True, smooth_sigma=1):
    """
    Plot Joint PSTH (JPSTH) for two units

    Parameters:
    -----------
    unit1_index : int
        Index of the first unit (x-axis)
    unit2_index : int
        Index of the second unit (y-axis)
    normalize : bool
        Whether to normalize the JPSTH by the product of the PSTHs
    smooth_sigma : float
        Sigma for Gaussian smoothing of the JPSTH matrix

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with the JPSTH plot
    """
    # Get unit IDs
    unit_ids = psth_result["unit_ids"]
    unit1_id = unit_ids[unit1_index]
    unit2_id = unit_ids[unit2_index]

    # Get time bins from PSTH
    time_bins = psth_result["time_bins"]
    bin_edges = np.concatenate(
        [time_bins, [time_bins[-1] + (time_bins[1] - time_bins[0])]]
    )
    bin_centers = time_bins
    bin_width = time_bins[1] - time_bins[0]
    n_bins = len(time_bins)

    # Get PSTHs for both units
    psth_data = psth_result["psth_rates"]
    unit1_psth = psth_data[:, unit1_index]
    unit2_psth = psth_data[:, unit2_index]

    # Get spike times for each trial
    raster_data1 = psth_result["raster_data"][unit1_index]
    raster_data2 = psth_result["raster_data"][unit2_index]

    spike_times1 = raster_data1["spike_times"]
    trial_indices1 = raster_data1["trials"]

    spike_times2 = raster_data2["spike_times"]
    trial_indices2 = raster_data2["trials"]

    # Build the JPSTH matrix
    jpsth_matrix = np.zeros((n_bins, n_bins))

    # Number of trials
    n_trials = psth_result["event_count"]

    # Get trial-by-trial spike counts in each bin for both units
    unit1_counts = np.zeros((n_trials, n_bins))
    unit2_counts = np.zeros((n_trials, n_bins))

    # For each trial, count spikes in each bin
    for trial_idx in range(n_trials):
        # Get spike times for this trial for unit 1
        mask1 = trial_indices1 == trial_idx
        trial_spikes1 = spike_times1[mask1]

        # Get spike times for this trial for unit 2
        mask2 = trial_indices2 == trial_idx
        trial_spikes2 = spike_times2[mask2]

        # Count spikes in each bin
        for bin_idx, (bin_start, bin_end) in enumerate(
            zip(bin_edges[:-1], bin_edges[1:])
        ):
            # Unit 1
            bin_spikes1 = np.sum(
                (trial_spikes1 >= bin_start) & (trial_spikes1 < bin_end)
            )
            unit1_counts[trial_idx, bin_idx] = bin_spikes1

            # Unit 2
            bin_spikes2 = np.sum(
                (trial_spikes2 >= bin_start) & (trial_spikes2 < bin_end)
            )
            unit2_counts[trial_idx, bin_idx] = bin_spikes2

    # Compute JPSTH by correlating spike counts bin-by-bin across all trials
    for i in range(n_bins):
        for j in range(n_bins):
            # Correlation between time bin i for unit 1 and time bin j for unit 2
            # across all trials
            if np.any(unit1_counts[:, i]) and np.any(unit2_counts[:, j]):
                jpsth_matrix[i, j] = np.corrcoef(
                    unit1_counts[:, i], unit2_counts[:, j]
                )[0, 1]
            else:
                jpsth_matrix[i, j] = 0

    # Apply Gaussian smoothing if requested
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter

        jpsth_matrix = gaussian_filter(jpsth_matrix, sigma=smooth_sigma)

    # Normalize if requested - this produces the "cross-correlogram"
    if normalize:
        # Compute expected JPSTH based on the product of the PSTHs
        expected = np.outer(unit1_psth, unit2_psth) / n_trials
        # Compute standard deviation for normalization
        std1 = np.sqrt(unit1_psth / n_trials)
        std2 = np.sqrt(unit2_psth / n_trials)
        std_outer = np.outer(std1, std2)
        std_outer[std_outer == 0] = 1  # Avoid division by zero

        # Normalize the JPSTH - this gives us the cross-correlation
        normalized_jpsth = (jpsth_matrix - expected) / std_outer
        display_matrix = normalized_jpsth
        vmin, vmax = -3, 3  # Typical range for correlation values
        cmap = "coolwarm"
        title = f"Normalized JPSTH: Unit {unit1_id} vs Unit {unit2_id}"
    else:
        display_matrix = jpsth_matrix
        vmin, vmax = -1, 1  # Raw correlation range
        cmap = "coolwarm"
        title = f"Joint PSTH: Unit {unit1_id} vs Unit {unit2_id}"

    # Create figure with GridSpec to manage the layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(
        2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.1, hspace=0.1
    )

    # Central JPSTH heatmap
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(
        display_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[min(time_bins), max(time_bins), max(time_bins), min(time_bins)],
    )

    # Top marginal PSTH (Unit 1)
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
    ax_top.bar(
        bin_centers,
        unit1_psth,
        width=bin_width,
        alpha=0.7,
        color="blue",
        label=f"Unit {unit1_id}",
    )
    ax_top.set_ylabel("Firing rate (Hz)")
    ax_top.set_title(title)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    # Right marginal PSTH (Unit 2)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
    ax_right.barh(
        bin_centers,
        unit2_psth,
        height=bin_width,
        alpha=0.7,
        color="green",
        label=f"Unit {unit2_id}",
    )
    ax_right.set_xlabel("Firing rate (Hz)")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    # Diagonal autocorrelation plot
    ax_diag = fig.add_subplot(gs[0, 1])
    # Extract the diagonal elements (shifted to center at 0)
    diag_indices = np.arange(n_bins)
    diags = []
    lags = []
    max_lag = min(
        20, n_bins // 2
    )  # Show up to 20 lags or half the bins, whichever is smaller

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            diag = np.diag(display_matrix, k=lag)
            i, j = -lag, 0
        else:
            diag = np.diag(display_matrix, k=lag)
            i, j = 0, lag

        lag_time = (
            (bin_centers[j] - bin_centers[i])
            if len(bin_centers) > max(i, j)
            else lag * bin_width
        )
        lags.append(lag_time)
        diags.append(np.mean(diag))

    ax_diag.plot(lags, diags, "k-", linewidth=2)
    ax_diag.axvline(x=0, color="r", linestyle="--", alpha=0.6)
    ax_diag.set_title("Cross-correlation")
    ax_diag.set_ylabel("Correlation")
    ax_diag.set_xlabel("Time lag (s)")

    # Set axis labels for the heatmap
    ax_heatmap.set_xlabel(f"Time from event (s) - Unit {unit1_id}")
    ax_heatmap.set_ylabel(f"Time from event (s) - Unit {unit2_id}")

    # Add vertical and horizontal lines at event onset (0)
    ax_heatmap.axvline(x=0, color="r", linestyle="--", alpha=0.6)
    ax_heatmap.axhline(y=0, color="r", linestyle="--", alpha=0.6)
    ax_top.axvline(x=0, color="r", linestyle="--", alpha=0.6)
    ax_right.axhline(y=0, color="r", linestyle="--", alpha=0.6)

    # Add colorbar
    cbar = fig.colorbar(
        im,
        ax=[ax_heatmap, ax_right],
        label="Correlation" if normalize else "Joint firing",
    )

    # Set legends
    ax_top.legend(loc="upper right")
    ax_right.legend(loc="upper right")

    plt.tight_layout()
    return fig


# Create and display the JPSTH
interesting_unit_pairs = [(26, 27), (20, 31), (36, 50)]
unit1_idx, unit2_idx = interesting_unit_pairs[0]  # Use first pair by default

fig = plot_joint_psth(unit1_index=unit1_idx, unit2_index=unit2_idx)
plt.show()
