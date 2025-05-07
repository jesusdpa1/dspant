"""
Create noise correlation visualization between neural units aligned to EMG contractions
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

sns.set_theme(style="darkgrid")
load_dotenv()
# %%
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

# Define time window for analysis (using this window to count spikes per trial)
PRE_EVENT = 0.1  # 100ms before event
POST_EVENT = 0.5  # 500ms after event - focus on response


def calculate_spike_counts(unit1_index, unit2_index):
    """
    Calculate spike counts per trial for two units

    Parameters:
    -----------
    unit1_index : int
        Index of the first unit
    unit2_index : int
        Index of the second unit

    Returns:
    --------
    unit1_counts : np.ndarray
        Spike counts for unit1 per trial
    unit2_counts : np.ndarray
        Spike counts for unit2 per trial
    """
    # Get unit IDs
    unit_ids = sorter_data.unit_ids
    unit1_id = unit_ids[unit1_index]
    unit2_id = unit_ids[unit2_index]

    # Initialize arrays to store spike counts
    n_trials = len(emg_onsets)
    unit1_counts = np.zeros(n_trials)
    unit2_counts = np.zeros(n_trials)

    # Calculate spike counts for each trial
    for trial_idx, onset_time in enumerate(emg_onsets):
        # Define time window for this trial
        start_time = onset_time - PRE_EVENT
        end_time = onset_time + POST_EVENT

        # Get spikes for unit1 in this window
        unit1_spikes = sorter_data.get_unit_spike_train(
            unit1_id, start_frame=int(start_time * fs), end_frame=int(end_time * fs)
        )
        unit1_counts[trial_idx] = len(unit1_spikes)

        # Get spikes for unit2 in this window
        unit2_spikes = sorter_data.get_unit_spike_train(
            unit2_id, start_frame=int(start_time * fs), end_frame=int(end_time * fs)
        )
        unit2_counts[trial_idx] = len(unit2_spikes)

    return unit1_counts, unit2_counts


def plot_noise_correlation(unit1_index=26, unit2_index=27):
    """
    Plot noise correlation (spike count correlation) between two units

    Parameters:
    -----------
    unit1_index : int
        Index of the first unit (x-axis)
    unit2_index : int
        Index of the second unit (y-axis)

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get unit IDs
    unit_ids = sorter_data.unit_ids
    unit1_id = unit_ids[unit1_index]
    unit2_id = unit_ids[unit2_index]

    # Calculate spike counts
    unit1_counts, unit2_counts = calculate_spike_counts(unit1_index, unit2_index)

    # Calculate correlation coefficient and p-value
    r, p_value = stats.pearsonr(unit1_counts, unit2_counts)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add jitter to avoid overlapping points
    jitter = 0.1
    unit1_jittered = unit1_counts + np.random.uniform(
        -jitter, jitter, size=unit1_counts.shape
    )
    unit2_jittered = unit2_counts + np.random.uniform(
        -jitter, jitter, size=unit2_counts.shape
    )

    # Plot individual points
    scatter = ax.scatter(
        unit1_jittered, unit2_jittered, alpha=0.7, s=50, c="skyblue", edgecolor="navy"
    )

    # Add regression line
    if len(unit1_counts) > 1:  # Need at least 2 points for regression
        m, b = np.polyfit(unit1_counts, unit2_counts, 1)
        x_range = np.array([max(0, min(unit1_counts) - 1), max(unit1_counts) + 1])
        ax.plot(x_range, m * x_range + b, "r-", linewidth=2)

    # Add correlation value and p-value to plot
    correlation_text = f"r = {r:.3f}, p = {p_value:.3f}"
    ax.text(
        0.05,
        0.95,
        correlation_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=14,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Add 2D kernel density estimate
    x_min, x_max = max(0, min(unit1_counts) - 1), max(unit1_counts) + 1
    y_min, y_max = max(0, min(unit2_counts) - 1), max(unit2_counts) + 1

    # Only add KDE if we have enough data points
    if len(unit1_counts) >= 10:
        # Create a 2D histogram
        nbins = 20
        h, xedges, yedges = np.histogram2d(
            unit1_counts,
            unit2_counts,
            bins=nbins,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        # Smooth the histogram
        h = h.T  # transpose for imshow
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # Add contour plot
        levels = np.linspace(0, h.max(), 10)
        if levels.size > 0 and h.max() > 0:
            contour = ax.contour(
                h, levels=levels, extent=extent, colors="k", alpha=0.3, linewidths=0.5
            )

    # Set labels and title
    ax.set_xlabel(f"Unit {unit1_id} Spike Count", fontsize=12)
    ax.set_ylabel(f"Unit {unit2_id} Spike Count", fontsize=12)
    ax.set_title(f"Noise Correlation: Unit {unit1_id} vs Unit {unit2_id}", fontsize=14)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Equal aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Add grid
    ax.grid(True, alpha=0.3)

    # Set ticks to integer values
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    return fig


# Create and display the plot for an interesting pair of units
interesting_unit_pairs = [(26, 27), (20, 31), (36, 50)]
unit1_idx, unit2_idx = interesting_unit_pairs[0]  # Use first pair by default

fig = plot_noise_correlation(unit1_index=unit1_idx, unit2_index=unit2_idx)
plt.show()


def plot_noise_correlation_matrix(unit_indices=None):
    """
    Plot a matrix of noise correlations between multiple units

    Parameters:
    -----------
    unit_indices : list of int, optional
        Indices of units to include in the matrix. If None, uses interesting units.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Use default units if none provided
    if unit_indices is None:
        unit_indices = [26, 20, 27, 31, 36, 50]  # Interesting units from original code

    # Get unit IDs
    unit_ids = sorter_data.unit_ids
    selected_unit_ids = [unit_ids[idx] for idx in unit_indices]
    n_units = len(unit_indices)

    # Create correlation matrix
    correlation_matrix = np.zeros((n_units, n_units))
    pvalue_matrix = np.zeros((n_units, n_units))

    # Calculate correlations between all pairs
    for i, unit1_index in enumerate(unit_indices):
        for j, unit2_index in enumerate(unit_indices):
            if i == j:
                # Diagonal elements (self-correlation) = 1
                correlation_matrix[i, j] = 1.0
                pvalue_matrix[i, j] = 0.0
            elif i < j:  # Only calculate for upper triangle
                # Calculate spike counts
                unit1_counts, unit2_counts = calculate_spike_counts(
                    unit1_index, unit2_index
                )

                # Calculate correlation
                r, p_value = stats.pearsonr(unit1_counts, unit2_counts)

                # Store in both upper and lower triangle
                correlation_matrix[i, j] = r
                correlation_matrix[j, i] = r

                pvalue_matrix[i, j] = p_value
                pvalue_matrix[j, i] = p_value

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use masked array to mark non-significant correlations
    mask = pvalue_matrix > 0.05  # Non-significant correlations

    # Plot heatmap
    cmap = plt.cm.coolwarm
    im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Correlation Coefficient")

    # Add text annotations with correlation values
    for i in range(n_units):
        for j in range(n_units):
            # Skip non-significant correlations
            if mask[i, j] and i != j:
                text_color = "gray"
                text = f"{correlation_matrix[i, j]:.2f}"
            else:
                text_color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"
                text = f"{correlation_matrix[i, j]:.2f}"

            ax.text(j, i, text, ha="center", va="center", color=text_color)

    # Set ticks and labels
    ax.set_xticks(np.arange(n_units))
    ax.set_yticks(np.arange(n_units))
    ax.set_xticklabels([f"Unit {unit_id}" for unit_id in selected_unit_ids])
    ax.set_yticklabels([f"Unit {unit_id}" for unit_id in selected_unit_ids])

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set title
    ax.set_title("Noise Correlation Matrix")

    plt.tight_layout()
    return fig


# Create and display the correlation matrix
fig_matrix = plot_noise_correlation_matrix()
plt.show()

# %%
