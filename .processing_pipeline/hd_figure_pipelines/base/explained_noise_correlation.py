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


# Create and display the plot for an interesting pair of units
interesting_unit_pairs = [(26, 27), (20, 31), (36, 50)]
unit1_idx, unit2_idx = interesting_unit_pairs[0]  # Use first pair by default


def plot_noise_correlation_grid(unit_indices=None, figsize=(15, 15), max_units=30):
    """
    Plot a grid of noise correlation plots for multiple units

    Parameters:
    -----------
    unit_indices : list of int, optional
        Indices of units to include in the grid. If None, uses all units (up to max_units).
    figsize : tuple, optional
        Figure size
    max_units : int, optional
        Maximum number of units to plot when unit_indices is None

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get all unit IDs
    unit_ids = sorter_data.unit_ids

    # If no unit indices provided, use all units (up to max_units)
    if unit_indices is None:
        # Get all valid unit indices
        all_unit_indices = list(range(len(unit_ids)))

        # Check if we have too many units
        if len(all_unit_indices) > max_units:
            print(
                f"Warning: Found {len(all_unit_indices)} units, limiting to {max_units}"
            )
            unit_indices = all_unit_indices[:max_units]
        else:
            unit_indices = all_unit_indices

    # Get selected unit IDs
    selected_unit_ids = [unit_ids[idx] for idx in unit_indices]
    n_units = len(unit_indices)

    # Adjust figure size based on number of units if needed
    if n_units > 6:
        scale_factor = n_units / 6  # Scale based on default 6 units
        figsize = (figsize[0] * scale_factor, figsize[1] * scale_factor)

    # Create figure with n_units × n_units subplots
    fig, axes = plt.subplots(
        n_units, n_units, figsize=figsize, sharex="col", sharey="row"
    )

    # Handle case of single unit (make axes a 2D array)
    if n_units == 1:
        axes = np.array([[axes]])

    # Calculate and store all spike counts first to avoid recalculation
    spike_counts = {}
    for i, unit_idx in enumerate(unit_indices):
        try:
            # Get spike counts for this unit
            unit_id = selected_unit_ids[i]
            unit_counts = np.zeros(len(emg_onsets))

            # Calculate spike counts for each trial
            for trial_idx, onset_time in enumerate(emg_onsets):
                # Define time window for this trial
                start_time = onset_time - PRE_EVENT
                end_time = onset_time + POST_EVENT

                # Get spikes for this unit in this window
                unit_spikes = sorter_data.get_unit_spike_train(
                    unit_id,
                    start_frame=int(start_time * fs),
                    end_frame=int(end_time * fs),
                )
                unit_counts[trial_idx] = len(unit_spikes)

            spike_counts[unit_idx] = unit_counts
        except Exception as e:
            print(f"Error calculating spike counts for unit {unit_id}: {e}")
            # Create empty array so plotting can continue
            spike_counts[unit_idx] = np.zeros(len(emg_onsets))

    # Plot each pair of units
    for i in range(n_units):
        for j in range(n_units):
            ax = axes[i, j]
            unit1_idx = unit_indices[j]  # x-axis
            unit2_idx = unit_indices[i]  # y-axis
            unit1_id = selected_unit_ids[j]
            unit2_id = selected_unit_ids[i]

            try:
                # Get pre-calculated spike counts
                unit1_counts = spike_counts[unit1_idx]
                unit2_counts = spike_counts[unit2_idx]

                # Check if there's enough variation in the data
                if (
                    len(np.unique(unit1_counts)) < 2
                    or len(np.unique(unit2_counts)) < 2
                    or np.std(unit1_counts) == 0
                    or np.std(unit2_counts) == 0
                ):
                    # Not enough variation, just plot the points without regression
                    no_regression = True
                else:
                    no_regression = False

                # Add jitter to avoid overlapping points
                jitter = 0.1
                unit1_jittered = unit1_counts + np.random.uniform(
                    -jitter, jitter, size=unit1_counts.shape
                )
                unit2_jittered = unit2_counts + np.random.uniform(
                    -jitter, jitter, size=unit2_counts.shape
                )

                # Plot individual points with flat color (no edge)
                ax.scatter(
                    unit1_jittered,
                    unit2_jittered,
                    alpha=0.7,
                    s=25,
                    c="#4285F4",
                    edgecolor=None,
                )

                # Calculate correlation coefficient and p-value
                try:
                    r, p_value = stats.pearsonr(unit1_counts, unit2_counts)
                except:
                    # Use nan if correlation can't be calculated
                    r, p_value = np.nan, np.nan

                # Add regression line if appropriate
                if len(unit1_counts) > 1 and not no_regression and not np.isnan(r):
                    try:
                        # Use polyfit with rcond parameter to improve stability
                        m, b = np.polyfit(unit1_counts, unit2_counts, 1, rcond=1e-10)
                        x_range = np.array(
                            [max(0, min(unit1_counts) - 1), max(unit1_counts) + 1]
                        )
                        ax.plot(x_range, m * x_range + b, "r-", linewidth=1.5)
                    except np.linalg.LinAlgError:
                        # Skip regression line if fitting fails
                        pass
                    except Exception as e:
                        print(
                            f"Error fitting regression line for units {unit1_id} and {unit2_id}: {e}"
                        )

                # Add correlation value to plot
                if not np.isnan(r):
                    correlation_text = f"r={r:.2f}"
                    if i != j:  # Don't add p-value on diagonal
                        correlation_text += f"\np={p_value:.3f}"
                    ax.text(
                        0.05,
                        0.95,
                        correlation_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )
                else:
                    # Show N/A if correlation couldn't be calculated
                    ax.text(
                        0.05,
                        0.95,
                        "r=N/A",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

                # Set axis limits
                try:
                    if len(unit1_counts) > 0 and np.max(unit1_counts) > np.min(
                        unit1_counts
                    ):
                        x_min, x_max = (
                            max(0, min(unit1_counts) - 1),
                            max(unit1_counts) + 1,
                        )
                        y_min, y_max = (
                            max(0, min(unit2_counts) - 1),
                            max(unit2_counts) + 1,
                        )
                        ax.set_xlim(x_min, x_max)
                        ax.set_ylim(y_min, y_max)
                    else:
                        # Set default limits if there's no variation
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                except:
                    # Set default limits if there's an error
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)

                # Set axis locators for integer ticks
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

            except Exception as e:
                print(
                    f"Error plotting correlation for units {unit1_id} and {unit2_id}: {e}"
                )
                # Display error message in the plot
                ax.text(
                    0.5,
                    0.5,
                    "Error plotting data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            # Highlight diagonal elements
            if i == j:
                ax.patch.set_facecolor("#f0f0f0")
                ax.patch.set_alpha(0.3)

            # Add labels only on the edges of the grid
            if i == n_units - 1:
                ax.set_xlabel(
                    f"Unit {unit1_id} Spike Count", fontsize=10, fontweight="bold"
                )
            if j == 0:
                ax.set_ylabel(
                    f"Unit {unit2_id} Spike Count", fontsize=10, fontweight="bold"
                )

    # Set overall title
    title = f"Noise Correlation Grid ({n_units} units)"
    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for title

    return fig


def plot_noise_correlation_matrix(
    unit_indices=None, style="color_with_text", show_triangle="lower", max_units=50
):
    """
    Plot a matrix of noise correlations between multiple units

    Parameters:
    -----------
    unit_indices : list of int, optional
        Indices of units to include in the matrix. If None, uses all units (up to max_units).
    style : str, optional
        Style of matrix visualization:
        - "color": just color indicating correlation strength
        - "color_with_text": color indicating direction and text showing r value
    show_triangle : str, optional
        Which triangle to show: "both", "upper", or "lower"
    max_units : int, optional
        Maximum number of units to plot when unit_indices is None

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Get all unit IDs
    unit_ids = sorter_data.unit_ids

    # If no unit indices provided, use all units (up to max_units)
    if unit_indices is None:
        # Get all valid unit indices
        all_unit_indices = list(range(len(unit_ids)))

        # Check if we have too many units
        if len(all_unit_indices) > max_units:
            print(
                f"Warning: Found {len(all_unit_indices)} units, limiting to {max_units}"
            )
            unit_indices = all_unit_indices[:max_units]
        else:
            unit_indices = all_unit_indices

    # Get selected unit IDs
    selected_unit_ids = [unit_ids[idx] for idx in unit_indices]
    n_units = len(unit_indices)

    # Adjust figure size based on number of units if needed
    figsize = (10, 8)
    if n_units > 10:
        scale_factor = n_units / 10  # Scale based on default size
        figsize = (figsize[0] * scale_factor, figsize[1] * scale_factor)

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
                try:
                    # Calculate spike counts
                    unit1_counts, unit2_counts = calculate_spike_counts(
                        unit1_index, unit2_index
                    )

                    # Check if there's enough variation for correlation
                    if (
                        len(np.unique(unit1_counts)) < 2
                        or len(np.unique(unit2_counts)) < 2
                        or np.std(unit1_counts) == 0
                        or np.std(unit2_counts) == 0
                    ):
                        r, p_value = 0.0, 1.0  # No correlation when no variation
                    else:
                        # Calculate correlation
                        r, p_value = stats.pearsonr(unit1_counts, unit2_counts)
                except Exception as e:
                    print(
                        f"Error calculating correlation for units {unit1_index} and {unit2_index}: {e}"
                    )
                    r, p_value = 0.0, 1.0  # Default values on error

                # Store in both upper and lower triangle
                correlation_matrix[i, j] = r
                correlation_matrix[j, i] = r

                pvalue_matrix[i, j] = p_value
                pvalue_matrix[j, i] = p_value

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for the triangle you want to hide
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    if show_triangle == "lower":
        # Mask the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    elif show_triangle == "upper":
        # Mask the lower triangle
        mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=0)

    # Create a masked array
    masked_corr = np.ma.array(correlation_matrix, mask=mask)

    # Use masked array to mark non-significant correlations
    sig_mask = pvalue_matrix > 0.05  # Non-significant correlations

    # Handle NaN values in the correlation matrix
    masked_corr = np.ma.masked_invalid(masked_corr)

    # Plot heatmap
    cmap = plt.cm.coolwarm
    cmap.set_bad("white")  # Set masked values to white
    im = ax.imshow(masked_corr, cmap=cmap, vmin=-1, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label="Correlation Coefficient")

    # Add text annotations with correlation values if requested
    if (
        style == "color_with_text" and n_units <= 20
    ):  # Only add text for reasonable number of units
        for i in range(n_units):
            for j in range(n_units):
                # Skip masked cells
                if mask[i, j]:
                    continue

                # Skip NaN values
                if np.isnan(correlation_matrix[i, j]):
                    continue

                # Format the correlation value
                text = f"{correlation_matrix[i, j]:.2f}"

                # Determine text color based on background
                text_color = "white" if abs(correlation_matrix[i, j]) > 0.5 else "black"

                # Use gray for non-significant values
                if sig_mask[i, j] and i != j:
                    text_color = "gray"

                # Bold text for diagonal elements
                fontweight = "bold" if i == j else "normal"

                # Add the text
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight=fontweight,
                )

    # Set ticks and labels with bold unit names
    # For large matrices, show ticks only for every nth unit to avoid overcrowding
    if n_units > 20:
        # Show every nth tick
        step = max(1, n_units // 20)
        tick_indices = np.arange(0, n_units, step)
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels(
            [f"Unit {selected_unit_ids[i]}" for i in tick_indices], fontweight="bold"
        )
        ax.set_yticklabels(
            [f"Unit {selected_unit_ids[i]}" for i in tick_indices], fontweight="bold"
        )
    else:
        # Show all ticks for smaller matrices
        ax.set_xticks(np.arange(n_units))
        ax.set_yticks(np.arange(n_units))
        ax.set_xticklabels(
            [f"Unit {unit_id}" for unit_id in selected_unit_ids], fontweight="bold"
        )
        ax.set_yticklabels(
            [f"Unit {unit_id}" for unit_id in selected_unit_ids], fontweight="bold"
        )

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set title in bold
    title = f"Noise Correlation Matrix ({n_units} units)"
    ax.set_title(title, fontweight="bold", fontsize=14)

    plt.tight_layout()
    return fig


# Create and display the grid view with updated styling
fig_grid = plot_noise_correlation_grid()
plt.show()
# %%
# Create and display the correlation matrix with only the lower triangle
fig_matrix = plot_noise_correlation_matrix(
    style="color_with_text", show_triangle="lower"
)
plt.show()


# %%
