"""
Enhanced crosscorrelogram visualization with KDE
Author: Jesus Penaloza
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from numba import jit, prange
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant_neuroproc.processors.spike_analytics.correlogram import (
    SpikeCovarianceAnalyzer,
)

# Set style for visualizations
sns.set_theme(style="darkgrid")
load_dotenv()

# %%
# Analysis Parameters

# %%
# Load spike sorting data
data_dir = Path(os.getenv("DATA_DIR"))
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)

# Load spike sorting data
sorter_data = load_kilosort(sorter_path)
fs = sorter_data.sampling_frequency

print(f"Data loaded successfully. Sampling rate: {fs} Hz")

# %%
# Import correlogram analyzer from dspant_neuroproc

# %%
# Set parameters for correlogram analysis
bin_size_ms = 5.0  # 1 ms bin size
window_size_ms = 100.0  # 100 ms window size
normalization = "rate"  # Normalize by firing rate

# Create analyzer
correlogram_analyzer = SpikeCovarianceAnalyzer(
    bin_size_ms=bin_size_ms, window_size_ms=window_size_ms, normalization=normalization
)

# %%
# Get a list of unit IDs with good quality and sufficient spikes
min_spikes = 200
good_units = [
    unit
    for unit in sorter_data.unit_ids
    if len(sorter_data.get_unit_spike_train(unit)) >= min_spikes
]

print(f"Found {len(good_units)} units with at least {min_spikes} spikes")

# %%
# Select two units for demonstration
if len(good_units) >= 2:
    unit1 = good_units[0]
    unit2 = good_units[1]

    print(
        f"Computing autocorrelograms and crosscorrelogram for units {unit1} and {unit2}"
    )

    # Compute autocorrelogram for unit1
    start_time = time.time()
    autocorr1 = correlogram_analyzer.compute_autocorrelogram(sorter_data, unit1)
    end_time = time.time()
    print(
        f"Autocorrelogram for unit {unit1} computed in {end_time - start_time:.4f} seconds"
    )

    # Compute autocorrelogram for unit2
    start_time = time.time()
    autocorr2 = correlogram_analyzer.compute_autocorrelogram(sorter_data, unit2)
    end_time = time.time()
    print(
        f"Autocorrelogram for unit {unit2} computed in {end_time - start_time:.4f} seconds"
    )

    # Compute crosscorrelogram
    start_time = time.time()
    crosscorr = correlogram_analyzer.compute_crosscorrelogram(sorter_data, unit1, unit2)
    end_time = time.time()
    print(
        f"Crosscorrelogram between units {unit1} and {unit2} computed in {end_time - start_time:.4f} seconds"
    )

    # Visualize results with KDE overlays
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Define colors
    bar_colors = ["blue", "green", "red"]
    kde_colors = ["navy", "darkgreen", "darkred"]

    # Plot correlogram data sets
    correlogram_data = [
        (autocorr1, f"Autocorrelogram for Unit {unit1}", 0),
        (autocorr2, f"Autocorrelogram for Unit {unit2}", 1),
        (crosscorr, f"Crosscorrelogram between Units {unit1} and {unit2}", 2),
    ]

    # Loop through data and plot histograms with KDE
    for data_dict, title, idx in correlogram_data:
        # Convert time bins to ms
        time_bins_ms = data_dict["time_bins"] * 1000
        correlogram_values = data_dict[
            "autocorrelogram" if "autocorrelogram" in data_dict else "crosscorrelogram"
        ]

        # Plot histogram
        axes[idx].bar(
            time_bins_ms,
            correlogram_values,
            width=bin_size_ms,
            alpha=0.7,
            color=bar_colors[idx],
        )

        # Add KDE overlay
        # Filter out zeros and use only non-zero points for KDE
        nonzero_mask = correlogram_values > 0
        if np.sum(nonzero_mask) > 5:  # Need enough points for KDE
            # Create KDE using only bins with data (repeat based on value for weighting)
            time_data = np.repeat(
                time_bins_ms[nonzero_mask],
                (correlogram_values[nonzero_mask] * 10).astype(int),
            )

            if len(time_data) > 0:
                kde = gaussian_kde(time_data, bw_method="scott")

                # Create smooth line for KDE
                x_kde = np.linspace(time_bins_ms.min(), time_bins_ms.max(), 1000)
                y_kde = kde(x_kde)

                # Apply Savitzky-Golay filter for smoother curve
                y_kde_smooth = savgol_filter(y_kde, window_length=15, polyorder=3)

                # Scale KDE to match histogram height
                scale_factor = (
                    np.max(correlogram_values) / np.max(y_kde_smooth)
                    if np.max(y_kde_smooth) > 0
                    else 1
                )
                y_kde_smooth *= scale_factor

                # Plot KDE
                axes[idx].plot(x_kde, y_kde_smooth, color=kde_colors[idx], linewidth=2)

        # Set labels
        axes[idx].set_title(title)
        axes[idx].set_ylabel("Firing Rate (Hz)")

        # Add vertical line at zero lag
        axes[idx].axvline(0, color="black", linestyle="--", alpha=0.5)

    # Set common x-label
    axes[2].set_xlabel("Time Lag (ms)")

    plt.tight_layout()
    plt.show()

    # Also compute Spike Time Tiling Coefficient
    sttc = correlogram_analyzer.compute_spike_time_tiling_coefficient(
        sorter_data, unit1, unit2, delta_t_ms=5.0
    )
    print(f"Spike Time Tiling Coefficient (±5ms window): {sttc:.4f}")
else:
    print("Not enough good units for demonstration")

# %%
# Time the computation for all units
print(f"\nComputing correlograms for all {len(good_units)} units...")

# Time the full transform method (which computes all correlograms)
start_time = time.time()
all_correlograms = correlogram_analyzer.transform(sorter_data, unit_ids=good_units)
end_time = time.time()

total_time = end_time - start_time
print(f"All correlograms computed in {total_time:.4f} seconds")

# Count the number of correlograms
n_autocorr = len(all_correlograms["autocorrelograms"])
n_crosscorr = len(all_correlograms["crosscorrelograms"])
total_correlograms = n_autocorr + n_crosscorr

print(f"Computed {n_autocorr} autocorrelograms and {n_crosscorr} crosscorrelograms")
print(f"Average time per correlogram: {(total_time / total_correlograms):.4f} seconds")

# %%
# Create a correlogram grid for multiple units
# Select a subset of units to display in the grid
max_grid_units = min(5, len(good_units))  # Limit to 5 units for readability
grid_units = good_units[:max_grid_units]
n_grid_units = len(grid_units)

# Create mapping from unit ID to index for easier lookup
unit_to_idx = {unit: i for i, unit in enumerate(grid_units)}

# Set up figure for correlogram grid
plt.figure(figsize=(3 * n_grid_units, 3 * n_grid_units))
grid_axes = plt.subplot2grid((n_grid_units, n_grid_units), (0, 0)).figure.subplots(
    n_grid_units, n_grid_units, sharex=True
)

# Generate colors for each unit
colors = plt.cm.viridis(np.linspace(0, 1, n_grid_units))
kde_colors = plt.cm.viridis(np.linspace(0, 0.7, n_grid_units))

# Plot autocorrelograms on diagonal
for i, unit in enumerate(grid_units):
    # Find autocorrelogram for this unit
    for ac in all_correlograms["autocorrelograms"]:
        if ac["unit_id"] == unit:
            # Convert time bins to ms
            time_bins_ms = ac["time_bins"] * 1000
            correlogram_values = ac["autocorrelogram"]

            # Plot histogram
            grid_axes[i, i].bar(
                time_bins_ms,
                correlogram_values,
                width=bin_size_ms,
                alpha=0.7,
                color=colors[i],
            )

            # Add KDE overlay (same code as above)
            nonzero_mask = correlogram_values > 0
            if np.sum(nonzero_mask) > 5:
                time_data = np.repeat(
                    time_bins_ms[nonzero_mask],
                    (correlogram_values[nonzero_mask] * 10).astype(int),
                )

                if len(time_data) > 0:
                    kde = gaussian_kde(time_data, bw_method="scott")
                    x_kde = np.linspace(time_bins_ms.min(), time_bins_ms.max(), 1000)
                    y_kde = kde(x_kde)
                    y_kde_smooth = savgol_filter(y_kde, window_length=15, polyorder=3)
                    scale_factor = (
                        np.max(correlogram_values) / np.max(y_kde_smooth)
                        if np.max(y_kde_smooth) > 0
                        else 1
                    )
                    y_kde_smooth *= scale_factor
                    grid_axes[i, i].plot(
                        x_kde, y_kde_smooth, color=kde_colors[i], linewidth=2
                    )

            # Set title and add zero line
            grid_axes[i, i].set_title(f"Unit {unit} auto")
            grid_axes[i, i].axvline(0, color="black", linestyle="--", alpha=0.5)
            break

# Plot crosscorrelograms off diagonal
for cc in all_correlograms["crosscorrelograms"]:
    unit1 = cc["unit1"]
    unit2 = cc["unit2"]
    if unit1 in grid_units and unit2 in grid_units:
        i = unit_to_idx[unit1]
        j = unit_to_idx[unit2]

        # Convert time bins to ms
        time_bins_ms = cc["time_bins"] * 1000
        correlogram_values = cc["crosscorrelogram"]

        # Plot histogram in both positions (symmetric)
        for pos in [(i, j), (j, i)]:
            row, col = pos

            # Plot histogram
            grid_axes[row, col].bar(
                time_bins_ms,
                correlogram_values,
                width=bin_size_ms,
                alpha=0.7,
                color=colors[col],  # Use color of column unit
            )

            # Add KDE overlay
            nonzero_mask = correlogram_values > 0
            if np.sum(nonzero_mask) > 5:
                time_data = np.repeat(
                    time_bins_ms[nonzero_mask],
                    (correlogram_values[nonzero_mask] * 10).astype(int),
                )

                if len(time_data) > 0:
                    kde = gaussian_kde(time_data, bw_method="scott")
                    x_kde = np.linspace(time_bins_ms.min(), time_bins_ms.max(), 1000)
                    y_kde = kde(x_kde)
                    y_kde_smooth = savgol_filter(y_kde, window_length=15, polyorder=3)
                    scale_factor = (
                        np.max(correlogram_values) / np.max(y_kde_smooth)
                        if np.max(y_kde_smooth) > 0
                        else 1
                    )
                    y_kde_smooth *= scale_factor
                    grid_axes[row, col].plot(
                        x_kde, y_kde_smooth, color=kde_colors[col], linewidth=2
                    )

            # Set title and add zero line
            unit_first = grid_units[row]
            unit_second = grid_units[col]
            grid_axes[row, col].set_title(f"Units {unit_first}-{unit_second}")
            grid_axes[row, col].axvline(0, color="black", linestyle="--", alpha=0.5)

# Set shared x and y labels
for i in range(n_grid_units):
    grid_axes[n_grid_units - 1, i].set_xlabel("Time Lag (ms)")
    grid_axes[i, 0].set_ylabel("Firing Rate (Hz)")

# Set global title and adjust layout
plt.suptitle(f"Correlogram Grid for {n_grid_units} Units", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave room for suptitle
plt.show()

# %%
# Create a heatmap of STTC values between all units
n_units = len(good_units)
sttc_matrix = np.zeros((n_units, n_units))

print("\nComputing STTC matrix for all unit pairs...")
start_time = time.time()

for i, unit1 in enumerate(good_units):
    for j, unit2 in enumerate(good_units):
        if i == j:
            # Autocorrelation is 1.0 by definition
            sttc_matrix[i, j] = 1.0
        elif i < j:  # Compute only for upper triangle
            sttc = correlogram_analyzer.compute_spike_time_tiling_coefficient(
                sorter_data, unit1, unit2, delta_t_ms=5.0
            )
            sttc_matrix[i, j] = sttc
            sttc_matrix[j, i] = sttc  # Matrix is symmetric

end_time = time.time()
print(f"STTC matrix computed in {end_time - start_time:.4f} seconds")

# Plot STTC matrix as a heatmap
plt.figure(figsize=(10, 8))
mask = np.zeros_like(sttc_matrix, dtype=bool)
mask[np.diag_indices_from(mask)] = True  # Mask diagonal (autocorrelation)

sns.heatmap(
    sttc_matrix,
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    xticklabels=good_units,  # Add unit IDs as tick labels
    yticklabels=good_units,  # Add unit IDs as tick labels
)
plt.title("Spike Time Tiling Coefficient (±5ms window)")
plt.xlabel("Unit ID")
plt.ylabel("Unit ID")
plt.tight_layout()
plt.show()

# %%
