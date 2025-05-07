"""
Create crosscorrelogram visualization
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

from dspant.io.loaders.phy_kilosort_loarder import load_kilosort

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
from dspant_neuroproc.processors.spike_analytics.correlogram import (
    SpikeCovarianceAnalyzer,
)

# %%
# Set parameters for correlogram analysis
bin_size_ms = 1.0  # 1 ms bin size
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

    # Visualize results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot autocorrelogram for unit1
    axes[0].bar(
        autocorr1["time_bins"] * 1000,
        autocorr1["autocorrelogram"],
        width=bin_size_ms,
        alpha=0.7,
        color="blue",
    )
    axes[0].set_title(f"Autocorrelogram for Unit {unit1}")
    axes[0].set_ylabel("Firing Rate (Hz)")

    # Plot autocorrelogram for unit2
    axes[1].bar(
        autocorr2["time_bins"] * 1000,
        autocorr2["autocorrelogram"],
        width=bin_size_ms,
        alpha=0.7,
        color="green",
    )
    axes[1].set_title(f"Autocorrelogram for Unit {unit2}")
    axes[1].set_ylabel("Firing Rate (Hz)")

    # Plot crosscorrelogram
    axes[2].bar(
        crosscorr["time_bins"] * 1000,
        crosscorr["crosscorrelogram"],
        width=bin_size_ms,
        alpha=0.7,
        color="red",
    )
    axes[2].set_title(f"Crosscorrelogram between Units {unit1} and {unit2}")
    axes[2].set_xlabel("Time Lag (ms)")
    axes[2].set_ylabel("Firing Rate (Hz)")

    # Add vertical line at zero lag
    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)

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
)
plt.title("Spike Time Tiling Coefficient (±5ms window)")
plt.xlabel("Unit ID Index")
plt.ylabel("Unit ID Index")
plt.tight_layout()
plt.show()
