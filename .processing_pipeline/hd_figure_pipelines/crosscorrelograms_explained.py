"""
Create a single crosscorrelogram visualization
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
from dspant_neuroproc.processors.spike_analytics.correlogram import (
    create_spike_correlogram_analyzer,
)

# Set style for visualizations
sns.set_theme(style="darkgrid")
load_dotenv()

# %%
# Analysis Parameters
BIN_SIZE_MS = 1  # Bin size in milliseconds
WINDOW_SIZE_MS = 100.0  # Window size for cross-correlation in ms (±50 ms)
UNIT_ID_1 = 27  # Reference unit
UNIT_ID_2 = 20  # Target unit
NORMALIZE = False  # Whether to normalize
METHOD = "fft"  # Method to use: 'direct' or 'fft'

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
# Get spike times directly for better performance
spikes1 = sorter_data.get_unit_spike_train(UNIT_ID_1) / fs
spikes2 = sorter_data.get_unit_spike_train(UNIT_ID_2) / fs

print(f"Unit {UNIT_ID_1}: {len(spikes1)} spikes")
print(f"Unit {UNIT_ID_2}: {len(spikes2)} spikes")

# %%
# Create correlogram analyzer with specified parameters
normalization = "rate" if NORMALIZE else "count"
correlogram_analyzer = create_spike_correlogram_analyzer(
    bin_size_ms=BIN_SIZE_MS,
    window_size_ms=WINDOW_SIZE_MS,
    normalization=normalization,
    method=METHOD,
)

# %%
# Compute correlogram
print(
    f"Computing correlogram between units {UNIT_ID_1} and {UNIT_ID_2} using {METHOD} method..."
)
is_autocorr = UNIT_ID_1 == UNIT_ID_2

start_time = time.time()
if is_autocorr:
    # Autocorrelogram (pass only spikes1)
    time_bins, ccg, _ = correlogram_analyzer.compute_correlogram(spikes1)
else:
    # Crosscorrelogram
    time_bins, ccg, _ = correlogram_analyzer.compute_correlogram(spikes1, spikes2)

elapsed = time.time() - start_time
print(f"Correlogram computed in {elapsed:.2f} seconds")

# %%
# Plot the correlogram
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate bar width
bar_width = time_bins[1] - time_bins[0] if len(time_bins) > 1 else 1.0

# Convert time_bins to milliseconds if needed
time_bins_ms = time_bins * 1000

# Plot as bars
ax.bar(time_bins_ms, ccg, width=bar_width * 1000, color="steelblue", alpha=0.7)
ax.axvline(x=0, color="red", linestyle="--", alpha=0.6, linewidth=1)

# Set title and labels
if is_autocorr:
    title = f"Autocorrelogram for Unit {UNIT_ID_1}"
else:
    title = f"Crosscorrelogram: Unit {UNIT_ID_1} → Unit {UNIT_ID_2}"

ax.set_title(title, fontsize=14)
ax.set_xlabel("Time lag (ms)", fontsize=12)

if NORMALIZE:
    ax.set_ylabel("Firing rate (Hz)", fontsize=12)
else:
    ax.set_ylabel("Count", fontsize=12)

# Set limits
ax.set_xlim(-WINDOW_SIZE_MS / 2, WINDOW_SIZE_MS / 2)
ax.set_ylim(bottom=0)

# Add statistics
n_spikes1 = len(spikes1)
n_spikes2 = len(spikes2)
ax.text(0.02, 0.95, f"Unit {UNIT_ID_1}: {n_spikes1} spikes", transform=ax.transAxes)
ax.text(0.02, 0.90, f"Unit {UNIT_ID_2}: {n_spikes2} spikes", transform=ax.transAxes)

# Add method and performance info
ax.text(0.02, 0.85, f"Method: {METHOD}", transform=ax.transAxes)
ax.text(0.02, 0.80, f"Computation time: {elapsed:.2f} seconds", transform=ax.transAxes)

plt.tight_layout()
plt.show()

# %%
# Save the figure
fig.savefig("single_correlogram.png", dpi=300, bbox_inches="tight")
print("Figure saved as 'single_correlogram.png'")


# %%
# Optionally compare performance between direct and FFT methods
def compare_methods():
    """Compare performance between direct and FFT methods"""
    print("\nComparing computation methods...")

    # Create analyzers for both methods
    direct_analyzer = create_spike_correlogram_analyzer(
        bin_size_ms=BIN_SIZE_MS,
        window_size_ms=WINDOW_SIZE_MS,
        normalization=normalization,
        method="direct",
    )

    fft_analyzer = create_spike_correlogram_analyzer(
        bin_size_ms=BIN_SIZE_MS,
        window_size_ms=WINDOW_SIZE_MS,
        normalization=normalization,
        method="fft",
    )

    # Time direct method
    start_time = time.time()
    if is_autocorr:
        direct_analyzer.compute_correlogram(spikes1)
    else:
        direct_analyzer.compute_correlogram(spikes1, spikes2)
    direct_time = time.time() - start_time

    # Time FFT method
    start_time = time.time()
    if is_autocorr:
        fft_analyzer.compute_correlogram(spikes1)
    else:
        fft_analyzer.compute_correlogram(spikes1, spikes2)
    fft_time = time.time() - start_time

    print(f"Direct method: {direct_time:.2f} seconds")
    print(f"FFT method: {fft_time:.2f} seconds")
    print(f"Speedup: {direct_time / fft_time:.2f}x")

    return direct_time, fft_time


# Run comparison if requested
# compare_methods()
