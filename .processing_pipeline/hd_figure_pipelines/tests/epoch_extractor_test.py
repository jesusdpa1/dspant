"""
Script to showcase spike density plots
Author: Jesus Penaloza
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv

from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.processors.extractors.epoch_extractor import EpochExtractor
from dspant.visualization.general_plots import plot_multi_channel_data

# Import the density estimation and plotting from dspant_neuroproc
from dspant_neuroproc.processors.spike_analytics.density_estimation import (
    SpikeDensityEstimator,
)
from dspant_neuroproc.visualization.spike_density_plots import (
    plot_combined_raster_density,
    plot_spike_density,
)

sns.set_theme(style="darkgrid")
load_dotenv()
# %%

data_dir = Path(os.getenv("DATA_DIR"))

sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

print(sorter_path)
# %%
# Load EMG data
sorter_data = load_kilosort(sorter_path)
# Print stream_emg summary
sorter_data.summarize()

# %%
# Define parameters for spike density estimation
bin_size_ms = 2.0
sigma_ms = 4.0
start_time_s = 0.0
end_time_s = None  # Use None to process the entire recording

# Create a spike density estimator
estimator = SpikeDensityEstimator(bin_size_ms=bin_size_ms, sigma_ms=sigma_ms)

# %%
# Estimate spike density
time_bins, spike_density, used_unit_ids = estimator.estimate(
    sorter=sorter_data, start_time_s=start_time_s, end_time_s=end_time_s
)

print(f"Processed {len(used_unit_ids)} units with {len(time_bins)} time bins")
print(f"Time range: {time_bins[0]:.2f}s - {time_bins[-1]:.2f}s")
print(f"Spike density shape: {spike_density.shape}")

# %%
# Create a density plot
plt.figure(figsize=(12, 8))
fig, ax = plot_spike_density(
    time_bins=time_bins,
    spike_density=spike_density,
    unit_ids=used_unit_ids,
    cmap="viridis",
    title="Spike Density Heatmap",
    time_range=(0, 10),
    sorting=True,
)
plt.tight_layout()
plt.show()

# %%
onsets = np.arange(0, 299)  # 10 evenly spaced onsets across the entire recording
print("Synthetic Onsets:", onsets)
# %%
# Use the time_bins and spike_density from previous estimation
epoch_extractor = EpochExtractor(
    time_bins, spike_density, fs=1 / np.mean(np.diff(time_bins))
)

# Test different epoch extraction strategies
# 1. Zero padding
epochs_zero = epoch_extractor.extract_epochs(
    onsets=onsets,
    window_size=1.0,  # 1-second window
    padding_strategy="zero",
)

# %%
# Select a specific epoch (e.g., the first epoch)
selected_epoch = epochs_zero[1, :, :].compute()  # Compute the dask array to numpy

# Use the plot_spike_density function
plt.figure(figsize=(12, 8))
fig, ax = plot_spike_density(
    time_bins=np.linspace(
        0, 1, selected_epoch.shape[0]
    ),  # Create time bins for the epoch
    spike_density=selected_epoch,
    unit_ids=used_unit_ids,
    cmap="viridis",
    title="Spike Density for Single Epoch",
    sorting=True,
)
plt.tight_layout()
plt.show()
# %%
