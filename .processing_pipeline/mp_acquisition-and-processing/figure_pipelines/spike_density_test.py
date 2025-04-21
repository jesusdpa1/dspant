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
from dspant.vizualization.general_plots import plot_multi_channel_data

# Import the density estimation and plotting from dspant_neuroproc
from dspant_neuroproc.processors.spike_analytics.density_estimation import (
    SpikeDensityEstimator,
)
from dspant_neuroproc.visualization.spike_density_plot import (
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
# Create a combined raster and density plot
fig, axes = plot_combined_raster_density(
    sorter=sorter_data,
    bin_size_ms=bin_size_ms,
    sigma_ms=sigma_ms,
    start_time_s=start_time_s,
    end_time_s=end_time_s,
    raster_alpha=0.5,
    density_cmap="inferno",
    time_range=[0, 1],
)

# Adjust title to include parameters
fig.suptitle(f"Spike Analysis (bin={bin_size_ms}ms, σ={sigma_ms}ms)", fontsize=14)
plt.show()

# %%
# Create a zoomed-in version for a specific time segment
# Assuming a recording of several seconds, let's focus on a 1-second window
if time_bins[-1] > 5.0:
    zoom_start = 2.0
    zoom_end = 3.0

    fig, axes = plot_combined_raster_density(
        sorter=sorter_data,
        bin_size_ms=bin_size_ms,
        sigma_ms=sigma_ms,
        start_time_s=zoom_start,
        end_time_s=zoom_end,
        raster_alpha=0.8,
        density_cmap="plasma",
    )

    fig.suptitle(f"Spike Analysis - Zoomed ({zoom_start}-{zoom_end}s)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
# Analyze the effect of different smoothing sigmas
sigmas = [5.0, 20.0, 50.0]
fig, axes = plt.subplots(len(sigmas), 1, figsize=(12, 10), sharex=True)

for i, sigma in enumerate(sigmas):
    # Create estimator with this sigma
    est = SpikeDensityEstimator(bin_size_ms=bin_size_ms, sigma_ms=sigma)

    # Get density estimation
    time_bins, density, _ = est.estimate(
        sorter=sorter_data,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        unit_ids=used_unit_ids[:5],  # Use only first 5 units for clarity
    )

    # Plot density for each unit
    for unit_idx, unit_id in enumerate(used_unit_ids[:5]):
        axes[i].plot(
            time_bins, density[:, unit_idx], label=f"Unit {unit_id}" if i == 0 else None
        )

    axes[i].set_ylabel(f"Firing rate (Hz)\nσ={sigma}ms")
    axes[i].set_title(f"Smoothing with σ={sigma}ms")

# Add legend to the top subplot only
axes[0].legend(loc="upper right")
axes[-1].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

# %%
# Create a more advanced visualization for specific units
# Find the most active units
unit_activity = np.mean(spike_density, axis=0)
most_active_indices = np.argsort(unit_activity)[-5:]  # Top 5 most active
most_active_units = [used_unit_ids[i] for i in most_active_indices]

# Plot the density for these units with colorbars
if len(most_active_units) > 0:
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 2]}
    )

    # Plot firing rates as lines
    for i, unit_idx in enumerate(most_active_indices):
        unit_id = used_unit_ids[unit_idx]
        axes[0].plot(
            time_bins,
            spike_density[:, unit_idx],
            label=f"Unit {unit_id}",
            linewidth=1.5,
        )

    axes[0].set_ylabel("Firing rate (Hz)")
    axes[0].set_title(
        f"Spike Density for Most Active Units (bin={bin_size_ms}ms, σ={sigma_ms}ms)"
    )
    axes[0].legend()

    # Plot heatmap for these units
    selected_density = spike_density[:, most_active_indices]
    im = axes[1].imshow(
        selected_density.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap="viridis",
        extent=[time_bins[0], time_bins[-1], -0.5, len(most_active_units) - 0.5],
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Unit")
    axes[1].set_yticks(np.arange(len(most_active_units)))
    axes[1].set_yticklabels(most_active_units)

    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label("Firing rate (Hz)")

    plt.tight_layout()
    plt.show()

# %%
