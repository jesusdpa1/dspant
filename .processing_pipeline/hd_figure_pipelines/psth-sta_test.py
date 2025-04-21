"""
Script to showcase spike density plots, PSTH, and STA analysis
Author: Jesus Penaloza
Extended by: Claude
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
from dspant.vizualization.general_plots import plot_multi_channel_data

# Import the density estimation and plotting from dspant_neuroproc
from dspant_neuroproc.processors.spike_analytics.density_estimation import (
    SpikeDensityEstimator,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer
from dspant_neuroproc.processors.spike_analytics.sta import SpikeTriggeredAnalyzer
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
# Create trigger events at 1-second intervals
# We'll generate events from the start of the recording to near the end
trigger_interval = 1.0  # 1 second interval
recording_duration = time_bins[-1]  # Total duration of the recording
trigger_onsets = np.arange(0.5, recording_duration - 1, trigger_interval)
print(f"Created {len(trigger_onsets)} trigger events at 1-second intervals")
print(f"First few trigger times: {trigger_onsets[:5]}")

# %%
# 1. PSTH Analysis
# Create a PSTH analyzer
psth_analyzer = PSTHAnalyzer(
    bin_size_ms=10.0,  # 10ms bins
    window_size_ms=500.0,  # 500ms window (+-250ms around trigger)
    sigma_ms=20.0,  # Gaussian smoothing
    baseline_window_ms=(-200, -50),  # Baseline period: 200-50ms before trigger
)

# Compute PSTH aligned to trigger events
psth_results = psth_analyzer.transform(
    sorter=sorter_data, events=trigger_onsets, unit_ids=used_unit_ids
)

# %%
# Visualize PSTH results
plt.figure(figsize=(12, 10))

# Create time axis in milliseconds
time_ms = psth_results["time_bins"] * 1000  # Convert to ms

# Plot PSTH for each unit
n_units = len(used_unit_ids)
n_rows = min(5, n_units)  # Display up to 5 units

for i in range(n_rows):
    unit_idx = i
    unit_id = used_unit_ids[unit_idx]

    plt.subplot(n_rows, 1, i + 1)

    # Plot raw PSTH
    plt.plot(
        time_ms,
        psth_results["psth_rates"][:, unit_idx],
        "k-",
        alpha=0.5,
        label=f"Raw PSTH",
    )

    # Plot smoothed PSTH if available
    if "psth_smoothed" in psth_results:
        plt.plot(
            time_ms,
            psth_results["psth_smoothed"][:, unit_idx],
            "r-",
            linewidth=2,
            label=f"Smoothed PSTH",
        )

    # Add standard error bands if available
    plt.fill_between(
        time_ms,
        psth_results["psth_rates"][:, unit_idx] - psth_results["psth_sem"][:, unit_idx],
        psth_results["psth_rates"][:, unit_idx] + psth_results["psth_sem"][:, unit_idx],
        color="gray",
        alpha=0.2,
    )

    # Add baseline window if available
    if "baseline_window" in psth_results:
        baseline_start, baseline_end = psth_results["baseline_window"]
        plt.axvspan(baseline_start * 1000, baseline_end * 1000, color="blue", alpha=0.1)

    # Add vertical line at trigger time
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    plt.title(f"Unit {unit_id}")
    plt.ylabel("Firing Rate (Hz)")
    if i == n_rows - 1:
        plt.xlabel("Time from trigger (ms)")
    plt.legend()
    plt.grid(True)

plt.suptitle("Peristimulus Time Histogram (PSTH)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# Visualize normalized PSTH if available
if "psth_normalized" in psth_results:
    plt.figure(figsize=(12, 10))

    for i in range(n_rows):
        unit_idx = i
        unit_id = used_unit_ids[unit_idx]

        plt.subplot(n_rows, 1, i + 1)

        # Plot normalized PSTH
        plt.plot(
            time_ms,
            psth_results["psth_normalized"][:, unit_idx],
            "g-",
            linewidth=2,
            label=f"Normalized PSTH",
        )

        # Add vertical line at trigger time
        plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

        # Add horizontal line at baseline (y=0)
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)

        plt.title(f"Unit {unit_id}")
        plt.ylabel("Normalized Change")
        if i == n_rows - 1:
            plt.xlabel("Time from trigger (ms)")
        plt.legend()
        plt.grid(True)

    plt.suptitle("Baseline-Normalized PSTH", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# %%
# 2. STA Analysis
# For STA, we need a continuous signal to correlate with spikes
# Let's generate some synthetic LFP data as an example
np.random.seed(42)  # For reproducibility

# Create a continuous signal with some oscillatory components
fs = sorter_data.sampling_frequency
t = np.arange(0, recording_duration, 1 / fs)
synthetic_lfp = np.zeros((len(t), 2))  # 2 channels

# Add slow oscillation (theta)
theta_freq = 6.0  # Hz
theta_amp = 1.0
synthetic_lfp[:, 0] = theta_amp * np.sin(2 * np.pi * theta_freq * t)

# Add faster oscillation (beta)
beta_freq = 20.0  # Hz
beta_amp = 0.5
synthetic_lfp[:, 1] = beta_amp * np.sin(2 * np.pi * beta_freq * t)

# Add noise
synthetic_lfp += 0.2 * np.random.randn(*synthetic_lfp.shape)

# %%
# Create a STA analyzer
sta_analyzer = SpikeTriggeredAnalyzer(
    window_size_ms=200.0,  # 200ms window (±100ms around spike)
    baseline_window_ms=(-180, -120),  # Baseline period: 180-120ms before spike
)

# Compute STA
sta_results = sta_analyzer.transform(
    sorter=sorter_data,
    signal=synthetic_lfp,
    unit_ids=used_unit_ids[:5],  # Analyze first few units for efficiency
    start_time_s=0,
    end_time_s=60,  # Analyze first minute only for efficiency
)

# %%
# Visualize STA results
plt.figure(figsize=(12, 10))

# Create time axis in milliseconds
time_ms = sta_results["time_bins"] * 1000  # Convert to ms

# Plot STA for each unit and channel
n_units = len(sta_results["unit_ids"])
n_channels = sta_results["sta"].shape[1]
channel_names = ["Theta", "Beta"]  # Names for our synthetic channels

for i in range(n_units):
    unit_idx = i
    unit_id = sta_results["unit_ids"][i]

    plt.subplot(n_units, 1, i + 1)

    for ch in range(n_channels):
        plt.plot(
            time_ms,
            sta_results["sta"][:, ch, unit_idx],
            "-",
            linewidth=2,
            label=f"Channel {channel_names[ch]}",
        )

    # Add baseline window if available
    if "baseline_window" in sta_results:
        baseline_start, baseline_end = sta_results["baseline_window"]
        plt.axvspan(baseline_start * 1000, baseline_end * 1000, color="blue", alpha=0.1)

    # Add vertical line at spike time
    plt.axvline(x=0, color="r", linestyle="--", alpha=0.5)

    plt.title(f"Unit {unit_id} (n={sta_results['spike_counts'][i]} spikes)")
    plt.ylabel("Signal Amplitude")
    if i == n_units - 1:
        plt.xlabel("Time from spike (ms)")
    plt.legend()
    plt.grid(True)

plt.suptitle("Spike-Triggered Average (STA)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# 3. Epoch extraction with 1-second interval triggers
# Use the trigger onsets for epoch extraction
epoch_extractor = EpochExtractor(
    time_array=time_bins, data_array=spike_density, fs=1 / np.mean(np.diff(time_bins))
)

# Extract epochs with different padding strategies
epochs_zero = epoch_extractor.extract_epochs(
    onsets=trigger_onsets[:-1],  # Exclude the last trigger to ensure complete epochs
    window_size=1.0,  # 1-second window
    padding_strategy="zero",
)

# %%
# Calculate mean, standard deviation, and coefficient of variation across epochs
epochs_mean = epochs_zero.mean(axis=0).compute()
epochs_std = epochs_zero.std(axis=0).compute()
epochs_cv = epochs_std / (
    epochs_mean + 1e-10
)  # Add small value to avoid division by zero

# %%
# Plot average epoch activity
plt.figure(figsize=(16, 10))

# 1. Mean across epochs
plt.subplot(2, 2, 1)
epoch_time = np.linspace(0, 1, epochs_mean.shape[0])
plot_spike_density(
    time_bins=epoch_time,
    spike_density=epochs_mean,
    unit_ids=used_unit_ids,
    cmap="viridis",
    title="Mean Activity Across Epochs",
    sorting=True,
    ax=plt.gca(),
)

# 2. Standard deviation across epochs
plt.subplot(2, 2, 2)
plot_spike_density(
    time_bins=epoch_time,
    spike_density=epochs_std,
    unit_ids=used_unit_ids,
    cmap="plasma",
    title="Std Dev Across Epochs",
    sorting=True,
    ax=plt.gca(),
)

# 3. Coefficient of variation across epochs
plt.subplot(2, 2, 3)
plot_spike_density(
    time_bins=epoch_time,
    spike_density=epochs_cv,
    unit_ids=used_unit_ids,
    cmap="coolwarm",
    title="Coefficient of Variation Across Epochs",
    sorting=True,
    ax=plt.gca(),
)

# 4. Example of a single epoch
plt.subplot(2, 2, 4)
example_epoch_idx = 5  # Choose epoch to display
example_epoch = epochs_zero[example_epoch_idx].compute()
plot_spike_density(
    time_bins=epoch_time,
    spike_density=example_epoch,
    unit_ids=used_unit_ids,
    cmap="cividis",
    title=f"Example Epoch (#{example_epoch_idx})",
    sorting=True,
    ax=plt.gca(),
)

plt.suptitle("Spike Density Patterns in 1-Second Epochs", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
# Create a stacked view of multiple epochs for comparison
n_epochs_to_show = 10
stacked_epochs = epochs_zero[:n_epochs_to_show].compute()

plt.figure(figsize=(12, 15))
for i in range(n_epochs_to_show):
    plt.subplot(n_epochs_to_show, 1, i + 1)

    # Get this epoch's data
    epoch_data = stacked_epochs[i]

    # Simple line plot of activity for clarity
    for unit_idx, unit_id in enumerate(used_unit_ids[:5]):  # Show first 5 units
        plt.plot(epoch_time, epoch_data[:, unit_idx], label=f"Unit {unit_id}")

    plt.title(f"Epoch {i} (t={trigger_onsets[i]:.2f}s)")
    plt.ylabel("Firing Rate (Hz)")
    plt.xlim(0, 1)

    if i == 0:
        plt.legend(loc="upper right")
    if i == n_epochs_to_show - 1:
        plt.xlabel("Time within Epoch (s)")

plt.suptitle("Comparison of Neural Activity Across Epochs", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
