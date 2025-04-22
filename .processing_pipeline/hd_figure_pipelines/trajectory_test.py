"""
Script to showcase spike density plots
Author: Jesus Penaloza
"""

# %%

import os
import time
from pathlib import Path

import dask.array as da
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.extractors.epoch_extractor import EpochExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.fir_filters import create_moving_average
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.double_threshold import (
    create_double_threshold_detector,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_absolute_threshold_detector,
)
from dspant_neuroproc.processors.neural_trajectories import PCATrajectoryAnalyzer

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


import matplotlib.pyplot as plt
import numpy as np


def plot_onset_detection_results(
    signal,
    onsets_df,
    fs,
    time_window=None,
    figsize=(12, 6),
    title="EMG Onset Detection Results",
    threshold=None,
    signal_label="Signal",
    highlight_color="red",
    show_threshold=True,
    debug_mode=False,  # Add debug mode to visualize the zero-crossing detection
):
    """
    Plot signal with highlighted onset/offset regions.

    Args:
        signal: The signal data (numpy or dask array)
        onsets_df: DataFrame with onset_idx, offset_idx columns
        fs: Sampling frequency in Hz
        time_window: Optional [start, end] in seconds to zoom
        figsize: Figure size tuple
        title: Plot title
        threshold: Optional threshold value to show
        signal_label: Label for the signal
        highlight_color: Color for highlighting detected events
        show_threshold: Whether to show threshold line
        debug_mode: Show additional debug information

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import polars as pl

    # Compute signal if it's a dask array
    if hasattr(signal, "compute"):
        signal = signal.compute()

    # Ensure signal is a 1D array
    if signal.ndim > 1:
        signal = signal[:, 0]  # Take first channel

    # Create time array
    t = np.arange(len(signal)) / fs

    # Create figure
    if debug_mode:
        fig, (ax, ax_debug) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 1.5), sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Apply time window if specified
    start_idx = 0
    if time_window is not None:
        start_idx = max(0, int(time_window[0] * fs))
        end_idx = min(len(signal), int(time_window[1] * fs))
        view_signal = signal[start_idx:end_idx]
        view_t = t[start_idx:end_idx]

        # Filter onsets within the window
        window_onsets = onsets_df.filter(
            (pl.col("onset_idx") >= start_idx) & (pl.col("onset_idx") <= end_idx)
        )
    else:
        view_signal = signal
        view_t = t
        window_onsets = onsets_df

    # Plot signal
    ax.plot(view_t, view_signal, label=signal_label, color="blue", linewidth=1.5)

    # Plot threshold if provided
    if threshold is not None and show_threshold:
        ax.axhline(
            y=threshold,
            color="green",
            linestyle="--",
            label=f"Threshold ({threshold:.2f})",
        )

    # Highlight onset-offset regions
    if len(window_onsets) > 0:
        for row in window_onsets.iter_rows(named=True):
            onset_idx = row["onset_idx"]
            offset_idx = row["offset_idx"]

            # Calculate correct time values, accounting for the window offset
            onset_time = (onset_idx - start_idx) / fs + (
                time_window[0] if time_window else 0
            )
            offset_time = (offset_idx - start_idx) / fs + (
                time_window[0] if time_window else 0
            )

            duration = row["duration"]

            # Shade the activation region
            ax.axvspan(onset_time, offset_time, alpha=0.3, color=highlight_color)

            # Mark the onset with a vertical line
            ax.axvline(x=onset_time, color=highlight_color, linestyle="-", linewidth=1)

            # Add annotation
            ax.annotate(
                f"{duration:.2f}s",
                xy=(onset_time, np.max(view_signal)),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=8,
                color=highlight_color,
            )

            # Debug: Mark actual signal value at onset and offset points
            if debug_mode:
                rel_onset_idx = onset_idx - start_idx
                rel_offset_idx = offset_idx - start_idx

                if 0 <= rel_onset_idx < len(view_signal):
                    onset_value = view_signal[rel_onset_idx]
                    ax.plot(onset_time, onset_value, "o", color="red", markersize=6)

                if 0 <= rel_offset_idx < len(view_signal):
                    offset_value = view_signal[rel_offset_idx]
                    ax.plot(
                        offset_time, offset_value, "s", color="purple", markersize=6
                    )

    # Add labels and legend
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Debug mode: Show the difference signal (signal - threshold)
    if debug_mode and threshold is not None:
        diff_signal = view_signal - threshold
        ax_debug.plot(view_t, diff_signal, label="Signal - Threshold", color="purple")
        ax_debug.axhline(y=0, color="red", linestyle="-", label="Zero line")

        # Mark zero-crossings
        zero_crossings = np.where(np.diff(np.signbit(diff_signal)))[0]
        for zc in zero_crossings:
            ax_debug.axvline(x=view_t[zc], color="green", linestyle="--", alpha=0.5)

        ax_debug.set_ylabel("Difference (Signal - Threshold)")
        ax_debug.legend(loc="upper right")
        ax_debug.grid(True, linestyle="--", alpha=0.7)

    # Tight layout for better appearance
    plt.tight_layout()

    return fig


# %%

data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/RawG.ant")
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

print(sorter_path)
# %%
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()


# Load EMG data
sorter_data = load_kilosort(sorter_path)
# Print stream_emg summary
sorter_data.summarize()

fs = stream_emg.fs  # Get sampling rate from the stream node
# %%
# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)
# Create processing node with filters
processor_emg = create_processing_node(stream_emg)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
filtered_emg = processor_emg.process(group=["filters"]).persist()
# %%
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data[0:1000000, 0]).persist()

st_tkeo_processor = create_double_threshold_detector(
    primary_threshold=0.045,
    secondary_threshold=0.1,
    min_event_spacing=0.01,
    min_contraction_duration=0.01,
)
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)
# %%

plot_ = plot_onset_detection_results(
    zscore_tkeo,
    tkeo_epochs,
    fs=fs,
    time_window=[0, 10],  # 10-second window
    title="TKEO EMG Signal with Detected Onsets",
    threshold=0.2,
    signal_label="Normalized TKEO EMG",
    highlight_color="red",
)

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
onsets = tkeo_epochs["onset_idx"] / fs
offsets = tkeo_epochs["offset_idx"] / fs
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
    offsets=offsets,
    padding_strategy="zero",
)

# %%
# Select a specific epoch (e.g., the first epoch)
selected_epoch = epochs_zero[0, :, :].compute()  # Compute the dask array to numpy

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

trajectory_analyzer = PCATrajectoryAnalyzer(
    n_components=3,
    compute_immediately=True,
)
# %%
a = trajectory_analyzer.fit_transform(epochs_zero)
# %%
b = trajectory_analyzer.plot_trajectories()

# %%
