"""
Create visualization showing neural activity aligned to EMG contractions
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

from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()

# %%
# Load data (using your existing code)
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

# %%
# Process EMG data
processor_emg = create_processing_node(stream_emg)

# Create filters
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)

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

# %%
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

# %%
# Extract EMG epochs around neural activity
# Define time window for analysis
PRE_EVENT = 0.5  # seconds before event
POST_EVENT = 1.0  # seconds after event

# Initialize EpochExtractor for TKEO data
epoch_extractor = WaveformExtractor(data=zscore_tkeo[:, :].compute(), fs=fs)

# Extract TKEO epochs aligned to EMG onsets
tkeo_epochs_aligned = epoch_extractor.extract_waveforms(
    spike_times=(emg_onsets * fs).astype(int),
    pre_samples=int(PRE_EVENT * fs),
    post_samples=int(POST_EVENT * fs),
    time_unit="samples",
    channel_selection=1,
)
# %%
# Calculate average TKEO and confidence intervals
avg_tkeo = da.mean(tkeo_epochs_aligned[0], axis=0)
plt.plot(avg_tkeo)
# %%
std_tkeo = da.std(tkeo_epochs_aligned[0], axis=0)
# %%
ci_lower = avg_tkeo - 1.96 * std_tkeo / np.sqrt(tkeo_epochs_aligned[0].shape[0])
ci_upper = avg_tkeo + 1.96 * std_tkeo / np.sqrt(tkeo_epochs_aligned[0].shape[0])

# Create time axis for epoch visualization (relative to event)
epoch_time = np.linspace(-PRE_EVENT, POST_EVENT, tkeo_epochs_aligned[0].shape[1])

# %%
# Create PSTH analysis
psth_analyzer = PSTHAnalyzer(
    bin_size_ms=10.0,
    window_size_ms=(0.5 + 1.0) * 1000,
    sigma_ms=0,  # No smoothing for raw PSTH
    baseline_window_ms=None,
)

# Compute PSTH aligned to EMG onsets
psth_result = psth_analyzer.transform(
    sorter=sorter_data,
    events=emg_onsets,
    pre_time_ms=0.5 * 1000,
    post_time_ms=1.0 * 1000,
    include_raster=True,
)


# %%
# Plot raster with PSTH and overlaid TKEO

# %%
# Plot raster with PSTH and overlaid TKEO - fixed time alignment
# %%
# Plot raster with PSTH and overlaid TKEO - fixed time alignment

# Get unit index and ID
unit_index = 27
unit_ids = psth_result["unit_ids"]
unit_id = unit_ids[unit_index]

# Calculate average TKEO and SEM
avg_tkeo = np.mean(tkeo_epochs_aligned[0], axis=0)
std_tkeo = np.std(tkeo_epochs_aligned[0], axis=0)
sem_tkeo = std_tkeo / np.sqrt(tkeo_epochs_aligned[0].shape[0])

# PRE_EVENT and POST_EVENT are in seconds, convert to samples
PRE_SAMPLES = int(PRE_EVENT * fs)
POST_SAMPLES = int(POST_EVENT * fs)

# Create proper time axis for TKEO epochs
# Convert from samples back to seconds
n_samples_tkeo = tkeo_epochs_aligned[0].shape[1]
epoch_time = np.linspace(-PRE_EVENT, POST_EVENT, n_samples_tkeo)

# Create figure with two subplots (shared x-axis)
figsize = (10, 8)
raster_height_ratio = 2.0
fig, axes = plt.subplots(
    2,
    1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"height_ratios": [raster_height_ratio, 1]},
)
raster_ax, psth_ax = axes

# Get time bins
time_bins = psth_result["time_bins"]

# Get raster data for this unit
raster_data = psth_result["raster_data"][unit_index]
spike_times = raster_data["spike_times"]
trial_indices = raster_data["trials"]

# Get number of events/trials
n_events = psth_result["event_count"]

# Plot raster
if len(spike_times) > 0:
    raster_ax.scatter(
        spike_times,
        trial_indices,
        marker="|",
        s=4,
        color="#2D3142",
        alpha=0.7,
        linewidths=1,
    )

# Set raster labels
raster_ax.set_ylabel("Trial")
raster_ax.set_title(f"Unit {unit_id} - {n_events} trials")
raster_ax.set_ylim(-0.5, n_events - 0.5)

# Get PSTH data and bin width
psth_data = psth_result["psth_rates"][:, unit_index]
bin_width = time_bins[1] - time_bins[0]

# Plot PSTH as bar plot
psth_ax.bar(
    time_bins,
    psth_data,
    width=bin_width,
    color="orange",
    alpha=0.7,
    align="center",
    label="Firing rate",
)

# Create secondary y-axis for TKEO
tkeo_ax = psth_ax.twinx()

# Plot TKEO average with SEM
tkeo_ax.plot(epoch_time, avg_tkeo[:, 0], color="blue", linewidth=2, label="TKEO")
tkeo_ax.fill_between(
    epoch_time,
    avg_tkeo[:, 0] - sem_tkeo[:, 0],
    avg_tkeo[:, 0] + sem_tkeo[:, 0],
    color="blue",
    alpha=0.2,
)

# Set PSTH labels
psth_ax.set_xlabel("Time from event onset (s)")
psth_ax.set_ylabel("Firing rate (Hz)", color="orange")
tkeo_ax.set_ylabel("TKEO (normalized)", color="blue")

# Color the y tick labels to match the data
psth_ax.tick_params(axis="y", labelcolor="orange")
tkeo_ax.tick_params(axis="y", labelcolor="blue")

# Add vertical line at event onset (time=0)
for ax in axes:
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)

# Add legends
lines1, labels1 = psth_ax.get_legend_handles_labels()
lines2, labels2 = tkeo_ax.get_legend_handles_labels()
psth_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# Improve layout
plt.tight_layout()
plt.show()

# %%

# %%
import ipywidgets as widgets
from IPython.display import clear_output, display

# Create a widget for unit selection
unit_ids = psth_result["unit_ids"]
unit_index_widget = widgets.IntSlider(
    value=27,
    min=0,
    max=len(unit_ids) - 1,
    step=1,
    description="Unit Index:",
    continuous_update=False,
)

# Create a placeholder for the output
output = widgets.Output()


def update_plot(change):
    with output:
        clear_output(wait=True)

        # Get unit index and ID
        unit_index = unit_index_widget.value
        unit_id = unit_ids[unit_index]

        # Calculate average TKEO and SEM
        avg_tkeo = np.mean(tkeo_epochs_aligned[0], axis=0)
        std_tkeo = np.std(tkeo_epochs_aligned[0], axis=0)
        sem_tkeo = std_tkeo / np.sqrt(tkeo_epochs_aligned[0].shape[0])

        # PRE_EVENT and POST_EVENT are in seconds, convert to samples
        PRE_SAMPLES = int(PRE_EVENT * fs)
        POST_SAMPLES = int(POST_EVENT * fs)

        # Create proper time axis for TKEO epochs
        # Convert from samples back to seconds
        n_samples_tkeo = tkeo_epochs_aligned[0].shape[1]
        epoch_time = np.linspace(-PRE_EVENT, POST_EVENT, n_samples_tkeo)

        # Create figure with two subplots (shared x-axis)
        figsize = (10, 8)
        raster_height_ratio = 2.0
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [raster_height_ratio, 1]},
        )
        raster_ax, psth_ax = axes

        # Get time bins
        time_bins = psth_result["time_bins"]

        # Get raster data for this unit
        raster_data = psth_result["raster_data"][unit_index]
        spike_times = raster_data["spike_times"]
        trial_indices = raster_data["trials"]

        # Get number of events/trials
        n_events = psth_result["event_count"]

        # Plot raster
        if len(spike_times) > 0:
            raster_ax.scatter(
                spike_times,
                trial_indices,
                marker="|",
                s=4,
                color="#2D3142",
                alpha=0.7,
                linewidths=1,
            )

        # Set raster labels
        raster_ax.set_ylabel("Trial")
        raster_ax.set_title(f"Unit {unit_id} - {n_events} trials")
        raster_ax.set_ylim(-0.5, n_events - 0.5)

        # Get PSTH data and bin width
        psth_data = psth_result["psth_rates"][:, unit_index]
        bin_width = time_bins[1] - time_bins[0]

        # Plot PSTH as bar plot
        psth_ax.bar(
            time_bins,
            psth_data,
            width=bin_width,
            color="orange",
            alpha=0.7,
            align="center",
            label="Firing rate",
        )

        # Create secondary y-axis for TKEO
        tkeo_ax = psth_ax.twinx()

        # Plot TKEO average with SEM
        tkeo_ax.plot(
            epoch_time, avg_tkeo[:, 0], color="blue", linewidth=2, label="TKEO"
        )
        tkeo_ax.fill_between(
            epoch_time,
            avg_tkeo[:, 0] - sem_tkeo[:, 0],
            avg_tkeo[:, 0] + sem_tkeo[:, 0],
            color="blue",
            alpha=0.2,
        )

        # Set PSTH labels
        psth_ax.set_xlabel("Time from event onset (s)")
        psth_ax.set_ylabel("Firing rate (Hz)", color="orange")
        tkeo_ax.set_ylabel("TKEO (normalized)", color="blue")

        # Color the y tick labels to match the data
        psth_ax.tick_params(axis="y", labelcolor="orange")
        tkeo_ax.tick_params(axis="y", labelcolor="blue")

        # Add vertical line at event onset (time=0)
        for ax in axes:
            ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)
            ax.grid(True, alpha=0.3)

        # Add legends
        lines1, labels1 = psth_ax.get_legend_handles_labels()
        lines2, labels2 = tkeo_ax.get_legend_handles_labels()
        psth_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # Improve layout
        plt.tight_layout()
        plt.show()


# Initial plot
update_plot(None)

# Display the widget and output
display(unit_index_widget)
display(output)

# Attach the update function to the widget
unit_index_widget.observe(update_plot, names="value")

# %%
# 26, 20, 27, 31, 36, 50
