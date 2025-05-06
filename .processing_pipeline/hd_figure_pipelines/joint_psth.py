"""
Create joint PSTH visualization showing correlation between neural units aligned to EMG onsets
Author: Jesus Penaloza
"""

# %%
import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.gridspec as gridspec
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

sns.set_theme(style="whitegrid")
load_dotenv()

# %%
# Analysis Parameters
PRE_EVENT = 0.2  # seconds before event
POST_EVENT = 0.2  # seconds after event
BIN_SIZE_MS = 5.0  # Milliseconds per bin
HIST_BINS = 40  # Number of bins for 2D histogram

# %%
# Load data
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
# Extract EMG epochs around onsets
epoch_extractor = WaveformExtractor(data=zscore_tkeo[:, :].compute(), fs=fs)

# Extract TKEO epochs aligned to EMG onsets
tkeo_epochs_aligned = epoch_extractor.extract_waveforms(
    spike_times=(emg_onsets * fs).astype(int),
    pre_samples=int(PRE_EVENT * fs),
    post_samples=int(POST_EVENT * fs),
    time_unit="samples",
    channel_selection=1,
)

# Create time axis for epoch visualization
epoch_time = np.linspace(-PRE_EVENT, POST_EVENT, tkeo_epochs_aligned[0].shape[1])

# %%
# Create PSTH analysis
psth_analyzer = PSTHAnalyzer(
    bin_size_ms=BIN_SIZE_MS,
    window_size_ms=(PRE_EVENT + POST_EVENT) * 1000,
    sigma_ms=0,  # No smoothing for raw PSTH
    baseline_window_ms=None,
)

# Compute PSTH aligned to EMG onsets for all units
psth_result = psth_analyzer.transform(
    sorter=sorter_data,
    events=emg_onsets,
    pre_time_ms=PRE_EVENT * 1000,
    post_time_ms=POST_EVENT * 1000,
    include_raster=True,
)


# %%
# Function to create joint PSTH plot
def create_joint_psth(unit1_idx, unit2_idx, psth_result, fig_size=(10, 10)):
    """Create joint PSTH visualization for two units"""
    # Get unit IDs
    unit_ids = psth_result["unit_ids"]
    unit1_id = unit_ids[unit1_idx]
    unit2_id = unit_ids[unit2_idx]

    # Get time bins and number of events/trials
    time_bins = psth_result["time_bins"]
    n_events = psth_result["event_count"]
    bin_width = time_bins[1] - time_bins[0]

    # Create figure with GridSpec for layout control
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(4, 4, figure=fig)

    # Create the three main axes
    ax_unit1 = fig.add_subplot(gs[0, 0:3])  # Top histogram (Unit 1)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])  # Main 2D histogram
    ax_unit2 = fig.add_subplot(gs[1:4, 3])  # Right histogram (Unit 2)

    # Create the colorbar axis
    ax_cbar = fig.add_subplot(gs[0, 3])

    # Get unit data
    psth_unit1 = psth_result["psth_rates"][:, unit1_idx]
    psth_unit2 = psth_result["psth_rates"][:, unit2_idx]

    # Get raster data for both units
    raster_unit1 = psth_result["raster_data"][unit1_idx]
    raster_unit2 = psth_result["raster_data"][unit2_idx]

    # Create binned data for 2D histogram
    bin_edges_x = np.linspace(-PRE_EVENT, POST_EVENT, HIST_BINS + 1)
    bin_edges_y = np.linspace(-PRE_EVENT, POST_EVENT, HIST_BINS + 1)

    # Create the joint histogram from the binned PSTH data directly
    # This creates a 2D visualization of firing rates
    joint_hist = np.zeros((HIST_BINS, HIST_BINS))

    # Create a count of spikes in each bin for each unit
    unit1_counts = np.zeros(HIST_BINS)
    unit2_counts = np.zeros(HIST_BINS)

    # Populate the spike counts for each bin
    for trial_idx in range(n_events):
        # Get spikes for this trial from both units
        unit1_trial_spikes = raster_unit1["spike_times"][
            raster_unit1["trials"] == trial_idx
        ]
        unit2_trial_spikes = raster_unit2["spike_times"][
            raster_unit2["trials"] == trial_idx
        ]

        # Bin the spikes
        if len(unit1_trial_spikes) > 0:
            hist1, _ = np.histogram(unit1_trial_spikes, bins=bin_edges_x)
            unit1_counts += hist1

        if len(unit2_trial_spikes) > 0:
            hist2, _ = np.histogram(unit2_trial_spikes, bins=bin_edges_y)
            unit2_counts += hist2

    # Create the joint histogram using outer product of spike counts
    # This shows where both units tend to fire relative to event onset
    for i in range(HIST_BINS):
        for j in range(HIST_BINS):
            # Simple multiplication of bin counts creates joint distribution
            joint_hist[i, j] = unit1_counts[i] * unit2_counts[j]

    # Plot the unit 1 PSTH
    ax_unit1.bar(
        time_bins,
        psth_unit1,
        width=bin_width,
        color="blue",
        alpha=0.7,
        label=f"Unit {unit1_id}",
    )
    ax_unit1.axvline(x=0, color="red", linestyle="--", alpha=0.8)
    ax_unit1.set_title(f"Unit {unit1_id}", fontsize=12)
    ax_unit1.set_ylabel("Firing Rate (Hz)")
    ax_unit1.set_xticklabels([])

    # Plot the unit 2 PSTH
    ax_unit2.barh(
        time_bins,
        psth_unit2,
        height=bin_width,
        color="green",
        alpha=0.7,
        label=f"Unit {unit2_id}",
    )
    ax_unit2.axhline(y=0, color="red", linestyle="--", alpha=0.8)
    ax_unit2.set_title(f"Unit {unit2_id}", fontsize=12)
    ax_unit2.set_xlabel("Firing Rate (Hz)")
    ax_unit2.set_yticklabels([])

    # Create colormap for the joint histogram
    cmap = plt.cm.jet
    cmap.set_under("white")  # Set color for values below vmin

    # Plot the joint histogram
    im = ax_joint.imshow(
        joint_hist.T,  # Transpose to match coordinate system
        extent=[-PRE_EVENT, POST_EVENT, -PRE_EVENT, POST_EVENT],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.1,  # Set minimum value to plot
        interpolation="gaussian",
    )

    # Add lines at t=0
    ax_joint.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax_joint.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Set labels
    ax_joint.set_xlabel(f"Time from EMG onset - Unit {unit1_id} (s)")
    ax_joint.set_ylabel(f"Time from EMG onset - Unit {unit2_id} (s)")

    # Add colorbar
    plt.colorbar(im, cax=ax_cbar, label="Joint Activity")

    # Add figure title
    fig.suptitle(
        f"Joint PSTH: Units {unit1_id} and {unit2_id} aligned to EMG onset", fontsize=14
    )

    plt.tight_layout()
    return fig


# %%
# Create joint PSTH for specified units
unit_id_1 = 10
unit_id_2 = 31

# Convert unit IDs to indices
unit_ids = psth_result["unit_ids"]
unit1_idx = unit_ids.index(unit_id_1) if unit_id_1 in unit_ids else 0
unit2_idx = unit_ids.index(unit_id_2) if unit_id_2 in unit_ids else 1

# Create and display the joint PSTH
fig = create_joint_psth(unit1_idx, unit2_idx, psth_result)
plt.show()

# %%
# Optional: create a widget for interactive unit selection
import ipywidgets as widgets
from IPython.display import clear_output, display

# Get unit IDs for dropdowns
unit_id_options = psth_result["unit_ids"]

# Create widgets
unit1_dropdown = widgets.Dropdown(
    options=unit_id_options,
    value=unit_id_options[unit1_idx],
    description="Unit 1:",
    style={"description_width": "initial"},
)

unit2_dropdown = widgets.Dropdown(
    options=unit_id_options,
    value=unit_id_options[unit2_idx],
    description="Unit 2:",
    style={"description_width": "initial"},
)

# Create output widget
output = widgets.Output()


# Define update function
def update_plot(change):
    with output:
        clear_output(wait=True)

        # Get selected unit IDs
        unit1_id = unit1_dropdown.value
        unit2_id = unit2_dropdown.value

        # Convert to indices
        unit1_idx = unit_ids.index(unit1_id)
        unit2_idx = unit_ids.index(unit2_id)

        # Create and display the joint PSTH
        fig = create_joint_psth(unit1_idx, unit2_idx, psth_result)
        plt.show()

        # Calculate cross-correlation metrics
        unit1_psth = psth_result["psth_rates"][:, unit1_idx]
        unit2_psth = psth_result["psth_rates"][:, unit2_idx]

        # Simple correlation coefficient
        corr = np.corrcoef(unit1_psth, unit2_psth)[0, 1]
        print(f"PSTH Correlation: {corr:.3f}")


# Register callbacks
unit1_dropdown.observe(update_plot, names="value")
unit2_dropdown.observe(update_plot, names="value")

# Create the UI
ui = widgets.HBox([unit1_dropdown, unit2_dropdown])
display(ui)
display(output)

# Initial plot
update_plot(None)

# %%
