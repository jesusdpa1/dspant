# %%

import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from scipy import stats

from dspant.engine import create_processing_node
from dspant.io.loaders.parquet_reader import ParquetReader
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.io.loaders.zarr_loader import ZarrReader
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.extractors.template_extractor import TemplateExtractor
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()
# %%
# Data loading configuration
BASE_DIR = Path(os.getenv("BASE_DIR"))
DATA_DIR = BASE_DIR.joinpath(
    r"hd_paper/small_contacts/25-03-26_4902-1_testSubject_topoMapping"
)

ZARR_PATH = DATA_DIR.joinpath(r"drv_zarr/drv_sliced_5min_00_baseline_emg.zarr")

SORTER_PATH = DATA_DIR.joinpath(
    r"drv_ks4/drv_sliced_5min_00_baseline/ks4_output_2025-06-17_13-43/sorter_output"
)

# %%
# cmr_processor = create_cmr_processor_rs()
# whiten_processor = create_whitening_processor_rs()
# cmr_data = cmr_processor.process(bandpassed_data, FS).persist()
# whitened_data = whiten_processor.process(cmr_data, FS).persist()
# Load data
# HD DATA
stream_emg = ZarrReader(str(ZARR_PATH))
stream_emg.load_metadata()
stream_emg.load_data()
FS = stream_emg.sampling_frequency

# SORTER DATA
sorter_data = load_kilosort(SORTER_PATH)

# %%
# PREPROCESSING
notch_filter = create_notch_filter(60, q=10, fs=FS)
bandpass_filter_hd = create_bandpass_filter(20, 2000, fs=FS, order=4)

# Create pre-processing functions
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)


# %%
notch_data = notch_processor.process(stream_emg.data, FS)
filtered_emg = bandpass_processor_hd.process(notch_data, FS)

# %%
a = plot_multi_channel_data(
    filtered_emg,
    fs=FS,
    channels=range(2),
    time_window=(0, 10),
    figsize=(15, 20),
)

# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=15)
tkeo_data = tkeo_processor.process(filtered_emg[0:1000000], fs=FS).persist()

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
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=FS).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)

# Convert onsets to seconds
emg_onsets = tkeo_epochs["onset_idx"].to_numpy() / FS

# %%
# Extract EMG epochs around neural activity
# Define time window for analysis
PRE_EVENT = 0.5  # seconds before event
POST_EVENT = 1.0  # seconds after event

# Initialize EpochExtractor for TKEO data
epoch_extractor = WaveformExtractor(data=zscore_tkeo[:, :], fs=FS)

# Extract TKEO epochs aligned to EMG onsets
tkeo_epochs_aligned = epoch_extractor.extract_waveforms(
    spike_times=(emg_onsets * FS).astype(int),
    pre_samples=int(PRE_EVENT * FS),
    post_samples=int(POST_EVENT * FS),
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

# Plot raster with PSTH and overlaid TKEO - fixed time alignment
# 49 48 38 10 18 29 14 19  #
# %%
"""
Neural Activity Visualization with MPU Styling
Updated to use mp_plotting_utils for publication-quality aesthetics
"""

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np

# Set publication style
mpu.set_publication_style()

# Define font sizes following the EMG plot pattern
TITLE_SIZE = 24 * 2  # 48pt (Main figure title)
SUBTITLE_SIZE = 20 * 2  # 40pt (Subplot titles)
AXIS_LABEL_SIZE = 18 * 2  # 36pt (Axis labels)
TICK_SIZE = 16 * 2  # 32pt (Tick labels)
LEGEND_SIZE = 18 * 2  # 36pt (Legend text)

# Define colors following the established palette
NAVY_BLUE = "#2D3142"  # Dark navy for raster spikes
ORANGE_PSTH = "#DE8F05"  # Orange for PSTH bars
BLUE_TKEO = "#1066DE"  # Blue for TKEO line
EVENT_LINE_COLOR = "red"  # Red for event onset line

# Line and marker styles
RASTER_MARKER_SIZE = 8  # Increased marker size for visibility
TKEO_LINEWIDTH = 4  # Thick line for TKEO
PSTH_ALPHA = 0.8  # PSTH bar transparency
TKEO_FILL_ALPHA = 0.2  # TKEO confidence interval transparency

# Get unit index and ID
unit_index = 49
unit_ids = psth_result["unit_ids"]
unit_id = unit_ids[unit_index]

# Calculate average TKEO and SEM
avg_tkeo = np.mean(tkeo_epochs_aligned[0], axis=0)
std_tkeo = np.std(tkeo_epochs_aligned[0], axis=0)
sem_tkeo = std_tkeo / np.sqrt(tkeo_epochs_aligned[0].shape[0])

# Create proper time axis for TKEO epochs
n_samples_tkeo = tkeo_epochs_aligned[0].shape[1]
epoch_time = np.linspace(-PRE_EVENT, POST_EVENT, n_samples_tkeo)

# Create figure with publication-quality sizing
figsize = (20, 16)  # Increased size for better visibility
raster_height_ratio = 2.0
fig, axes = plt.subplots(
    2,
    1,
    figsize=figsize,
    sharex=True,
    gridspec_kw={"height_ratios": [raster_height_ratio, 1]},
)
raster_ax, psth_ax = axes

# Get time bins and raster data
time_bins = psth_result["time_bins"]
raster_data = psth_result["raster_data"][unit_index]
spike_times = raster_data["spike_times"]
trial_indices = raster_data["trials"]
n_events = psth_result["event_count"]

# ===== RASTER PLOT =====
if len(spike_times) > 0:
    # Vectorized approach using vlines - much faster than loop
    raster_ax.vlines(
        x=spike_times,
        ymin=trial_indices - 0.4,
        ymax=trial_indices + 0.4,
        colors=NAVY_BLUE,
        linewidth=2,
        alpha=0.8,
    )

# Set raster y-limits
raster_ax.set_ylim(-0.5, n_events - 0.5)

# Format raster axis using mpu
mpu.format_axis(
    raster_ax,
    title=f"Unit {unit_id} - {n_events} trials",
    ylabel="Trial",
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# ===== PSTH PLOT =====
# Get PSTH data and bin width
psth_data = psth_result["psth_rates"][:, unit_index]
bin_width = time_bins[1] - time_bins[0]

# Plot PSTH as bar plot
psth_ax.bar(
    time_bins,
    psth_data,
    width=bin_width,
    color=ORANGE_PSTH,
    alpha=PSTH_ALPHA,
    align="center",
    label="Firing Rate",
)

# Create secondary y-axis for TKEO
tkeo_ax = psth_ax.twinx()

# Plot TKEO average with SEM
tkeo_ax.plot(
    epoch_time, avg_tkeo[:, 0], color=BLUE_TKEO, linewidth=TKEO_LINEWIDTH, label="TKEO"
)
tkeo_ax.fill_between(
    epoch_time,
    avg_tkeo[:, 0] - sem_tkeo[:, 0],
    avg_tkeo[:, 0] + sem_tkeo[:, 0],
    color=BLUE_TKEO,
    alpha=TKEO_FILL_ALPHA,
)

# Format PSTH axis using mpu
mpu.format_axis(
    psth_ax,
    title="Neural Activity Aligned to EMG Onset",
    xlabel="Time from Event Onset (s)",
    ylabel="Firing Rate (Hz)",
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Format TKEO axis labels with custom font sizes
# tkeo_ax.set_ylabel(
#     "TKEO (normalized)",
#     fontsize=AXIS_LABEL_SIZE,
#     color=BLUE_TKEO,
# )

tkeo_ax.tick_params(
    axis="y",
    labelcolor=BLUE_TKEO,
    labelsize=TICK_SIZE,
)

# Color the PSTH y tick labels to match the data
psth_ax.tick_params(
    axis="y",
    labelcolor=ORANGE_PSTH,
    labelsize=TICK_SIZE,
)


# raster_ax.yaxis.set_visible(True)
# psth_ax.yaxis.set_visible(True)
# tkeo_ax.yaxis.set_visible(False)

raster_ax.yaxis.set_visible(False)
psth_ax.yaxis.set_visible(False)
tkeo_ax.yaxis.set_visible(False)

# Add event onset line to both subplots
for ax in axes:
    ax.axvline(x=0, color=EVENT_LINE_COLOR, linestyle="--", linewidth=3, alpha=0.8)
    ax.grid(True, alpha=0.3)

# Apply tight layout
plt.tight_layout()

# Save figure with high DPI for publication
# Uncomment and modify path as needed:
# FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
# FIGURE_PATH = FIGURE_DIR.joinpath("neural_activity_emg_correlation.png")
# mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

# %%
