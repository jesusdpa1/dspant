"""
Create visualization showing neural activity aligned to EMG contractions [RUST implementaiton]
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
print(tkeo_data.shape)
# %%
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
# TESTING RUST IMPLEMENTATION
# Create PSTH analysis
start_time = time.time()
psth_analyzer = PSTHAnalyzer(
    bin_size_ms=10.0,
    window_size_ms=(PRE_EVENT + POST_EVENT) * 1000,
    sigma_ms=25.0,  # Use smoothing to test Rust implementation
    baseline_window_ms=(-400, -200),
)

# Compute PSTH aligned to EMG onsets
print("Computing PSTH with Rust implementation...")
psth_result = psth_analyzer.transform(
    sorter=sorter_data,
    events=emg_onsets.astype(np.float32),  # Ensure float32 for Rust
    pre_time_ms=PRE_EVENT * 1000,
    post_time_ms=POST_EVENT * 1000,
)
end_time = time.time()
print(f"PSTH computation completed in {end_time - start_time:.4f} seconds")

# %%
# Plot PSTH results for a specific unit
unit_index = 37  # Change to visualize different units
unit_id = psth_result["unit_ids"][unit_index]

# Get PSTH data (preferring smoothed if available)
if "psth_smoothed" in psth_result:
    psth_data = psth_result["psth_smoothed"][:, unit_index]
    print("Using smoothed PSTH data")
else:
    psth_data = psth_result["psth_rates"][:, unit_index]
    print("Using raw PSTH data")

# Get time bins
time_bins = psth_result["time_bins"]

# Plot PSTH
plt.figure(figsize=(10, 6))
plt.bar(time_bins, psth_data, width=psth_analyzer.bin_size_ms / 1000, alpha=0.6)

# Add EMG onset line
plt.axvline(x=0, color="r", linestyle="--", alpha=0.8)

plt.title(f"Neural Activity (Unit {unit_id}) Around EMG Contractions")
plt.xlabel("Time relative to EMG contraction (s)")
plt.ylabel("Firing Rate (Hz)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Plot raster data for the same unit
raster_data = psth_result["raster_data"][unit_index]

plt.figure(figsize=(10, 10))
plt.scatter(
    raster_data["spike_times"],
    raster_data["trials"],
    marker="|",
    s=20,
    color="black",
    alpha=0.7,
)
plt.axvline(x=0, color="r", linestyle="--", alpha=0.8)
plt.title(f"Spike Raster for Unit {unit_id}")
plt.xlabel("Time relative to EMG contraction (s)")
plt.ylabel("Trial number")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
