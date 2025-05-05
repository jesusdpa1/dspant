"""
Script to showcase psth with rasters aligned
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

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import (
    PSTHAnalyzer,
    plot_psth_with_raster,
)
from dspant_viz.core.data_models import SpikeData
from dspant_viz.widgets.psth_raster_inspector import PSTHRasterInspector

sns.set_theme(style="darkgrid")
load_dotenv()

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
zscore_tkeo = zscore_processor.process(tkeo_data[0:1000000, :1]).persist()

st_tkeo_processor = create_single_threshold_detector(
    threshold_method="absolute",
    threshold_value=0.045,
    refractory_period=0.01,
    min_contraction_duration=0.01,
)
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)
# %%
emg_onsets = tkeo_epochs["onset_idx"].to_numpy() / fs  # in seconds needs to be updated
print(f"Found {len(emg_onsets)} EMG onsets")

# Create and configure the PSTH analyzer
psth_analyzer = PSTHAnalyzer(
    bin_size_ms=10.0,  # 10ms bins
    window_size_ms=1000.0,  # 1 second window
    sigma_ms=20.0,  # 20ms smoothing
    baseline_window_ms=(-500, -100),  # Baseline period
)

# Compute PSTH aligned to EMG onsets
psth_result = psth_analyzer.transform(
    sorter=sorter_data,
    events=emg_onsets,
    pre_time_ms=1000.0,  # 500ms before onset
    post_time_ms=1000.0,  # 500ms after onset
    include_raster=True,  # Include raster data
)

# Print some basic stats
print(f"Analyzed {len(psth_result['unit_ids'])} units")
print(f"PSTH shape: {psth_result['psth_rates'].shape}")

# %%
# Plot the PSTH with raster for the first unit
fig, axes = plot_psth_with_raster(
    psth_result=psth_result,
    unit_index=45,  # First unit
    show_smoothed=True,
    raster_color="navy",
    psth_color="darkorange",
    title=f"Neural response to EMG activity",
)

plt.show()

# %%

multi_unit_ = {}
for unit_id in sorter_data.unit_ids:
    multi_unit_[unit_id] = sorter_data.get_unit_spike_train(unit_id) / fs

multi_unit_data = SpikeData(spikes=multi_unit_)
# %%
pre_time = 1.0  # 1 second before event
post_time = 1.5  # 1.5 seconds after event

inspector_ = PSTHRasterInspector(
    spike_data=multi_unit_data,
    event_times=emg_onsets,  # Add events
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.01,
    backend="plotly",
    raster_color="navy",
    psth_color="crimson",
    show_sem=True,
    raster_height_ratio=2.5,
    sigma=0.02,  # Add smoothing
)

print("Interactive Plotly version (use slider to switch units):")
inspector_.display()
# %%
