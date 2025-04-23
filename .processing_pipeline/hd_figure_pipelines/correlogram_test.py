"""
Script to showcase correlogram plots
Author: Jesus Penaloza
Needs to be fixed
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
from dspant_neuroproc.processors.spike_analytics.correlogram import (
    create_spike_correlogram_analyzer,
)

# Import the density estimation and plotting from dspant_neuroproc
from dspant_neuroproc.processors.spike_analytics.density_estimation import (
    SpikeDensityEstimator,
)
from dspant_neuroproc.visualization.correlogram_plots import plot_crosscorrelogram
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
# Correlogram Analysis Test Script

# Create correlogram analyzer
correlogram_analyzer = create_spike_correlogram_analyzer(
    bin_size_ms=1.0,  # 1 ms bin size
    window_size_ms=100.0,  # 100 ms window
)

# Compute correlograms
correlogram_results = correlogram_analyzer.transform(
    sorter_data,  # Your loaded sorter data
    start_time_s=0.0,  # Optional: start time
    end_time_s=None,  # Optional: end time (None means entire recording)
    unit_ids=None,  # Optional: specific units (None means all units)
)
# %%
# Visualize Autocorrelograms
plt.figure(figsize=(15, 10))
for i, autocorr in enumerate(correlogram_results["autocorrelograms"]):
    plt.subplot(2, 3, i + 1)  # Adjust grid as needed
    plot_crosscorrelogram(
        {
            "unit_id": autocorr["unit_id"],
            "crosscorrelogram": autocorr["autocorrelogram"],
            "time_bins": autocorr["time_bins"],
        },
        title=f"Autocorrelogram Unit {autocorr['unit_id']}",
    )
plt.tight_layout()
plt.show()
# %%
# Visualize Crosscorrelograms
plt.figure(figsize=(15, 10))
for i, crosscorr in enumerate(correlogram_results["crosscorrelograms"]):
    plt.subplot(2, 3, i + 1)  # Adjust grid as needed
    plot_crosscorrelogram(
        {
            "unit1": crosscorr["unit1"],
            "unit2": crosscorr["unit2"],
            "crosscorrelogram": crosscorr["crosscorrelogram"],
            "time_bins": crosscorr["time_bins"],
        },
        title=f"Crosscorrelogram Units {crosscorr['unit1']}-{crosscorr['unit2']}",
    )
plt.tight_layout()
plt.show()

# %%
