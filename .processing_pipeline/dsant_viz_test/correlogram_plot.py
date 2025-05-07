"""
Interactive Correlogram Visualization for Neural Data
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv

# Import dspant components
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort

# Import dspant_viz components
from dspant_viz.core.data_models import SpikeData
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.widgets.correlogram_inspector import CorrelogramInspector

# Set style for visualizations
sns.set_theme(style="darkgrid")
apply_matplotlib_theme("ggplot")  # For matplotlib
apply_plotly_theme("seaborn")  # For plotly
load_dotenv()

# %%
# Analysis Parameters
BIN_SIZE_MS = 2.0  # Bin size in milliseconds
WINDOW_SIZE_MS = 100.0  # Window size for cross-correlation in ms (±50 ms)
MAX_UNITS = 20  # Maximum number of units to include in the analysis
SELECTED_UNITS = None  # Specific unit IDs to include, if None, use first MAX_UNITS

# %%
# Load spike sorting data
data_dir = Path(os.getenv("DATA_DIR"))
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)

# Load spike sorting data
sorter_data = load_kilosort(sorter_path)
fs = sorter_data.sampling_frequency

# Get unit IDs - limit to MAX_UNITS for better performance
if SELECTED_UNITS is not None:
    unit_ids = SELECTED_UNITS
else:
    unit_ids = sorter_data.unit_ids[:MAX_UNITS]

print(
    f"Data loaded successfully. Using {len(unit_ids)} units out of {len(sorter_data.unit_ids)} total."
)
print(f"Sampling rate: {fs} Hz")

# %%
# Convert to SpikeData format for dspant_viz
spike_dict = {}

for unit_id in unit_ids:
    # Get spike times for this unit in seconds
    unit_spikes = sorter_data.get_unit_spike_train(unit_id=unit_id) / fs

    # Check for empty spike trains
    if len(unit_spikes) == 0:
        print(f"Warning: Unit {unit_id} has no spikes!")
        continue

    spike_dict[unit_id] = unit_spikes
    print(f"Unit {unit_id}: {len(unit_spikes)} spikes")

# Create SpikeData object
neural_data = SpikeData(spikes=spike_dict)

print(
    f"Converted data to SpikeData format with {len(neural_data.get_unit_ids())} units."
)

# %%
# Create interactive correlogram inspector
# Convert ms to seconds for dspant_viz parameters
bin_width_s = BIN_SIZE_MS / 1000.0
window_size_s = WINDOW_SIZE_MS / 1000.0 / 2.0  # Half of total window

# Create inspector with Plotly backend (more interactive)
inspector_plotly = CorrelogramInspector(
    spike_data=neural_data,
    backend="plotly",
    bin_width=bin_width_s,
    window_size=window_size_s,
    normalize=True,
)

# %%
inspector_plotly.display()

# %%
