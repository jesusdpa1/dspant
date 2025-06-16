"""
spikes overlay
Author: jpenalozaa
Description: Creates an upper triangular grid of cross-correlograms for spike units
"""

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
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode
from dspant.pattern.detection.peak import create_threshold_detector
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()
# %%
# r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/00_baseline/drv_00_baseline/HDEG.ant"

# Data loading configuration
data_dir = Path(os.getenv("DATA_DIR"))
hd_path = data_dir.joinpath(
    r"testSubject/24-09-06_5042-2_testSubject_DST-and-contusion/drv/drv_00_baseline/EMGM.ant"
)

# sorter_path = data_dir.joinpath(
#     r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
# sorter_data = load_kilosort(sorter_path, load_templates=True)
# # )

# %%
stream_hd = StreamNode(str(hd_path))
stream_hd.load_metadata()
stream_hd.load_data()
stream_hd.summarize()
# %%
# Get sampling rate
FS = stream_hd.fs

# %%
notch_filter = create_notch_filter(60, q=10, fs=FS)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=4)
# Create pre-processing functions
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)

bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)

cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()
# %%
# Create and apply filters for HD data
processor_hd = create_processing_node(stream_hd)

# Add processors
processor_hd.add_processor(
    [notch_processor, bandpass_processor_hd],
    group="filters",
)

# %%
# Slice recordings 5min
START_TIME = int(0.0 * FS)  # Start time in seconds
END_TIME = int(5.0 * 60 * FS)  # End time in seconds
# Create time range for processing
time_range = slice(START_TIME, END_TIME)
print(time_range)
# %%
filtered_hd = processor_hd.process(group=["filters"]).persist()[START_TIME:END_TIME, :]

# %%
cmr_hd_small = cmr_processor.process(
    filtered_hd,
    fs=FS,
).persist()

# %%
whiten_hd_small = whiten_processor.process(cmr_hd_small, fs=FS).persist()

# %%
START_ = int(10.25 * FS)
END_ = int((10.25 + 3) * FS)

a = plot_multi_channel_data(
    data=whiten_hd_small,
    fs=FS,
    time_window=(1, 2),
    figsize=(15, 20),
    color_mode="single",
)

# %%
# HD Multi-Channel Recording Visualization - Compact Layout
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np

TRANSPARENT_BACKGROUND = True

# Set publication style
mpu.set_publication_style()

# Configuration
# small contacts = [0, 2, 5, 9, 15, 23, 27, 30]
CHANNELS = [0, 1, 2, 4, 6, 18, 22, 25]  # 8 channels to plot
RECORDING_START_TIME = 10.05  # Start time in seconds
WINDOW_DURATION = 0.35  # Window duration in seconds
ROW_SPACING = 35.0  # Vertical spacing between channels
FIGURE_SIZE = (15, 12)
AMPLIFICATION = 2.0  # Trace amplitude scaling factor

# All traces in dark navy
DARK_NAVY = mpu.PRIMARY_COLOR  # Use mpu's primary color (dark navy)

# Font sizes using mpu scaling
FONT_SIZE = 35
TITLE_SIZE = int(FONT_SIZE * 1.0)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.8)
TICK_SIZE = int(FONT_SIZE * 0.8)
LABEL_SIZE = int(FONT_SIZE * 0.8)

# Extract HD data for the specified time window
start_sample = int(RECORDING_START_TIME * FS)
end_sample = int((RECORDING_START_TIME + WINDOW_DURATION) * FS)

# Get HD data for selected channels
hd_data = whiten_hd_small[start_sample:end_sample, CHANNELS].compute()
hd_time = np.arange(hd_data.shape[0]) / FS

# Normalize each channel
hd_normalized = np.zeros_like(hd_data, dtype=float)
for i in range(len(CHANNELS)):
    channel_data = hd_data[:, i]
    # Z-score normalization
    channel_mean = np.mean(channel_data)
    channel_std = np.std(channel_data)
    if channel_std > 0:
        hd_normalized[:, i] = (channel_data - channel_mean) / channel_std
    else:
        hd_normalized[:, i] = channel_data - channel_mean

# Create plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

total_channels = len(CHANNELS)

# Plot each HD channel
for ch_idx, channel in enumerate(CHANNELS):
    # Calculate y position (top to bottom)
    y_position = (total_channels - 1 - ch_idx) * ROW_SPACING

    # Plot channel data in dark navy with amplification
    ax.plot(
        hd_time,
        (hd_normalized[:, ch_idx] * AMPLIFICATION) + y_position,
        color=DARK_NAVY,
        linewidth=1.5,
        alpha=0.8,
    )

    # Add channel label
    ax.text(
        -0.02,
        y_position,
        f"Ch {ch_idx + 1}",
        ha="right",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        color=DARK_NAVY,
        transform=ax.get_yaxis_transform(),
    )

# Set compact axis limits - minimal margins
y_bottom = -ROW_SPACING * 0.8  # Small margin below last channel
y_top = (
    total_channels - 1
) * ROW_SPACING + ROW_SPACING * 0.5  # Minimal margin above first channel

# Format axis
ax.set_xlim(0, WINDOW_DURATION)
ax.set_ylim(y_bottom, y_top)
ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE)
ax.tick_params(labelsize=TICK_SIZE)

# Remove y-axis ticks and spines
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Apply styling conditionally
if not TRANSPARENT_BACKGROUND:
    mpu.finalize_figure(fig, title=None)

plt.tight_layout()
mpu.set_publication_style(use_seaborn=True)

# # Re-apply transparency if needed
# if TRANSPARENT_BACKGROUND:
#     fig.patch.set_alpha(0.0)
#     ax.patch.set_alpha(0.0)

plt.show()

print(f"\nDisplayed: {len(CHANNELS)} HD channels")
print(f"Channels: {CHANNELS}")
print(
    f"Time window: {RECORDING_START_TIME}s - {RECORDING_START_TIME + WINDOW_DURATION}s"
)
print(f"Data source: whiten_hd_small (filtered + CMR + whitened)")
print(f"Trace amplification: {AMPLIFICATION}")
print(f"Compact layout with minimal margins")

# %%

