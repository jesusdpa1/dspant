"""
EMG Contractile and Breathing Metrics Visualization with Y-Spread Channels
Author: Jesus Penaloza (Updated with TKEO envelope and metrics analysis)
"""

# %%
import os
import time
from pathlib import Path

import dotenv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)

sns.set_theme(style="darkgrid")

# %%
# Data loading (keep your existing data loading code)
data_path = Path(os.getenv("DATA_DIR"))
base_path = data_path.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
emg_stream_path = base_path.joinpath(r"RawG.ant")

# %%
# Load EMG data
stream_emg = StreamNode(str(emg_stream_path))
stream_emg.load_metadata()
stream_emg.load_data()
fs = stream_emg.fs

# %%
# Create and apply filters
processor_emg = create_processing_node(stream_emg)

# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(20, fs)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
lowpass_processor = FilterProcessor(
    filter_func=lowpass_filter.get_filter_function(), overlap_samples=40
)

# Add processors
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters
filter_data = processor_emg.process(group=["filters"]).persist()

# %%
# TKEO Processing
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=15)
tkeo_envelope = tkeo_processor.process(filter_data, fs=fs).persist()

# %%
# Create visualization


# Define adjustable parameters for the visualization
class PlotParameters:
    # Data selection parameters
    data_start = int(fs * 5)  # Start time in seconds
    data_duration = int(fs * 8)  # Duration in seconds
    zoom_start = int(fs * 6)  # Zoomed section start
    zoom_duration = int(fs * 1)  # 1 second zoom

    # Ttot zoom parameters
    zoom_start_ttot = int(fs * 6.8)  # Ttot zoomed section start
    zoom_duration_ttot = int(fs * 1.5)  # 4 seconds zoom for Ttot

    # Contractile metric parameters (in samples from zoom_start)
    onset_position = int(fs * 0.32)  # Position of onset marker
    peak_position = int(fs * 0.51)  # Position of peak marker
    offset_position = int(fs * 0.6)  # Position of offset marker

    # Breathing metric parameters
    ti_start = int(fs * 0.32)  # Ti start (onset)
    ti_end = int(fs * 0.51)  # Ti end (peak)
    te_start = int(fs * 0.51)  # Te start (peak)
    te_end = int(fs * 0.6)  # Te end (next onset)
    ttot_start = int(fs * 0.24)  # Ttot start (first onset)
    ttot_end = int(fs * 0.9)  # Ttot end (second onset)

    # Channel to visualize
    channel = 1

    # Y-spread parameters
    y_spread = 5.0  # Vertical spacing between channels
    emg_offset = 0.0  # Baseline for EMG
    tkeo_offset = 1.0  # Baseline for TKEO (above EMG)


# Extract data segments
params = PlotParameters()
data_segment = filter_data[
    params.data_start : params.data_start + params.data_duration, :
]
tkeo_segment = tkeo_envelope[
    params.data_start : params.data_start + params.data_duration, :
]

# Create time arrays
time_full = np.arange(len(data_segment)) / fs
time_zoom = np.arange(params.zoom_duration) / fs
time_zoom_ttot = np.arange(params.zoom_duration_ttot) / fs

# Extract zoomed sections
zoom_data = data_segment[
    params.zoom_start - params.data_start : params.zoom_start
    - params.data_start
    + params.zoom_duration,
    :,
]
zoom_tkeo = tkeo_segment[
    params.zoom_start - params.data_start : params.zoom_start
    - params.data_start
    + params.zoom_duration,
    :,
]

# Extract zoomed sections for Ttot
zoom_data_ttot = data_segment[
    params.zoom_start_ttot - params.data_start : params.zoom_start_ttot
    - params.data_start
    + params.zoom_duration_ttot,
    :,
]
zoom_tkeo_ttot = tkeo_segment[
    params.zoom_start_ttot - params.data_start : params.zoom_start_ttot
    - params.data_start
    + params.zoom_duration_ttot,
    :,
]


# Normalize TKEO to EMG scale for y-spread visualization
def normalize_tkeo_for_spread(emg_data, tkeo_data, emg_max_amplitude=None):
    """Normalize TKEO data to match EMG max amplitude for y-spread visualization."""
    if emg_max_amplitude is None:
        emg_max_amplitude = np.max(np.abs(emg_data))

    # Normalize TKEO to have same scale as max EMG amplitude
    tkeo_max = np.max(np.abs(tkeo_data))
    if tkeo_max > 0:
        normalized_tkeo = tkeo_data / tkeo_max * emg_max_amplitude
    else:
        normalized_tkeo = tkeo_data

    return normalized_tkeo


# Set up visualization style
TITLE_SIZE = 20
SUBTITLE_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_SIZE = 14
ENVELOPE_LINEWIDTH = 4  # Thicker line for envelope

# Get colorblind palette for breathing metrics
colorblind_colors = sns.color_palette("colorblind")
NAVY_BLUE = "#2D3142"  # Dark navy blue for time series
ORANGE_ENVELOPE = colorblind_colors[1]  # Orange color for envelope

# Create figure with 2x3 grid
fig = plt.figure(figsize=(36, 24))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

# Get max amplitude for normalization
emg_max_amp = np.max(np.abs(data_segment[:, params.channel]))

# Normalize TKEO data for y-spread
tkeo_segment_normalized = normalize_tkeo_for_spread(
    data_segment[:, params.channel], tkeo_segment[:, params.channel], emg_max_amp
)
zoom_tkeo_normalized = normalize_tkeo_for_spread(
    zoom_data[:, params.channel], zoom_tkeo[:, params.channel], emg_max_amp
)
zoom_tkeo_ttot_normalized = normalize_tkeo_for_spread(
    zoom_data_ttot[:, params.channel], zoom_tkeo_ttot[:, params.channel], emg_max_amp
)

# ===== Row 1: Full EMG with TKEO in y-spread (3x1) =====
ax1 = fig.add_subplot(gs[0, :])

# Plot EMG with y offset
emg_plot_data = data_segment[:, params.channel] / emg_max_amp * 1.5 + params.emg_offset
ax1.plot(
    time_full,
    emg_plot_data,
    color=NAVY_BLUE,
    linewidth=2,
    label="EMG Signal",
)

# Plot TKEO envelope with y offset (placed above EMG)
tkeo_plot_data = tkeo_segment_normalized / emg_max_amp * 1.5 + params.tkeo_offset
ax1.plot(
    time_full,
    tkeo_plot_data,
    color=ORANGE_ENVELOPE,
    linewidth=ENVELOPE_LINEWIDTH,
    alpha=0.9,
    label="TKEO Envelope",
)

# Set y-ticks with channel labels
y_ticks = [params.emg_offset, params.tkeo_offset]
y_labels = ["EMG", "TKEO"]
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels)

# Highlight zoomed sections
# First zoom section
zoom_start_time = (params.zoom_start - params.data_start) / fs
zoom_end_time = zoom_start_time + params.zoom_duration / fs
highlight1 = patches.Rectangle(
    (zoom_start_time, ax1.get_ylim()[0]),
    params.zoom_duration / fs,
    ax1.get_ylim()[1] - ax1.get_ylim()[0],
    color="yellow",
    alpha=0.3,
    linewidth=0,
)
ax1.add_patch(highlight1)

# Second zoom section (for Ttot)
zoom_start_time_ttot = (params.zoom_start_ttot - params.data_start) / fs
zoom_end_time_ttot = zoom_start_time_ttot + params.zoom_duration_ttot / fs
highlight2 = patches.Rectangle(
    (zoom_start_time_ttot, ax1.get_ylim()[0]),
    params.zoom_duration_ttot / fs,
    ax1.get_ylim()[1] - ax1.get_ylim()[0],
    color="lightgreen",
    alpha=0.3,
    linewidth=0,
)
ax1.add_patch(highlight2)

ax1.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE)
ax1.set_ylabel("Signals", fontsize=AXIS_LABEL_SIZE)
ax1.set_title("EMG Signal with TKEO Envelope (Y-Spread)", fontsize=SUBTITLE_SIZE)
ax1.legend(fontsize=AXIS_LABEL_SIZE)
ax1.tick_params(labelsize=TICK_SIZE)
# ax1.grid(True, alpha=1)

# ===== Row 2, Column 1: EMG Contractile Metrics =====
ax2 = fig.add_subplot(gs[1, 0])

# Plot zoomed EMG and TKEO with y offset
zoom_emg_data = zoom_data[:, params.channel] / emg_max_amp * 1.5 + params.emg_offset
zoom_tkeo_data = zoom_tkeo_normalized / emg_max_amp * 1.5 + params.tkeo_offset

ax2.plot(time_zoom, zoom_emg_data, color=NAVY_BLUE, linewidth=2)
ax2.plot(
    time_zoom,
    zoom_tkeo_data,
    color=ORANGE_ENVELOPE,
    linewidth=ENVELOPE_LINEWIDTH,
    alpha=0.9,
)

# Set y-ticks with channel labels
ax2.set_yticks(y_ticks)
ax2.set_yticklabels(y_labels)

# Add contractile metric lines across both channels
onset_time = params.onset_position / fs
peak_time = params.peak_position / fs
offset_time = params.offset_position / fs

ax2.axvline(x=onset_time, color="green", linestyle="--", linewidth=3, label="Onset")
ax2.axvline(x=peak_time, color="blue", linestyle="--", linewidth=3, label="Peak")
ax2.axvline(x=offset_time, color="purple", linestyle="--", linewidth=3, label="Offset")

ax2.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE)
ax2.set_ylabel("Signals", fontsize=AXIS_LABEL_SIZE)
ax2.set_title("EMG Contractile Metrics", fontsize=SUBTITLE_SIZE)
ax2.legend(fontsize=AXIS_LABEL_SIZE)
ax2.tick_params(labelsize=TICK_SIZE)
# ax2.grid(True, alpha=0.3)

# ===== Row 2, Column 2: Single Breath (Ti and Te) =====
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_zoom, zoom_emg_data, color=NAVY_BLUE, linewidth=2)
ax3.plot(
    time_zoom,
    zoom_tkeo_data,
    color=ORANGE_ENVELOPE,
    linewidth=ENVELOPE_LINEWIDTH,
    alpha=0.9,
)

# Set y-ticks with channel labels
ax3.set_yticks(y_ticks)
ax3.set_yticklabels(y_labels)

# Add Ti block (spans both EMG and TKEO)
ti_start_time = params.ti_start / fs
ti_end_time = params.ti_end / fs
ti_block = patches.Rectangle(
    (ti_start_time, ax3.get_ylim()[0]),
    ti_end_time - ti_start_time,
    ax3.get_ylim()[1] - ax3.get_ylim()[0],
    color="lightblue",
    alpha=0.5,
    label="Ti",
)
ax3.add_patch(ti_block)

# Add Te block
te_start_time = params.te_start / fs
te_end_time = min(params.te_end / fs, time_zoom[-1])
te_block = patches.Rectangle(
    (te_start_time, ax3.get_ylim()[0]),
    te_end_time - te_start_time,
    ax3.get_ylim()[1] - ax3.get_ylim()[0],
    color="lightcoral",
    alpha=0.5,
    label="Te",
)
ax3.add_patch(te_block)

ax3.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE)
ax3.set_ylabel("Signals", fontsize=AXIS_LABEL_SIZE)
ax3.set_title("Single Breath: Ti and Te", fontsize=SUBTITLE_SIZE)
ax3.legend(fontsize=AXIS_LABEL_SIZE)
ax3.tick_params(labelsize=TICK_SIZE)
# ax3.grid(True, alpha=0.3)

# ===== Row 2, Column 3: Two Breaths with Ttot =====
ax4 = fig.add_subplot(gs[1, 2])

# Plot EMG and TKEO with y offset for Ttot zoom
ttot_emg_data = (
    zoom_data_ttot[:, params.channel] / emg_max_amp * 1.5 + params.emg_offset
)
ttot_tkeo_data = zoom_tkeo_ttot_normalized / emg_max_amp * 1.5 + params.tkeo_offset

ax4.plot(time_zoom_ttot, ttot_emg_data, color=NAVY_BLUE, linewidth=2)
ax4.plot(
    time_zoom_ttot,
    ttot_tkeo_data,
    color=ORANGE_ENVELOPE,
    linewidth=ENVELOPE_LINEWIDTH,
    alpha=0.9,
)

# Set y-ticks with channel labels
ax4.set_yticks(y_ticks)
ax4.set_yticklabels(y_labels)

# Add Ttot block
ttot_start_time = params.ttot_start / fs
ttot_end_time = params.ttot_end / fs
ttot_block = patches.Rectangle(
    (ttot_start_time, ax4.get_ylim()[0]),
    ttot_end_time - ttot_start_time,
    ax4.get_ylim()[1] - ax4.get_ylim()[0],
    color="lightgreen",
    alpha=0.5,
    label="Ttot",
)
ax4.add_patch(ttot_block)

ax4.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE)
ax4.set_ylabel("Signals", fontsize=AXIS_LABEL_SIZE)
ax4.set_title("Two Breaths: Total Cycle Time (Ttot)", fontsize=SUBTITLE_SIZE)
ax4.legend(fontsize=AXIS_LABEL_SIZE)
ax4.tick_params(labelsize=TICK_SIZE)
# ax4.grid(True, alpha=0.3)

# Overall title
fig.suptitle(
    "EMG Signal Analysis: Contractile and Breathing Metrics",
    fontsize=TITLE_SIZE,
    y=1.02,
)

# Adjust layout
plt.tight_layout()
# plt.savefig(
#     "emg_contractile_breathing_metrics_2x3.png", dpi=300, bbox_inches="tight"
# )
plt.show()

# %%
# Interactive parameter adjustment helper
# Print current parameter values for easy adjustment
print("Current Parameter Values:")
print("-" * 50)
print(f"Data start: {params.data_start / fs:.1f}s")
print(f"Data duration: {params.data_duration / fs:.1f}s")
print(f"Zoom start: {params.zoom_start / fs:.1f}s (relative to data start)")
print(f"Zoom duration: {params.zoom_duration / fs:.1f}s")
print(f"Zoom start (Ttot): {params.zoom_start_ttot / fs:.1f}s")
print(f"Zoom duration (Ttot): {params.zoom_duration_ttot / fs:.1f}s")
print("-" * 50)
print("Contractile Metrics (relative to zoom start):")
print(f"Onset position: {params.onset_position / fs:.3f}s")
print(f"Peak position: {params.peak_position / fs:.3f}s")
print(f"Offset position: {params.offset_position / fs:.3f}s")
print("-" * 50)
print("Breathing Metrics (relative to zoom start):")
print(f"Ti start: {params.ti_start / fs:.3f}s")
print(f"Ti end: {params.ti_end / fs:.3f}s")
print(f"Te start: {params.te_start / fs:.3f}s")
print(f"Te end: {params.te_end / fs:.3f}s")
print(f"Ttot start: {params.ttot_start / fs:.3f}s")
print(f"Ttot end: {params.ttot_end / fs:.3f}s")
print("-" * 50)
print("Y-Spread Parameters:")
print(f"Y-spread: {params.y_spread}")
print(f"EMG offset: {params.emg_offset}")
print(f"TKEO offset: {params.tkeo_offset}")
print("-" * 50)

# %%
