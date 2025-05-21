"""
EMG Contractile and Breathing Metrics Visualization with Y-Spread Channels
Author: Jesus Penaloza (Updated with TKEO envelope and metrics analysis)
Using mp_plotting_utils for standardized publication-quality visualization
"""

# %%
import os
import time
from pathlib import Path

import dotenv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

# Set publication style
mpu.set_publication_style()

# %%
# Data loading (keep your existing data loading code)
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath(r"RawG.ant")

# %%
# Load EMG data
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()
FS = stream_emg.fs

# %%
# Create and apply filters
processor_emg = create_processing_node(stream_emg)

# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=60, fs=FS)
lowpass_filter = create_lowpass_filter(20, FS)

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
tkeo_envelope = tkeo_processor.process(filter_data, fs=FS).persist()

# %%
# Create visualization


# Define adjustable parameters for the visualization
class PlotParameters:
    # Data selection parameters
    data_start = int(FS * 5)  # Start time in seconds
    data_duration = int(FS * 8)  # Duration in seconds
    zoom_start = int(FS * 6)  # Zoomed section start
    zoom_duration = int(FS * 1.5)  # 1 second zoom

    # Ttot zoom parameters
    zoom_start_ttot = int(FS * 6.8)  # Ttot zoomed section start
    zoom_duration_ttot = int(FS * 1.5)  # 4 seconds zoom for Ttot

    # Contractile metric parameters (in samples from zoom_start)
    onset_position = int(FS * 0.32)  # Position of onset marker
    peak_position = int(FS * 0.51)  # Position of peak marker
    offset_position = int(FS * 0.6)  # Position of offset marker

    # Breathing metric parameters
    ti_start = int(FS * 0.32)  # Ti start (onset)
    ti_end = int(FS * 0.51)  # Ti end (peak)
    te0_start = int(FS * 0.51)  # Te start (peak)
    te0_end = int(FS * 0.61)  # Te end (next onset)
    te1_start = int(FS * 0.61)  # Te start (peak)
    te1_end = int(FS * 1.02)  # Te end (next onset)
    ttot_start = int(FS * 0.24)  # Ttot start (first onset)
    ttot_end = int(FS * 0.9)  # Ttot end (second onset)

    # Channel to visualize
    channel = 1

    # Y-spread parameters
    y_spread = 5.0  # Vertical spacing between channels
    emg_offset = 0.0  # Baseline for EMG
    tkeo_offset = 1.0  # Baseline for TKEO (above EMG)


# Extract data segments
PARAMS = PlotParameters()
data_segment = filter_data[
    PARAMS.data_start : PARAMS.data_start + PARAMS.data_duration, :
]
tkeo_segment = tkeo_envelope[
    PARAMS.data_start : PARAMS.data_start + PARAMS.data_duration, :
]

# Create time arrays
time_full = np.arange(len(data_segment)) / FS
time_zoom = np.arange(PARAMS.zoom_duration) / FS
time_zoom_ttot = np.arange(PARAMS.zoom_duration_ttot) / FS

# Extract zoomed sections
zoom_data = data_segment[
    PARAMS.zoom_start - PARAMS.data_start : PARAMS.zoom_start
    - PARAMS.data_start
    + PARAMS.zoom_duration,
    :,
]
zoom_tkeo = tkeo_segment[
    PARAMS.zoom_start - PARAMS.data_start : PARAMS.zoom_start
    - PARAMS.data_start
    + PARAMS.zoom_duration,
    :,
]

# Extract zoomed sections for Ttot
zoom_data_ttot = data_segment[
    PARAMS.zoom_start_ttot - PARAMS.data_start : PARAMS.zoom_start_ttot
    - PARAMS.data_start
    + PARAMS.zoom_duration_ttot,
    :,
]
zoom_tkeo_ttot = tkeo_segment[
    PARAMS.zoom_start_ttot - PARAMS.data_start : PARAMS.zoom_start_ttot
    - PARAMS.data_start
    + PARAMS.zoom_duration_ttot,
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


# Define colors for visualization - preserve all original colors
NAVY_BLUE = "#2D3142"  # Dark navy blue for time series (original)
ORANGE_ENVELOPE = "#DE8F05"  # Original orange for envelope
HIGHLIGHT_YELLOW = "#FFCC00"  # Original yellow for zoom highlight
HIGHLIGHT_GREEN = "#90EE90"  # Original light green for Ttot highlight
TI_COLOR = "#AEDFF7"  # Original light blue for Ti block
TE0_COLOR = "#F9BFC1"  # Original light coral for Te0 block
TE1_COLOR = "#DE4010"  # Original red for Te1 block
TTOT_COLOR = HIGHLIGHT_GREEN  # Light green for Ttot block
ONSET_COLOR = "green"  # Original color for onset line
PEAK_COLOR = "blue"  # Original color for peak line
OFFSET_COLOR = "purple"  # Original color for offset line

# Font sizes - increased for better visibility
TITLE_SIZE = 24 * 2  # Increased from 20
SUBTITLE_SIZE = 20 * 2  # Increased from 18
AXIS_LABEL_SIZE = 18 * 2  # Increased from 16
TICK_SIZE = 16 * 2  # Increased from 14
LEGEND_SIZE = 18 * 2  # Added for legend text

# Line styles
ENVELOPE_LINEWIDTH = 4  # Thicker line for envelope

# Create figure with 2x3 grid
fig = plt.figure(figsize=(36, 24))
gs = GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

# Get max amplitude for normalization
emg_max_amp = np.max(np.abs(data_segment[:, PARAMS.channel]))

# Normalize TKEO data for y-spread
tkeo_segment_normalized = normalize_tkeo_for_spread(
    data_segment[:, PARAMS.channel], tkeo_segment[:, PARAMS.channel], emg_max_amp
)
zoom_tkeo_normalized = normalize_tkeo_for_spread(
    zoom_data[:, PARAMS.channel], zoom_tkeo[:, PARAMS.channel], emg_max_amp
)
zoom_tkeo_ttot_normalized = normalize_tkeo_for_spread(
    zoom_data_ttot[:, PARAMS.channel], zoom_tkeo_ttot[:, PARAMS.channel], emg_max_amp
)

# ===== Row 1: Full EMG with TKEO in y-spread (3x1) =====
ax1 = fig.add_subplot(gs[0, :])

# Plot EMG with y offset
emg_plot_data = data_segment[:, PARAMS.channel] / emg_max_amp * 1.5 + PARAMS.emg_offset
ax1.plot(
    time_full,
    emg_plot_data,
    color=NAVY_BLUE,
    linewidth=2,
    label="EMG Signal",
)

# Plot TKEO envelope with y offset (placed above EMG)
tkeo_plot_data = tkeo_segment_normalized / emg_max_amp * 1.5 + PARAMS.tkeo_offset
ax1.plot(
    time_full,
    tkeo_plot_data,
    color=ORANGE_ENVELOPE,
    linewidth=ENVELOPE_LINEWIDTH,
    alpha=0.9,
    label="TKEO Envelope",
)

# Set y-ticks with channel labels
y_ticks = [PARAMS.emg_offset, PARAMS.tkeo_offset]
y_labels = ["EMG", "TKEO"]
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels)

# Highlight zoomed sections
# First zoom section
zoom_start_time = (PARAMS.zoom_start - PARAMS.data_start) / FS
zoom_end_time = zoom_start_time + PARAMS.zoom_duration / FS
highlight1 = patches.Rectangle(
    (zoom_start_time, ax1.get_ylim()[0]),
    PARAMS.zoom_duration / FS,
    ax1.get_ylim()[1] - ax1.get_ylim()[0],
    color=HIGHLIGHT_YELLOW,
    alpha=0.3,
    linewidth=0,
)
ax1.add_patch(highlight1)

# Second zoom section (for Ttot)
zoom_start_time_ttot = (PARAMS.zoom_start_ttot - PARAMS.data_start) / FS
zoom_end_time_ttot = zoom_start_time_ttot + PARAMS.zoom_duration_ttot / FS
highlight2 = patches.Rectangle(
    (zoom_start_time_ttot, ax1.get_ylim()[0]),
    PARAMS.zoom_duration_ttot / FS,
    ax1.get_ylim()[1] - ax1.get_ylim()[0],
    color=HIGHLIGHT_GREEN,
    alpha=0.3,
    linewidth=0,
)
ax1.add_patch(highlight2)

# Format axis using mpu with larger font sizes
mpu.format_axis(
    ax1,
    title="EMG Signal with TKEO Envelope (Y-Spread)",
    xlabel="Time [s]",
    ylabel="Signals",
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
mpu.add_legend(ax1, fontsize=LEGEND_SIZE)

# ===== Row 2, Column 1: EMG Contractile Metrics =====
ax2 = fig.add_subplot(gs[1, 0])

# Plot zoomed EMG and TKEO with y offset
zoom_emg_data = zoom_data[:, PARAMS.channel] / emg_max_amp * 1.5 + PARAMS.emg_offset
zoom_tkeo_data = zoom_tkeo_normalized / emg_max_amp * 1.5 + PARAMS.tkeo_offset

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
onset_time = PARAMS.onset_position / FS
peak_time = PARAMS.peak_position / FS
offset_time = PARAMS.offset_position / FS

ax2.axvline(x=onset_time, color=ONSET_COLOR, linestyle="--", linewidth=3, label="Onset")
ax2.axvline(x=peak_time, color=PEAK_COLOR, linestyle="--", linewidth=3, label="Peak")
ax2.axvline(
    x=offset_time, color=OFFSET_COLOR, linestyle="--", linewidth=3, label="Offset"
)

# Format axis using mpu with larger font sizes
mpu.format_axis(
    ax2,
    title="EMG Contractile Metrics",
    xlabel="Time [s]",
    ylabel="Signals",
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
mpu.add_legend(ax2, fontsize=LEGEND_SIZE)

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
ti_start_time = PARAMS.ti_start / FS
ti_end_time = PARAMS.ti_end / FS
ti_block = patches.Rectangle(
    (ti_start_time, ax3.get_ylim()[0]),
    ti_end_time - ti_start_time,
    ax3.get_ylim()[1] - ax3.get_ylim()[0],
    color=TI_COLOR,
    alpha=0.5,
    label="Ti",
)
ax3.add_patch(ti_block)

# Add Te0 block
te_start_time = PARAMS.te0_start / FS
te_end_time = min(PARAMS.te0_end / FS, time_zoom[-1])
te_block = patches.Rectangle(
    (te_start_time, ax3.get_ylim()[0]),
    te_end_time - te_start_time,
    ax3.get_ylim()[1] - ax3.get_ylim()[0],
    color=TE0_COLOR,
    alpha=0.5,
    label="Te0",
)
ax3.add_patch(te_block)

# Add Te1 block
te1_start_time = PARAMS.te1_start / FS
te1_end_time = min(PARAMS.te1_end / FS, time_zoom[-1])
te1_block = patches.Rectangle(
    (te1_start_time, ax3.get_ylim()[0]),
    te1_end_time - te1_start_time,
    ax3.get_ylim()[1] - ax3.get_ylim()[0],
    color=TE1_COLOR,
    alpha=0.5,
    label="Te1",
)
ax3.add_patch(te1_block)

# Format axis using mpu with larger font sizes
mpu.format_axis(
    ax3,
    title="Single Breath: Ti and Putative Te",
    xlabel="Time [s]",
    ylabel="Signals",
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
mpu.add_legend(ax3, fontsize=LEGEND_SIZE)

# ===== Row 2, Column 3: Two Breaths with Ttot =====
ax4 = fig.add_subplot(gs[1, 2])

# Plot EMG and TKEO with y offset for Ttot zoom
ttot_emg_data = (
    zoom_data_ttot[:, PARAMS.channel] / emg_max_amp * 1.5 + PARAMS.emg_offset
)
ttot_tkeo_data = zoom_tkeo_ttot_normalized / emg_max_amp * 1.5 + PARAMS.tkeo_offset

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
ttot_start_time = PARAMS.ttot_start / FS
ttot_end_time = PARAMS.ttot_end / FS
ttot_block = patches.Rectangle(
    (ttot_start_time, ax4.get_ylim()[0]),
    ttot_end_time - ttot_start_time,
    ax4.get_ylim()[1] - ax4.get_ylim()[0],
    color=TTOT_COLOR,
    alpha=0.5,
    label="Ttot",
)
ax4.add_patch(ttot_block)

# Format axis using mpu with larger font sizes
mpu.format_axis(
    ax4,
    title="Two Breaths: Total Cycle Time (Ttot)",
    xlabel="Time [s]",
    ylabel="Signals",
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
mpu.add_legend(ax4, fontsize=LEGEND_SIZE)

# Finalize the figure with mpu using larger font size for title
mpu.finalize_figure(
    fig,
    title="EMG Signal Analysis: Contractile and Breathing Metrics",
    title_y=1.02,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout before adding panel labels
plt.tight_layout()

# Add panel labels - using adaptive panel label function with larger font
# Add panel labels - using the mpu module's function with larger font
mpu.add_panel_label(
    ax1,
    "A",
    x_offset_factor=0.01,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2,
    "B",
    x_offset_factor=0.02,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3,
    "C",
    x_offset_factor=0.02,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax4,
    "D",
    x_offset_factor=0.02,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)

# Save figure if needed
mpu.save_figure(fig, "emg_contractile_breathing_metrics.png", dpi=600)

plt.show()

# %%
# Interactive parameter adjustment helper
# Print current parameter values for easy adjustment
print("Current Parameter Values:")
print("-" * 50)
print(f"Data start: {PARAMS.data_start / FS:.1f}s")
print(f"Data duration: {PARAMS.data_duration / FS:.1f}s")
print(f"Zoom start: {PARAMS.zoom_start / FS:.1f}s (relative to data start)")
print(f"Zoom duration: {PARAMS.zoom_duration / FS:.1f}s")
print(f"Zoom start (Ttot): {PARAMS.zoom_start_ttot / FS:.1f}s")
print(f"Zoom duration (Ttot): {PARAMS.zoom_duration_ttot / FS:.1f}s")
print("-" * 50)
print("Contractile Metrics (relative to zoom start):")
print(f"Onset position: {PARAMS.onset_position / FS:.3f}s")
print(f"Peak position: {PARAMS.peak_position / FS:.3f}s")
print(f"Offset position: {PARAMS.offset_position / FS:.3f}s")
print("-" * 50)
print("Breathing Metrics (relative to zoom start):")
print(f"Ti start: {PARAMS.ti_start / FS:.3f}s")
print(f"Ti end: {PARAMS.ti_end / FS:.3f}s")
print(f"Te0 start: {PARAMS.te0_start / FS:.3f}s")
print(f"Te0 end: {PARAMS.te0_end / FS:.3f}s")
print(f"Te1 start: {PARAMS.te1_start / FS:.3f}s")
print(f"Te1 end: {PARAMS.te1_end / FS:.3f}s")
print(f"Ttot start: {PARAMS.ttot_start / FS:.3f}s")
print(f"Ttot end: {PARAMS.ttot_end / FS:.3f}s")
print("-" * 50)
print("Y-Spread Parameters:")
print(f"Y-spread: {PARAMS.y_spread}")
print(f"EMG offset: {PARAMS.emg_offset}")
print(f"TKEO offset: {PARAMS.tkeo_offset}")
print("-" * 50)

# %%
