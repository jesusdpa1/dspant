"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import time

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import (
    FilterProcessor,
    create_ffc_notch,
    create_wp_harmonic_removal,
)
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data

# Set publication style
mpu.set_publication_style()

# %%
# Constants in UPPER CASE
BASE_PATH = r"E:\jpenalozaa\papers\2025_mp_emg diaphragm acquisition and processing"
EMG_STREAM_PATH = BASE_PATH + r"/noisy_recording.ant"

# %%
# Load EMG data
stream_emg = StreamNode(EMG_STREAM_PATH)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

FS = stream_emg.fs  # Get sampling rate from the stream node
# %%
# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=60, fs=FS)
# %%
# Create processing node with filters
processor_emg = create_processing_node(stream_emg)
# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
# %%
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
# %%
# View summary of the processing node
processor_emg.summarize()
# %%
ffc_filter = create_ffc_notch(60)
whp_filter = create_wp_harmonic_removal(60)
# %%
# Apply filters and plot results
raw_data = stream_emg.data.persist()
filter_data = processor_emg.process(group=["filters"]).persist()
ffc_data = ffc_filter.process(filter_data, FS).persist()
whp_data = whp_filter.process(filter_data[0:1000000], FS).persist()
# %%
START = int(FS * 5)
END = int(FS * 10)
base_data = filter_data[START:END, :]

# %%
raw_fig = plot_multi_channel_data(raw_data, fs=FS, time_window=[0, 10])
filtered_fig = plot_multi_channel_data(filter_data, fs=FS, time_window=[0, 10])
ffc_fig = plot_multi_channel_data(ffc_data, fs=FS, time_window=[0, 10])
whp_fig = plot_multi_channel_data(whp_data, fs=FS, time_window=[0, 10])
# %%

# Define color palette - maintaining dark navy and using colorblind-friendly colors
DARK_GREY_NAVY = "#2D3142"  # Dark navy for raw signals
FILTER_COLORS = {
    "bandpass": mpu.COLORS["blue"],
    "ffc": mpu.COLORS["orange"],
    "wavelet": mpu.COLORS["green"],
}
HIGHLIGHT_COLOR = mpu.COLORS["purple"]  # For zoom highlighting

# Define data range
DATA_START = 0
DATA_END = int(10 * FS)
ZOOM_START = int(1.9 * FS)
ZOOM_END = int(2.2 * FS)

# Get time values for the zoom highlight box
ZOOM_START_TIME = ZOOM_START / FS
ZOOM_END_TIME = ZOOM_END / FS
ZOOM_WIDTH = ZOOM_END_TIME - ZOOM_START_TIME

# Create figure with GridSpec for custom layout
# 4 rows, 5 columns with the right side being 1/4 of the left
fig = plt.figure(figsize=(20, 16))
gs = mpu.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1])

# Calculate time arrays
time_array = np.arange(DATA_END - DATA_START) / FS
zoom_time_array = np.arange(ZOOM_END - ZOOM_START) / FS


# Improved panel label function
def add_panel_label(
    ax,
    label,
    position="top-left",
    offset_factor=0.1,
    fontsize=None,
    fontweight="bold",
    color="black",
):
    """
    Add a panel label (A, B, C, etc.) to a subplot with adaptive positioning.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the label to
    label : str
        Label text (typically a single letter like 'A', 'B', etc.)
    position : str
        Position of the label relative to the subplot. Options:
        'top-left' (default), 'top-right', 'bottom-left', 'bottom-right'
    offset_factor : float
        Factor to determine the offset relative to subplot width/height.
        Smaller values place the label closer to the subplot.
        Typical values range from 0.05 to 0.2.
    fontsize : int, optional
        Font size for the label. If None, uses the FONT_SIZES["panel_label"]
    fontweight : str
        Font weight for the label
    color : str
        Color for the label text
    """
    # Get the position of the axes in figure coordinates
    bbox = ax.get_position()
    fig = plt.gcf()

    # Set default font size if not specified
    if fontsize is None:
        fontsize = mpu.FONT_SIZES["panel_label"]

    # Calculate offset based on subplot size and offset factor
    # This will scale the offset proportionally to the subplot size
    x_offset = bbox.width * offset_factor
    y_offset = bbox.height * offset_factor

    # Determine position coordinates based on selected position
    if position == "top-left":
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset
    elif position == "top-right":
        x = bbox.x1 + x_offset
        y = bbox.y1 + y_offset
    elif position == "bottom-left":
        x = bbox.x0 - x_offset
        y = bbox.y0 - y_offset
    elif position == "bottom-right":
        x = bbox.x1 + x_offset
        y = bbox.y0 - y_offset
    else:
        # Default to top-left if invalid position
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset

    # Determine text alignment based on position
    if "left" in position:
        ha = "right"
    else:
        ha = "left"

    if "top" in position:
        va = "bottom"
    else:
        va = "top"

    # Position the label outside the subplot
    fig.text(
        x,
        y,
        label,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha=ha,
        color=color,
    )


# Plot 1: Original Raw Data (spanning first 4 columns)
ax_raw = fig.add_subplot(gs[0, 0:4])
ax_raw.plot(
    time_array, raw_data[DATA_START:DATA_END, 0], color=DARK_GREY_NAVY, linewidth=2
)
mpu.format_axis(
    ax_raw,
    title="Raw EMG Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Plot 2: Bandpass Filtered Data
ax_bp = fig.add_subplot(gs[1, 0:4])
ax_bp.plot(
    time_array,
    filter_data[DATA_START:DATA_END, 0],
    color=FILTER_COLORS["bandpass"],
    linewidth=2,
)
mpu.format_axis(
    ax_bp,
    title="Bandpass + Notch Filtered Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Plot 3: FFC Filtered Data
ax_ffc = fig.add_subplot(gs[2, 0:4])
ax_ffc.plot(
    time_array,
    ffc_data[DATA_START:DATA_END, 0],
    color=FILTER_COLORS["ffc"],
    linewidth=2,
)
mpu.format_axis(
    ax_ffc,
    title="FFC Notch Filtered Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Add highlight box for FFC zoomed region using the zoom variables
y_min, y_max = ax_ffc.get_ylim()
height = y_max - y_min
ffc_rect = plt.Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_ffc.add_patch(ffc_rect)

# Plot 4: Wavelet Pack Harmonic Removal
ax_whp = fig.add_subplot(gs[3, 0:4])
ax_whp.plot(
    time_array[: len(whp_data)],
    whp_data[DATA_START : min(DATA_END, len(whp_data)), 0],
    color=FILTER_COLORS["wavelet"],
    linewidth=2,
)
mpu.format_axis(
    ax_whp,
    title="Wavelet Packet Harmonic Removal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Add highlight box for Wavelet zoomed region using the zoom variables
y_min, y_max = ax_whp.get_ylim()
height = y_max - y_min
whp_rect = plt.Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_whp.add_patch(whp_rect)

# Plot 5: FFC Zoomed (spanning 2 rows on the right)
ax_ffc_zoom = fig.add_subplot(gs[0:2, 4])
ax_ffc_zoom.plot(
    zoom_time_array,
    ffc_data[ZOOM_START:ZOOM_END, 0],
    color=FILTER_COLORS["ffc"],
    linewidth=2,
)
mpu.format_axis(
    ax_ffc_zoom,
    title="FFC Filter (Zoomed)",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, ZOOM_WIDTH),
)

# Plot 6: Wavelet Zoomed (spanning 2 rows on the right)
ax_whp_zoom = fig.add_subplot(gs[2:4, 4])
ax_whp_zoom.plot(
    zoom_time_array[: min(len(zoom_time_array), ZOOM_END - ZOOM_START)],
    whp_data[ZOOM_START : min(ZOOM_END, len(whp_data)), 0],
    color=FILTER_COLORS["wavelet"],
    linewidth=2,
)
mpu.format_axis(
    ax_whp_zoom,
    title="Wavelet Filter (Zoomed)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, ZOOM_WIDTH),
)

# Finalize the figure with our utility function
mpu.finalize_figure(
    fig,
    title="EMG Signal Filtering Comparison",
    title_y=0.98,
)

# Apply tight layout to finalize positions before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

# Add panel labels with adaptive positioning
add_panel_label(ax_raw, "A", offset_factor=0.05)
add_panel_label(ax_bp, "B", offset_factor=0.05)
add_panel_label(ax_ffc, "C", offset_factor=0.05)
add_panel_label(ax_whp, "D", offset_factor=0.05)
add_panel_label(ax_ffc_zoom, "E", offset_factor=0.05)
add_panel_label(ax_whp_zoom, "F", offset_factor=0.05)

# Save figure if needed
mpu.save_figure(fig, "emg_filtering_comparison.png", dpi=600)

plt.show()

# %%
