"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import os
import time
from pathlib import Path

import dask.array as da
import dotenv
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import seaborn as sns
from dspant_emgproc.processors.activity_detection.double_threshold import (
    create_double_threshold_detector,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.fir_filters import create_moving_average
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()

# %%
base_path = Path(os.getenv("DATA_DIR"))
data_path = base_path.joinpath(
    r"2025_mp_emg diaphragm acquisition and processing/Sample Ventilator Trace"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_right_path = data_path.joinpath(r"emg_r.ant")
emg_left_path = data_path.joinpath(r"emg_l.ant")
insp_path = data_path.joinpath(r"insp.ant")

# %%

# Load EMG data
stream_emg_r = StreamNode(str(emg_right_path))
stream_emg_r.load_metadata()
stream_emg_r.load_data()
# Print stream_emg summary
stream_emg_r.summarize()


stream_emg_l = StreamNode(str(emg_left_path))
stream_emg_l.load_metadata()
stream_emg_l.load_data()
# Print stream_emg summary
stream_emg_l.summarize()


stream_insp = StreamNode(str(insp_path))
stream_insp.load_metadata()
stream_insp.load_data()
# Print stream_emg summary
stream_insp.summarize()


# %%
# Create and visualize filters before applying them
fs = stream_emg_l.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)
# %%
# bandpass_plot = bandpass_filter.plot_frequency_response()
# notch_plot = notch_filter.plot_frequency_response()
# lowpass_plot = lowpass_filter.plot_frequency_response()
# %%

# Create processing node with filters
processor_emg_r = create_processing_node(stream_emg_r)
processor_emg_l = create_processing_node(stream_emg_l)
processor_insp = create_processing_node(stream_insp)
# %%

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
# %%

# Add processors to the processing node
processor_emg_r.add_processor([notch_processor, bandpass_processor], group="filters")
processor_emg_l.add_processor([notch_processor, bandpass_processor], group="filters")
processor_insp.add_processor([notch_processor, lowpass_processor], group="filters")
# Apply filters and plot results
filtered_emg_r = processor_emg_r.process(group=["filters"]).persist()
filtered_emg_l = processor_emg_l.process(group=["filters"]).persist()
filtered_insp = processor_insp.process(group=["filters"]).persist()

# %%
filtered_data = da.concatenate([filtered_emg_r, filtered_emg_l, filtered_insp], axis=1)

# %%
multichannel_fig = plot_multi_channel_data(filtered_data, fs=fs, time_window=[100, 110])

# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
tkeo_data = tkeo_processor.process(filtered_emg_l, fs=fs).persist()
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()
zscore_insp = zscore_processor.process(filtered_insp).persist()
# %%
tkeo_fig = plot_multi_channel_data(tkeo_data, fs=fs, time_window=[100, 110])

# %%
"""
time stamps
neural trigger on = 0 to 30
neural trigger off = 365 to 395
# neural triggered tidal volume increased = 1090 - 1120
# neural triggered tidal volume decreased = 1245 - 1275
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the style
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 1
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Define time range for the plots
start_time = 1240  # seconds 375, 1245
end_time = 1240 + 25  # seconds 400,
zoom_width = 5  # seconds 5

# Calculate the zoom range
zoom_start_time = start_time + (end_time - start_time - zoom_width) / 2
zoom_end_time = zoom_start_time + zoom_width

# Convert to sample indices
start_idx = int(start_time * fs)
end_idx = int(end_time * fs)
zoom_start_idx = int(zoom_start_time * fs)
zoom_end_idx = int(zoom_end_time * fs)

# Create figure with GridSpec for custom layout (6 rows, 6 columns)
# Using a 6x6 grid to achieve 4:2 ratio (4 columns for left, 2 for right)
fig = plt.figure(figsize=(20, 8))
gs = GridSpec(6, 6)

# Darker grey with navy tint for raw EMG
dark_grey_navy = "#2D3142"

# Choose colors for different signals
tkeo_color = palette[0]  # First color from palette
insp_color = palette[1]  # Second color from palette
emg_r_color = "#2D3142"
highlight_color = palette[5]  # Fourth color for highlight areas

# Calculate time arrays
time_array = np.arange(start_idx, end_idx) / fs - start_time  # Normalize to start at 0
zoom_time_array = (
    np.arange(zoom_start_idx, zoom_end_idx) / fs - zoom_start_time
)  # Normalize to start at 0

# Prepare the data
# Convert dask arrays to numpy if needed
zscore_tkeo_np = (
    zscore_tkeo[start_idx:end_idx].compute()
    if hasattr(zscore_tkeo, "compute")
    else zscore_tkeo[start_idx:end_idx]
)
zscore_insp_np = (
    zscore_insp[start_idx:end_idx].compute()
    if hasattr(zscore_insp, "compute")
    else zscore_insp[start_idx:end_idx]
)
filtered_emg_r_np = (
    filtered_emg_l[start_idx:end_idx].compute()
    if hasattr(filtered_emg_l, "compute")
    else filtered_emg_l[start_idx:end_idx]
)

# Ensure we have 1D arrays
if zscore_tkeo_np.ndim > 1:
    zscore_tkeo_np = zscore_tkeo_np[:, 0]
if zscore_insp_np.ndim > 1:
    zscore_insp_np = zscore_insp_np[:, 0]
if filtered_emg_r_np.ndim > 1:
    filtered_emg_r_np = filtered_emg_r_np[:, 0]


# Min-max normalize signals for better visualization
def min_max_normalize(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)


# Also prepare zoom data
zscore_tkeo_zoom = zscore_tkeo_np[zoom_start_idx - start_idx : zoom_end_idx - start_idx]
zscore_insp_zoom = zscore_insp_np[zoom_start_idx - start_idx : zoom_end_idx - start_idx]
filtered_emg_r_zoom = filtered_emg_r_np[
    zoom_start_idx - start_idx : zoom_end_idx - start_idx
]

# Normalize the signals for visualization
tkeo_norm = min_max_normalize(zscore_tkeo_np)
insp_norm = min_max_normalize(zscore_insp_np)
tkeo_zoom_norm = min_max_normalize(zscore_tkeo_zoom)
insp_zoom_norm = min_max_normalize(zscore_insp_zoom)

# Convert thresholds to the normalized scale
tkeo_min = np.min(zscore_tkeo_np)
tkeo_max = np.max(zscore_tkeo_np)
insp_min = np.min(zscore_insp_np)
insp_max = np.max(zscore_insp_np)


# Define colors with alpha for fill
tkeo_fill_color = to_rgba(tkeo_color, 0.4)
insp_fill_color = to_rgba(insp_color, 0.4)

# Define the merged color for where the two signals overlap
# By blending the two colors
insp_rgb = mcolors.to_rgb(insp_fill_color)
tkeo_rgb = mcolors.to_rgb(tkeo_fill_color)
merged_color = [
    (insp_rgb[0] + tkeo_rgb[0]) / 2,
    (insp_rgb[1] + tkeo_rgb[1]) / 2,
    (insp_rgb[2] + tkeo_rgb[2]) / 2,
    0.6,  # Higher alpha for the overlap
]

# Plot 1: Overlapping TKEO and Inspiration with shared axes (spanning 3 rows, 4 columns)
ax_overlap = fig.add_subplot(gs[0:3, 0:4])

# Plot overlapping signals
ax_overlap.plot(time_array, tkeo_norm, color=tkeo_color, linewidth=2, label="TKEO EMG")
ax_overlap.plot(
    time_array, insp_norm, color=insp_color, linewidth=2, label="Inspiration"
)

# Fill beneath each line
ax_overlap.fill_between(time_array, 0, tkeo_norm, color=tkeo_fill_color)
ax_overlap.fill_between(
    time_array, 0, insp_norm, color=insp_fill_color, where=(insp_norm > tkeo_norm)
)
# Add a special fill for where they overlap
overlap = np.minimum(tkeo_norm, insp_norm)
ax_overlap.fill_between(time_array, 0, overlap, color=merged_color)

# Set axis properties
ax_overlap.set_xlim(0, end_time - start_time)
ax_overlap.set_ylim(-0.05, 1.05)  # Add a little padding to the normalized [0,1] range
ax_overlap.set_xlabel("")  # No x-label for top plot
ax_overlap.set_ylabel("Normalized Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_overlap.tick_params(labelsize=TICK_SIZE)
ax_overlap.set_title(
    "EMG TKEO and Respiratory Signals", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Add highlight for zoom area
zoom_start_rel = zoom_start_time - start_time
zoom_width_rel = zoom_width
overlap_rect = Rectangle(
    (zoom_start_rel, -0.05),
    zoom_width_rel,
    1.1,  # Full height plus padding
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.15,
)
ax_overlap.add_patch(overlap_rect)

# Plot 2: Right side EMG (spanning 3 rows, 4 columns)
ax_emg_r = fig.add_subplot(gs[3:6, 0:4])
ax_emg_r.plot(time_array, filtered_emg_r_np, color=emg_r_color, linewidth=2)
ax_emg_r.set_xlim(0, end_time - start_time)
ax_emg_r.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_emg_r.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_emg_r.tick_params(labelsize=TICK_SIZE)
ax_emg_r.set_title("Right Side EMG Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Add highlight for zoom area on EMG right
emg_r_rect = Rectangle(
    (zoom_start_rel, ax_emg_r.get_ylim()[0]),
    zoom_width_rel,
    ax_emg_r.get_ylim()[1] - ax_emg_r.get_ylim()[0],
    linewidth=2,
    edgecolor=highlight_color,
    facecolor=highlight_color,
    alpha=0.15,
)
ax_emg_r.add_patch(emg_r_rect)

# Plot 3: Zoomed overlapping signals (spanning 3 rows, 2 columns)
ax_zoom_overlap = fig.add_subplot(gs[0:3, 4:6])

# Plot overlapping signals in zoomed view
ax_zoom_overlap.plot(
    zoom_time_array, tkeo_zoom_norm, color=tkeo_color, linewidth=2, label="TKEO EMG"
)
ax_zoom_overlap.plot(
    zoom_time_array, insp_zoom_norm, color=insp_color, linewidth=2, label="Inspiration"
)

# Fill beneath each line in zoomed view
ax_zoom_overlap.fill_between(zoom_time_array, 0, tkeo_zoom_norm, color=tkeo_fill_color)
ax_zoom_overlap.fill_between(
    zoom_time_array,
    0,
    insp_zoom_norm,
    color=insp_fill_color,
    where=(insp_zoom_norm > tkeo_zoom_norm),
)
# Add a special fill for where they overlap
zoom_overlap = np.minimum(tkeo_zoom_norm, insp_zoom_norm)
ax_zoom_overlap.fill_between(zoom_time_array, 0, zoom_overlap, color=merged_color)

# Set axis properties
ax_zoom_overlap.set_xlim(0, zoom_width)
ax_zoom_overlap.set_ylim(
    -0.05, 1.05
)  # Add a little padding to the normalized [0,1] range
ax_zoom_overlap.set_xlabel("")  # No x-label for top plot
ax_zoom_overlap.set_ylabel(
    "Normalized Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold"
)
ax_zoom_overlap.tick_params(labelsize=TICK_SIZE)
ax_zoom_overlap.set_title("Zoomed Signals", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot 4: Onset comparison with only vertical lines (spanning 3 rows, 2 columns)
ax_onset_compare = fig.add_subplot(gs[3:6, 4:6])
ax_onset_compare.set_xlim(0, zoom_width)
ax_onset_compare.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_onset_compare.set_ylabel("", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax_onset_compare.tick_params(labelsize=TICK_SIZE)
ax_onset_compare.set_title(
    "Onset Timing Comparison", fontsize=SUBTITLE_SIZE, weight="bold"
)

# Remove y-axis ticks since we're only showing onset lines
ax_onset_compare.set_yticks([])

# Find all onsets within the zoom window
tkeo_onsets_zoom = []
insp_onsets_zoom = []

# Add onset markers with vertical lines and annotations
for tkeo_onset in tkeo_onsets_zoom:
    # Draw a vertical line spanning the full height
    ax_onset_compare.axvline(x=tkeo_onset, color=tkeo_color, linestyle="-", linewidth=3)


for insp_onset in insp_onsets_zoom:
    # Draw a vertical line spanning the full height
    ax_onset_compare.axvline(x=insp_onset, color=insp_color, linestyle="-", linewidth=3)

# Add legend for onset comparison
custom_lines = [
    Line2D([0], [0], color=tkeo_color, lw=3),
    Line2D([0], [0], color=insp_color, lw=3),
]
ax_onset_compare.legend(
    custom_lines, ["TKEO Onset", "Insp Onset"], loc="lower right", fontsize=TICK_SIZE
)

# Add overall title
plt.suptitle(
    "EMG and Respiratory Activity Detection",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    y=0.98,
)

# Adjust layout


plt.show()
# %%

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter


# Custom formatter class to control decimal places in scientific notation
class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"  # 1 decimal place


# Min-max normalize signals for better visualization
def min_max_normalize(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    return (signal - min_val) / (max_val - min_val)


# Set the colorblind-friendly palette
sns.set_palette("colorblind")
PALETTE = sns.color_palette("colorblind")

# Define font sizes with appropriate scaling
FONT_SIZE = 14
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.7)
TICK_SIZE = int(FONT_SIZE * 0.7)
LEGEND_SIZE = int(FONT_SIZE * 0.7)
SCIENTIFIC_NOTATION_SIZE = int(FONT_SIZE * 0.5)

# Dark grey with navy tint for EMG
DARK_GREY_NAVY = "#2D3142"

# Choose colors for different signals
TKEO_COLOR = PALETTE[0]
INSP_COLOR = PALETTE[3]
EMG_COLOR = DARK_GREY_NAVY
HIGHLIGHT_COLOR = "#FFFB00"  # For zoom rectangle
ENTRAINED_COLOR = "#2D3142"  # Different color for entrained highlight

# Define colors with alpha for fill
TKEO_FILL_COLOR = to_rgba(TKEO_COLOR, 0.4)
INSP_FILL_COLOR = to_rgba(INSP_COLOR, 0.4)

# Define the merged color for overlap
insp_rgb = mcolors.to_rgb(INSP_FILL_COLOR)
tkeo_rgb = mcolors.to_rgb(TKEO_FILL_COLOR)
MERGED_COLOR = [
    (insp_rgb[0] + tkeo_rgb[0]) / 2,
    (insp_rgb[1] + tkeo_rgb[1]) / 2,
    (insp_rgb[2] + tkeo_rgb[2]) / 2,
    0.6,
]

# Define time windows
# Neural Trigger OFF
OFF_START = 365 + 10
OFF_END = 395

# Neural Trigger ON
ON_START = 0
ON_END = 30 - 10

# Zoom parameters (2 seconds in the middle of each window)
ZOOM_DURATION = 4

# Entrained highlight parameters
ENTRAINED_START = 1.5  # Start at 1.5s in zoomed view
ENTRAINED_DURATION = 1.0  # 1 second duration

# Calculate zoom start times (middle of each window)
off_duration = OFF_END - OFF_START
on_duration = ON_END - ON_START
zoom_off_start = OFF_START + (off_duration - ZOOM_DURATION) / 2
zoom_off_end = zoom_off_start + ZOOM_DURATION
zoom_on_start = ON_START + (on_duration - ZOOM_DURATION) / 2
zoom_on_end = zoom_on_start + ZOOM_DURATION

# Convert to sample indices
off_start_idx = int(OFF_START * fs)
off_end_idx = int(OFF_END * fs)
zoom_off_start_idx = int(zoom_off_start * fs)
zoom_off_end_idx = int(zoom_off_end * fs)

on_start_idx = int(ON_START * fs)
on_end_idx = int(ON_END * fs)
zoom_on_start_idx = int(zoom_on_start * fs)
zoom_on_end_idx = int(zoom_on_end * fs)

# Extract data for each time window
# Neural Trigger OFF data
filtered_emg_off = filtered_emg_l[off_start_idx:off_end_idx]
tkeo_off = zscore_tkeo[off_start_idx:off_end_idx]
insp_off = zscore_insp[off_start_idx:off_end_idx]

# Neural Trigger ON data
filtered_emg_on = filtered_emg_l[on_start_idx:on_end_idx]
tkeo_on = zscore_tkeo[on_start_idx:on_end_idx]
insp_on = zscore_insp[on_start_idx:on_end_idx]

# Zoom data OFF
filtered_emg_off_zoom = filtered_emg_l[zoom_off_start_idx:zoom_off_end_idx]
tkeo_off_zoom = zscore_tkeo[zoom_off_start_idx:zoom_off_end_idx]
insp_off_zoom = zscore_insp[zoom_off_start_idx:zoom_off_end_idx]

# Zoom data ON
filtered_emg_on_zoom = filtered_emg_l[zoom_on_start_idx:zoom_on_end_idx]
tkeo_on_zoom = zscore_tkeo[zoom_on_start_idx:zoom_on_end_idx]
insp_on_zoom = zscore_insp[zoom_on_start_idx:zoom_on_end_idx]

# Convert dask arrays to numpy if needed
arrays_to_convert = [
    "filtered_emg_off",
    "tkeo_off",
    "insp_off",
    "filtered_emg_on",
    "tkeo_on",
    "insp_on",
    "filtered_emg_off_zoom",
    "tkeo_off_zoom",
    "insp_off_zoom",
    "filtered_emg_on_zoom",
    "tkeo_on_zoom",
    "insp_on_zoom",
]

for arr_name in arrays_to_convert:
    arr = locals()[arr_name]
    if hasattr(arr, "compute"):
        arr = arr.compute()
    # Ensure 1D arrays
    if arr.ndim > 1:
        arr = arr[:, 0]
    locals()[arr_name] = arr

# Normalize TKEO and Insp for overlay plots
tkeo_off_norm = min_max_normalize(tkeo_off)
insp_off_norm = min_max_normalize(insp_off)
tkeo_on_norm = min_max_normalize(tkeo_on)
insp_on_norm = min_max_normalize(insp_on)

tkeo_off_zoom_norm = min_max_normalize(tkeo_off_zoom)
insp_off_zoom_norm = min_max_normalize(insp_off_zoom)
tkeo_on_zoom_norm = min_max_normalize(tkeo_on_zoom)
insp_on_zoom_norm = min_max_normalize(insp_on_zoom)

# Create time arrays
time_off = np.arange(len(filtered_emg_off)) / fs
time_on = np.arange(len(filtered_emg_on)) / fs
time_off_zoom = np.arange(len(filtered_emg_off_zoom)) / fs
time_on_zoom = np.arange(len(filtered_emg_on_zoom)) / fs

# Calculate zoom rectangle positions (relative to each plot's time axis)
zoom_off_start_rel = zoom_off_start - OFF_START
zoom_on_start_rel = zoom_on_start - ON_START

# Create figure with nested GridSpec
fig = plt.figure(figsize=(7, 6))

# Outer GridSpec: 2 rows, 1 column
outer_gs = GridSpec(2, 1, figure=fig, hspace=0.1)

# Inner GridSpec for Neural Trigger OFF (top)
gs_off = GridSpecFromSubplotSpec(
    2, 3, subplot_spec=outer_gs[0], width_ratios=[1, 0.8, 1.2], hspace=0.07, wspace=0.1
)

# Inner GridSpec for Neural Trigger ON (bottom)
gs_on = GridSpecFromSubplotSpec(
    2, 3, subplot_spec=outer_gs[1], width_ratios=[1, 0.8, 1.2], hspace=0.07, wspace=0.1
)

# ===== NEURAL TRIGGER OFF (TOP) =====

# Row 1, Col 1-2: TKEO + Insp overlay (Neural Trigger OFF)
ax_overlay_off = fig.add_subplot(gs_off[0, 0:2])
ax_overlay_off.plot(time_off, tkeo_off_norm, color=TKEO_COLOR, linewidth=1.2)
ax_overlay_off.plot(time_off, insp_off_norm, color=INSP_COLOR, linewidth=1.2)
ax_overlay_off.fill_between(time_off, 0, tkeo_off_norm, color=TKEO_FILL_COLOR)
ax_overlay_off.fill_between(
    time_off,
    0,
    insp_off_norm,
    color=INSP_FILL_COLOR,
    where=(insp_off_norm > tkeo_off_norm),
)
overlap_off = np.minimum(tkeo_off_norm, insp_off_norm)
ax_overlay_off.fill_between(time_off, 0, overlap_off, color=MERGED_COLOR)

mpu.format_axis(
    ax_overlay_off,
    title="Neural Trigger OFF",
    xlabel="",
    ylabel="A.U.",
    xlim=(0, OFF_END - OFF_START),
    ylim=(-0.05, 1.05),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

rect_off_overlay = Rectangle(
    (zoom_off_start_rel, -0.05),
    ZOOM_DURATION,
    1.1,
    linewidth=1.2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_overlay_off.add_patch(rect_off_overlay)

# Row 1, Col 3: TKEO + Insp zoom (Neural Trigger OFF)
ax_overlay_off_zoom = fig.add_subplot(gs_off[0, 2])
ax_overlay_off_zoom.plot(
    time_off_zoom, tkeo_off_zoom_norm, color=TKEO_COLOR, linewidth=1.2
)
ax_overlay_off_zoom.plot(
    time_off_zoom, insp_off_zoom_norm, color=INSP_COLOR, linewidth=1.2
)
ax_overlay_off_zoom.fill_between(
    time_off_zoom, 0, tkeo_off_zoom_norm, color=TKEO_FILL_COLOR
)
ax_overlay_off_zoom.fill_between(
    time_off_zoom,
    0,
    insp_off_zoom_norm,
    color=INSP_FILL_COLOR,
    where=(insp_off_zoom_norm > tkeo_off_zoom_norm),
)
overlap_off_zoom = np.minimum(tkeo_off_zoom_norm, insp_off_zoom_norm)
ax_overlay_off_zoom.fill_between(time_off_zoom, 0, overlap_off_zoom, color=MERGED_COLOR)

mpu.format_axis(
    ax_overlay_off_zoom,
    title="Zoomed",
    xlabel="",
    ylabel="",
    xlim=(0, ZOOM_DURATION),
    ylim=(-0.05, 1.05),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Add entrained highlight box on overlay zoom
rect_entrained_overlay = Rectangle(
    (ENTRAINED_START, -0.05),
    ENTRAINED_DURATION,
    1.1,
    linewidth=1.2,
    edgecolor=ENTRAINED_COLOR,
    facecolor=ENTRAINED_COLOR,
    alpha=0.2,
)
ax_overlay_off_zoom.add_patch(rect_entrained_overlay)

# Add "Entrained" label with background
ax_overlay_off_zoom.text(
    ENTRAINED_START + ENTRAINED_DURATION / 2,
    0.95,
    "Entrained",
    ha="center",
    va="top",
    fontsize=TICK_SIZE,
    color=ENTRAINED_COLOR,
    weight="bold",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor=ENTRAINED_COLOR,
        linewidth=1,
    ),
)

# Row 2, Col 1-2: Filtered EMG (Neural Trigger OFF)
ax_emg_off = fig.add_subplot(gs_off[1, 0:2])
ax_emg_off.plot(time_off, filtered_emg_off, color=EMG_COLOR, linewidth=1.2)

mpu.format_axis(
    ax_emg_off,
    xlabel="",
    ylabel="Amplitude",
    ylim=(-0.30, 0.30),
    xlim=(0, OFF_END - OFF_START),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# apply_scientific_notation(ax_emg_off, SCIENTIFIC_NOTATION_SIZE)

y_min, y_max = ax_emg_off.get_ylim()
rect_off_emg = Rectangle(
    (zoom_off_start_rel, y_min),
    ZOOM_DURATION,
    y_max - y_min,
    linewidth=1.2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_emg_off.add_patch(rect_off_emg)

# Row 2, Col 3: Filtered EMG zoom (Neural Trigger OFF)
ax_emg_off_zoom = fig.add_subplot(gs_off[1, 2])
ax_emg_off_zoom.plot(
    time_off_zoom, filtered_emg_off_zoom, color=EMG_COLOR, linewidth=1.2
)

mpu.format_axis(
    ax_emg_off_zoom,
    xlabel="",
    ylabel="",
    ylim=(-0.30, 0.30),
    xlim=(0, ZOOM_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# apply_scientific_notation(ax_emg_off_zoom, SCIENTIFIC_NOTATION_SIZE)

# Add entrained highlight box on EMG zoom
y_min_zoom, y_max_zoom = ax_emg_off_zoom.get_ylim()
rect_entrained_emg = Rectangle(
    (ENTRAINED_START, y_min_zoom),
    ENTRAINED_DURATION,
    y_max_zoom - y_min_zoom,
    linewidth=1.2,
    edgecolor=ENTRAINED_COLOR,
    facecolor=ENTRAINED_COLOR,
    alpha=0.2,
)
ax_emg_off_zoom.add_patch(rect_entrained_emg)

# Add "Entrained" label with background
ax_emg_off_zoom.text(
    ENTRAINED_START + ENTRAINED_DURATION / 2,
    y_max_zoom * 0.9,
    "Entrained",
    ha="center",
    va="top",
    fontsize=TICK_SIZE,
    color=ENTRAINED_COLOR,
    weight="bold",
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor=ENTRAINED_COLOR,
        linewidth=1,
    ),
)

# ===== NEURAL TRIGGER ON (BOTTOM) =====

# Row 1, Col 1-2: TKEO + Insp overlay (Neural Trigger ON)
ax_overlay_on = fig.add_subplot(gs_on[0, 0:2])
ax_overlay_on.plot(time_on, tkeo_on_norm, color=TKEO_COLOR, linewidth=1.2)
ax_overlay_on.plot(time_on, insp_on_norm, color=INSP_COLOR, linewidth=1.2)
ax_overlay_on.fill_between(time_on, 0, tkeo_on_norm, color=TKEO_FILL_COLOR)
ax_overlay_on.fill_between(
    time_on, 0, insp_on_norm, color=INSP_FILL_COLOR, where=(insp_on_norm > tkeo_on_norm)
)
overlap_on = np.minimum(tkeo_on_norm, insp_on_norm)
ax_overlay_on.fill_between(time_on, 0, overlap_on, color=MERGED_COLOR)

mpu.format_axis(
    ax_overlay_on,
    title="Neural Trigger ON",
    xlabel="",
    ylabel="A.U.",
    xlim=(0, ON_END - ON_START),
    ylim=(-0.05, 1.05),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

rect_on_overlay = Rectangle(
    (zoom_on_start_rel, -0.05),
    ZOOM_DURATION,
    1.1,
    linewidth=1.2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_overlay_on.add_patch(rect_on_overlay)

# Row 1, Col 3: TKEO + Insp zoom (Neural Trigger ON)
ax_overlay_on_zoom = fig.add_subplot(gs_on[0, 2])
ax_overlay_on_zoom.plot(
    time_on_zoom, tkeo_on_zoom_norm, color=TKEO_COLOR, linewidth=1.2
)
ax_overlay_on_zoom.plot(
    time_on_zoom, insp_on_zoom_norm, color=INSP_COLOR, linewidth=1.2
)
ax_overlay_on_zoom.fill_between(
    time_on_zoom, 0, tkeo_on_zoom_norm, color=TKEO_FILL_COLOR
)
ax_overlay_on_zoom.fill_between(
    time_on_zoom,
    0,
    insp_on_zoom_norm,
    color=INSP_FILL_COLOR,
    where=(insp_on_zoom_norm > tkeo_on_zoom_norm),
)
overlap_on_zoom = np.minimum(tkeo_on_zoom_norm, insp_on_zoom_norm)
ax_overlay_on_zoom.fill_between(time_on_zoom, 0, overlap_on_zoom, color=MERGED_COLOR)

mpu.format_axis(
    ax_overlay_on_zoom,
    title="Zoomed",
    xlabel="",
    ylabel="",
    xlim=(0, ZOOM_DURATION),
    ylim=(-0.05, 1.05),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Row 2, Col 1-2: Filtered EMG (Neural Trigger ON)
ax_emg_on = fig.add_subplot(gs_on[1, 0:2])
ax_emg_on.plot(time_on, filtered_emg_on, color=EMG_COLOR, linewidth=1.2)

mpu.format_axis(
    ax_emg_on,
    xlabel="Time [s]",
    ylabel="Amplitude",
    ylim=(-0.30, 0.30),
    xlim=(0, ON_END - ON_START),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# apply_scientific_notation(ax_emg_on, SCIENTIFIC_NOTATION_SIZE)

y_min, y_max = ax_emg_on.get_ylim()
rect_on_emg = Rectangle(
    (zoom_on_start_rel, y_min),
    ZOOM_DURATION,
    y_max - y_min,
    linewidth=1.2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_emg_on.add_patch(rect_on_emg)

# Row 2, Col 3: Filtered EMG zoom (Neural Trigger ON)
ax_emg_on_zoom = fig.add_subplot(gs_on[1, 2])
ax_emg_on_zoom.plot(time_on_zoom, filtered_emg_on_zoom, color=EMG_COLOR, linewidth=1.2)

mpu.format_axis(
    ax_emg_on_zoom,
    xlabel="Time [s]",
    ylabel="",
    ylim=(-0.30, 0.30),
    xlim=(0, ZOOM_DURATION),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# apply_scientific_notation(ax_emg_on_zoom, SCIENTIFIC_NOTATION_SIZE)

# Hide tick labels
all_axes = [
    ax_overlay_off,
    ax_overlay_off_zoom,
    ax_emg_off,
    ax_emg_off_zoom,
    ax_overlay_on,
    ax_overlay_on_zoom,
    ax_emg_on,
    ax_emg_on_zoom,
]

# Hide x-tick labels for all except bottom row
for ax in [
    ax_overlay_off,
    ax_overlay_off_zoom,
    ax_emg_off,
    ax_emg_off_zoom,
    ax_overlay_on,
    ax_overlay_on_zoom,
]:
    ax.set_xticklabels([])

# Hide y-tick labels for zoomed columns
for ax in [ax_overlay_off_zoom, ax_emg_off_zoom, ax_overlay_on_zoom, ax_emg_on_zoom]:
    ax.set_yticklabels([])

# Align y-axis labels
label_x = -0.1
for ax in all_axes:
    ax.yaxis.set_label_coords(label_x, 0.5)

# Add legend
legend_elements = [
    Line2D([0], [0], color=INSP_COLOR, lw=2, label="Insp Pressure"),
    Line2D([0], [0], color=TKEO_COLOR, lw=2, label="EMGenv"),
    Line2D([0], [0], color=EMG_COLOR, lw=2, label="EMGdia"),
]

fig.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(0.90, 1),
    fontsize=LEGEND_SIZE,
    frameon=True,
    fancybox=False,
    shadow=False,
    ncol=3,
)

mpu.add_panel_label(
    ax_overlay_off,
    "A",
    x_offset_factor=0.1,
    y_offset_factor=0.1,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_overlay_off_zoom,
    "B",
    x_offset_factor=0.05,
    y_offset_factor=0.1,
    fontsize=SUBTITLE_SIZE,
)

# Row 3 labels
mpu.add_panel_label(
    ax_overlay_on,
    "C",
    x_offset_factor=0.1,
    y_offset_factor=0.05,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_overlay_on_zoom,
    "D",
    x_offset_factor=0.05,
    y_offset_factor=0.05,
    fontsize=SUBTITLE_SIZE,
)

# Use mpu.finalize_figure
mpu.finalize_figure(
    fig,
    title_y=0.96,
    left_margin=0.08,
    top_margin=0.10,
    title_fontsize=TITLE_SIZE,
)


for ax in all_axes:
    ax.tick_params(axis="both", pad=-3, labelsize=TICK_SIZE)

ax_overlay_off.set_yticks([0, 0.5, 1.0])
ax_overlay_off_zoom.set_yticks([0, 0.5, 1.0])
ax_emg_off.set_yticks([-0.20, 0.0, 0.20])
ax_emg_off_zoom.set_yticks([-0.20, 0.0, 0.20])
ax_emg_on.set_yticks([-0.20, 0.0, 0.20])
ax_emg_on_zoom.set_yticks([-0.20, 0.0, 0.20])

ax_emg_on_zoom.set_xticklabels(np.linspace(8, 12, 5))
ax_emg_on_zoom.tick_params(axis="x", rotation=45)
ax_emg_on.tick_params(axis="x", rotation=45)
# Figure output path
FIGURE_TITLE = "fig09_vent_explained"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")
# Save the figure
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

# %%
