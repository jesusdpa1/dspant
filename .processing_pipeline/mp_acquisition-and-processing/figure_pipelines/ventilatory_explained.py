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
import numpy as np
import polars as pl
import seaborn as sns

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
from dspant_emgproc.processors.activity_detection.double_threshold import (
    create_double_threshold_detector,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_absolute_threshold_detector,
)

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()
# %%
base_path = Path(os.getenv("DATA_DIR"))
data_path = base_path.joinpath(
    r"papers\2025_mp_emg diaphragm acquisition and processing\Sample Ventilator Trace"
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

st_tkeo_processor = create_double_threshold_detector(
    primary_threshold=0.045,
    secondary_threshold=0.1,
    min_event_spacing=0.01,
    min_contraction_duration=0.01,
)

st_insp_processor = create_double_threshold_detector(
    primary_threshold=0.2,
    secondary_threshold=0.25,
    min_event_spacing=0.001,
    min_contraction_duration=0.001,
)

tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)

insp_epochs = st_insp_processor.process(zscore_insp, fs=fs).compute()
insp_epochs = st_insp_processor.to_dataframe(insp_epochs)
# %%
tkeo_fig = plot_multi_channel_data(tkeo_data, fs=fs, time_window=[100, 110])

# %%
"""
time stamps
neural trigger on = 0 to 30
neural trigger off = 365 to 395
neural triggered tidal volume increased = 1090 - 1120
neural triggered tidal volume decreased = 1245 - 1275
"""

# %%
import matplotlib.pyplot as plt
import numpy as np


def plot_onset_detection_results(
    signal,
    onsets_df,
    fs,
    time_window=None,
    figsize=(12, 6),
    title="EMG Onset Detection Results",
    threshold=None,
    signal_label="Signal",
    highlight_color="red",
    show_threshold=True,
    debug_mode=False,  # Add debug mode to visualize the zero-crossing detection
):
    """
    Plot signal with highlighted onset/offset regions.

    Args:
        signal: The signal data (numpy or dask array)
        onsets_df: DataFrame with onset_idx, offset_idx columns
        fs: Sampling frequency in Hz
        time_window: Optional [start, end] in seconds to zoom
        figsize: Figure size tuple
        title: Plot title
        threshold: Optional threshold value to show
        signal_label: Label for the signal
        highlight_color: Color for highlighting detected events
        show_threshold: Whether to show threshold line
        debug_mode: Show additional debug information

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import polars as pl

    # Compute signal if it's a dask array
    if hasattr(signal, "compute"):
        signal = signal.compute()

    # Ensure signal is a 1D array
    if signal.ndim > 1:
        signal = signal[:, 0]  # Take first channel

    # Create time array
    t = np.arange(len(signal)) / fs

    # Create figure
    if debug_mode:
        fig, (ax, ax_debug) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 1.5), sharex=True
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Apply time window if specified
    start_idx = 0
    if time_window is not None:
        start_idx = max(0, int(time_window[0] * fs))
        end_idx = min(len(signal), int(time_window[1] * fs))
        view_signal = signal[start_idx:end_idx]
        view_t = t[start_idx:end_idx]

        # Filter onsets within the window
        window_onsets = onsets_df.filter(
            (pl.col("onset_idx") >= start_idx) & (pl.col("onset_idx") <= end_idx)
        )
    else:
        view_signal = signal
        view_t = t
        window_onsets = onsets_df

    # Plot signal
    ax.plot(view_t, view_signal, label=signal_label, color="blue", linewidth=1.5)

    # Plot threshold if provided
    if threshold is not None and show_threshold:
        ax.axhline(
            y=threshold,
            color="green",
            linestyle="--",
            label=f"Threshold ({threshold:.2f})",
        )

    # Highlight onset-offset regions
    if len(window_onsets) > 0:
        for row in window_onsets.iter_rows(named=True):
            onset_idx = row["onset_idx"]
            offset_idx = row["offset_idx"]

            # Calculate correct time values, accounting for the window offset
            onset_time = (onset_idx - start_idx) / fs + (
                time_window[0] if time_window else 0
            )
            offset_time = (offset_idx - start_idx) / fs + (
                time_window[0] if time_window else 0
            )

            duration = row["duration"]

            # Shade the activation region
            ax.axvspan(onset_time, offset_time, alpha=0.3, color=highlight_color)

            # Mark the onset with a vertical line
            ax.axvline(x=onset_time, color=highlight_color, linestyle="-", linewidth=1)

            # Add annotation
            ax.annotate(
                f"{duration:.2f}s",
                xy=(onset_time, np.max(view_signal)),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=8,
                color=highlight_color,
            )

            # Debug: Mark actual signal value at onset and offset points
            if debug_mode:
                rel_onset_idx = onset_idx - start_idx
                rel_offset_idx = offset_idx - start_idx

                if 0 <= rel_onset_idx < len(view_signal):
                    onset_value = view_signal[rel_onset_idx]
                    ax.plot(onset_time, onset_value, "o", color="red", markersize=6)

                if 0 <= rel_offset_idx < len(view_signal):
                    offset_value = view_signal[rel_offset_idx]
                    ax.plot(
                        offset_time, offset_value, "s", color="purple", markersize=6
                    )

    # Add labels and legend
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add grid for better readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Debug mode: Show the difference signal (signal - threshold)
    if debug_mode and threshold is not None:
        diff_signal = view_signal - threshold
        ax_debug.plot(view_t, diff_signal, label="Signal - Threshold", color="purple")
        ax_debug.axhline(y=0, color="red", linestyle="-", label="Zero line")

        # Mark zero-crossings
        zero_crossings = np.where(np.diff(np.signbit(diff_signal)))[0]
        for zc in zero_crossings:
            ax_debug.axvline(x=view_t[zc], color="green", linestyle="--", alpha=0.5)

        ax_debug.set_ylabel("Difference (Signal - Threshold)")
        ax_debug.legend(loc="upper right")
        ax_debug.grid(True, linestyle="--", alpha=0.7)

    # Tight layout for better appearance
    plt.tight_layout()

    return fig


# %%

plot_ = plot_onset_detection_results(
    zscore_insp,
    insp_epochs,
    fs=fs,
    time_window=[1240, 1250],  # 10-second window
    title="TKEO EMG Signal with Detected Onsets",
    threshold=0.2,
    signal_label="Normalized TKEO EMG",
    highlight_color="red",
)

plot_ = plot_onset_detection_results(
    zscore_tkeo,
    tkeo_epochs,
    fs=fs,
    time_window=[1240, 1250],  # 10-second window
    title="TKEO EMG Signal with Detected Onsets",
    threshold=0.045,
    signal_label="Normalized TKEO EMG",
    highlight_color="red",
)
# %%

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
TITLE_SIZE = 18
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

for row in tkeo_epochs.iter_rows(named=True):
    if zoom_start_idx <= row["onset_idx"] <= zoom_end_idx:
        tkeo_onsets_zoom.append((row["onset_idx"] - zoom_start_idx) / fs)

for row in insp_epochs.iter_rows(named=True):
    if zoom_start_idx <= row["onset_idx"] <= zoom_end_idx:
        insp_onsets_zoom.append((row["onset_idx"] - zoom_start_idx) / fs)

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
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

plt.show()

# %%
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
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Define time range for the plots
start_time = 1240  # seconds 375, 1245
end_time = 1240 + 25  # seconds 400,
zoom_width = 5  # seconds 5

# Calculate the zoom range

# Convert to sample indices
start_idx = int(start_time * fs)
end_idx = int(end_time * fs)


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

# Find all onsets within the zoom window
tkeo_onsets_zoom = []
insp_onsets_zoom = []

for row in tkeo_epochs.iter_rows(named=True):
    if zoom_start_idx <= row["onset_idx"] <= zoom_end_idx:
        tkeo_onsets_zoom.append((row["onset_idx"] - zoom_start_idx) / fs)

for row in insp_epochs.iter_rows(named=True):
    if zoom_start_idx <= row["onset_idx"] <= zoom_end_idx:
        insp_onsets_zoom.append((row["onset_idx"] - zoom_start_idx) / fs)

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
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Define the start and end times in seconds
start_time = 0
end_time = 2000  # adjust this value as needed

# Convert the start and end times to sample indices
start_idx = int(start_time * fs)
end_idx = int(end_time * fs)

# Filter the tkeo_epochs DataFrame
filtered_epochs = tkeo_epochs.filter(
    (pl.col("onset_idx") >= start_idx) & (pl.col("onset_idx") <= end_idx)
)

# Calculate the onset_idx_diff and onset_idx_hz for the filtered epochs
filtered_onset_idx_diff = np.diff(filtered_epochs["onset_idx"].to_numpy()) / fs
filtered_onset_idx_hz = 1 / filtered_onset_idx_diff

# Calculate the moving average over a 1-second window
window_size = int(fs // 500)  # approximate window size in samples
if window_size == 0:
    window_size = 1
smoothed_diff = np.convolve(
    filtered_onset_idx_hz, np.ones(window_size) / window_size, mode="same"
)

# Plot the original and smoothed time differences
plt.figure(figsize=(10, 6))
plt.plot(filtered_onset_idx_hz, label="Original")
plt.plot(smoothed_diff, label="Smoothed")
plt.xlabel("Onset Index")
plt.ylabel("hz")
plt.title("Time Difference between Consecutive Onsets")
plt.legend()
plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Calculate the angles for the polar plot
angles = np.linspace(0, 2 * np.pi, len(filtered_onset_idx_hz), endpoint=False)

# Create a color map based on time
time_array = np.arange(len(filtered_onset_idx_hz))
cmap = sns.color_palette("viridis", as_cmap=True)
norm = plt.Normalize(time_array.min(), time_array.max())

# Plot the breathing frequency in a polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for i in range(len(angles) - 1):
    ax.plot(
        angles[i : i + 2],
        filtered_onset_idx_hz[i : i + 2],
        color=cmap(norm(time_array[i])),
    )
ax.set_title("Breathing Frequency")
plt.show()

# Plot the smoothed breathing frequency in a polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
for i in range(len(angles) - 1):
    ax.plot(
        angles[i : i + 2], smoothed_diff[i : i + 2], color=cmap(norm(time_array[i]))
    )
ax.set_title("Smoothed Breathing Frequency")
plt.show()
# %%


multi_data = da.concatenate(
    [filtered_emg_l[:49949994, :], tkeo_data.compute(), filtered_insp[:49949994, :]],
    axis=1,
)

# %%
multichannel_fig = plot_multi_channel_data(multi_data, fs=fs, time_window=[1000, 1400])

# %%
import numpy as np

# Calculate the onset times
onset_times = tkeo_epochs["onset_idx"].to_numpy() / fs

# Calculate the time differences between onsets
time_diffs = np.diff(onset_times)

# Calculate the instantaneous frequency (in Hz)
instantaneous_freq = 1 / time_diffs

# Create an array of time points that correspond to the original signal
time_array = np.arange(len(filtered_emg_l)) / fs

# Create an array to store the interpolated frequency values
interpolated_freq = np.zeros(len(time_array), dtype=np.float32)

# Interpolate the frequency values
for i in range(len(onset_times) - 1):
    start_idx = int(onset_times[i] * fs)
    end_idx = int(onset_times[i + 1] * fs)
    interpolated_freq[start_idx:end_idx] = instantaneous_freq[i]

# Handle the last segment
start_idx = int(onset_times[-1] * fs)
interpolated_freq[start_idx:] = instantaneous_freq[-1]

# Plot the interpolated frequency waveform
plt.figure(figsize=(10, 6))
plt.plot(time_array, interpolated_freq)
# plt.plot(time_array, a)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Interpolated Frequency Waveform")
plt.show()
# %%

# %%
a = da.array(interpolated_freq[:, np.newaxis])

# %%
multi_data = da.concatenate(
    [
        a[:49949994, :],
        filtered_emg_l[:49949994, :],
        tkeo_data.compute(),
        filtered_insp[:49949994, :],
    ],
    axis=1,
)

# %%
multichannel_fig = plot_multi_channel_data(multi_data, fs=fs, time_window=[900, 1400])

# %%
