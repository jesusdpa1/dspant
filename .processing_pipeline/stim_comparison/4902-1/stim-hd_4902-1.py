"""
Functions to showcase differneces between stim with different electrodes
Data: 25-02-26_9881-2_testSubject_topoMapping
Analysis: MEPs EMG
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Union

import dask.array as da
import dotenv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from dspant.engine import create_processing_node
from dspant.nodes import EpocNode, StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.extractors.template_extractor import TemplateExtractor
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)


def ls_(path_: Union[Path, str]):
    """
    List directories within the given path.

    Args:
    - path_: The path to list directories from. Can be a string or a Path object.
    Returns:
    - A list of Path objects representing directories.
    """
    if isinstance(path_, str):
        path_ = Path(path_)

    # Ensure path_ is a directory
    if not path_.is_dir():
        raise ValueError(f"{path_} is not a directory")

    # Get directories
    path_list = [p for p in path_.iterdir() if p.is_dir()]
    for i, path_val in enumerate(path_list):
        print(f"[{i}] {path_val.name}")
    return path_list


sns.set_theme(style="darkgrid")
dotenv.load_dotenv()

# %%
base_path = Path(os.getenv("DATA_DIR"))
drv_path = base_path.joinpath(
    r"topoMapping\25-03-26_4902-1_testSubject_topoMapping\drv"
)

dir_list = ls_(drv_path)


# %%
PATH_SELECT = 4
data_path = dir_list[PATH_SELECT]
emg_path = data_path.joinpath(r"RawG.ant")
hd_path = data_path.joinpath(r"HDEG.ant")
stim_path = data_path.joinpath(r"MonA.ant")
epoch_stim_path = data_path.joinpath(r"AmpA.ant")
chnA_path = data_path.joinpath(r"ChnA.ant")
# insp_path = data_path.joinpath(r"insp.ant")
# %%
# Load EMG data
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

stream_hd = StreamNode(str(hd_path))
stream_hd.load_metadata()
stream_hd.load_data()
# Print stream_emg summary
stream_hd.summarize()


stream_stim = StreamNode(str(stim_path))
stream_stim.load_metadata()
stream_stim.load_data()
# Print stream_emg summary
stream_stim.summarize()


DATA_LENGTH = int(
    min(
        stream_emg.data.shape[0],
        stream_hd.data.shape[0],
        stream_stim.data.shape[0],
    )
)
# %%
epoch_stim = EpocNode(data_path=str(epoch_stim_path))
epoch_stim.load_metadata()
epoch_stim.load_data()
# Print stream_emg summary
epoch_stim.summarize()

chnA_stim = EpocNode(data_path=str(chnA_path))
chnA_stim.load_metadata()
chnA_stim.load_data()
# Print stream_emg summary
chnA_stim.summarize()
# %%
# rename columns before joining
# epoch renamining
# epoch renaming
column_mapping_epoch = {
    "data": "current_amplitude",
    "onset": "onset_amp",
    "offset": "offset_amp",
}
epoch_stim.data = epoch_stim.data.rename(column_mapping_epoch)

# channel renaming
column_mapping_channel = {
    "data": "channel",
    "onset": "onset_chn",
    "offset": "offset_chn",
}
chnA_stim.data = chnA_stim.data.rename(column_mapping_channel)

# Now you can join them on offset_amp and offset_chn
final_epochs = pl.concat([epoch_stim.data, chnA_stim.data], how="horizontal")

# %%
# Process EMG
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(30, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)

bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=fs, order=5)
notch_filter_hd = create_notch_filter(60, q=60, fs=fs)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

notch_processor_hd = FilterProcessor(
    filter_func=notch_filter_hd.get_filter_function(), overlap_samples=40
)
bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)
# %%
# spatial
cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()
# %%
processor_emg = create_processing_node(stream_emg)
processor_hd = create_processing_node(stream_hd)

# Add processors to the processing node
processor_emg.add_processor(
    [
        bandpass_processor,
        notch_processor,
    ],
    group="filters",
)

processor_hd.add_processor(
    [
        bandpass_processor_hd,
        notch_processor_hd,
    ],
    group="filters",
)
processor_hd.add_processor(
    [
        cmr_processor,
        whiten_processor,
    ],
    group="spatial",
)
# Apply filters and plot results
filtered_emg = processor_emg.process(group=["filters"]).persist()
filtered_hd = processor_hd.process().persist()
# %%

filtered_stim_data = da.concatenate(
    [
        stream_stim.data[:DATA_LENGTH, :1],
        filtered_emg[:DATA_LENGTH, :],
    ],
    axis=1,
)

hd_stim_data = da.concatenate(
    [
        stream_stim.data[:DATA_LENGTH, :1],
        filtered_hd[:DATA_LENGTH, :],
    ],
    axis=1,
)
# %%
# stim_chn[10] = 60-250
#
# Get first and last values for each channel
result = final_epochs.group_by("channel").agg(
    [
        pl.col("onset_amp").first().alias("start"),
        pl.col("onset_amp").last().alias("end"),
    ]
)

print(result)
print(final_epochs["channel"].unique(maintain_order=True))

# %%
STIM_CHANNEL_SELECT = 10

times_filtered = result.filter((pl.col("channel") == STIM_CHANNEL_SELECT))


# START_PLOT = int(times_filtered["start"][0]) - 50
# END_PLOT = int(times_filtered["end"][0]) + 50

# multichannel_fig = plot_multi_channel_data(
#     filtered_stim_data, fs=fs, time_window=[START_PLOT, END_PLOT]
# )
START_PLOT = 1277.3
END_PLOT = 1277.7
emg_stim_fig = plot_multi_channel_data(
    filtered_stim_data, fs=fs, time_window=[START_PLOT, END_PLOT]
)

hd_stim_fig = plot_multi_channel_data(
    hd_stim_data,
    fs=fs,
    time_window=[START_PLOT, END_PLOT],
    figsize=(15, 15),
    color_mode="unique",
    color="darkorange",
)
# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=20)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()

# %%
tkeo_fig = plot_multi_channel_data(
    tkeo_data,
    fs=fs,
    time_window=[00, 300],
)
# %%

data_organized = da.concatenate(
    [
        stream_stim.data[:DATA_LENGTH, :1].compute(),
        tkeo_data[:DATA_LENGTH, :].compute(),
        filtered_emg[:DATA_LENGTH, :].compute(),
    ],
    axis=1,
)
# %%
agg_fig = plot_multi_channel_data(data_organized, fs=fs, time_window=[250, 300])

# %%
emg_waveform_analyzer = WaveformExtractor(filtered_emg[:DATA_LENGTH, :].compute(), fs)
template_analyzer = TemplateExtractor()

# %%
STIM_AMPLITUDE = -250
CHANNEL = 3

PRE_SAMPLES = int(fs * 0.02)
POST_SAMPLES = int(fs * 0.02)

filter_onsets = final_epochs.filter(
    (pl.col("current_amplitude") == STIM_AMPLITUDE) & (pl.col("channel") == CHANNEL)
)["onset_amp"]

print(filter_onsets)
# %%
waveforms_data = emg_waveform_analyzer.extract_waveforms(
    spike_times=filter_onsets, pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES
)

# %%
template_data = template_analyzer.extract_template_distributions(waveforms_data[0])

# %%
plt.plot(template_data["template_mean"])


# %%

# Pre-calculate all templates and statistics
print("Extracting templates...")
template_results = {}
row_max_abs_values = defaultdict(
    lambda: [0, 0]
)  # Dictionary to store max abs values per amplitude (row)

# Get unique stimulation amplitudes and channels from the final_epochs dataframe
unique_amplitudes = final_epochs["current_amplitude"].unique().sort()
unique_channels = final_epochs["channel"].unique().sort()

# Precompute all templates
for amplitude in unique_amplitudes:
    template_results[amplitude] = {}

    for stim_channel in unique_channels:
        # Get the evoked responses for this combination
        onsets = final_epochs.filter(
            (pl.col("current_amplitude") == amplitude)
            & (pl.col("channel") == stim_channel)
        )["onset_amp"]

        # Skip if no data for this combination
        if len(onsets) == 0:
            template_results[amplitude][stim_channel] = None
            continue

        try:
            # Extract waveforms
            waveforms_data = emg_waveform_analyzer.extract_waveforms(
                spike_times=onsets, pre_samples=PRE_SAMPLES, post_samples=POST_SAMPLES
            )

            # Extract template statistics
            template_data = template_analyzer.extract_template_distributions(
                waveforms_data[0]
            )

            # Store the template data
            template_results[amplitude][stim_channel] = {
                "template_data": template_data,
                "n_waveforms": len(onsets),
            }

            # Calculate and store waveform time axis
            waveform_length = template_data["template_mean"].shape[0]
            waveform_time = np.linspace(
                -PRE_SAMPLES / fs, POST_SAMPLES / fs, waveform_length
            )
            waveform_time_ms = waveform_time * 1000

            template_results[amplitude][stim_channel]["time_ms"] = waveform_time_ms

            # Update row-specific max for each recording channel
            for ch_idx in range(template_data["n_channels"]):
                mean_waveform = template_data["template_mean"][:, ch_idx]
                std_waveform = template_data["template_std"][:, ch_idx]

                # Find max absolute value including standard deviation
                max_abs = max(
                    abs(np.max(mean_waveform + std_waveform)),
                    abs(np.min(mean_waveform - std_waveform)),
                )

                row_max_abs_values[amplitude][ch_idx] = max(
                    row_max_abs_values[amplitude][ch_idx], max_abs
                )

        except Exception as e:
            template_results[amplitude][stim_channel] = {"error": str(e)}
            print(
                f"Error processing amplitude {amplitude}, channel {stim_channel}: {str(e)}"
            )


# %%
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


# Helper function to adjust color brightness
def adjust_color_brightness(color, factor):
    """Adjust brightness of a color by a factor (0-1)"""
    # Convert to HSV color space
    c = mcolors.rgb_to_hsv(color[:3])

    # Adjust value (brightness)
    c[2] = min(c[2] * factor, 1.0)

    # Convert back to RGB
    return mcolors.hsv_to_rgb(c)


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
SUBTITLE_SIZE = 14  # Reduced for more compact layout
AXIS_LABEL_SIZE = 12  # Reduced for more compact layout
TICK_SIZE = 10  # Reduced for more compact layout
CAPTION_SIZE = 11  # Reduced for more compact layout

# NEW: Define plot orientation - easily switch between layouts
# 'amplitude_as_rows': amplitude on rows (stim channels on columns)
# 'amplitude_as_columns': amplitude on columns (stim channels on rows)
PLOT_ORIENTATION = "amplitude_as_columns"  # Change to 'amplitude_as_rows' if needed

# Fix the y-axis scaling issue - recalculate max abs values for EACH stim channel
# New nested dictionary structure: {amplitude: {stim_channel: [ch1_max, ch2_max]}}
row_max_abs_values = defaultdict(lambda: defaultdict(lambda: [0, 0]))

for amplitude in unique_amplitudes:
    for stim_channel in unique_channels:
        template_result = template_results[amplitude][stim_channel]

        if template_result is None or "error" in template_result:
            continue

        template_data = template_result["template_data"]

        for ch_idx in range(template_data["n_channels"]):
            mean_waveform = template_data["template_mean"][:, ch_idx]
            std_waveform = template_data["template_std"][:, ch_idx]

            # Find the actual max value in this waveform (mean ± std)
            max_abs = max(
                abs(np.max(mean_waveform + std_waveform)),
                abs(np.min(mean_waveform - std_waveform)),
            )

            # Update the max abs value for this amplitude, stim channel and recording channel
            row_max_abs_values[amplitude][stim_channel][ch_idx] = max(
                row_max_abs_values[amplitude][stim_channel][ch_idx], max_abs
            )

# Add padding to max values - using less padding to keep scale closer to actual data
for amp in row_max_abs_values:
    for stim_ch in row_max_abs_values[amp]:
        for i in range(len(row_max_abs_values[amp][stim_ch])):
            row_max_abs_values[amp][stim_ch][i] *= 1.05  # Just 5% padding

# Create a color map for amplitudes/channels based on orientation
# Using base colors for the data lines and error bands
if PLOT_ORIENTATION == "amplitude_as_rows":
    row_colors = {
        amp: palette[idx % len(palette)] for idx, amp in enumerate(unique_amplitudes)
    }
    col_colors = {
        ch: sns.color_palette("Blues", n_colors=len(unique_channels))[idx]
        for idx, ch in enumerate(unique_channels)
    }
else:  # amplitude_as_columns
    col_colors = {
        amp: palette[idx % len(palette)] for idx, amp in enumerate(unique_amplitudes)
    }
    row_colors = {
        ch: sns.color_palette("Blues", n_colors=len(unique_channels))[idx]
        for idx, ch in enumerate(unique_channels)
    }

# For labels, always use black instead of the color (as requested)
label_color = "black"

print("Template extraction complete. Starting plotting...")

# ============== PLOTTING PHASE ==============

# Create figure dimensions based on orientation
if PLOT_ORIENTATION == "amplitude_as_rows":
    n_rows = len(unique_amplitudes)
    n_cols = len(unique_channels)
    row_items = unique_amplitudes
    col_items = unique_channels
    # Figure size adjusted for orientation
    figsize = (len(unique_channels) * 5, len(unique_amplitudes) * 2.5)
else:  # amplitude_as_columns
    n_rows = len(unique_channels)
    n_cols = len(unique_amplitudes)
    row_items = unique_channels
    col_items = unique_amplitudes
    # Figure size adjusted for orientation - make columns narrower
    figsize = (len(unique_amplitudes) * 4, len(unique_channels) * 3)

# Create figure with GridSpec - REDUCED HEIGHT for tighter vertical spacing
fig = plt.figure(figsize=figsize)

# Outer grid: determined by orientation
gs_outer = GridSpec(
    n_rows, n_cols, figure=fig, wspace=0.25, hspace=0.15
)  # Even tighter spacing

# Create the plots using precomputed data
for row_idx, row_item in enumerate(row_items):
    for col_idx, col_item in enumerate(col_items):
        # Get amplitude and stim channel based on orientation
        if PLOT_ORIENTATION == "amplitude_as_rows":
            amplitude = row_item
            stim_channel = col_item
        else:  # amplitude_as_columns
            amplitude = col_item
            stim_channel = row_item

        # Get the template data for this combination
        template_result = template_results[amplitude][stim_channel]

        # Skip if no data for this combination
        if template_result is None:
            # Create an empty subplot with message
            ax = fig.add_subplot(gs_outer[row_idx, col_idx])
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                fontsize=TICK_SIZE,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Check if there was an error
        if "error" in template_result:
            # Create an empty subplot with error message
            ax = fig.add_subplot(gs_outer[row_idx, col_idx])
            ax.text(
                0.5,
                0.5,
                f"Error: {template_result['error']}",
                ha="center",
                va="center",
                fontsize=TICK_SIZE - 2,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Get the template data and time axis
        template_data = template_result["template_data"]
        waveform_time_ms = template_result["time_ms"]
        n_waveforms = template_result["n_waveforms"]

        # Calculate time limits
        time_min = waveform_time_ms[0]
        time_max = waveform_time_ms[-1]

        # Create inner grid for this cell (1 row, 2 columns for the two recording channels)
        # With equal width/height ratio for each subplot and NO SPACE between them
        inner_gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_outer[row_idx, col_idx], wspace=0
        )

        # Get color based on the orientation - for the data itself
        if PLOT_ORIENTATION == "amplitude_as_rows":
            base_color = row_colors[amplitude]
        else:  # amplitude_as_columns
            base_color = col_colors[amplitude]

        # Get the max values for THIS specific combination of amplitude and stim channel
        y_limit_ch1 = row_max_abs_values[amplitude][stim_channel][0]
        y_limit_ch2 = row_max_abs_values[amplitude][stim_channel][1]

        # Use the same max limit for both channels (the larger of the two)
        # to ensure the pair shares the same y-axis scale
        pair_y_limit = max(y_limit_ch1, y_limit_ch2)

        # Create subplots for each recording channel
        shared_y_axis = None  # Will store the first axis to share with the second

        for ch_idx in range(template_data["n_channels"]):
            # For the second channel (right), share y-axis with the first channel (left)
            if ch_idx == 0:
                ax = fig.add_subplot(inner_gs[0, ch_idx])
                shared_y_axis = ax
            else:
                ax = fig.add_subplot(inner_gs[0, ch_idx], sharey=shared_y_axis)

            # Plot mean waveform
            mean_waveform = template_data["template_mean"][:, ch_idx]
            std_waveform = template_data["template_std"][:, ch_idx]

            # Adjust color darkness based on channel
            channel_color = adjust_color_brightness(base_color, 1.0 - 0.3 * ch_idx)

            # Plot with shaded error regions
            ax.plot(
                waveform_time_ms, mean_waveform, color=channel_color, linewidth=1.5
            )  # Thinner lines
            ax.fill_between(
                waveform_time_ms,
                mean_waveform - std_waveform,
                mean_waveform + std_waveform,
                color=channel_color,
                alpha=0.25,  # Reduced for cleaner appearance
            )

            # Add vertical line at stimulus onset (time=0)
            ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)

            # Set x-axis to the full time range
            ax.set_xlim(time_min, time_max)

            # Use the SAME y-axis limit for BOTH channels in this pair
            # This ensures the pair shares the same scale
            ax.set_ylim(-pair_y_limit, pair_y_limit)

            # Add horizontal line at y=0
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.2)

            # Formatting - COMPACT LABELS AND TICKS
            if row_idx == n_rows - 1:  # Only bottom row shows x labels
                ax.set_xlabel("Time (ms)", fontsize=AXIS_LABEL_SIZE)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            # Y-axis settings:
            # - Left channel shows y-axis
            # - Right channel hides y-axis
            if ch_idx == 0:  # Left channel
                ax.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE)
            else:  # Right channel
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)  # Hide y tick labels on right plot
                ax.tick_params(left=False)  # Hide y ticks on right plot

            ax.tick_params(labelsize=TICK_SIZE)

            # Force square aspect ratio for the data box
            ax.set_box_aspect(1)

            # Add simplified channel title only to the top row
            if row_idx == 0:
                ax.set_title(
                    f"CH{ch_idx + 1:02d}", fontsize=SUBTITLE_SIZE, weight="bold"
                )

            # Add number of trials to the subplot - smaller and less intrusive
            ax.text(
                0.05,
                0.95,
                f"n={n_waveforms}",
                transform=ax.transAxes,
                fontsize=TICK_SIZE - 2,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

    # Add row label based on orientation
    row_label_ax = fig.add_subplot(gs_outer[row_idx, :], frameon=False)
    row_label_ax.set_xticks([])
    row_label_ax.set_yticks([])

    # Prepare label based on orientation
    if PLOT_ORIENTATION == "amplitude_as_rows":
        amp = row_item

        # Calculate average y-axis range for this row to show in the label
        avg_ch1_range = np.mean(
            [row_max_abs_values[amp][sc][0] for sc in row_max_abs_values[amp]]
        )
        avg_ch2_range = np.mean(
            [row_max_abs_values[amp][sc][1] for sc in row_max_abs_values[amp]]
        )

        # Format label for amplitude
        label_text = f"{amp} μA\n±{avg_ch1_range:.2e}/±{avg_ch2_range:.2e}"
    else:  # amplitude_as_columns
        # Format label for stim channel
        stim_ch = row_item
        label_text = f"Stim Channel {stim_ch}"

    # Add the row label - now ALWAYS BLACK AND BOLD
    row_label_ax.text(
        -0.05,
        0.5,
        label_text,
        ha="right",
        va="center",
        fontsize=SUBTITLE_SIZE - 1,
        weight="bold",
        color=label_color,  # Use black for all labels
        transform=row_label_ax.transAxes,
    )

# Add column labels based on orientation
for col_idx, col_item in enumerate(col_items):
    col_label_ax = fig.add_subplot(gs_outer[:, col_idx], frameon=False)
    col_label_ax.set_xticks([])
    col_label_ax.set_yticks([])

    # Prepare label based on orientation
    if PLOT_ORIENTATION == "amplitude_as_rows":
        # Format label for stim channel
        stim_ch = col_item
        label_text = f"Stim Channel {stim_ch}"
    else:  # amplitude_as_columns
        # Format label for amplitude
        amp = col_item
        label_text = f"{amp} μA"

    # Add the column label - now ALWAYS BLACK AND BOLD
    col_label_ax.text(
        0.5,
        1.01,
        label_text,
        ha="center",
        va="bottom",
        fontsize=SUBTITLE_SIZE,
        weight="bold",
        color=label_color,  # Use black for all labels
        transform=col_label_ax.transAxes,
    )

# Add a legend showing colors based on orientation
legend_ax = fig.add_subplot(111, frameon=False)
legend_ax.set_xticks([])
legend_ax.set_yticks([])

# Create legend handles based on orientation
handles = []
labels = []

# Add orientation-specific title
if PLOT_ORIENTATION == "amplitude_as_rows":
    title = "Evoked Responses (Amplitudes as Rows, Channels as Columns)"
else:  # amplitude_as_columns
    title = "Evoked Responses (Amplitudes as Columns, Channels as Rows)"

plt.suptitle(
    title,
    fontsize=TITLE_SIZE,
    fontweight="bold",
    y=0.98,
)

# Use tight_layout with MINIMAL PADDING
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.1, w_pad=0.1)

print(f"Plot complete with orientation: {PLOT_ORIENTATION}")
plt.show()

# %%
