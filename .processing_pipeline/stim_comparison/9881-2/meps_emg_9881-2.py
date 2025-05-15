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
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()

# %%
base_path = Path(os.getenv("DATA_DIR"))
data_path = base_path.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_15-25-33_meps"
)
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

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
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

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
processor_emg = create_processing_node(stream_emg)
# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters and plot results
filtered_emg = processor_emg.process(group=["filters"]).persist()

# %%
filtered_data = da.concatenate(
    [filtered_emg[:69156864, 1:2], stream_stim.data[:, :1]], axis=1
)

# %%

multichannel_fig = plot_multi_channel_data(filtered_data, fs=fs, time_window=[400, 440])

# %%

tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=20)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()

# %%
tkeo_fig = plot_multi_channel_data(tkeo_data, fs=fs, time_window=[100, 110])
# %%

data_organized = da.concatenate(
    [
        stream_stim.data[:, :1].compute(),
        tkeo_data[:69156864, 1:2].compute(),
        filtered_emg[:69156864, 1:2].compute(),
    ],
    axis=1,
)
# %%
agg_fig = plot_multi_channel_data(data_organized, fs=fs, time_window=[400, 800])
# %%
emg_waveform_analyzer = WaveformExtractor(filtered_emg[:69156864, :], fs)
template_analyzer = TemplateExtractor()

# %%
STIM_AMPLITUDE = -750
CHANNEL = 4

PRE_SAMPLES = int(fs * 0.005)
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

# Create a color map for amplitudes to ensure consistent coloring
amplitude_colors = {}
for idx, amp in enumerate(unique_amplitudes):
    amplitude_colors[amp] = palette[idx % len(palette)]

print("Template extraction complete. Starting plotting...")

# ============== PLOTTING PHASE ==============

# Create figure dimensions
n_amplitudes = len(unique_amplitudes)
n_channels = len(unique_channels)

# Create figure with GridSpec - REDUCED HEIGHT for tighter vertical spacing
fig = plt.figure(
    figsize=(n_channels * 5, n_amplitudes * 2.5)
)  # Even more reduced height

# Outer grid: amplitudes as rows, stim channels as columns
# REDUCED VERTICAL SPACING
gs_outer = GridSpec(
    n_amplitudes, n_channels, figure=fig, wspace=0.25, hspace=0.15
)  # Even tighter spacing

# Create the plots using precomputed data
for row_idx, amplitude in enumerate(unique_amplitudes):
    for col_idx, stim_channel in enumerate(unique_channels):
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

        # Get the amplitude color from our mapping
        base_color = amplitude_colors[amplitude]

        # Get the max values for THIS specific combination of amplitude and stim channel
        # This is the key change - using stim channel specific y-axis limits
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
            if row_idx == n_amplitudes - 1:  # Only bottom row shows x labels
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

    # Add row label for the amplitude (now on the left)
    row_label_ax = fig.add_subplot(gs_outer[row_idx, :], frameon=False)
    row_label_ax.set_xticks([])
    row_label_ax.set_yticks([])

    # Get the color for this amplitude and use it for the row label
    amp_color = amplitude_colors[amplitude]

    # Calculate average y-axis range for this row to show in the label
    # Get the average of max values across all stim channels for this amplitude
    avg_ch1_range = np.mean(
        [row_max_abs_values[amplitude][sc][0] for sc in row_max_abs_values[amplitude]]
    )
    avg_ch2_range = np.mean(
        [row_max_abs_values[amplitude][sc][1] for sc in row_max_abs_values[amplitude]]
    )

    # Format the label to match the example image
    row_label_ax.text(
        -0.05,
        0.5,
        f"{amplitude} μA\n±{avg_ch1_range:.2e}/±{avg_ch2_range:.2e}",
        ha="right",
        va="center",
        fontsize=SUBTITLE_SIZE - 1,  # Smaller font
        weight="bold",
        color=amp_color,
        transform=row_label_ax.transAxes,
    )

# Add column labels for channels at the top - MOVED CLOSER TO PLOTS
for col_idx, stim_channel in enumerate(unique_channels):
    col_label_ax = fig.add_subplot(gs_outer[:, col_idx], frameon=False)
    col_label_ax.set_xticks([])
    col_label_ax.set_yticks([])
    col_label_ax.text(
        0.5,
        1.01,
        f"Stim Channel {stim_channel}",
        ha="center",
        va="bottom",
        fontsize=SUBTITLE_SIZE,
        weight="bold",
        transform=col_label_ax.transAxes,
    )

# Add a legend showing the amplitude colors - MOVED TO SAVE VERTICAL SPACE
legend_ax = fig.add_subplot(111, frameon=False)
legend_ax.set_xticks([])
legend_ax.set_yticks([])

# Create legend handles
handles = []
labels = []
for amp, color in amplitude_colors.items():
    handles.append(plt.Line2D([0], [0], color=color, lw=3))
    labels.append(f"{amp} μA")

# Add overall title
plt.suptitle(
    "Evoked Responses by Stimulation Amplitude and Channel",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    y=0.98,
)

# Use tight_layout with MINIMAL PADDING
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.1, w_pad=0.1)

print("Plot complete.")
plt.show()

# %%
