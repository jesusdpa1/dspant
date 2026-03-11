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
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.extractors.template_extractor import TemplateExtractor
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()
# %%

# Data loading configuration
data_dir = Path(os.getenv("DATA_DIR"))
hd_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/HDEG.ant")
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/RawG.ant")

sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)


stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()

stream_hd = StreamNode(str(hd_path))
stream_hd.load_metadata()
stream_hd.load_data()
# %%

sorter_data = load_kilosort(sorter_path, load_templates=True)

# %%
# Get sampling rate
FS = stream_hd.fs

# %%
notch_filter = create_notch_filter(60, q=10, fs=FS)
bandpass_filter_hd = create_bandpass_filter(300, 6000, fs=FS, order=4)
bandpass_filter_emg = create_bandpass_filter(20, 2000, fs=FS, order=4)
# Create pre-processing functions
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)

bandpass_processor_hd = FilterProcessor(
    filter_func=bandpass_filter_hd.get_filter_function(), overlap_samples=40
)

bandpass_processor_emg = FilterProcessor(
    filter_func=bandpass_filter_emg.get_filter_function(), overlap_samples=40
)

cmr_processor = create_cmr_processor_rs()
whiten_processor = create_whitening_processor_rs()
# %%
# Create and apply filters for HD data
processor_hd = create_processing_node(stream_hd)
processor_emg = create_processing_node(stream_emg)
# Add processors
processor_hd.add_processor(
    [notch_processor, bandpass_processor_hd],
    group="filters",
)

processor_emg.add_processor(
    [notch_processor, bandpass_processor_emg],
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
filtered_emg = processor_emg.process(group=["filters"]).persist()[
    START_TIME:END_TIME, :
]

# %%

cmr_hd_small = cmr_processor.process(
    filtered_hd,
    fs=FS,
).persist()
# %%
whiten_hd_small = whiten_processor.process(cmr_hd_small, fs=FS).persist()

# %%

# %%
# Fresh start - EMG and Neural Template Visualization
# Fresh start - EMG and Neural Template Visualization
import matplotlib.pyplot as plt
import numpy as np

# Configuration
SELECTED_UNITS = [21, 20, 47, 22, 6, 15, 8]  # 7 templates
TEMPLATE_COLOR = "#1A0083"  # Blue for templates
EMG_COLOR = "#830000"  # Red for EMG
RECORDING_START_TIME = 10.0  # Start time in seconds
WINDOW_DURATION = 1.0  # 1 second window
ROW_SPACING = 3.0  # Vertical spacing between rows
FIGURE_SIZE = (15, 12)

# Template extraction parameters
NUM_SPIKES_TO_TEMPLATE = 500
PRE_SAMPLES_MS = 2.0
POST_SAMPLES_MS = 2.0
MIN_SPIKES_THRESHOLD = 50


def extract_template_and_reconstruct(unit_id, start_time_s, duration_s):
    """Extract template and reconstruct signal over time window"""
    # Get unit's primary channel
    templates = sorter_data.templates_data["templates"]
    unit_template = templates[unit_id]
    max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
    unit_channel = max_channel - 1

    # Get all spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    if len(all_spike_times) < MIN_SPIKES_THRESHOLD:
        return None

    # Extract template from subset of spikes
    if len(all_spike_times) > NUM_SPIKES_TO_TEMPLATE:
        np.random.seed(42)
        sampled_indices = np.random.choice(
            len(all_spike_times), NUM_SPIKES_TO_TEMPLATE, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
    else:
        sampled_spikes = all_spike_times

    # Extract template waveforms
    pre_samples = int(PRE_SAMPLES_MS * FS / 1000)
    post_samples = int(POST_SAMPLES_MS * FS / 1000)

    waveforms = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whiten_hd_small.shape[0]:
            waveform = whiten_hd_small[start_idx:end_idx, unit_channel].compute()
            waveforms.append(waveform)

    if not waveforms:
        return None

    # Create normalized template
    waveforms_array = np.array(waveforms)
    template_mean = np.mean(waveforms_array, axis=0)
    template_normalized = (template_mean - np.mean(template_mean)) / np.std(
        template_mean
    )

    # Reconstruct signal in time window
    start_sample = int(start_time_s * FS)
    end_sample = int((start_time_s + duration_s) * FS)
    window_length = end_sample - start_sample

    # Create empty signal array
    reconstructed_signal = np.zeros(window_length)

    # Find spikes in window
    spikes_in_window = all_spike_times[
        (all_spike_times >= start_sample) & (all_spike_times < end_sample)
    ]

    # Place template at each spike location
    for spike_time in spikes_in_window:
        spike_pos = spike_time - start_sample
        template_start = spike_pos - pre_samples
        template_end = spike_pos + post_samples + 1

        if template_start >= 0 and template_end <= window_length:
            reconstructed_signal[template_start:template_end] += template_normalized
        elif template_start < window_length and template_end > 0:
            # Handle partial overlap
            window_start = max(0, template_start)
            window_end = min(window_length, template_end)
            template_offset_start = max(0, -template_start)
            template_offset_end = template_offset_start + (window_end - window_start)

            reconstructed_signal[window_start:window_end] += template_normalized[
                template_offset_start:template_offset_end
            ]

    # Create time array
    time_s = np.arange(window_length) / FS

    return {
        "reconstructed_signal": reconstructed_signal,
        "time_s": time_s,
        "unit_id": unit_id,
        "n_spikes_in_window": len(spikes_in_window),
    }


# Extract neural data
print(f"Processing units: {SELECTED_UNITS}")
reconstructed_data = {}

for unit_id in SELECTED_UNITS:
    result = extract_template_and_reconstruct(
        unit_id, RECORDING_START_TIME, WINDOW_DURATION
    )
    if result:
        reconstructed_data[unit_id] = result
        print(f"Unit {unit_id}: {result['n_spikes_in_window']} spikes")

# Extract EMG data
start_sample = int(RECORDING_START_TIME * FS)
end_sample = int((RECORDING_START_TIME + WINDOW_DURATION) * FS)

emg_data = filtered_emg[start_sample:end_sample, 0].compute()  # First channel only
emg_time = np.arange(len(emg_data)) / FS

# Normalize EMG - better normalization
emg_mean = np.mean(emg_data)
emg_std = np.std(emg_data)
if emg_std > 0:
    emg_normalized = (emg_data - emg_mean) / emg_std
else:
    emg_normalized = emg_data - emg_mean

# Create plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

valid_units = [unit_id for unit_id in SELECTED_UNITS if unit_id in reconstructed_data]
total_rows = 1 + len(valid_units)  # EMG + neural units

# Plot EMG at top
emg_y_position = (total_rows - 1) * ROW_SPACING
ax.plot(
    emg_time, emg_normalized + emg_y_position, color=EMG_COLOR, linewidth=1.5, alpha=0.8
)
ax.text(
    -0.02,
    emg_y_position,
    "EMG",
    ha="right",
    va="center",
    fontsize=10,
    fontweight="bold",
    color=EMG_COLOR,
    transform=ax.get_yaxis_transform(),
)

# Plot neural units below
for row_idx, unit_id in enumerate(valid_units):
    reconstruction = reconstructed_data[unit_id]
    y_position = (len(valid_units) - 1 - row_idx) * ROW_SPACING

    ax.plot(
        reconstruction["time_s"],
        reconstruction["reconstructed_signal"] + y_position,
        color=TEMPLATE_COLOR,
        linewidth=1.5,
        alpha=0.8,
    )

    ax.text(
        -0.02,
        y_position,
        f"Unit {unit_id}",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=TEMPLATE_COLOR,
        transform=ax.get_yaxis_transform(),
    )

# Format plot
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Signals", fontsize=12)
ax.set_title(f"EMG and Neural Activity - {WINDOW_DURATION}s Window", fontsize=14)
ax.set_xlim(0, WINDOW_DURATION)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.show()

print(f"\nDisplayed: 1 EMG channel + {len(valid_units)} neural units")
print(
    f"Time window: {RECORDING_START_TIME}s - {RECORDING_START_TIME + WINDOW_DURATION}s"
)

# %%
#
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np

# Set publication style
mpu.set_publication_style()
TRANSPARENT_BACKGROUND = True  # Set to False for white background

# Use same units and colors as the perspective plot
SELECTED_UNITS = [21, 20, 47, 22, 6, 15, 8, 12, 34, 18, 25, 9]  # 12 templates
TEMPLATE_COLORS = [
    "#00833B",  # Green
    "#830000",  # Red
    "#835100",  # Orange
    "#3CFE53",  # Light green
    "#FE4343",  # Light red
    "#FA8C12",  # Orange variant
    "#8B0000",  # Dark red
    "#228B22",  # Forest green
    "#FF6347",  # Tomato
    "#DAA520",  # Goldenrod
    "#CD853F",  # Peru
    "#A0522D",  # Sienna
]

DARK_GREY_NAVY = "#2D3142"  # Dark navy for EMG
RECORDING_START_TIME = 10.0  # Start time in seconds
WINDOW_DURATION = 1.0  # 1 second window
ROW_SPACING = 2.0  # Vertical spacing between rows
PERSPECTIVE_SHIFT = 0.1  # Horizontal shift for 3D perspective effect
FIGURE_SIZE = (20, 12)

# Font sizes using mpu scaling
FONT_SIZE = 35
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.8)
LABEL_SIZE = int(FONT_SIZE * 0.8)

# Template extraction parameters
NUM_SPIKES_TO_TEMPLATE = 500
PRE_SAMPLES_MS = 2.0
POST_SAMPLES_MS = 2.0
MIN_SPIKES_THRESHOLD = 50


def extract_template_and_reconstruct(unit_id, start_time_s, duration_s):
    """Extract template and reconstruct signal over time window"""
    # Get unit's primary channel
    templates = sorter_data.templates_data["templates"]
    unit_template = templates[unit_id]
    max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
    unit_channel = max_channel - 1

    # Get all spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    if len(all_spike_times) < MIN_SPIKES_THRESHOLD:
        return None

    # Extract template from subset of spikes
    if len(all_spike_times) > NUM_SPIKES_TO_TEMPLATE:
        np.random.seed(42)
        sampled_indices = np.random.choice(
            len(all_spike_times), NUM_SPIKES_TO_TEMPLATE, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
    else:
        sampled_spikes = all_spike_times

    # Extract template waveforms
    pre_samples = int(PRE_SAMPLES_MS * FS / 1000)
    post_samples = int(POST_SAMPLES_MS * FS / 1000)

    waveforms = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whiten_hd_small.shape[0]:
            waveform = whiten_hd_small[start_idx:end_idx, unit_channel].compute()
            waveforms.append(waveform)

    if not waveforms:
        return None

    # Create normalized template
    waveforms_array = np.array(waveforms)
    template_mean = np.mean(waveforms_array, axis=0)
    template_normalized = (template_mean - np.mean(template_mean)) / np.std(
        template_mean
    )

    # Reconstruct signal in time window
    start_sample = int(start_time_s * FS)
    end_sample = int((start_time_s + duration_s) * FS)
    window_length = end_sample - start_sample

    # Create empty signal array
    reconstructed_signal = np.zeros(window_length)

    # Find spikes in window
    spikes_in_window = all_spike_times[
        (all_spike_times >= start_sample) & (all_spike_times < end_sample)
    ]

    # Place template at each spike location
    for spike_time in spikes_in_window:
        spike_pos = spike_time - start_sample
        template_start = spike_pos - pre_samples
        template_end = spike_pos + post_samples + 1

        if template_start >= 0 and template_end <= window_length:
            reconstructed_signal[template_start:template_end] += template_normalized
        elif template_start < window_length and template_end > 0:
            # Handle partial overlap
            window_start = max(0, template_start)
            window_end = min(window_length, template_end)
            template_offset_start = max(0, -template_start)
            template_offset_end = template_offset_start + (window_end - window_start)

            reconstructed_signal[window_start:window_end] += template_normalized[
                template_offset_start:template_offset_end
            ]

    # Create time array
    time_s = np.arange(window_length) / FS

    return {
        "reconstructed_signal": reconstructed_signal,
        "time_s": time_s,
        "unit_id": unit_id,
        "n_spikes_in_window": len(spikes_in_window),
    }


# Extract neural data
print(f"Processing units: {SELECTED_UNITS}")
reconstructed_data = {}

for unit_id in SELECTED_UNITS:
    result = extract_template_and_reconstruct(
        unit_id, RECORDING_START_TIME, WINDOW_DURATION
    )
    if result:
        reconstructed_data[unit_id] = result
        print(f"Unit {unit_id}: {result['n_spikes_in_window']} spikes")

# Extract EMG data
start_sample = int(RECORDING_START_TIME * FS)
end_sample = int((RECORDING_START_TIME + WINDOW_DURATION) * FS)

emg_data = filtered_emg[start_sample:end_sample, 0].compute()  # First channel only
emg_time = np.arange(len(emg_data)) / FS

# Normalize EMG - z-score normalization
emg_mean = np.mean(emg_data)
emg_std = np.std(emg_data)
if emg_std > 0:
    emg_normalized = (emg_data - emg_mean) / emg_std
else:
    emg_normalized = emg_data - emg_mean

# Create plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

valid_units = [unit_id for unit_id in SELECTED_UNITS if unit_id in reconstructed_data]
total_rows = 1 + len(valid_units)  # EMG + neural units

# Plot EMG at top with perspective shift and alpha
emg_y_position = (total_rows - 1) * ROW_SPACING
emg_x_shift = (total_rows - 1) * PERSPECTIVE_SHIFT
shifted_emg_time = emg_time + emg_x_shift

ax.plot(
    shifted_emg_time,
    emg_normalized + emg_y_position,
    color=DARK_GREY_NAVY,
    linewidth=1.5,
    alpha=0.8,  # Alpha only for EMG
)
ax.text(
    shifted_emg_time[0] - 0.05,
    emg_y_position,
    "EMG",
    ha="right",
    va="center",
    fontsize=LABEL_SIZE,
    fontweight="bold",
    color=DARK_GREY_NAVY,
)

# Plot neural units below with perspective shifts
for row_idx, unit_id in enumerate(valid_units):
    reconstruction = reconstructed_data[unit_id]
    y_position = (len(valid_units) - 1 - row_idx) * ROW_SPACING
    x_shift = (len(valid_units) - 1 - row_idx) * PERSPECTIVE_SHIFT
    shifted_time = reconstruction["time_s"] + x_shift

    # Use different muted color for each unit
    unit_color = TEMPLATE_COLORS[row_idx % len(TEMPLATE_COLORS)]

    # Plot reconstructed signal only (no template at end)
    ax.plot(
        shifted_time,
        reconstruction["reconstructed_signal"] + y_position,
        color=unit_color,
        linewidth=1.5,
        alpha=0.8,  # No alpha for neural units
    )

    ax.text(
        shifted_time[0] - 0.05,
        y_position,
        f"Unit {unit_id}",
        ha="right",
        va="center",
        fontsize=LABEL_SIZE,
        fontweight="bold",
        color=unit_color,
    )

# Draw perspective connection lines
for i in range(total_rows - 1):
    y1 = i * ROW_SPACING
    y2 = (i + 1) * ROW_SPACING
    x1_start = i * PERSPECTIVE_SHIFT
    x1_end = WINDOW_DURATION + i * PERSPECTIVE_SHIFT
    x2_start = (i + 1) * PERSPECTIVE_SHIFT
    x2_end = WINDOW_DURATION + (i + 1) * PERSPECTIVE_SHIFT

    # Left perspective lines
    ax.plot([x1_start, x2_start], [y1, y2], "k--", alpha=0.3, linewidth=1)
    # Right perspective lines
    ax.plot([x1_end, x2_end], [y1, y2], "k--", alpha=0.3, linewidth=1)

# Format plot - clean minimal style with mpu aesthetics
ax.set_xticks([])  # No x ticks
ax.set_yticks([])  # No y ticks
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
# Set background based on option
if TRANSPARENT_BACKGROUND:
    fig.patch.set_alpha(0.0)  # Transparent figure background
    ax.patch.set_alpha(0.0)  # Transparent axes background
# Set background color and ensure it covers the entire plot area


# Apply mpu styling
mpu.finalize_figure(fig, title=None)
plt.tight_layout()
plt.show()

print(f"\nDisplayed: 1 EMG channel + {len(valid_units)} neural units")
print(
    f"Time window: {RECORDING_START_TIME}s - {RECORDING_START_TIME + WINDOW_DURATION}s"
)
print(f"Colors: Muted palette (no blues), EMG with alpha transparency")

# %%

# TEMPLATES

# Template extraction constants (reuse from previous code)
NUM_SPIKES_TO_TEMPLATE = 500
PRE_SAMPLES_MS = 2.5  # ms before spike
POST_SAMPLES_MS = 2.5  # ms after spike
MIN_SPIKES_THRESHOLD = 50

# Grid layout constants - TWO COLUMNS
TEMPLATES_PER_ROW = 6  # Two columns
X_SPACING = 2.0  # Horizontal spacing between columns
Y_SPACING = 8.0  # Vertical spacing between rows
TIME_SCALE = 0.2
FIGURE_SIZE = (20, 8)  # Wider for two columns

# Background options
TRANSPARENT_BACKGROUND = True  # Set to False for white background

# Font sizes using mpu scaling
FONT_SIZE = 35
LABEL_SIZE = int(FONT_SIZE * 0.7)

# Extract templates for specified units (reuse extraction function from previous code)
print(f"Extracting templates for units: {SELECTED_UNITS}")

templates_data = {}
for unit_id in SELECTED_UNITS:
    # Get unit's primary channel
    templates = sorter_data.templates_data["templates"]
    unit_template = templates[unit_id]
    max_channel = np.argmax(np.max(np.abs(unit_template), axis=0))
    unit_channel = max_channel - 1

    # Get spike times for this unit
    unit_mask = sorter_data.spike_clusters == unit_id
    all_spike_times = sorter_data.spike_times[unit_mask]

    # Check if unit has enough spikes
    if len(all_spike_times) < MIN_SPIKES_THRESHOLD:
        print(f"Unit {unit_id}: Only {len(all_spike_times)} spikes, below threshold")
        continue

    # Subsample spikes if necessary
    if len(all_spike_times) > NUM_SPIKES_TO_TEMPLATE:
        np.random.seed(42)
        sampled_indices = np.random.choice(
            len(all_spike_times), NUM_SPIKES_TO_TEMPLATE, replace=False
        )
        sampled_spikes = all_spike_times[sampled_indices]
    else:
        sampled_spikes = all_spike_times

    # Extract waveforms around spike times
    pre_samples = int(PRE_SAMPLES_MS * FS / 1000)
    post_samples = int(POST_SAMPLES_MS * FS / 1000)

    waveforms = []
    for spike_time in sampled_spikes:
        start_idx = spike_time - pre_samples
        end_idx = spike_time + post_samples + 1

        if start_idx >= 0 and end_idx < whiten_hd_small.shape[0]:
            waveform = whiten_hd_small[start_idx:end_idx, unit_channel].compute()
            waveforms.append(waveform)

    if waveforms:
        waveforms_array = np.array(waveforms)
        template_mean = np.mean(waveforms_array, axis=0)

        # Z-score normalization for this unit
        template_normalized = (template_mean - np.mean(template_mean)) / np.std(
            template_mean
        )

        # Create time array in milliseconds
        time_ms = np.arange(pre_samples + post_samples + 1) / FS * 1000 - PRE_SAMPLES_MS

        templates_data[unit_id] = {
            "template": template_normalized,
            "time_ms": time_ms,
            "n_waveforms": len(waveforms),
            "channel": unit_channel,
        }

        print(
            f"Unit {unit_id}: Template from {len(waveforms)} spikes on channel {unit_channel}"
        )

print(f"Successfully extracted {len(templates_data)} templates")

# Create clean plot
fig, ax = plt.subplots(figsize=FIGURE_SIZE)

# Set background based on option
if TRANSPARENT_BACKGROUND:
    fig.patch.set_alpha(0.0)  # Transparent figure background
    ax.patch.set_alpha(0.0)  # Transparent axes background
else:
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

# Plot templates in two-column grid
valid_units = [unit_id for unit_id in SELECTED_UNITS if unit_id in templates_data]

for i, unit_id in enumerate(valid_units):
    # Calculate grid position (two columns)
    row = i // TEMPLATES_PER_ROW
    col = i % TEMPLATES_PER_ROW

    # Grid coordinates
    x_pos = col * X_SPACING
    y_pos = -row * Y_SPACING

    # Get template data
    template_info = templates_data[unit_id]

    # Get specific color for this unit (same order as perspective plot)
    color = TEMPLATE_COLORS[i % len(TEMPLATE_COLORS)]

    # Plot normalized template
    ax.plot(
        template_info["time_ms"] * TIME_SCALE + x_pos,
        template_info["template"] + y_pos,
        color=color,
        linewidth=7,
        alpha=0.8,
    )

    # Add unit label at left of template
    ax.text(
        x_pos - 0.6,  # Left of template
        y_pos + 1.2,  # Above template
        f"U{unit_id}",
        fontsize=LABEL_SIZE,
        ha="left",
        va="bottom",
        fontweight="bold",
        color=color,
    )

# Remove all visual elements except the waveforms
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

# Apply mpu styling and finalize
mpu.finalize_figure(fig, title=None)
plt.tight_layout()
plt.show()

print(f"\nTemplate Summary:")
print(f"- {len(templates_data)} templates plotted")
print(f"- Template window: {PRE_SAMPLES_MS}ms pre + {POST_SAMPLES_MS}ms post")
print(f"- Subsampled to {NUM_SPIKES_TO_TEMPLATE} spikes per unit")
print(f"- Units plotted: {list(templates_data.keys())}")

# %%
