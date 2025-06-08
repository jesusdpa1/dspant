"""
Merged EMG Rectification and Envelope Analysis
Author: Jesus Penaloza (Updated with combined rectification and envelope methods)
Using mp_plotting_utils for standardized publication-quality visualization
Modified to use scientific notation on all y-axes with custom color scheme
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
from dspant.processors.filters.fir_filters import (
    create_moving_average,
    create_moving_rms,
)
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)

# Set publication style
mpu.set_publication_style()
# %%
# Define font sizes with appropriate scaling - increased for better visibility

# Define custom color scheme
NAVY_BLUE = "#2D3142"
ABS = "#DE8F05"
ABS_ENV = "#DE8F05"
ABS_ENV_FILL = "#FFC800"
SQUARE = "#00A410"
SQUARE_ENV = "#00A410"
SQUARE_ENV_FILL = "#90FF4B"
TKEO = "#1066DE"
TKEO_ENV = "#1066DE"
TKEO_ENV_FILL = "#31B7FA"

# Define paths using environment variables
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")

# Load EMG data
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()
FS = stream_emg.fs

# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=60, fs=FS)
lowpass_filter = create_lowpass_filter(20, FS)

# Create processing node with filters
processor_emg = create_processing_node(stream_emg)

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

# Add processors to the processing node
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# Apply filters
filter_data = processor_emg.process(group=["filters"]).persist()
# %%
# Extract data segment
START = int(FS * 5)
END = int(FS * 10)
base_data = filter_data[START:END, :]

# Create different EMG processing methods
abs_processor = RectificationProcessor("abs")
square_processor = RectificationProcessor("square")
tkeo_processor = create_tkeo_envelope_rs("modified", rectify=False, smooth=False)

# Process data with different rectification methods
abs_data = abs_processor.process(base_data).persist()
square_data = square_processor.process(base_data).persist()

# Create square + TKEO data
tkeo_data = tkeo_processor.process(base_data).persist()

# Create envelope methods
window_size = int(0.050 * FS)  # 50ms time window
moving_avg_processor = create_moving_average(window_size=window_size, center=False)
moving_rms_processor = create_moving_rms(window_size=window_size, center=False)

# Apply envelope methods
abs_moving_avg = moving_avg_processor.process(abs_data).compute()
moving_rms_envelope = moving_rms_processor.process(base_data).compute()
tkeo_lowpass = lowpass_processor.process(tkeo_data, fs=FS).compute()

# Select channel for visualization
CHANNEL_TO_PLOT = 0

# Calculate time array for x-axis
time_array = np.arange(len(base_data[:, CHANNEL_TO_PLOT])) / FS
len_to_plot = len(base_data[:, CHANNEL_TO_PLOT])
max_time = len_to_plot / FS


# Function to apply scientific notation to y-axis with larger font size
def apply_scientific_notation(ax, fontsize=24):
    """Apply scientific notation formatting to the y-axis with custom font size"""
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits(
        (-2, 2)
    )  # Use scientific notation for numbers outside 0.01 to 100
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(-2, 2))

    # Set the font size for the tick labels (including scientific notation)
    ax.tick_params(axis="y", labelsize=fontsize)

    # Also set the offset text (the exponent part) font size
    ax.yaxis.offsetText.set_fontsize(fontsize)


FONT_SIZE = 35
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)
LEGEND_SIZE = int(FONT_SIZE * 0.5)

# Larger font size for scientific notation
SCIENTIFIC_NOTATION_SIZE = int(FONT_SIZE * 0.5)

# Create figure with 3-row grid: top row spanning all columns, middle and bottom rows with 3 columns each
fig = plt.figure(figsize=(24, 18))
gs = GridSpec(3, 3, height_ratios=[1, 1.2, 1.2])

# ===== Row 1: Filtered Signal (spanning all columns) =====
ax_filtered = fig.add_subplot(gs[0, :])
ax_filtered.plot(
    time_array,
    base_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=NAVY_BLUE,
    linewidth=2,
)

mpu.format_axis(
    ax_filtered,
    title="Filtered Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_filtered, SCIENTIFIC_NOTATION_SIZE)

# ===== Row 2: Rectification Methods =====
# No uniform scaling for row 2 - each plot uses its own optimal scaling

# Column 1: Absolute Rectification
ax_abs = fig.add_subplot(gs[1, 0])
ax_abs.plot(
    time_array,
    abs_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=ABS,
    linewidth=2,
)
ax_abs.fill_between(
    time_array,
    abs_data[:len_to_plot, CHANNEL_TO_PLOT],
    0,
    color=ABS,
    alpha=0.3,
)

mpu.format_axis(
    ax_abs,
    title="Abs",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Auto-scale for this plot
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_abs, SCIENTIFIC_NOTATION_SIZE)

# Column 2: Square Rectification
ax_square = fig.add_subplot(gs[1, 1])
ax_square.plot(
    time_array,
    square_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=SQUARE,
    linewidth=2,
)
ax_square.fill_between(
    time_array,
    square_data[:len_to_plot, CHANNEL_TO_PLOT],
    0,
    color=SQUARE,
    alpha=0.3,
)

mpu.format_axis(
    ax_square,
    title="Square",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Auto-scale for this plot
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_square, SCIENTIFIC_NOTATION_SIZE)

# Column 3: TKEO
ax_square_tkeo = fig.add_subplot(gs[1, 2])
ax_square_tkeo.plot(
    time_array,
    tkeo_data[:len_to_plot, CHANNEL_TO_PLOT],
    color=TKEO,
    linewidth=2,
)
ax_square_tkeo.fill_between(
    time_array,
    tkeo_data[:len_to_plot, CHANNEL_TO_PLOT],
    0,
    color=TKEO,
    alpha=0.3,
)

mpu.format_axis(
    ax_square_tkeo,
    title="TKEO",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Auto-scale for this plot
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_square_tkeo, SCIENTIFIC_NOTATION_SIZE)

# ===== Row 3: Envelope Methods =====
# Get y-limits for abs and RMS only (columns 1 and 2) to make them consistent
abs_rms_data = [
    abs_moving_avg[:len_to_plot, CHANNEL_TO_PLOT],
    moving_rms_envelope[:len_to_plot, CHANNEL_TO_PLOT],
]
abs_rms_max = max(np.max(data) for data in abs_rms_data)
abs_rms_ylim = (0, abs_rms_max * 1.05)

# Column 1: Abs + Moving Average
ax_abs_ma = fig.add_subplot(gs[2, 0])
ax_abs_ma.plot(
    time_array,
    abs_moving_avg[:len_to_plot, CHANNEL_TO_PLOT],
    color=ABS_ENV,
    linewidth=2,
)
ax_abs_ma.fill_between(
    time_array,
    abs_moving_avg[:len_to_plot, CHANNEL_TO_PLOT],
    0,
    color=ABS_ENV_FILL,
    alpha=0.3,
)

mpu.format_axis(
    ax_abs_ma,
    title="Abs + Moving Average",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=abs_rms_ylim,  # Shared scaling with RMS
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_abs_ma, SCIENTIFIC_NOTATION_SIZE)

# Column 2: Moving RMS
ax_rms = fig.add_subplot(gs[2, 1])
ax_rms.plot(
    time_array,
    moving_rms_envelope[:len_to_plot, CHANNEL_TO_PLOT],
    color=SQUARE_ENV,
    linewidth=2,
)
ax_rms.fill_between(
    time_array,
    moving_rms_envelope[:len_to_plot, CHANNEL_TO_PLOT],
    0,
    color=SQUARE_ENV_FILL,
    alpha=0.3,
)

mpu.format_axis(
    ax_rms,
    title="Moving RMS",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=abs_rms_ylim,  # Shared scaling with Abs
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_rms, SCIENTIFIC_NOTATION_SIZE)

# Column 3: TKEO + Lowpass (independent scaling)
ax_square_tkeo_lp = fig.add_subplot(gs[2, 2])
ax_square_tkeo_lp.plot(
    time_array,
    tkeo_lowpass[:len_to_plot, CHANNEL_TO_PLOT],
    color=TKEO_ENV,
    linewidth=2,
)
ax_square_tkeo_lp.fill_between(
    time_array,
    tkeo_lowpass[:len_to_plot, CHANNEL_TO_PLOT],
    0,
    color=TKEO_ENV_FILL,
    alpha=0.3,
)

mpu.format_axis(
    ax_square_tkeo_lp,
    title="TKEO + Lowpass",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, max_time),
    ylim=(0, None),  # Auto-scale for this plot
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
apply_scientific_notation(ax_square_tkeo_lp, SCIENTIFIC_NOTATION_SIZE)

# Finalize the figure with mpu
mpu.finalize_figure(
    fig,
    # title="EMG Signal Processing: Rectification and Envelope Methods",
    title_y=0.98,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Add panel labels
mpu.add_panel_label(
    ax_filtered,
    "A",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Row 2 labels
mpu.add_panel_label(
    ax_abs,
    "B",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_square,
    "C",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_square_tkeo,
    "D",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Row 3 labels
mpu.add_panel_label(
    ax_abs_ma,
    "E",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_rms,
    "F",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax_square_tkeo_lp,
    "G",
    x_offset_factor=0.05,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
# %%
# Figure output path
FIGURE_TITLE = "merged_emg_rectification_envelope_analysis"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")
# Save the figure
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print("EMG rectification and envelope analysis complete.")

# %%
