"""
ECG Template Subtraction Script - Final Layout
Simplified script for ECG removal using template subtraction
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.pattern.detection.peak import create_positive_peak_detector
from dspant.pattern.subtraction.correlation import (
    create_correlation_subtractor,
)
from dspant.processors.extractors.template_extractor import extract_template
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)

# Set publication style
mpu.set_publication_style()

# Load environment variables
load_dotenv()

# Define font sizes with appropriate scaling (consistent with first script)
FONT_SIZE = 25
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)

# %%

FIGURE_TITLE = "ecg_template_subtraction"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

# %%
# Define paths
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")

# %%
# Load streams
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()

# Get sampling rate
FS = stream_emg.fs
# %%
# Create filters
bandpass_filter = create_bandpass_filter(10, 2000, fs=FS, order=5)
notch_filter = create_notch_filter(60, q=30, fs=FS)
lowpass_filter = create_lowpass_filter(200, fs=FS, order=5)

# Create filter processors
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
# Set up processing nodes
processor_emg = create_processing_node(stream_emg)

# Add processors to the nodes
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")

# %%
# Apply filters
filter_emg = processor_emg.process(group=["filters"]).persist()

# %%
START_ = int(FS * 0)  # Starting from beginning of recording
END_ = int(FS * 10)  # 10 seconds of data
time_slice = slice(START_, END_)
# Get common-mode reference
reference_ecg = lowpass_processor.process(filter_emg, FS)

plt.plot(reference_ecg[time_slice])

# %%
# Slice the reference signal for peak detection
reference_data = reference_ecg[time_slice, 0].persist()

# Create a peak detector optimized for R-peaks
r_peak_detector = create_positive_peak_detector(
    threshold=8.0,
    threshold_mode="mad",
    refractory_period=0.1,
)

# Detect R-peaks in the reference signal
peak_results = r_peak_detector.detect(reference_data, fs=FS)
r_peak_indices = peak_results["index"].compute()
print(f"Detected {len(r_peak_indices)} R-peaks in the reference signal.")

# Plot reference signal with detected peaks
plt.figure(figsize=(15, 5))
plt.plot(reference_data, label="Reference Signal")
plt.plot(
    r_peak_indices,
    reference_data[r_peak_indices],
    "r*",
    markersize=10,
    label="Detected R-peaks",
)
plt.title("ECG Reference Signal with Detected R-peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Extract ECG templates
window_ms = 30  # Window size in milliseconds
window_samples = int((window_ms / 1000) * FS)
half_win = window_samples // 2
waveform_processor = WaveformExtractor(filter_emg[:, 0], FS)

# Extract waveforms
ecg_waveforms = waveform_processor.extract_waveforms(
    r_peak_indices, pre_samples=half_win, post_samples=half_win, time_unit="samples"
)

# Extract template
ecg_template = extract_template(ecg_waveforms[0], axis=0)
template = ecg_template[:, 0]  # Use first channel

# Plot the extracted template
plt.figure(figsize=(12, 4))
plt.plot(template)
plt.title("ECG Template (Using Extractors)")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert to numpy arrays for processing
emg_data = filter_emg[time_slice, 0].compute()

# %%
# Using the new subtraction function with the new interface
# Use the convenience function for direct template subtraction

# Alternatively, using the class-based approach
subtractor = create_correlation_subtractor()
cleaned_emg_alt = subtractor.process(
    data=filter_emg[time_slice, :1],
    template=ecg_template[:, :1],
    indices=r_peak_indices,
    fs=FS,
    mode=None,
).compute()

# %%

"""
Enhanced Visualization for ECG removal using template subtraction
Using a 2x5 grid layout with highlight boxes and improved organization
"""

# Define constants
HIGHLIGHT_COLOR = "#FFCC00"  # Bright yellow for highlight box

# Time ranges
FULL_START = 0
FULL_END = int(FS * 10)  # Show 10 seconds of data
ZOOM_START = int(FS * 2)  # Start zooming at 2 seconds
ZOOM_END = int(FS * 3)  # End zooming at 3 seconds

# Calculate time values
full_time = np.arange(FULL_END - FULL_START) / FS
zoom_time = np.arange(ZOOM_END - ZOOM_START) / FS
zoom_duration = (ZOOM_END - ZOOM_START) / FS  # Duration of zoomed region in seconds

# Convert to absolute times for highlight box
ZOOM_START_TIME = ZOOM_START / FS
ZOOM_END_TIME = ZOOM_END / FS
ZOOM_WIDTH = ZOOM_END_TIME - ZOOM_START_TIME

# Create the figure with a custom grid layout
# 2 rows, 5 columns layout
FIG = plt.figure(figsize=(15, 8))
GS = GridSpec(2, 5, width_ratios=[1.1, 1.1, 1.1, 1.1, 1.1])

# Row 1: Unfiltered signal with ECG (spans columns 0-2, 3 columns total)
AX1 = FIG.add_subplot(GS[0, 0:3])
AX1.plot(
    full_time, filter_emg[FULL_START:FULL_END, 0], color=mpu.PRIMARY_COLOR, linewidth=2
)
mpu.format_axis(
    AX1,
    title="EMG Signal with ECG Artifacts",
    xlabel=None,  # No xlabel for the top plot
    ylabel="Amplitude",
    xlim=(0, 10),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Add highlight box for the zoom region in the first plot
y_min, y_max = AX1.get_ylim()
height = y_max - y_min
rect1 = Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.3,
)
AX1.add_patch(rect1)

# Mark R-peaks on the first plot
for peak_idx in r_peak_indices:
    if FULL_START <= peak_idx < FULL_END:
        AX1.axvline(
            x=peak_idx / FS, color=mpu.COLORS["orange"], linestyle="--", alpha=0.5
        )

# Row 2: Cleaned signal (spans columns 0-2, 3 columns total)
AX2 = FIG.add_subplot(GS[1, 0:3])
AX2.plot(
    full_time,
    cleaned_emg_alt[FULL_START:FULL_END, 0],
    color=mpu.COLORS["blue"],
    linewidth=2,
)
mpu.format_axis(
    AX2,
    title="EMG Signal with ECG Artifacts Removed",
    xlabel="Time (s)",
    ylabel="Amplitude",
    xlim=(0, 10),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Add highlight box for the zoom region in the second plot
y_min, y_max = AX2.get_ylim()
height = y_max - y_min
rect2 = Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.3,
)
AX2.add_patch(rect2)

# Row 1, Column 3: Zoomed ECG signal (1 column)
AX_ZOOM_ECG = FIG.add_subplot(GS[0, 3])
AX_ZOOM_ECG.plot(
    zoom_time, filter_emg[ZOOM_START:ZOOM_END, 0], color=mpu.PRIMARY_COLOR, linewidth=2
)
mpu.format_axis(
    AX_ZOOM_ECG,
    title="Zoomed Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, zoom_duration),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Mark R-peaks on the zoomed plot
for peak_idx in r_peak_indices:
    if ZOOM_START <= peak_idx < ZOOM_END:
        # Adjust the peak index to match our zoomed time array
        t_idx = peak_idx - ZOOM_START
        AX_ZOOM_ECG.axvline(
            x=t_idx / FS, color=mpu.COLORS["orange"], linestyle="--", alpha=0.7
        )

# Get the y-axis limits from the ECG zoom plot
zoom_ylim = AX_ZOOM_ECG.get_ylim()

# Row 1, Column 4: ECG Template (1 column)
AX_TEMPLATE = FIG.add_subplot(GS[0, 4])
template_time = np.arange(len(template)) / FS
AX_TEMPLATE.plot(template_time, template, color=mpu.COLORS["orange"], linewidth=2)
mpu.format_axis(
    AX_TEMPLATE,
    title="ECG Template",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, len(template) / FS),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)


# Row 2, Columns 3-4: Zoomed Cleaned signal (spans 2 columns)
AX_ZOOM_CLEAN = FIG.add_subplot(GS[1, 3:5])
AX_ZOOM_CLEAN.plot(
    zoom_time,
    cleaned_emg_alt[ZOOM_START:ZOOM_END, 0],
    color=mpu.COLORS["blue"],
    linewidth=2,
)
mpu.format_axis(
    AX_ZOOM_CLEAN,
    title="Zoomed Cleaned Signal",
    xlabel="Time (s)",
    ylabel="Amplitude",
    xlim=(0, zoom_duration),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)

# Apply the same y-axis limits to the cleaned signal zoom plot
AX_ZOOM_CLEAN.set_ylim(zoom_ylim)

# Format y-axis to scientific notation for all plots
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 2))  # Forces scientific notation

for ax in [AX1, AX2, AX_ZOOM_ECG, AX_ZOOM_CLEAN, AX_TEMPLATE]:
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(
        style="scientific", axis="y", scilimits=(0, 0), useMathText=True
    )
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

# Add panel labels with the specified formatting
mpu.add_panel_label(
    AX1,
    "A",
    x_offset_factor=0.2,
    y_offset_factor=0.04,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    AX_ZOOM_ECG,
    "B",
    x_offset_factor=-0.1,
    y_offset_factor=0.04,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    AX_TEMPLATE,
    "C",
    x_offset_factor=-0.4,
    y_offset_factor=0.04,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    AX2,
    "D",
    x_offset_factor=0.2,
    y_offset_factor=-0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    AX_ZOOM_CLEAN,
    "E",
    x_offset_factor=-0.03,
    y_offset_factor=-0.01,
    fontsize=SUBTITLE_SIZE,
)

# Finalize the figure
mpu.finalize_figure(
    FIG,
    # title="ECG Artifact Removal using Template Subtraction",
    title_y=0.98,
    title_fontsize=TITLE_SIZE,
)

# Apply tight layout before saving
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the figure
# mpu.save_figure(FIG, FIGURE_PATH, dpi=600)

plt.show()

print(f"Figure saved to: {FIGURE_PATH}")
print("ECG template subtraction analysis complete.")

# %%
