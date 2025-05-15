"""
Simplified script for ECG removal using template subtraction
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.signal import correlate

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.pattern.detection.peak import create_positive_peak_detector
from dspant.pattern.subtraction.correlation import (
    create_correlation_subtractor,
    subtract_templates,
)
from dspant.processors.extractors.template_extractor import extract_template
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs

# Load environment variables
load_dotenv()
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
END_ = int(FS * 10)  # 5 seconds of data
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
plt.title(f"ECG Template (Using Extractors)")
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

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

# %%
# Set publication style
mpu.set_publication_style()

# Define constants
FS = 2000  # Sampling rate (placeholder - in real code, get this from the data)
HIGHLIGHT_COLOR = "#FFCC00"  # Bright yellow for highlight box

# Time ranges
FULL_START = 0
FULL_END = int(FS * 100)  # Show 10 seconds of data
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
fig = plt.figure(figsize=(18, 9))
gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1])

# Row 1: Unfiltered signal with ECG (spans columns 1-4)
ax1 = fig.add_subplot(gs[0, 0:4])
ax1.plot(
    full_time, filter_emg[FULL_START:FULL_END, 0], color=mpu.PRIMARY_COLOR, linewidth=2
)
mpu.format_axis(
    ax1,
    title="Filtered EMG Signal with ECG Artifacts",
    xlabel=None,  # No xlabel for the top plot
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Add highlight box for the zoom region in the first plot
y_min, y_max = ax1.get_ylim()
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
ax1.add_patch(rect1)

# Mark R-peaks on the first plot
for peak_idx in r_peak_indices:
    if FULL_START <= peak_idx < FULL_END:
        ax1.axvline(
            x=peak_idx / FS, color=mpu.COLORS["orange"], linestyle="--", alpha=0.5
        )

# Row 2: Cleaned signal (spans columns 1-4)
ax2 = fig.add_subplot(gs[1, 0:4])
ax2.plot(
    full_time,
    cleaned_emg_alt[FULL_START:FULL_END, 0],
    color=mpu.COLORS["blue"],
    linewidth=2,
)
mpu.format_axis(
    ax2,
    title="EMG Signal with ECG Artifacts Removed",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Add highlight box for the zoom region in the second plot
y_min, y_max = ax2.get_ylim()
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
ax2.add_patch(rect2)

# Create an inset axes for the ECG template in the second row, bottom left corner
ax_inset = fig.add_axes([0.15, 0.1, 0.1, 0.1])  # [left, bottom, width, height]
template_time = np.arange(len(template)) / FS
ax_inset.plot(template_time, template, color=mpu.COLORS["orange"], linewidth=2)
mpu.format_axis(
    ax_inset,
    title="ECG Template",
    xlabel="Time [s]",
    ylabel="Amp",
    xlim=(0, len(template) / FS),
)

# Row 1, Column 5: Zoomed ECG signal
ax_zoom_ecg = fig.add_subplot(gs[0, 4])
ax_zoom_ecg.plot(
    zoom_time, filter_emg[ZOOM_START:ZOOM_END, 0], color=mpu.PRIMARY_COLOR, linewidth=2
)
mpu.format_axis(
    ax_zoom_ecg,
    title="Zoomed ECG Artifacts",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, zoom_duration),
)

# Mark R-peaks on the zoomed plot
for peak_idx in r_peak_indices:
    if ZOOM_START <= peak_idx < ZOOM_END:
        # Adjust the peak index to match our zoomed time array
        t_idx = peak_idx - ZOOM_START
        ax_zoom_ecg.axvline(
            x=t_idx / FS, color=mpu.COLORS["orange"], linestyle="--", alpha=0.7
        )

# Row 2, Column 5: Zoomed Cleaned signal
ax_zoom_clean = fig.add_subplot(gs[1, 4])
ax_zoom_clean.plot(
    zoom_time,
    cleaned_emg_alt[ZOOM_START:ZOOM_END, 0],
    color=mpu.COLORS["blue"],
    linewidth=2,
)
mpu.format_axis(
    ax_zoom_clean,
    title="Zoomed Cleaned Signal",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, zoom_duration),
)

# Add panel labels with the specified formatting
mpu.add_panel_label(
    ax1,
    "A",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax2,
    "B",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_zoom_ecg,
    "C",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_zoom_clean,
    "D",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_inset,
    "E",
    x_offset_factor=0.1,
    y_offset_factor=0.01,
)

# Finalize the figure
mpu.finalize_figure(
    fig,
    title="ECG Artifact Removal using Template Subtraction",
    title_y=0.98,
    hspace=0.3,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title

# Save the figure
mpu.save_figure(fig, "ecg_removal_results.png", dpi=600)

plt.show()

# %%
