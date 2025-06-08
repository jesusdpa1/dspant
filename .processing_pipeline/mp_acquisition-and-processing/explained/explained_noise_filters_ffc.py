# Apply VisuShrink wavelet filtering only to the window that will be shown (more efficient)
"""
Functions to extract onset detection - Complete test script with Rust acceleration
Author: Jesus Penaloza (Updated with VisuShrink wavelet filtering)
"""

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
import polars as pl
import pywt
import seaborn as sns
from dotenv import load_dotenv
from scipy import signal

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

load_dotenv()

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
# Define energy-based wavelet filtering function with aggressive thresholding
import numpy as np
import pywt


def wavelet_filter(signal_data, wavelet="db4", levels=6, sigma=None):
    """
    Apply energy-based wavelet denoising to EMG signal using soft thresholding
    with secondary cleanup pass for isolated coefficients

    Parameters:
    -----------
    signal_data : array-like
        Input signal to denoise
    wavelet : str
        Wavelet to use for decomposition (default: 'db4')
    levels : int
        Number of decomposition levels (default: 6)
    sigma : float
        Noise standard deviation (if None, estimated from signal)

    Returns:
    --------
    denoised_signal : array
        Denoised signal using energy-based soft thresholding with cleanup
    """
    # Ensure signal is 1D
    if signal_data.ndim > 1:
        signal_1d = signal_data.flatten()
    else:
        signal_1d = signal_data.copy()

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal_1d, wavelet, level=levels)

    # Estimate noise standard deviation if not provided
    if sigma is None:
        # Use robust median absolute deviation estimator on finest detail coefficients
        detail_coeffs = coeffs[-1]  # Finest detail coefficients
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745

    # Energy-based threshold: Remove 95% of coefficients (very aggressive)
    # Calculate threshold from all detail coefficients combined
    all_detail_coeffs = np.concatenate(
        [coeffs[i].flatten() for i in range(1, len(coeffs))]
    )
    threshold = np.percentile(
        np.abs(all_detail_coeffs), 95
    )  # Remove 95% of coefficients

    # Define soft thresholding function
    def soft_threshold(x, thresh):
        """
        Soft thresholding function:
        - If |x| <= thresh: return 0
        - If x > thresh: return x - thresh
        - If x < -thresh: return x + thresh
        """
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)

    # Apply soft thresholding to detail coefficients only
    coeffs_thresh = list(coeffs)
    for i in range(1, len(coeffs)):  # Skip approximation coefficients (index 0)
        coeffs_thresh[i] = soft_threshold(coeffs[i], threshold)

    # AGGRESSIVE STRATEGY 4: Secondary cleanup pass
    # Remove isolated small coefficients that might be remaining noise
    for i in range(1, len(coeffs_thresh)):
        coeff_level = coeffs_thresh[i]

        # Calculate local energy and remove isolated weak coefficients
        local_energy_threshold = np.std(coeff_level) * 0.2
        weak_mask = np.abs(coeff_level) < local_energy_threshold

        # Check for isolated coefficients (surrounded by zeros or very small values)
        if len(coeff_level) > 4:  # Only for sufficiently long coefficient arrays
            for j in range(2, len(coeff_level) - 2):
                if weak_mask[j]:
                    # Check if surrounded by weak coefficients
                    neighbors = coeff_level[j - 2 : j + 3]
                    neighbor_energy = np.mean(np.abs(neighbors))
                    if neighbor_energy < local_energy_threshold:
                        coeff_level[j] = 0

        coeffs_thresh[i] = coeff_level

    # Reconstruct signal from thresholded coefficients
    denoised_signal = pywt.waverec(coeffs_thresh, wavelet)

    # Ensure output length matches input (handle boundary effects)
    if len(denoised_signal) != len(signal_1d):
        denoised_signal = denoised_signal[: len(signal_1d)]

    return denoised_signal


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
# Define color palette - maintaining dark navy and using colorblind-friendly colors
DARK_GREY_NAVY = "#2D3142"  # Dark navy for raw signals
FILTER_COLORS = {
    "bandpass": mpu.COLORS["blue"],
    "visushrink": mpu.COLORS["purple"],  # New color for VisuShrink
    "ffc": mpu.COLORS["orange"],
    "wavelet": mpu.COLORS["green"],
}
HIGHLIGHT_COLOR = mpu.COLORS["red"]  # For zoom highlighting

# Define data range
DATA_START = 0
DATA_END = int(10 * FS)
ZOOM_START = int(2.02 * FS)
ZOOM_END = int(2.1 * FS)

# Get time values for the zoom highlight box
ZOOM_START_TIME = ZOOM_START / FS
ZOOM_END_TIME = ZOOM_END / FS
ZOOM_WIDTH = ZOOM_END_TIME - ZOOM_START_TIME

# Apply VisuShrink wavelet filtering only to the window that will be shown (more efficient)
raw_window = raw_data[DATA_START:DATA_END, 0].compute()
wavelet_filtered_window = wavelet_filter(raw_window)

# Create figure with GridSpec for custom layout
# 5 rows, 5 columns with the right side being 1/4 of the left
fig = plt.figure(figsize=(20, 20))
gs = mpu.GridSpec(5, 5, width_ratios=[1, 1, 1, 1, 1])

# Calculate time arrays
time_array = np.arange(DATA_END - DATA_START) / FS
zoom_time_array = np.arange(ZOOM_END - ZOOM_START) / FS

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

# Add highlight box for Raw zoomed region
y_min, y_max = ax_raw.get_ylim()
height = y_max - y_min
raw_rect = plt.Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_raw.add_patch(raw_rect)

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

# Add highlight box for Bandpass zoomed region
y_min, y_max = ax_bp.get_ylim()
height = y_max - y_min
bp_rect = plt.Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_bp.add_patch(bp_rect)

# Plot 3: Wavelet Filtered Data (NEW)
ax_vs = fig.add_subplot(gs[2, 0:4])
ax_vs.plot(
    time_array,
    wavelet_filtered_window,
    color=FILTER_COLORS["visushrink"],
    linewidth=2,
)
mpu.format_axis(
    ax_vs,
    title="Wavelet Denoised Signal",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, 10),
)

# Add highlight box for  zoomed region
y_min, y_max = ax_vs.get_ylim()
height = y_max - y_min
vs_rect = plt.Rectangle(
    (ZOOM_START_TIME, y_min),
    ZOOM_WIDTH,
    height,
    linewidth=2,
    edgecolor=HIGHLIGHT_COLOR,
    facecolor=HIGHLIGHT_COLOR,
    alpha=0.2,
)
ax_vs.add_patch(vs_rect)

# Plot 4: FFC Filtered Data
ax_ffc = fig.add_subplot(gs[3, 0:4])
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

# Plot 5: Wavelet Pack Harmonic Removal
ax_whp = fig.add_subplot(gs[4, 0:4])
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

# Right side plots - Zoomed views
# Plot 6: Raw Zoomed (NEW - top right)
ax_raw_zoom = fig.add_subplot(gs[0, 4])
ax_raw_zoom.plot(
    zoom_time_array,
    raw_data[ZOOM_START:ZOOM_END, 0],
    color=DARK_GREY_NAVY,
    linewidth=2,
)
mpu.format_axis(
    ax_raw_zoom,
    title="Raw (Zoomed)",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, ZOOM_WIDTH),
)

# Plot 7: Bandpass Zoomed
ax_bp_zoom = fig.add_subplot(gs[1, 4])
ax_bp_zoom.plot(
    zoom_time_array,
    filter_data[ZOOM_START:ZOOM_END, 0],
    color=FILTER_COLORS["bandpass"],
    linewidth=2,
)
mpu.format_axis(
    ax_bp_zoom,
    title="Bandpass (Zoomed)",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, ZOOM_WIDTH),
)

# Plot 8: VisuShrink Zoomed
ax_vs_zoom = fig.add_subplot(gs[2, 4])
visushrink_zoom_start = ZOOM_START - DATA_START
visushrink_zoom_end = ZOOM_END - DATA_START
ax_vs_zoom.plot(
    zoom_time_array,
    wavelet_filtered_window[visushrink_zoom_start:visushrink_zoom_end],
    color=FILTER_COLORS["visushrink"],
    linewidth=2,
)
mpu.format_axis(
    ax_vs_zoom,
    title="VisuShrink (Zoomed)",
    xlabel=None,
    ylabel="Amplitude",
    xlim=(0, ZOOM_WIDTH),
)

# Plot 9: FFC Zoomed
ax_ffc_zoom = fig.add_subplot(gs[3, 4])
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

# Plot 10: Wavelet Zoomed
ax_whp_zoom = fig.add_subplot(gs[4, 4])
ax_whp_zoom.plot(
    zoom_time_array[: min(len(zoom_time_array), ZOOM_END - ZOOM_START)],
    whp_data[ZOOM_START : min(ZOOM_END, len(whp_data)), 0],
    color=FILTER_COLORS["wavelet"],
    linewidth=2,
)
mpu.format_axis(
    ax_whp_zoom,
    title="Wavelet Packet (Zoomed)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, ZOOM_WIDTH),
)

# Finalize the figure with our utility function
mpu.finalize_figure(
    fig,
    # title="EMG Signal Filtering Comparison with VisuShrink Wavelet Denoising",
    title_y=0.98,
)

# Apply tight layout to finalize positions before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle

# Add panel labels with adaptive positioning
mpu.add_panel_label(
    ax_raw,
    "A",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_bp,
    "C",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_vs,
    "E",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_ffc,
    "G",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_whp,
    "I",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_raw_zoom,
    "B",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_bp_zoom,
    "D",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_vs_zoom,
    "F",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_ffc_zoom,
    "H",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)
mpu.add_panel_label(
    ax_whp_zoom,
    "J",
    x_offset_factor=0.02,
    y_offset_factor=0.01,
)

# Save and show figure
FIGURE_TITLE = "emg_filtering_with_visushrink"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()

# %%
