"""
EMG Signal Processing Concepts Visualization
Demonstrates amplification, anti-aliasing filtering, ADC conversion, and sampling effects
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec

from dspant.nodes import StreamNode
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_lowpass_filter,
)

# %%
# Load environment variables
load_dotenv()

# Set publication style
mpu.set_publication_style()

# Define paths
DATA_DIR = Path(os.getenv("DATA_DIR"))
BASE_PATH = DATA_DIR.joinpath(
    r"topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
)
EMG_STREAM_PATH = BASE_PATH.joinpath("RawG.ant")

# Load streams
stream_emg = StreamNode(str(EMG_STREAM_PATH))
stream_emg.load_metadata()
stream_emg.load_data()

# Get sampling rate
FS = stream_emg.fs

# Get 1 second of raw data
START_ = int(FS * 0)
END_ = int(FS * 1)  # 1 second of data
time_slice = slice(START_, END_)

# Extract raw signal (keep as dask array initially)
raw_signal_dask = stream_emg.data[time_slice, 0]
raw_signal = raw_signal_dask.compute()  # Only compute when we need numpy array
time_vec = np.arange(len(raw_signal)) / FS

# Add realistic noise to simulate pre-amplification conditions
np.random.seed(42)  # For reproducibility
noise_level = np.std(raw_signal) * 0.5
noise = np.random.normal(0, noise_level, len(raw_signal))
noisy_raw_signal = raw_signal + noise

# Create amplified versions
amplified_10x = noisy_raw_signal * 10
amplified_100x = noisy_raw_signal * 100

# Create anti-aliasing filter
nyquist_freq = FS / 2
cutoff_freq = 500  # Hz - typical for EMG
anti_alias_filter = create_lowpass_filter(cutoff_freq, fs=FS, order=8)
filter_processor = FilterProcessor(
    filter_func=anti_alias_filter.get_filter_function(), overlap_samples=40
)

# Apply anti-aliasing filter
# Convert to dask array for processing, then compute result
import dask.array as da

amplified_100x_dask = da.from_array(amplified_100x.reshape(-1, 1), chunks="auto")
filtered_signal_dask = filter_processor.process(amplified_100x_dask, FS)
filtered_signal = filtered_signal_dask[:, 0].compute()

# Create high-frequency interference for aliasing demonstration
high_freq_component = 0.2 * amplified_100x.max() * np.sin(2 * np.pi * 1200 * time_vec)
signal_with_interference = amplified_100x + high_freq_component

# Simulate ADC conversion
adc_bits = 16
adc_range = 2**adc_bits
max_voltage = np.max(np.abs(filtered_signal)) * 1.1  # Add some headroom
quantization_step = (2 * max_voltage) / adc_range

# Quantize the signal
quantized_signal = np.round(filtered_signal / quantization_step) * quantization_step

# Simulate different sampling rates for aliasing demonstration
fs_adequate = FS  # Original sampling rate
fs_inadequate = 400  # Inadequate sampling rate (< 2 * max frequency)

# Downsample for aliasing demo
decimation_factor = int(FS / fs_inadequate)
time_decimated = time_vec[::decimation_factor]
signal_decimated = signal_with_interference[::decimation_factor]
filtered_decimated = filtered_signal[::decimation_factor]

# Create figure
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

# Colors
original_color = mpu.PRIMARY_COLOR
amplified_color = mpu.COLORS["orange"]
filtered_color = mpu.COLORS["green"]
quantized_color = mpu.COLORS["red"]
aliased_color = mpu.COLORS["purple"]

# ================== QUADRANT 1: AMPLIFICATION ==================
ax1 = fig.add_subplot(gs[0, 0])

# Calculate more dramatic separation for amplification stages
max_signal = np.max(np.abs(noisy_raw_signal))

# Position signals with much more separation
offset_1x = -max_signal * 0.8  # Raw signal close to x-axis (slightly below)
offset_10x = max_signal * 25  # Significant separation for 10x
offset_100x = max_signal * 80  # Much larger separation for 100x

ax1.plot(
    time_vec,
    noisy_raw_signal + offset_1x,
    color=original_color,
    linewidth=2,
    label="Raw Signal (1×)",
)
ax1.plot(
    time_vec,
    amplified_10x + offset_10x,
    color=amplified_color,
    linewidth=2,
    label="Pre-amplified (10×)",
)
ax1.plot(
    time_vec,
    amplified_100x + offset_100x,
    color=mpu.COLORS["red"],
    linewidth=2,
    label="Final Amplified (100×)",
)

# Add horizontal reference lines
ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)
ax1.axhline(y=offset_10x, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax1.axhline(y=offset_100x, color="gray", linestyle="--", alpha=0.5, linewidth=1)

# Add amplification arrows with better positioning
arrow_props = dict(
    arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black", lw=2.5
)
# Arrow from raw to 10x
ax1.annotate(
    "×10 Gain",
    xy=(0.25, offset_1x + max_signal * 3),
    xytext=(0.25, offset_10x - max_signal * 3),
    arrowprops=arrow_props,
    fontsize=11,
    ha="center",
    weight="bold",
)
# Arrow from 10x to 100x
ax1.annotate(
    "×10 Gain",
    xy=(0.75, offset_10x + max_signal * 8),
    xytext=(0.75, offset_100x - max_signal * 8),
    arrowprops=arrow_props,
    fontsize=11,
    ha="center",
    weight="bold",
)

# Add stage labels with better positioning
ax1.text(
    0.02,
    offset_1x + max_signal * 0.3,
    "Electrode\nSignal",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
    weight="bold",
)
ax1.text(
    0.02,
    offset_10x + max_signal * 2,
    "Pre-amplifier\nStage",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    weight="bold",
)
ax1.text(
    0.02,
    offset_100x + max_signal * 8,
    "Secondary\nAmplifier",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.8),
    weight="bold",
)

# Add gain specifications
ax1.text(
    0.85,
    offset_1x,
    "~µV range",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
)
ax1.text(
    0.85,
    offset_10x,
    "~10µV range",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
)
ax1.text(
    0.85,
    offset_100x,
    "~mV range",
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
)

mpu.format_axis(
    ax1,
    title="A. Multi-Stage Differential Amplification",
    xlabel="Time [s]",
    ylabel="Signal Level (Offset for Clarity)",
    xlim=(0, 1),
)
ax1.set_yticks([])  # Remove y-ticks as requested

# ================== QUADRANT 2: ANTI-ALIASING FILTER ==================
ax2 = fig.add_subplot(gs[0, 1])

# Show signal with and without high-frequency interference
ax2.plot(
    time_vec,
    signal_with_interference,
    color=mpu.COLORS["red"],
    linewidth=1.5,
    alpha=0.7,
    label=f"Signal + {1200}Hz Interference",
)
ax2.plot(
    time_vec,
    filtered_signal,
    color=filtered_color,
    linewidth=2.5,
    label=f"Anti-aliased (LP {cutoff_freq}Hz)",
)

# Add filter cutoff indicator
ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=1)

# Add text box explaining the concept
filter_text = f"Low-pass filter @ {cutoff_freq}Hz\nRemoves frequencies > Nyquist/2\nPrevents aliasing artifacts"
ax2.text(
    0.02,
    0.98,
    filter_text,
    transform=ax2.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8),
)

mpu.format_axis(
    ax2,
    title="B. Anti-Aliasing Filtering",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 1),
)
mpu.add_legend(ax2, loc="lower right")

# ================== QUADRANT 3: ADC CONVERSION ==================
ax3 = fig.add_subplot(gs[1, 0])

# Show analog-to-digital conversion process similar to reference image
zoom_start, zoom_end = 0.1, 0.15
zoom_mask = (time_vec >= zoom_start) & (time_vec <= zoom_end)
time_zoom = time_vec[zoom_mask]
analog_zoom = filtered_signal[zoom_mask]

# Create step-like digital representation
digital_steps = np.round(analog_zoom / quantization_step) * quantization_step
time_digital = np.repeat(time_zoom, 2)[1:]
digital_stepped = np.repeat(digital_steps, 2)[:-1]

# Plot analog signal as smooth curve
ax3.plot(
    time_zoom,
    analog_zoom,
    color=filtered_color,
    linewidth=3,
    label="Analog Signal",
    alpha=0.8,
)

# Plot digital representation as stepped/bar-like
ax3.plot(
    time_digital,
    digital_stepped,
    color=quantized_color,
    linewidth=2.5,
    label=f"{adc_bits}-bit Digital",
    drawstyle="steps-post",
)

# Add sample points as vertical lines (like the reference image)
sample_times = time_zoom[::2]  # Subsample for clarity
for i, t in enumerate(sample_times):
    if i < len(digital_steps[::2]):
        ax3.axvline(t, color="gray", linestyle=":", alpha=0.7, linewidth=1)
        # Add small digital value bars
        digital_val = digital_steps[::2][i] if i < len(digital_steps[::2]) else 0
        ax3.plot(
            [t, t], [0, digital_val], color=quantized_color, linewidth=2, alpha=0.7
        )

# Add quantization levels as horizontal lines (fewer for clarity)
y_min, y_max = ax3.get_ylim()
n_levels = 12  # Reduced number for better visibility
level_spacing = (y_max - y_min) / n_levels
for i in range(n_levels + 1):
    level = y_min + i * level_spacing
    ax3.axhline(level, color="lightgray", alpha=0.4, linewidth=0.5, linestyle="-")

# Add ADC process labels
ax3.text(
    0.02,
    0.98,
    "Analog\nSignal",
    transform=ax3.transAxes,
    fontsize=11,
    verticalalignment="top",
    color=filtered_color,
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
)

ax3.text(
    0.02,
    0.02,
    f"{adc_bits}-bit\nQuantization",
    transform=ax3.transAxes,
    fontsize=11,
    verticalalignment="bottom",
    color=quantized_color,
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
)

# Add sampling info
ax3.text(
    0.98,
    0.5,
    f"Fs = {FS}Hz\nResolution = {quantization_step:.1e}V\n{2**adc_bits:,} levels",
    transform=ax3.transAxes,
    fontsize=10,
    verticalalignment="center",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
)

mpu.format_axis(
    ax3,
    title="C. Analog-to-Digital Conversion",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(zoom_start, zoom_end),
)
mpu.add_legend(ax3, loc="upper left")

# ================== QUADRANT 4: SAMPLING & ALIASING EFFECTS ==================
ax4 = fig.add_subplot(gs[1, 1])

# Show signals with vertical separation for clarity
separation = np.max(np.abs(signal_with_interference)) * 1.5

# Original signal (adequate sampling)
ax4.plot(
    time_vec,
    signal_with_interference,
    color=original_color,
    linewidth=1,
    alpha=0.7,
    label=f"Original ({FS}Hz sampling)",
)

# Filtered signal (adequate sampling) - offset
ax4.plot(
    time_vec,
    filtered_signal + separation,
    color=filtered_color,
    linewidth=1.5,
    label=f"Anti-aliased ({FS}Hz sampling)",
)

# Inadequately sampled signal - larger offset
ax4.plot(
    time_decimated,
    signal_decimated + 2 * separation,
    color=aliased_color,
    linewidth=2,
    marker="o",
    markersize=2,
    label=f"Aliased ({fs_inadequate}Hz sampling)",
)

# Properly sampled after filtering - largest offset
ax4.plot(
    time_decimated,
    filtered_decimated + 3 * separation,
    color=mpu.COLORS["blue"],
    linewidth=2,
    marker="s",
    markersize=2,
    label=f"Proper sampling ({fs_inadequate}Hz)",
)

# Add Nyquist frequency indicators
nyquist_inadequate = fs_inadequate / 2
ax4.text(
    0.02,
    2 * separation + np.max(signal_decimated) * 0.8,
    f"Nyquist: {nyquist_inadequate}Hz\n(< {1200}Hz interference!)",
    fontsize=9,
    color=aliased_color,
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
)

ax4.text(
    0.02,
    3 * separation + np.max(filtered_decimated) * 0.8,
    f"Nyquist: {nyquist_inadequate}Hz\n(> {cutoff_freq}Hz cutoff ✓)",
    fontsize=9,
    color=mpu.COLORS["blue"],
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7),
)

mpu.format_axis(
    ax4,
    title="D. Sampling Rate Effects & Aliasing Prevention",
    xlabel="Time [s]",
    ylabel="Signal Level (Separated for Clarity)",
    xlim=(0, 1),
)
ax4.set_yticks([])  # Remove y-ticks for clarity
mpu.add_legend(ax4, loc="center right")

# Add frequency domain inset in bottom right of this quadrant
ax4_inset = fig.add_axes([0.75, 0.15, 0.2, 0.15])
freqs = np.fft.fftfreq(len(signal_with_interference), 1 / FS)
fft_interference = np.abs(np.fft.fft(signal_with_interference))
fft_filtered = np.abs(np.fft.fft(filtered_signal))

# Plot only positive frequencies
pos_freqs = freqs[: len(freqs) // 2]
ax4_inset.loglog(
    pos_freqs[1:],
    fft_interference[1 : len(freqs) // 2],
    color=mpu.COLORS["red"],
    alpha=0.7,
    linewidth=1.5,
    label="With interference",
)
ax4_inset.loglog(
    pos_freqs[1:],
    fft_filtered[1 : len(freqs) // 2],
    color=filtered_color,
    linewidth=2,
    label="Filtered",
)
ax4_inset.axvline(cutoff_freq, color="black", linestyle="--", alpha=0.7, linewidth=2)
ax4_inset.axvline(
    nyquist_inadequate, color=aliased_color, linestyle=":", alpha=0.8, linewidth=2
)
ax4_inset.set_xlim(10, 2000)
ax4_inset.set_ylim(1e1, 1e6)
ax4_inset.set_title("Frequency Domain", fontsize=10, weight="bold")
ax4_inset.tick_params(labelsize=8)
ax4_inset.text(
    cutoff_freq * 1.1, 1e5, f"{cutoff_freq}Hz\nCutoff", fontsize=8, color="black"
)
ax4_inset.text(
    nyquist_inadequate * 0.6,
    1e3,
    f"{nyquist_inadequate}Hz\nNyquist",
    fontsize=8,
    color=aliased_color,
)
ax4_inset.legend(fontsize=8, loc="upper right")

# Add overall title
fig.suptitle(
    "EMG Signal Processing Pipeline: From Electrode to Digital Domain",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# Add panel labels
mpu.add_panel_label(ax1, "A", x_offset_factor=0.02, y_offset_factor=-0.05)
mpu.add_panel_label(ax2, "B", x_offset_factor=0.02, y_offset_factor=-0.05)
mpu.add_panel_label(ax3, "C", x_offset_factor=0.02, y_offset_factor=-0.05)
mpu.add_panel_label(ax4, "D", x_offset_factor=0.02, y_offset_factor=-0.05)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath("emg_signal_processing_concepts.png")
mpu.save_figure(fig, FIGURE_PATH, dpi=600)

plt.show()
# %%
