"""
/filters_explained_with_phase.py
Author: jpenalozaa
Description: Comprehensive filter demonstration including phase delay comparison
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from dotenv import load_dotenv
from matplotlib.ticker import FormatStrFormatter
from scipy import signal

load_dotenv()

# Define colors - maintaining original dark navy and blue scheme
ORIGINAL_SIGNAL_COLOR = "#2d3142"  # Dark navy for original signal
FILTERED_SIGNAL_COLOR = "#0173b2"  # Blue for filtered signal
CUTOFF_COLOR = "#de8f05"  # Orange for cutoff markers
LFILTER_COLOR = "#de8f05"  # Orange for lfilter
FILTFILT_COLOR = "#029e73"  # Green for filtfilt
ALPHA_VALUE = 1

# Define filter parameters
LOWPASS_CUTOFF = 100
HIGHPASS_CUTOFF = 10
BANDPASS_LOW = 20
BANDPASS_HIGH = 2000
NOTCH_FREQ = 60
FILTER_ORDER = 4
SAMPLING_FREQ = 8000

# Phase comparison filter parameters
PHASE_CUTOFF_FREQ = 50  # Hz
PHASE_SAMPLING_FREQ = 1000  # Hz

# Define font sizes with appropriate scaling
FONT_SIZE = 25
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.6)
TICK_SIZE = int(FONT_SIZE * 0.5)


# Function to design filters
def design_butterworth_filter(filter_type, cutoff, order=4, fs=8000):
    nyquist = 0.5 * fs

    if filter_type == "bandpass":
        low, high = cutoff
        b, a = signal.butter(order, [low / nyquist, high / nyquist], btype="band")
    elif filter_type == "notch":
        # Quality factor for notch filter
        Q = 1
        b, a = signal.iirnotch(cutoff / nyquist, Q)
    elif filter_type == "lowpass":
        b, a = signal.butter(order, cutoff / nyquist, btype="low")
    elif filter_type == "highpass":
        b, a = signal.butter(order, cutoff / nyquist, btype="high")

    return b, a


# Function to design a Butterworth lowpass filter for phase comparison
def design_butterworth_lowpass(cutoff, order=4, fs=1000):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Function to plot filter frequency response
def plot_filter_response(ax, filter_type, cutoff, order=4, fs=8000, title=None):
    # Design filter
    if filter_type == "bandpass":
        b, a = design_butterworth_filter("bandpass", cutoff, order, fs)
        cutoff_marker = cutoff
    elif filter_type == "notch":
        b, a = design_butterworth_filter("notch", cutoff, order, fs)
        cutoff_marker = [cutoff]
    elif filter_type == "lowpass":
        b, a = design_butterworth_filter("lowpass", cutoff, order, fs)
        cutoff_marker = [cutoff]
    elif filter_type == "highpass":
        b, a = design_butterworth_filter("highpass", cutoff, order, fs)
        cutoff_marker = [cutoff]

    # Compute frequency response
    w, h = signal.freqz(b, a, worN=2000)
    freq = w * fs / (2 * np.pi)

    # Plot frequency response
    ax.semilogx(freq, 20 * np.log10(abs(h)), color=FILTERED_SIGNAL_COLOR, linewidth=2)

    # Format the axis with our utility function
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Frequency [Hz]",
        ylabel="Magnitude [dB]",
        xlim=(1, 10000),
        ylim=(-80, 5),
        xscale="log",
        title_fontsize=SUBTITLE_SIZE,
        label_fontsize=AXIS_LABEL_SIZE,
        tick_fontsize=TICK_SIZE,
    )

    # Add cutoff frequency markers
    for cf in cutoff_marker:
        mpu.add_cutoff_marker(
            ax,
            x=cf,
            label=f"{cf} Hz",
            y_pos=-75,
            color=CUTOFF_COLOR,
            linestyle="--",
            alpha=0.7,
            fontsize=AXIS_LABEL_SIZE,
            fontweight="bold",
        )

    return b, a


# Generate synthetic time domain signals
def generate_test_signals(fs=8000, duration=0.5):
    t = np.linspace(0, duration, int(fs * duration))

    # Base signal (5 Hz sine wave)
    base_signal = np.sin(2 * np.pi * 5 * t)

    # Mixed signal with both low (5 Hz) and high frequency (500 Hz) components
    mixed_signal = base_signal + 0.5 * np.sin(2 * np.pi * 500 * t)

    # Create signal with components outside bandpass range
    bandpass_test = (
        base_signal
        + 0.5 * np.sin(2 * np.pi * 5000 * t)
        + 0.7 * np.sin(2 * np.pi * 10 * t)
    )

    # Add 60 Hz noise for notch test
    notch_test = base_signal + 0.8 * np.sin(2 * np.pi * 60 * t)

    return t, mixed_signal, mixed_signal, bandpass_test, notch_test


# Generate test signal for phase comparison
def generate_phase_test_signal(fs=1000, duration=1.0):
    # Create time array
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate a multi-frequency signal to better show phase effects
    signal_5hz = np.sin(2 * np.pi * 5 * t)
    signal_20hz = 0.5 * np.sin(2 * np.pi * 20 * t)
    signal_40hz = 0.3 * np.sin(2 * np.pi * 40 * t)
    signal_100hz = 0.2 * np.sin(2 * np.pi * 100 * t)

    # Combine all components
    composite_signal = signal_5hz + signal_20hz + signal_40hz + signal_100hz

    # Add some impulses to better showcase the filter's response
    impulse_locations = [int(0.25 * fs), int(0.5 * fs), int(0.75 * fs)]
    for loc in impulse_locations:
        composite_signal[loc] += 1.5

    return t, composite_signal


# Function to plot time domain signals
def plot_time_domain(ax, t, original_signal, filtered_signal, title=None):
    ax.plot(
        t,
        original_signal,
        color=ORIGINAL_SIGNAL_COLOR,
        alpha=0.3,
        linewidth=1.5,
        label="Original Signal",
    )
    ax.plot(
        t,
        filtered_signal,
        color=FILTERED_SIGNAL_COLOR,
        linewidth=2,
        label="Filtered Signal",
    )

    # Format the axis with our utility function
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Time [s]",
        ylabel="Amplitude",
        xlim=(0, 0.2),  # Show only first 0.2 seconds for better visibility
        title_fontsize=SUBTITLE_SIZE,
        label_fontsize=AXIS_LABEL_SIZE,
        tick_fontsize=TICK_SIZE,
    )

    # Add legend
    mpu.add_legend(ax, loc="upper right")


# Set publication style
mpu.set_publication_style()

# Create figure with 5x2 grid (preserving original A-H, adding I-J for phase)
fig, gs = mpu.create_figure_grid(
    rows=5,
    cols=2,
    height_ratios=[1, 1, 1, 1, 1],
    figsize=(14, 18),
)

# Generate test signals for filter demonstrations
t, lowpass_test, highpass_test, bandpass_test, notch_test = generate_test_signals(
    SAMPLING_FREQ
)

# Generate test signal for phase comparison
t_phase, phase_test_signal = generate_phase_test_signal(PHASE_SAMPLING_FREQ)

# Design all filters
lowpass_b, lowpass_a = design_butterworth_filter(
    "lowpass", LOWPASS_CUTOFF, FILTER_ORDER, SAMPLING_FREQ
)
highpass_b, highpass_a = design_butterworth_filter(
    "highpass", HIGHPASS_CUTOFF, FILTER_ORDER, SAMPLING_FREQ
)
bandpass_b, bandpass_a = design_butterworth_filter(
    "bandpass", [BANDPASS_LOW, BANDPASS_HIGH], FILTER_ORDER, SAMPLING_FREQ
)
notch_b, notch_a = design_butterworth_filter(
    "notch", NOTCH_FREQ, FILTER_ORDER, SAMPLING_FREQ
)

# Design filter for phase comparison
phase_b, phase_a = design_butterworth_lowpass(
    PHASE_CUTOFF_FREQ, FILTER_ORDER, PHASE_SAMPLING_FREQ
)

# Apply filters
lowpass_filtered = signal.filtfilt(lowpass_b, lowpass_a, lowpass_test)
highpass_filtered = signal.filtfilt(highpass_b, highpass_a, highpass_test)
bandpass_filtered = signal.filtfilt(bandpass_b, bandpass_a, bandpass_test)
notch_filtered = signal.filtfilt(notch_b, notch_a, notch_test)

# Apply phase comparison filters
phase_lfilter = signal.lfilter(phase_b, phase_a, phase_test_signal)
phase_filtfilt = signal.filtfilt(phase_b, phase_a, phase_test_signal)

# Row 1: Lowpass and Highpass time domain
ax1_1 = plt.subplot(gs[0, 0])
plot_time_domain(
    ax1_1,
    t,
    lowpass_test,
    lowpass_filtered,
    "Signal Before and After Lowpass Filtering",
)

ax1_2 = plt.subplot(gs[0, 1])
plot_time_domain(
    ax1_2,
    t,
    highpass_test,
    highpass_filtered,
    "Signal Before and After Highpass Filtering",
)

# Row 2: Lowpass and Highpass frequency response
ax2_1 = plt.subplot(gs[1, 0])
plot_filter_response(
    ax2_1,
    "lowpass",
    LOWPASS_CUTOFF,
    FILTER_ORDER,
    SAMPLING_FREQ,
    f"Butterworth Lowpass ({LOWPASS_CUTOFF} Hz) Order {FILTER_ORDER}",
)

ax2_2 = plt.subplot(gs[1, 1])
plot_filter_response(
    ax2_2,
    "highpass",
    HIGHPASS_CUTOFF,
    FILTER_ORDER,
    SAMPLING_FREQ,
    f"Butterworth Highpass ({HIGHPASS_CUTOFF} Hz) Order {FILTER_ORDER}",
)

# Row 3: Bandpass and Notch time domain
ax3_1 = plt.subplot(gs[2, 0])
plot_time_domain(
    ax3_1,
    t,
    bandpass_test,
    bandpass_filtered,
    "Signal Before and After Bandpass Filtering",
)

ax3_2 = plt.subplot(gs[2, 1])
plot_time_domain(
    ax3_2, t, notch_test, notch_filtered, "Signal Before and After Notch Filtering"
)

# Row 4: Bandpass and Notch frequency response
ax4_1 = plt.subplot(gs[3, 0])
plot_filter_response(
    ax4_1,
    "bandpass",
    [BANDPASS_LOW, BANDPASS_HIGH],
    FILTER_ORDER,
    SAMPLING_FREQ,
    f"Butterworth Bandpass ({BANDPASS_LOW}-{BANDPASS_HIGH} Hz) Order {FILTER_ORDER}",
)

ax4_2 = plt.subplot(gs[3, 1])
plot_filter_response(
    ax4_2,
    "notch",
    NOTCH_FREQ,
    FILTER_ORDER,
    SAMPLING_FREQ,
    f"{NOTCH_FREQ} Hz Notch Filter",
)

# Row 5: Phase comparison plots (NEW - panels I and J)
ax5_1 = plt.subplot(gs[4, 0])
ax5_1.plot(
    t_phase,
    phase_test_signal,
    label="Original Signal",
    color=ORIGINAL_SIGNAL_COLOR,
    linewidth=1.5,
)
ax5_1.plot(
    t_phase,
    phase_lfilter,
    label="lfilter (with phase delay)",
    color=LFILTER_COLOR,
    alpha=ALPHA_VALUE,
    linewidth=2,
)

mpu.format_axis(
    ax5_1,
    title="Single-Pass Filtering (Phase Delay)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 1),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
mpu.add_legend(ax5_1, fontsize=TICK_SIZE)

ax5_2 = plt.subplot(gs[4, 1])
ax5_2.plot(
    t_phase,
    phase_test_signal,
    label="Original Signal",
    color=ORIGINAL_SIGNAL_COLOR,
    linewidth=1.5,
)
ax5_2.plot(
    t_phase,
    phase_filtfilt,
    label="filtfilt (zero phase)",
    color=FILTFILT_COLOR,
    alpha=ALPHA_VALUE,
    linewidth=2,
)

mpu.format_axis(
    ax5_2,
    title="Zero-Phase Filtering (No Delay)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 1),
    title_fontsize=SUBTITLE_SIZE,
    label_fontsize=AXIS_LABEL_SIZE,
    tick_fontsize=TICK_SIZE,
)
mpu.add_legend(ax5_2, fontsize=TICK_SIZE)

# Finalize the figure
mpu.finalize_figure(
    fig,
    title="Comprehensive Filter Analysis: Types and Phase Behavior",
    title_y=0.98,
    hspace=0.4,
    title_fontsize=TITLE_SIZE,
)

# Use tight_layout to finalize positions before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Format all axes to show 1 decimal place
all_axes = [ax1_1, ax1_2, ax2_1, ax2_2, ax3_1, ax3_2, ax4_1, ax4_2, ax5_1, ax5_2]
for ax in all_axes:
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

# Add panel labels (now including I and J)
mpu.add_panel_label(
    ax1_1,
    "A",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax1_2,
    "B",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2_1,
    "C",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2_2,
    "D",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3_1,
    "E",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3_2,
    "F",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax4_1,
    "G",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax4_2,
    "H",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax5_1,
    "I",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax5_2,
    "J",
    x_offset_factor=0.03,
    y_offset_factor=0.02,
    fontsize=SUBTITLE_SIZE,
)

# Save the figure
FIGURE_TITLE = "comprehensive_filter_analysis"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")

mpu.save_figure(fig, FIGURE_PATH, dpi=600)

# Show the plot
plt.show()

# %%
