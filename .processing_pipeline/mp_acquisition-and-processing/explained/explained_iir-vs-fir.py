"""
/fir_vs_iir_filters.py
Author: jpenalozaa
Description: Code to visualize and compare FIR (Moving Average) and IIR (Butterworth) filters
Modified: Using mp_plotting_utils for standardized publication formatting with colorblind-friendly palette
Added yellow highlight boxes to show zoom regions
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt

# Import our plotting utilities
import mp_plotting_utils as mpu
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from scipy import signal

# %%
# Define filter parameters
CUTOFF_FREQ = 50  # Hz - cutoff frequency for both filters
BUTTERWORTH_ORDER = 4  # Filter order for Butterworth
MA_LENGTH = 51  # Length of moving average filter (make odd for symmetry)
SAMPLING_FREQ = 8000  # Hz - sampling frequency
NYQUIST = SAMPLING_FREQ / 2

# Define font sizes with appropriate scaling
FONT_SIZE = 14
TITLE_SIZE = int(FONT_SIZE * 1)
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.7)
TICK_SIZE = int(FONT_SIZE * 0.7)

# Define consistent colors for our signal types - using colorblind-friendly palette
ORIGINAL_SIGNAL_COLOR = mpu.PRIMARY_COLOR  # Dark navy blue for original signal
FIR_SIGNAL_COLOR = mpu.COLORS["blue"]  # Blue for FIR filter
IIR_SIGNAL_COLOR = mpu.COLORS["orange"]  # Orange for IIR filter
CUTOFF_COLOR = mpu.COLORS["green"]  # Green for cutoff markers

# Set figure path
FIGURE_TITLE = "iir_vs_fir"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")


# Function to design Butterworth IIR filter
def design_butterworth_filter(cutoff, order=4, fs=8000):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyquist, btype="low")
    return b, a


# Function to design Moving Average FIR filter
def design_moving_average_filter(length):
    b = np.ones(length) / length  # Simple average of 'length' points
    a = [1.0]  # FIR filter denominator is always 1.0
    return b, a


# Generate synthetic test signals
def generate_test_signals(fs=8000, duration=1.0):
    t = np.linspace(0, duration, int(fs * duration))

    # Create a multi-component signal to better show filter effects
    # Low frequency component (5 Hz)
    low_freq = np.sin(2 * np.pi * 5 * t)

    # Mid frequency component (50 Hz) - near our cutoff
    mid_freq = 0.7 * np.sin(2 * np.pi * 50 * t)

    # High frequency component (200 Hz)
    high_freq = 0.5 * np.sin(2 * np.pi * 200 * t)

    # Very high frequency component (1000 Hz)
    very_high_freq = 0.3 * np.sin(2 * np.pi * 1000 * t)

    # Combine all components
    mixed_signal = low_freq + mid_freq + high_freq + very_high_freq

    # Add some noise
    noise = 0.1 * np.random.randn(len(t))
    noisy_signal = mixed_signal + noise

    # Create a step signal to test step response
    step_signal = np.zeros_like(t)
    step_idx = int(len(t) * 0.2)  # Step at 20% of the signal
    step_signal[step_idx:] = 1.0

    return t, noisy_signal, step_signal


# Function to add highlight box for zoom regions
def add_zoom_highlight(ax, x_start, x_end, highlight_color="#FFCC00", alpha=0.3):
    """
    Add a yellow highlight box to indicate zoom region
    """
    y_min, y_max = ax.get_ylim()
    height = y_max - y_min
    width = x_end - x_start

    rect = Rectangle(
        (x_start, y_min),
        width,
        height,
        linewidth=2,
        edgecolor=highlight_color,
        facecolor=highlight_color,
        alpha=alpha,
    )
    ax.add_patch(rect)


# Function to plot filter frequency response
def plot_filter_response(
    ax, b, a, fs=8000, title=None, color=FIR_SIGNAL_COLOR, filter_name=""
):
    # Compute frequency response
    w, h = signal.freqz(b, a, worN=8000)
    freq = w * fs / (2 * np.pi)

    # Plot frequency response
    ax.semilogx(
        freq, 20 * np.log10(abs(h)), color=color, linewidth=2, label=filter_name
    )

    # Format the axis with our utility function
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Frequency [Hz]",
        ylabel="Magnitude [dB]",
        xlim=(1, fs / 2),
        ylim=(-80, 5),
        xscale="log",
        title_fontsize=SUBTITLE_SIZE,
        label_fontsize=AXIS_LABEL_SIZE,
        tick_fontsize=TICK_SIZE,
    )

    # Add cutoff frequency marker
    mpu.add_cutoff_marker(
        ax,
        x=CUTOFF_FREQ,
        label=f"{CUTOFF_FREQ} Hz",
        y_pos=-75,
        color=CUTOFF_COLOR,
        fontsize=AXIS_LABEL_SIZE,
    )

    # Add legend
    mpu.add_legend(ax, loc="upper right", fontsize=TICK_SIZE)


# Function to plot time domain signals
def plot_time_domain(
    ax,
    t,
    original_signal,
    fir_filtered,
    iir_filtered,
    title=None,
    xlim=None,
    add_zoom_box=False,
    zoom_xlim=None,
    loc="best",
):
    ax.plot(
        t,
        original_signal,
        color=ORIGINAL_SIGNAL_COLOR,
        alpha=0.5,
        linewidth=1.5,
        label="Original Signal",
    )
    ax.plot(
        t,
        fir_filtered,
        color=FIR_SIGNAL_COLOR,
        linewidth=2,
        label="FIR Filtered (Moving Avg)",
    )
    ax.plot(
        t,
        iir_filtered,
        color=IIR_SIGNAL_COLOR,
        linewidth=2,
        label="IIR Filtered (Butter)",
    )

    # Set default xlim if not provided
    if xlim is None:
        xlim = (0, 0.2)  # Show only first 0.2 seconds for better visibility

    # Format the axis with our utility function
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Time [s]",
        ylabel="Amplitude",
        xlim=xlim,
        title_fontsize=SUBTITLE_SIZE,
        label_fontsize=AXIS_LABEL_SIZE,
        tick_fontsize=TICK_SIZE,
    )

    # Add zoom highlight box if requested
    if add_zoom_box and zoom_xlim is not None:
        add_zoom_highlight(ax, zoom_xlim[0], zoom_xlim[1])

    # Add legend
    mpu.add_legend(ax, loc=loc, fontsize=TICK_SIZE)


# Function to plot phase response
def plot_phase_response(ax, b, a, fs=8000, title=None):
    # Compute frequency response
    w, h = signal.freqz(b, a, worN=8000)
    freq = w * fs / (2 * np.pi)

    # Extract phase in degrees
    phase = np.unwrap(np.angle(h)) * 180 / np.pi

    # Plot phase response
    ax.semilogx(
        freq, phase, color=FIR_SIGNAL_COLOR, linewidth=2, label="FIR (Moving Avg)"
    )

    # Compute IIR phase response
    b_iir, a_iir = design_butterworth_filter(CUTOFF_FREQ, BUTTERWORTH_ORDER, fs)
    w_iir, h_iir = signal.freqz(b_iir, a_iir, worN=8000)
    freq_iir = w_iir * fs / (2 * np.pi)
    phase_iir = np.unwrap(np.angle(h_iir)) * 180 / np.pi

    # Plot IIR phase response
    ax.semilogx(
        freq_iir,
        phase_iir,
        color=IIR_SIGNAL_COLOR,
        linewidth=2,
        label="IIR (Butter)",
    )

    # Format the axis with our utility function
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Frequency [Hz]",
        ylabel="Phase [degrees]",
        xlim=(1, fs / 2),
        xscale="log",
        title_fontsize=SUBTITLE_SIZE,
        label_fontsize=AXIS_LABEL_SIZE,
        tick_fontsize=TICK_SIZE,
    )

    # Add cutoff frequency marker
    mpu.add_cutoff_marker(
        ax,
        x=CUTOFF_FREQ,
        color=CUTOFF_COLOR,
        linestyle="--",
        alpha=0.7,
        fontsize=AXIS_LABEL_SIZE,
    )

    # Add legend
    mpu.add_legend(ax, loc="lower left", fontsize=TICK_SIZE)


# Set publication style with colorblind-friendly palette
mpu.set_publication_style(use_seaborn=True)
HIGHLIGHT_COLOR = "#FFCC00"  # Bright yellow for highlight box

# Define zoom regions
STEP_ZOOM_START = 0.1
STEP_ZOOM_END = 0.4
NOISY_ZOOM_START = 0.02
NOISY_ZOOM_END = 0.05

# Create figure with grid
fig, gs = mpu.create_figure_grid(
    rows=3,
    cols=2,
    height_ratios=[1.1, 1.1, 1.1],
    figsize=(7, 9),
)

# Generate test signals
t, noisy_signal, step_signal = generate_test_signals(SAMPLING_FREQ)

# Design filters
butter_b, butter_a = design_butterworth_filter(
    CUTOFF_FREQ, BUTTERWORTH_ORDER, SAMPLING_FREQ
)
ma_b, ma_a = design_moving_average_filter(MA_LENGTH)

# Apply filters to signals
# Using filtfilt for zero-phase filtering
butter_filtered_noisy = signal.filtfilt(butter_b, butter_a, noisy_signal)
ma_filtered_noisy = signal.filtfilt(ma_b, ma_a, noisy_signal)

# For step response, use lfilter to show actual filter behavior including phase
butter_filtered_step = signal.lfilter(butter_b, butter_a, step_signal)
ma_filtered_step = signal.lfilter(ma_b, ma_a, step_signal)

# 1. Row: Frequency response comparison
ax1_1 = plt.subplot(gs[0, 0])
plot_filter_response(
    ax1_1,
    butter_b,
    butter_a,
    SAMPLING_FREQ,
    "IIR Filter: Butter Frequency Response",
    IIR_SIGNAL_COLOR,
    f"Butter {BUTTERWORTH_ORDER}th Order",
)

ax1_2 = plt.subplot(gs[0, 1])
plot_filter_response(
    ax1_2,
    ma_b,
    ma_a,
    SAMPLING_FREQ,
    "FIR Filter: MA Frequency Response",
    FIR_SIGNAL_COLOR,
    f"MA (Length {MA_LENGTH})",
)

# 2. Row: Phase response and step response
ax2_1 = plt.subplot(gs[1, 0])
plot_phase_response(ax2_1, ma_b, ma_a, SAMPLING_FREQ, "Phase Response Comparison")

ax2_2 = plt.subplot(gs[1, 1])
plot_time_domain(
    ax2_2,
    t,
    step_signal,
    ma_filtered_step,
    butter_filtered_step,
    "Step Response Comparison",
    (STEP_ZOOM_START, STEP_ZOOM_END),
    loc="lower right",  # Focus on the step transition part
)

# 3. Row: Time domain comparison - noisy signal filtering and zoomed view
ax3_1 = plt.subplot(gs[2, 0])
plot_time_domain(
    ax3_1,
    t,
    noisy_signal,
    ma_filtered_noisy,
    butter_filtered_noisy,
    "Noisy Signal Filtering Comparison",
    (0, 0.1),  # Show only first 0.1 seconds
    add_zoom_box=True,  # Add yellow highlight box
    zoom_xlim=(NOISY_ZOOM_START, NOISY_ZOOM_END),  # Highlight the zoomed region
    loc="lower right",  # Focus on the filtered signal part
)

ax3_2 = plt.subplot(gs[2, 1])
plot_time_domain(
    ax3_2,
    t,
    noisy_signal,
    ma_filtered_noisy,
    butter_filtered_noisy,
    "Zoomed View of Filtered Signals",
    (NOISY_ZOOM_START, NOISY_ZOOM_END),  # Zoomed to show details
    loc="lower right",  # Focus on the filtered signal part
)

# Finalize the figure with our utility function
mpu.finalize_figure(
    fig,
    # title="FIR vs. IIR Filters: Moving Average vs. Butterworth Comparison",
    title_y=0.96,
    left_margin=0.01,
    hspace=0.4,
    wspace=0.3,
    top_margin=0.1,
    title_fontsize=TITLE_SIZE,
)

# Format all axes to show 1 decimal place
all_axes = [ax1_1, ax1_2, ax2_1, ax2_2, ax3_1, ax3_2]
for ax in all_axes:
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

# Now add panel labels after layout adjustments
mpu.add_panel_label(
    ax1_1,
    "A",
    x_offset_factor=0.18,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax1_2,
    "B",
    x_offset_factor=0.12,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2_1,
    "C",
    x_offset_factor=0.18,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax2_2,
    "D",
    x_offset_factor=0.12,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3_1,
    "E",
    x_offset_factor=0.18,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)
mpu.add_panel_label(
    ax3_2,
    "F",
    x_offset_factor=0.12,
    y_offset_factor=0.01,
    fontsize=SUBTITLE_SIZE,
)

# Save the figure using our utility function


label_x = -0.22  # Adjust this value to move labels left/right
for ax in all_axes:
    ax.yaxis.set_label_coords(label_x, 0.5)

# mpu.save_figure(fig, FIGURE_PATH, dpi=600)
# Show the plot
plt.show()
# %%
