"""
/fir_vs_iir_filters.py
Author: jpenalozaa
Description: Code to visualize and compare FIR (Moving Average) and IIR (Butterworth) filters
Modified: Using mp_plotting_utils for standardized publication formatting with colorblind-friendly palette
#TODO UPDATE
"""

# %%
import matplotlib.pyplot as plt

# Import our plotting utilities
import mp_plotting_utils as mpu
import numpy as np
from scipy import signal

# Define filter parameters
CUTOFF_FREQ = 50  # Hz - cutoff frequency for both filters
BUTTERWORTH_ORDER = 4  # Filter order for Butterworth
MA_LENGTH = 51  # Length of moving average filter (make odd for symmetry)
SAMPLING_FREQ = 8000  # Hz - sampling frequency
NYQUIST = SAMPLING_FREQ / 2

# Define consistent colors for our signal types - using colorblind-friendly palette
ORIGINAL_SIGNAL_COLOR = mpu.PRIMARY_COLOR  # Dark navy blue for original signal
FIR_SIGNAL_COLOR = mpu.COLORS["blue"]  # Blue for FIR filter
IIR_SIGNAL_COLOR = mpu.COLORS["orange"]  # Orange for IIR filter
CUTOFF_COLOR = mpu.COLORS["green"]  # Green for cutoff markers


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
    )

    # Add cutoff frequency marker
    mpu.add_cutoff_marker(
        ax,
        x=CUTOFF_FREQ,
        label=f"{CUTOFF_FREQ} Hz",
        y_pos=-75,
        color=CUTOFF_COLOR,
    )

    # Add legend
    mpu.add_legend(ax, loc="lower left")


# Function to plot time domain signals
def plot_time_domain(
    ax, t, original_signal, fir_filtered, iir_filtered, title=None, xlim=None
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
        label="IIR Filtered (Butterworth)",
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
    )

    # Add legend
    mpu.add_legend(ax, loc="lower left")


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
        label="IIR (Butterworth)",
    )

    # Format the axis with our utility function
    mpu.format_axis(
        ax,
        title=title,
        xlabel="Frequency [Hz]",
        ylabel="Phase [degrees]",
        xlim=(1, fs / 2),
        xscale="log",
    )

    # Add cutoff frequency marker
    mpu.add_cutoff_marker(
        ax,
        x=CUTOFF_FREQ,
        color=CUTOFF_COLOR,
        linestyle="--",
        alpha=0.7,
    )

    # Add legend
    mpu.add_legend(ax, loc="lower left")


# Set publication style with colorblind-friendly palette
mpu.set_publication_style(use_seaborn=True)

# Create figure with grid
fig, gs = mpu.create_figure_grid(
    rows=3,
    cols=2,
    height_ratios=[1, 1, 1],
    figsize=(14, 12),
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
    "IIR Filter: Butterworth Frequency Response",
    IIR_SIGNAL_COLOR,
    f"Butterworth {BUTTERWORTH_ORDER}th Order",
)

ax1_2 = plt.subplot(gs[0, 1])
plot_filter_response(
    ax1_2,
    ma_b,
    ma_a,
    SAMPLING_FREQ,
    "FIR Filter: Moving Average Frequency Response",
    FIR_SIGNAL_COLOR,
    f"Moving Average (Length {MA_LENGTH})",
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
    (0.1, 0.4),  # Focus on the step transition part
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
)

ax3_2 = plt.subplot(gs[2, 1])
plot_time_domain(
    ax3_2,
    t,
    noisy_signal,
    ma_filtered_noisy,
    butter_filtered_noisy,
    "Zoomed View of Filtered Signals",
    (0.02, 0.05),  # Zoomed to show details
)

# Finalize the figure with our utility function
mpu.finalize_figure(
    fig,
    title="FIR vs. IIR Filters: Moving Average vs. Butterworth Comparison",
    title_y=0.98,
    left_margin=0.01,
    hspace=0.4,
    top_margin=0.12,
)

# Now add panel labels after layout adjustments
mpu.add_panel_label(ax1_1, "A", x_offset=-0.04, y_offset=0.02)
mpu.add_panel_label(ax1_2, "B", x_offset=-0.04, y_offset=0.02)
mpu.add_panel_label(ax2_1, "C", x_offset=-0.04, y_offset=0.02)
mpu.add_panel_label(ax2_2, "D", x_offset=-0.04, y_offset=0.02)
mpu.add_panel_label(ax3_1, "E", x_offset=-0.04, y_offset=0.02)
mpu.add_panel_label(ax3_2, "F", x_offset=-0.04, y_offset=0.02)

# Save the figure using our utility function
mpu.save_figure(fig, "fir_vs_iir_filters.png", dpi=600)

# Show the plot
plt.show()
# %%
