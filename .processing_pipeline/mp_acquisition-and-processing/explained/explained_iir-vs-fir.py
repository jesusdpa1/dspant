"""
/fir_vs_iir_filters.py
Author: jpenalozaa
Description: Code to visualize and compare FIR (Moving Average) and IIR (Butterworth) filters
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import signal

# Set color constants
ORIGINAL_SIGNAL_COLOR = "#2d3142"
FIR_SIGNAL_COLOR = "#0173b2"
IIR_SIGNAL_COLOR = "#d55e00"

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the figure size and style
plt.figure(figsize=(14, 12))
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Define filter parameters
CUTOFF_FREQ = 50  # Hz - cutoff frequency for both filters
BUTTERWORTH_ORDER = 4  # Filter order for Butterworth
MA_LENGTH = 51  # Length of moving average filter (make odd for symmetry)
SAMPLING_FREQ = 8000  # Hz - sampling frequency
NYQUIST = SAMPLING_FREQ / 2

# Create a grid for multiple subplots - 3x2 grid
gs = GridSpec(3, 2, height_ratios=[1, 1, 1])


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
    ax.set_xlim(1, fs / 2)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.set_ylabel("Magnitude [dB]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.tick_params(labelsize=TICK_SIZE)

    # Add cutoff frequency marker
    ax.axvline(x=CUTOFF_FREQ, color="r", linestyle="--", alpha=0.7)
    ax.text(
        CUTOFF_FREQ,
        -75,
        f"{CUTOFF_FREQ} Hz",
        color="r",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Add legend in the bottom left corner
    ax.legend(loc="lower left", fontsize=TICK_SIZE)

    if title:
        ax.set_title(title, fontsize=SUBTITLE_SIZE, weight="bold")


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

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, 0.2)  # Show only first 0.2 seconds for better visibility

    ax.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(loc="lower left", fontsize=TICK_SIZE)

    if title:
        ax.set_title(title, fontsize=SUBTITLE_SIZE, weight="bold")


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

    ax.set_xlim(1, fs / 2)
    ax.set_xlabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.set_ylabel("Phase [degrees]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.tick_params(labelsize=TICK_SIZE)

    # Add cutoff frequency marker
    ax.axvline(x=CUTOFF_FREQ, color="r", linestyle="--", alpha=0.7)

    # Add legend in bottom left corner
    ax.legend(loc="lower left", fontsize=TICK_SIZE)

    if title:
        ax.set_title(title, fontsize=SUBTITLE_SIZE, weight="bold")


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

# Add a title for the entire figure
plt.suptitle(
    "FIR vs. IIR Filters: Moving Average vs. Butterworth Comparison",
    fontsize=TITLE_SIZE,
    weight="bold",
    y=0.98,
)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.4)
plt.savefig("./img/fir_vs_iir_filters.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
