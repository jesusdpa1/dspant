"""
/filter_phase_comparison.py
Author: jpenalozaa
Description: Code to demonstrate phase delay in filtering operations (lfilter vs filtfilt)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Define custom colors
ORIGINAL_COLOR = "#2d3142"
LFILTER_COLOR = "#de8f05"
FILTFILT_COLOR = "#029e73"
ALPHA_VALUE = 1

# Set the figure style
plt.style.use("seaborn-v0_8-darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
ANNOTATION_SIZE = 10

# Define filter parameters
FILTER_ORDER = 4
CUTOFF_FREQ = 50  # Hz
SAMPLING_FREQ = 1000  # Hz


# Function to design a Butterworth lowpass filter
def design_butterworth_lowpass(cutoff, order=4, fs=1000):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Generate test signal that clearly shows phase differences
def generate_test_signal(fs=1000, duration=1.0):
    # Create time array
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Generate a multi-frequency signal to better show phase effects
    # Main component at 5 Hz
    signal_5hz = np.sin(2 * np.pi * 5 * t)
    # Add a 20 Hz component
    signal_20hz = 0.5 * np.sin(2 * np.pi * 20 * t)
    # Add a 40 Hz component (will be affected by the filter)
    signal_40hz = 0.3 * np.sin(2 * np.pi * 40 * t)
    # Add a higher frequency component that will be heavily filtered
    signal_100hz = 0.2 * np.sin(2 * np.pi * 100 * t)

    # Combine all components
    composite_signal = signal_5hz + signal_20hz + signal_40hz + signal_100hz

    # Add some impulses to better showcase the filter's response
    impulse_locations = [int(0.25 * fs), int(0.5 * fs), int(0.75 * fs)]
    for loc in impulse_locations:
        composite_signal[loc] += 1.5

    return t, composite_signal


# Create a 2x1 grid for our plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Generate our test signal
t, test_signal = generate_test_signal(SAMPLING_FREQ)

# Design our filter
b, a = design_butterworth_lowpass(CUTOFF_FREQ, FILTER_ORDER, SAMPLING_FREQ)

# Apply the filter both ways
filtered_signal = signal.lfilter(b, a, test_signal)  # Single-pass (with phase delay)
filtfilt_signal = signal.filtfilt(b, a, test_signal)  # Zero-phase filtering

# Calculate the filter's frequency response (magnitude and phase)
w, h = signal.freqz(b, a, worN=8000)
frequencies = w * SAMPLING_FREQ / (2 * np.pi)
magnitude = 20 * np.log10(abs(h))
phase = np.unwrap(np.angle(h))

# Plot for lfilter (single-pass filtering)
ax1.plot(t, test_signal, label="Original Signal", color=ORIGINAL_COLOR, linewidth=1.5)
ax1.plot(
    t,
    filtered_signal,
    label="lfilter (with phase delay)",
    color=LFILTER_COLOR,
    alpha=ALPHA_VALUE,
    linewidth=2,
)
ax1.set_title("Single-Pass Filtering (lfilter)", fontsize=SUBTITLE_SIZE, weight="bold")
ax1.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.tick_params(labelsize=TICK_SIZE)
ax1.legend(fontsize=TICK_SIZE)
ax1.set_xlim(0, 1)

# Plot for filtfilt (zero-phase filtering)
ax2.plot(t, test_signal, label="Original Signal", color=ORIGINAL_COLOR, linewidth=1.5)
ax2.plot(
    t,
    filtfilt_signal,
    label="filtfilt (zero phase)",
    color=FILTFILT_COLOR,
    alpha=ALPHA_VALUE,
    linewidth=2,
)
ax2.set_title("Zero-Phase Filtering (filtfilt)", fontsize=SUBTITLE_SIZE, weight="bold")
ax2.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.tick_params(labelsize=TICK_SIZE)
ax2.legend(fontsize=TICK_SIZE)
ax2.set_xlim(0, 1)

# Add a small inset with the frequency response
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create inset for phase response in the first plot
axins1 = inset_axes(ax1, width="40%", height="30%", loc="upper right")
axins1.semilogx(frequencies, phase * 180 / np.pi, color=LFILTER_COLOR, linewidth=2)
axins1.set_title("Phase Response", fontsize=TICK_SIZE)
axins1.set_xlim(1, SAMPLING_FREQ / 2)
axins1.set_xlabel("Frequency [Hz]", fontsize=ANNOTATION_SIZE)
axins1.set_ylabel("Phase [degrees]", fontsize=ANNOTATION_SIZE)
axins1.tick_params(labelsize=ANNOTATION_SIZE)
axins1.axvline(x=CUTOFF_FREQ, color="r", linestyle="--", alpha=0.7)
axins1.grid(True, which="both", ls="--", alpha=0.5)

# Create inset for magnitude response in the second plot
axins2 = inset_axes(ax2, width="40%", height="30%", loc="upper right")
axins2.semilogx(frequencies, magnitude, color=FILTFILT_COLOR, linewidth=2)
axins2.set_title("Magnitude Response", fontsize=TICK_SIZE)
axins2.set_xlim(1, SAMPLING_FREQ / 2)
axins2.set_ylim(-80, 5)
axins2.set_xlabel("Frequency [Hz]", fontsize=ANNOTATION_SIZE)
axins2.set_ylabel("Magnitude [dB]", fontsize=ANNOTATION_SIZE)
axins2.tick_params(labelsize=ANNOTATION_SIZE)
axins2.axvline(x=CUTOFF_FREQ, color="r", linestyle="--", alpha=0.7)
axins2.grid(True, which="both", ls="--", alpha=0.5)

# Add a main title
fig.suptitle(
    "Comparison of Phase Delay in Filtering Methods",
    fontsize=TITLE_SIZE,
    weight="bold",
    y=0.98,
)

plt.tight_layout()
plt.subplots_adjust(top=0.92, hspace=0.3)

plt.savefig("./img/explained_phase-delay-filters.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
