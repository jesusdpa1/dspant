"""
/aliasing_explained.py
Author: jpenalozaa
Description: Code to visualize the effects of aliasing and antialiasing in signal processing
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import signal

# Set color constants
ORIGINAL_SIGNAL_COLOR = "#2d3142"
SAMPLED_SIGNAL_COLOR = "#0173b2"
ALIASED_SIGNAL_COLOR = "#d55e00"
RECONSTRUCTED_SIGNAL_COLOR = "#009e73"

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the style
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Signal parameters
ORIGINAL_FREQ = 5  # Hz - our original signal frequency
ALIASING_FREQ = 15  # Hz - this will cause aliasing with our sampling rates
DURATION = 1.0  # seconds
PROPER_SAMPLING_RATE = 100  # Hz - above Nyquist for our signals
LOW_SAMPLING_RATE = 20  # Hz - will cause aliasing for the higher frequency
ANTIALIASING_CUTOFF = 10  # Hz - cutoff for our antialiasing filter

# Nyquist frequency calculations
PROPER_NYQUIST = PROPER_SAMPLING_RATE / 2
LOW_NYQUIST = LOW_SAMPLING_RATE / 2

# Create a figure
plt.figure(figsize=(14, 16))

# Create a grid for multiple subplots - 4x2 grid
gs = GridSpec(4, 2, height_ratios=[1, 1, 1, 1])

# Generate time arrays for continuous signals with very fine resolution
t_continuous = np.linspace(0, DURATION, int(1000 * DURATION))

# Generate original signals
original_signal_low_freq = np.sin(2 * np.pi * ORIGINAL_FREQ * t_continuous)
original_signal_high_freq = np.sin(2 * np.pi * ALIASING_FREQ * t_continuous)

# Generate time arrays for different sampling rates
t_proper_sampling = np.linspace(0, DURATION, int(PROPER_SAMPLING_RATE * DURATION))
t_low_sampling = np.linspace(0, DURATION, int(LOW_SAMPLING_RATE * DURATION))

# Sample signals at different rates
sampled_low_freq_proper = np.sin(2 * np.pi * ORIGINAL_FREQ * t_proper_sampling)
sampled_high_freq_proper = np.sin(2 * np.pi * ALIASING_FREQ * t_proper_sampling)
sampled_low_freq_low = np.sin(2 * np.pi * ORIGINAL_FREQ * t_low_sampling)
sampled_high_freq_low = np.sin(2 * np.pi * ALIASING_FREQ * t_low_sampling)

# Calculate the aliased frequency when undersampled
aliased_freq = abs(
    ALIASING_FREQ - LOW_SAMPLING_RATE * np.round(ALIASING_FREQ / LOW_SAMPLING_RATE)
)
aliased_signal = np.sin(2 * np.pi * aliased_freq * t_continuous)


# Design a lowpass filter for antialiasing
def design_antialiasing_filter(cutoff, order=4, fs=PROPER_SAMPLING_RATE):
    nyquist = 0.5 * fs
    b, a = signal.butter(order, cutoff / nyquist, btype="low")
    return b, a


b, a = design_antialiasing_filter(ANTIALIASING_CUTOFF)

# Apply the antialiasing filter to the high frequency signal before sampling
filtered_high_freq = signal.filtfilt(b, a, original_signal_high_freq)
sampled_filtered_high_freq = signal.filtfilt(b, a, original_signal_high_freq)[
    :: int(1000 / LOW_SAMPLING_RATE)
]


# Function to plot the frequency domain representation
def plot_frequency_domain(
    ax, t, signal_data, fs, title=None, color=ORIGINAL_SIGNAL_COLOR
):
    # Compute the FFT
    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    fft_freq = np.fft.fftfreq(n, 1 / fs)

    # Plot only the positive frequencies up to Nyquist frequency
    positive_mask = fft_freq > 0
    nyquist_mask = fft_freq <= fs / 2
    combined_mask = positive_mask & nyquist_mask

    ax.plot(
        fft_freq[combined_mask],
        2.0 / n * np.abs(fft_result[combined_mask]),
        color=color,
        linewidth=2,
    )
    ax.set_xlabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.set_ylabel("Magnitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.tick_params(labelsize=TICK_SIZE)

    # Add vertical lines for notable frequencies
    ax.axvline(x=ORIGINAL_FREQ, color="green", linestyle="--", alpha=0.7)
    ax.text(
        ORIGINAL_FREQ,
        0.1,
        f"{ORIGINAL_FREQ} Hz",
        color="green",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    ax.axvline(x=ALIASING_FREQ, color="red", linestyle="--", alpha=0.7)
    ax.text(
        ALIASING_FREQ,
        0.1,
        f"{ALIASING_FREQ} Hz",
        color="red",
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    if aliased_freq > 0 and aliased_freq < fs / 2:
        ax.axvline(x=aliased_freq, color="orange", linestyle="--", alpha=0.7)
        ax.text(
            aliased_freq,
            0.1,
            f"{aliased_freq:.1f} Hz\n(Aliased)",
            color="orange",
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    if title:
        ax.set_title(title, fontsize=SUBTITLE_SIZE, weight="bold")


# Function to plot the time domain signals
def plot_time_domain(
    ax,
    t_cont,
    original,
    t_sampled=None,
    sampled=None,
    aliased=None,
    reconstructed=None,
    title=None,
):
    ax.plot(
        t_cont,
        original,
        color=ORIGINAL_SIGNAL_COLOR,
        linewidth=1.5,
        label="Original Signal",
    )

    if t_sampled is not None and sampled is not None:
        ax.scatter(
            t_sampled,
            sampled,
            color=SAMPLED_SIGNAL_COLOR,
            s=30,
            label="Sampled Points",
            zorder=3,
        )

    if aliased is not None:
        ax.plot(
            t_cont,
            aliased,
            color=ALIASED_SIGNAL_COLOR,
            linewidth=2,
            linestyle="--",
            label="Aliased Signal",
        )

    if reconstructed is not None:
        ax.plot(
            t_cont,
            reconstructed,
            color=RECONSTRUCTED_SIGNAL_COLOR,
            linewidth=2,
            linestyle="-.",
            label="Reconstructed Signal",
        )

    ax.set_xlim(
        0, min(0.5, DURATION)
    )  # Show only part of the signal for better visibility
    ax.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.legend(loc="upper right", fontsize=TICK_SIZE)

    if title:
        ax.set_title(title, fontsize=SUBTITLE_SIZE, weight="bold")


# Plot the antialiasing filter frequency response
def plot_filter_response(ax, b, a, fs, title=None):
    # Compute frequency response
    w, h = signal.freqz(b, a, worN=2000)
    freq = w * fs / (2 * np.pi)

    # Plot frequency response
    ax.plot(freq, 20 * np.log10(abs(h)), color=SAMPLED_SIGNAL_COLOR, linewidth=2)
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-80, 5)
    ax.set_xlabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.set_ylabel("Magnitude [dB]", fontsize=AXIS_LABEL_SIZE, weight="bold")
    ax.tick_params(labelsize=TICK_SIZE)

    # Add cutoff frequency markers
    ax.axvline(x=ANTIALIASING_CUTOFF, color="r", linestyle="--", alpha=0.7)
    ax.text(
        ANTIALIASING_CUTOFF,
        -20,
        f"{ANTIALIASING_CUTOFF} Hz\nCutoff",
        color="r",
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Mark the Nyquist frequencies
    ax.axvline(x=LOW_NYQUIST, color="orange", linestyle="--", alpha=0.7)
    ax.text(
        LOW_NYQUIST,
        -40,
        f"{LOW_NYQUIST} Hz\nNyquist (Low Rate)",
        color="orange",
        horizontalalignment="center",
        verticalalignment="center",
    )

    ax.axvline(x=PROPER_NYQUIST, color="green", linestyle="--", alpha=0.7)
    ax.text(
        PROPER_NYQUIST,
        -60,
        f"{PROPER_NYQUIST} Hz\nNyquist (Proper Rate)",
        color="green",
        horizontalalignment="center",
        verticalalignment="center",
    )

    if title:
        ax.set_title(title, fontsize=SUBTITLE_SIZE, weight="bold")


# Create a simple reconstruction of the sampled signal (zero-order hold)
def reconstruct_signal(t_sampled, samples, t_continuous):
    reconstructed = np.interp(t_continuous, t_sampled, samples)
    return reconstructed


# First row: Time domain comparison of proper sampling
ax1_1 = plt.subplot(gs[0, 0])
plot_time_domain(
    ax1_1,
    t_continuous,
    original_signal_low_freq,
    t_proper_sampling,
    sampled_low_freq_proper,
    title="Low Frequency Signal (5 Hz)\nProperly Sampled (100 Hz)",
)

ax1_2 = plt.subplot(gs[0, 1])
plot_time_domain(
    ax1_2,
    t_continuous,
    original_signal_high_freq,
    t_proper_sampling,
    sampled_high_freq_proper,
    title="High Frequency Signal (15 Hz)\nProperly Sampled (100 Hz)",
)

# Second row: Time domain with aliasing
ax2_1 = plt.subplot(gs[1, 0])
plot_time_domain(
    ax2_1,
    t_continuous,
    original_signal_low_freq,
    t_low_sampling,
    sampled_low_freq_low,
    title="Low Frequency Signal (5 Hz)\nLow Sampling Rate (20 Hz) - No Aliasing",
)

# For the high frequency signal, show both original and aliased waveform
reconstructed_high = reconstruct_signal(
    t_low_sampling, sampled_high_freq_low, t_continuous
)
ax2_2 = plt.subplot(gs[1, 1])
plot_time_domain(
    ax2_2,
    t_continuous,
    original_signal_high_freq,
    t_low_sampling,
    sampled_high_freq_low,
    aliased=aliased_signal,
    title=f"High Frequency Signal (15 Hz)\nUndersampled (20 Hz) - Aliasing at {aliased_freq:.1f} Hz",
)

# Third row: Antialiasing implementation
ax3_1 = plt.subplot(gs[2, 0])
plot_filter_response(
    ax3_1,
    b,
    a,
    PROPER_SAMPLING_RATE,
    title=f"Antialiasing Filter Response\nLowpass at {ANTIALIASING_CUTOFF} Hz",
)

ax3_2 = plt.subplot(gs[2, 1])
plot_time_domain(
    ax3_2,
    t_continuous,
    original_signal_high_freq,
    t_low_sampling,
    sampled_filtered_high_freq,
    reconstructed=filtered_high_freq,
    title="High Frequency Signal After Antialiasing Filter\nSampled at 20 Hz - Aliasing Prevented",
)

# Fourth row: Frequency domain representation
ax4_1 = plt.subplot(gs[3, 0])
plot_frequency_domain(
    ax4_1,
    t_proper_sampling,
    sampled_high_freq_proper,
    PROPER_SAMPLING_RATE,
    title="Frequency Domain - Properly Sampled Signal",
)

ax4_2 = plt.subplot(gs[3, 1])
plot_frequency_domain(
    ax4_2,
    t_low_sampling,
    sampled_high_freq_low,
    LOW_SAMPLING_RATE,
    title="Frequency Domain - Aliased Signal",
    color=ALIASED_SIGNAL_COLOR,
)

# Add a title for the entire figure
plt.suptitle(
    "Understanding Aliasing and Antialiasing in Signal Processing",
    fontsize=TITLE_SIZE,
    weight="bold",
    y=0.99,
)

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.4)
# plt.savefig("aliasing_explained.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
