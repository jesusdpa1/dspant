"""
/filter_phase_comparison.py
Author: jpenalozaa
Description: Code to demonstrate phase delay in filtering operations (lfilter vs filtfilt)
"""

# %%
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal

# Define custom colors - maintaining the original color scheme
ORIGINAL_COLOR = "#2d3142"  # Dark navy for original signal
LFILTER_COLOR = "#de8f05"  # Orange for lfilter
FILTFILT_COLOR = "#029e73"  # Green for filtfilt
CUTOFF_COLOR = mpu.COLORS["purple"]  # Purple for cutoff lines
ALPHA_VALUE = 1

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


# Improved panel label function
def add_panel_label(
    ax,
    label,
    position="top-left",
    offset_factor=0.1,
    fontsize=None,
    fontweight="bold",
    color="black",
):
    """
    Add a panel label (A, B, C, etc.) to a subplot with adaptive positioning.
    """
    # Get the position of the axes in figure coordinates
    bbox = ax.get_position()
    fig = plt.gcf()

    # Set default font size if not specified
    if fontsize is None:
        fontsize = mpu.FONT_SIZES["panel_label"]

    # Calculate offset based on subplot size and offset factor
    x_offset = bbox.width * offset_factor
    y_offset = bbox.height * offset_factor

    # Determine position coordinates based on selected position
    if position == "top-left":
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset
    elif position == "top-right":
        x = bbox.x1 + x_offset
        y = bbox.y1 + y_offset
    elif position == "bottom-left":
        x = bbox.x0 - x_offset
        y = bbox.y0 - y_offset
    elif position == "bottom-right":
        x = bbox.x1 + x_offset
        y = bbox.y0 - y_offset
    else:
        # Default to top-left if invalid position
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset

    # Determine text alignment based on position
    if "left" in position:
        ha = "right"
    else:
        ha = "left"

    if "top" in position:
        va = "bottom"
    else:
        va = "top"

    # Position the label outside the subplot
    fig.text(
        x,
        y,
        label,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha=ha,
        color=color,
    )


# Set publication style
mpu.set_publication_style()

# Create a figure with a 2x1 grid for our plots
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

# Format the axis with mpu
mpu.format_axis(
    ax1,
    title="Single-Pass Filtering (lfilter)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 1),
)
mpu.add_legend(ax1, fontsize=mpu.FONT_SIZES["tick_label"])

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

# Format the axis with mpu
mpu.format_axis(
    ax2,
    title="Zero-Phase Filtering (filtfilt)",
    xlabel="Time [s]",
    ylabel="Amplitude",
    xlim=(0, 1),
)
mpu.add_legend(ax2, fontsize=mpu.FONT_SIZES["tick_label"])

# Create inset for phase response in the first plot
axins1 = inset_axes(ax1, width="40%", height="30%", loc="upper right")
axins1.semilogx(frequencies, phase * 180 / np.pi, color=LFILTER_COLOR, linewidth=2)
mpu.format_axis(
    axins1,
    title="Phase Response",
    xlabel="Frequency [Hz]",
    ylabel="Phase [degrees]",
    xlim=(1, SAMPLING_FREQ / 2),
    xscale="log",
    title_fontsize=mpu.FONT_SIZES["tick_label"],
    label_fontsize=mpu.FONT_SIZES["annotation"],
    tick_fontsize=mpu.FONT_SIZES["annotation"],
)
axins1.axvline(x=CUTOFF_FREQ, color=CUTOFF_COLOR, linestyle="--", alpha=0.7)
axins1.grid(True, which="both", ls="--", alpha=0.5)

# Create inset for magnitude response in the second plot
axins2 = inset_axes(ax2, width="40%", height="30%", loc="upper right")
axins2.semilogx(frequencies, magnitude, color=FILTFILT_COLOR, linewidth=2)
mpu.format_axis(
    axins2,
    title="Magnitude Response",
    xlabel="Frequency [Hz]",
    ylabel="Magnitude [dB]",
    xlim=(1, SAMPLING_FREQ / 2),
    ylim=(-80, 5),
    xscale="log",
    title_fontsize=mpu.FONT_SIZES["tick_label"],
    label_fontsize=mpu.FONT_SIZES["annotation"],
    tick_fontsize=mpu.FONT_SIZES["annotation"],
)
axins2.axvline(x=CUTOFF_FREQ, color=CUTOFF_COLOR, linestyle="--", alpha=0.7)
axins2.grid(True, which="both", ls="--", alpha=0.5)

# Finalize the figure with mpu
mpu.finalize_figure(
    fig,
    title="Comparison of Phase Delay in Filtering Methods",
    title_y=0.96,
    hspace=0.4,
)

# Apply tight layout before adding panel labels
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Add panel labels with adaptive positioning
add_panel_label(ax1, "A", offset_factor=0.05)
add_panel_label(ax2, "B", offset_factor=0.05)


# Save the figure using mpu
mpu.save_figure(fig, "explained_phase-delay-filters.png", dpi=600)

plt.show()

# %%
