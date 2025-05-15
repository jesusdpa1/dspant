"""
/filters_explained.py
Author: jpenalozaa
Description: Code to plot filter configurations using mp_plotting_utils for standardized visualization
"""

# %%
import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np
from scipy import signal

# Define colors - maintaining original dark navy and blue scheme
ORIGINAL_SIGNAL_COLOR = "#2d3142"  # Dark navy for original signal (keeping original)
FILTERED_SIGNAL_COLOR = "#0173b2"  # Blue for filtered signal (keeping original)
CUTOFF_COLOR = mpu.COLORS["green"]  # Green for cutoff markers (colorblind-friendly)

# Define filter parameters
LOWPASS_CUTOFF = 100
HIGHPASS_CUTOFF = 10  # Increased to better show its effect on our test signal
BANDPASS_LOW = 20
BANDPASS_HIGH = 2000
NOTCH_FREQ = 60
FILTER_ORDER = 4
SAMPLING_FREQ = 8000


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
        )

    return b, a


# Generate synthetic time domain signals
def generate_test_signals(fs=8000, duration=0.5):
    t = np.linspace(0, duration, int(fs * duration))

    # Base signal (5 Hz sine wave)
    base_signal = np.sin(2 * np.pi * 5 * t)

    # Mixed signal with both low (5 Hz) and high frequency (500 Hz) components
    # Use this for both lowpass and highpass test to better compare effects
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
    )

    # Add legend
    mpu.add_legend(ax, loc="upper right")


# Improved panel label function
def add_panel_label(
    ax,
    label,
    position="top-left",
    offset_factor=0.1,
    fontsize=20,
    fontweight="bold",
    fontfamily="Montserrat",
    color="black",
):
    """
    Add a panel label (A, B, C, etc.) to a subplot with adaptive positioning.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the label to
    label : str
        Label text (typically a single letter like 'A', 'B', etc.)
    position : str
        Position of the label relative to the subplot. Options:
        'top-left' (default), 'top-right', 'bottom-left', 'bottom-right'
    offset_factor : float
        Factor to determine the offset relative to subplot width/height.
        Smaller values place the label closer to the subplot.
        Typical values range from 0.05 to 0.2.
    fontsize : int
        Font size for the label
    fontweight : str
        Font weight for the label
    fontfamily : str
        Font family for the label
    color : str
        Color for the label text
    """
    # Get the position of the axes in figure coordinates
    bbox = ax.get_position()
    fig = plt.gcf()

    # Calculate width and height of the figure
    fig_width, fig_height = fig.get_size_inches()

    # Calculate offset based on subplot size and offset factor
    # This will scale the offset proportionally to the subplot size
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
        fontfamily=fontfamily,
    )


# Set publication style
mpu.set_publication_style()

# Create figure with grid
fig, gs = mpu.create_figure_grid(
    rows=4,
    cols=2,
    height_ratios=[1, 1, 1, 1],
    figsize=(14, 14),
)

# Generate test signals
t, lowpass_test, highpass_test, bandpass_test, notch_test = generate_test_signals(
    SAMPLING_FREQ
)

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

# Apply filters
lowpass_filtered = signal.filtfilt(lowpass_b, lowpass_a, lowpass_test)
highpass_filtered = signal.filtfilt(highpass_b, highpass_a, highpass_test)
bandpass_filtered = signal.filtfilt(bandpass_b, bandpass_a, bandpass_test)
notch_filtered = signal.filtfilt(notch_b, notch_a, notch_test)

# 1. Row: Time domain signals for lowpass and highpass
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

# 2. Row: Filter responses for lowpass and highpass
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

# 3. Row: Time domain signals for bandpass and notch
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

# 4. Row: Filter responses for bandpass and notch
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

# Finalize the figure
mpu.finalize_figure(
    fig,
    title="Different Filtering Techniques: Lowpass, Highpass, Bandpass, and Notch",
    title_y=1.02,
    hspace=0.4,
)

# Use tight_layout to finalize positions before adding panel labels
plt.tight_layout()

# Add panel labels with the improved function - using consistent offset_factor
add_panel_label(ax1_1, "A", position="top-left", offset_factor=0.05)
add_panel_label(ax1_2, "B", position="top-left", offset_factor=0.05)
add_panel_label(ax2_1, "C", position="top-left", offset_factor=0.05)
add_panel_label(ax2_2, "D", position="top-left", offset_factor=0.05)
add_panel_label(ax3_1, "E", position="top-left", offset_factor=0.05)
add_panel_label(ax3_2, "F", position="top-left", offset_factor=0.05)
add_panel_label(ax4_1, "G", position="top-left", offset_factor=0.05)
add_panel_label(ax4_2, "H", position="top-left", offset_factor=0.05)

# Save the figure
mpu.save_figure(fig, "explained_filters.png", dpi=600)

# Show the plot
plt.show()

# %%
