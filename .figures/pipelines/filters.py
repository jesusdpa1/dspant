# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal

from dspant.processor.filters import (
    ButterFilter,
)

sns.set_theme(style="darkgrid")


def create_combined_filter_plots(fs, fig_size=(16, 12)):
    """
    Create a 2x2 grid of filter response plots with bandpass, notch, lowpass, and highpass filters.

    Args:
        fs: Sampling frequency in Hz
        fig_size: Figure size as (width, height) in inches

    Returns:
        Matplotlib figure object
    """
    # Create a figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=fig_size)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Create filters
    bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
    notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)
    lowpass_filter = ButterFilter("lowpass", 100, order=4, fs=fs)
    highpass_filter = ButterFilter("highpass", 1, order=4, fs=fs)

    # Common parameters
    worN = 8000
    y_min = -80
    freq_scale = "log"
    cutoff_lines = True

    # Plot bandpass filter in the first subplot
    plot_filter_in_axis(
        axes[0],
        bandpass_filter,
        fs,
        worN,
        freq_scale,
        y_min,
        cutoff_lines,
        "Butterworth Bandpass (20-2000 Hz) Order 4",
    )

    # Plot notch filter in the second subplot
    plot_filter_in_axis(
        axes[1],
        notch_filter,
        fs,
        worN,
        freq_scale,
        y_min,
        cutoff_lines,
        "60 Hz Notch Filter",
    )

    # Plot lowpass filter in the third subplot
    plot_filter_in_axis(
        axes[2],
        lowpass_filter,
        fs,
        worN,
        freq_scale,
        y_min,
        cutoff_lines,
        "Butterworth Lowpass (100 Hz) Order 4",
    )

    # Plot highpass filter in the fourth subplot
    plot_filter_in_axis(
        axes[3],
        highpass_filter,
        fs,
        worN,
        freq_scale,
        y_min,
        cutoff_lines,
        "Butterworth Highpass (1 Hz) Order 4",
    )

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_filter_in_axis(
    ax, filter_obj, fs, worN, freq_scale, y_min, cutoff_lines, title
):
    """
    Plot filter frequency response in the specified axis.

    Args:
        ax: Matplotlib axis to plot on
        filter_obj: ButterFilter object
        fs: Sampling frequency in Hz
        worN: Number of frequency points
        freq_scale: Frequency scale ("linear" or "log")
        y_min: Minimum value for y-axis
        cutoff_lines: Whether to show cutoff lines
        title: Plot title
    """
    # Make sure we have filter coefficients
    if filter_obj._sos is None:
        filter_obj.fs = fs
        filter_obj._create_filter_coefficients()

    # Calculate frequency response
    w, h = signal.sosfreqz(filter_obj._sos, worN=worN, fs=fs)

    # Plot magnitude response with log scale
    if freq_scale == "log":
        ax.semilogx(w, 20 * np.log10(abs(h)), "b", linewidth=2)
        # Avoid zero frequency for log scale
        min_freq = max(w[1], 0.1)
        ax.set_xlim(min_freq, fs / 2)
    else:
        ax.plot(w, 20 * np.log10(abs(h)), "b", linewidth=2)

    # Set y-axis limits
    ax.set_ylim(bottom=y_min)
    ax.set_ylabel("Magnitude [dB]")

    # Add cutoff lines
    if cutoff_lines:
        if filter_obj.filter_type in ["bandpass", "bandstop"]:
            cutoffs = (
                [filter_obj.cutoff[0], filter_obj.cutoff[1]]
                if isinstance(filter_obj.cutoff, (tuple, list))
                else [filter_obj.cutoff]
            )
        else:
            cutoffs = (
                [filter_obj.cutoff]
                if not isinstance(filter_obj.cutoff, (tuple, list))
                else filter_obj.cutoff
            )

        for cutoff in cutoffs:
            ax.axvline(x=cutoff, color="r", linestyle="--", alpha=0.7)
            # Add text label near the cutoff line
            text_y = ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(
                cutoff * 1.05,
                text_y,
                f"{cutoff} Hz",
                rotation=90,
                color="r",
                alpha=0.9,
                fontsize=8,
            )

    # Add grid
    ax.grid(True, which="major", linestyle="-", alpha=0.4)
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    # Set x-axis label
    ax.set_xlabel("Frequency [Hz]")

    # Set title
    ax.set_title(title)


# Example usage:
# fig = create_combined_filter_plots(fs=stream_emg.fs)
# plt.savefig("combined_filters.png", dpi=300, bbox_inches='tight')
# plt.show()

# %%

# Create the combined filter plot
fig = create_combined_filter_plots(fs=12000)

# Save if needed
plt.savefig("combined_filters.png", dpi=300, bbox_inches="tight")

# Display
plt.show()

# %%
