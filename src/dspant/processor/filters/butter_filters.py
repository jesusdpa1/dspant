from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ...engine.base import ProcessingFunction
from ..filters.base import parallel_filter_channels


class ButterFilter:
    """
    Butterworth filter implementation with frequency response plotting.

    This class encapsulates filter creation, application, and visualization.
    It doesn't use inheritance, but instead uses a composition approach
    where filter types are specified by parameters.
    """

    def __init__(
        self,
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"],
        cutoff: Union[float, Tuple[float, float]],
        order: int = 4,
        q: float = 30,
        fs: Optional[float] = None,
    ):
        """
        Initialize a Butterworth filter.

        Args:
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
            cutoff: Cutoff frequency in Hz. For bandpass/bandstop, provide a tuple (low, high)
            order: Filter order
            q: Quality factor (for bandstop/notch filters)
            fs: Optional sampling rate. If provided, cutoff is interpreted as Hz,
                otherwise as normalized frequency (0-1)
        """
        self.filter_type = filter_type
        self.cutoff = cutoff
        self.order = order
        self.q = q
        self.fs = fs
        self._sos = None

        # Validate parameters
        self._validate_parameters()

        # Create the filter coefficients if fs is provided
        if fs is not None:
            self._create_filter_coefficients()

    def _validate_parameters(self):
        """Validate filter parameters"""
        if self.filter_type not in ["lowpass", "highpass", "bandpass", "bandstop"]:
            raise ValueError(f"Invalid filter type: {self.filter_type}")

        if self.filter_type in ["bandpass", "bandstop"]:
            if not isinstance(self.cutoff, (list, tuple)) or len(self.cutoff) != 2:
                raise ValueError(
                    f"For {self.filter_type} filter, cutoff must be a tuple of (low, high)"
                )
            if self.cutoff[0] >= self.cutoff[1]:
                raise ValueError(
                    f"For {self.filter_type} filter, low cutoff must be less than high cutoff"
                )
        else:
            if isinstance(self.cutoff, (list, tuple)):
                raise ValueError(
                    f"For {self.filter_type} filter, cutoff must be a single value"
                )

        if self.order < 1:
            raise ValueError("Filter order must be at least 1")

    def _create_filter_coefficients(self):
        """Create filter coefficients based on filter type and parameters"""
        nyquist = 0.5 * self.fs if self.fs else 1.0

        if self.filter_type == "lowpass":
            cutoff_norm = self.cutoff / nyquist if self.fs else self.cutoff
            self._sos = signal.butter(
                self.order, cutoff_norm, btype="lowpass", output="sos"
            )

        elif self.filter_type == "highpass":
            cutoff_norm = self.cutoff / nyquist if self.fs else self.cutoff
            self._sos = signal.butter(
                self.order, cutoff_norm, btype="highpass", output="sos"
            )

        elif self.filter_type == "bandpass":
            cutoff_low_norm = self.cutoff[0] / nyquist if self.fs else self.cutoff[0]
            cutoff_high_norm = self.cutoff[1] / nyquist if self.fs else self.cutoff[1]
            self._sos = signal.butter(
                self.order,
                [cutoff_low_norm, cutoff_high_norm],
                btype="bandpass",
                output="sos",
            )

        elif self.filter_type == "bandstop":
            cutoff_low_norm = self.cutoff[0] / nyquist if self.fs else self.cutoff[0]
            cutoff_high_norm = self.cutoff[1] / nyquist if self.fs else self.cutoff[1]
            self._sos = signal.butter(
                self.order,
                [cutoff_low_norm, cutoff_high_norm],
                btype="bandstop",
                output="sos",
            )

    def filter(
        self, data: np.ndarray, fs: Optional[float] = None, parallel: bool = False
    ) -> np.ndarray:
        """
        Apply the filter to input data.

        Args:
            data: Input data array
            fs: Sampling frequency (required if not provided at initialization)
            parallel: Whether to use parallel processing for multi-channel data

        Returns:
            Filtered data array
        """
        # Update sampling rate if necessary
        if fs is not None and fs != self.fs:
            self.fs = fs
            self._create_filter_coefficients()

        # Check if we have filter coefficients
        if self._sos is None:
            if self.fs is None:
                raise ValueError("Sampling frequency (fs) must be provided")
            self._create_filter_coefficients()

        # Apply filter
        if parallel and data.ndim > 1 and data.shape[1] > 1:
            return parallel_filter_channels(data, self._sos)
        else:
            return signal.sosfiltfilt(self._sos, data, axis=0)

    def get_filter_function(self) -> ProcessingFunction:
        """
        Get a filter function compatible with FilterProcessor.

        Returns:
            Function that applies the filter
        """
        # Create a filter function with the parameters from this instance
        filter_args = {
            "type": self.filter_type,
            "cutoff": self.cutoff,
            "order": self.order,
            "q": self.q if self.filter_type == "bandstop" else None,
        }

        def filter_function(
            chunk: np.ndarray, fs: float, parallel: bool = False, **kwargs
        ) -> np.ndarray:
            # Update this instance with the provided sampling rate
            if fs != self.fs:
                self.fs = fs
                self._create_filter_coefficients()

            return self.filter(chunk, fs, parallel)

        # Attach parameters for introspection
        filter_function.filter_args = filter_args
        return filter_function

    def plot_frequency_response(
        self,
        fs: Optional[float] = None,
        worN: int = 8000,
        fig_size: Tuple[int, int] = (10, 6),
        show_phase: bool = True,
        show_group_delay: bool = False,
        freq_scale: Literal["linear", "log"] = "linear",
        cutoff_lines: bool = True,
        grid: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot the frequency response of the filter.

        Args:
            fs: Sampling frequency (Hz). If None, uses instance fs or normalized frequency
            worN: Number of frequency points to compute
            fig_size: Figure size as (width, height) in inches
            show_phase: Whether to show phase response
            show_group_delay: Whether to show group delay
            freq_scale: Frequency scale ("linear" or "log")
            cutoff_lines: Whether to show cutoff frequency lines
            grid: Whether to show grid
            title: Custom title for the plot
            save_path: Path to save the figure, if provided

        Returns:
            Matplotlib figure object
        """
        # Make sure we have filter coefficients
        if self._sos is None:
            if fs is not None:
                self.fs = fs
            if self.fs is None:
                self.fs = 1.0  # Use normalized frequency if no fs is provided
            self._create_filter_coefficients()

        # Use provided fs or instance fs
        plot_fs = fs if fs is not None else self.fs

        # Create figure
        num_plots = 1 + show_phase + show_group_delay
        fig, axes = plt.subplots(num_plots, 1, figsize=fig_size, sharex=True)
        if num_plots == 1:
            axes = [axes]  # Make it a list for consistency

        # Calculate frequency response
        w, h = signal.sosfreqz(self._sos, worN=worN, fs=plot_fs)

        # Plot magnitude response
        ax_mag = axes[0]
        ax_mag.plot(w, 20 * np.log10(abs(h)), "b", linewidth=2)
        ax_mag.set_ylabel("Magnitude [dB]")

        # Set frequency scale
        if freq_scale == "log" and plot_fs is not None:
            ax_mag.set_xscale("log")
            min_freq = max(w[1], 0.1)  # Avoid zero frequency for log scale
            ax_mag.set_xlim(min_freq, plot_fs / 2)

        # Add cutoff lines
        if cutoff_lines:
            if self.filter_type in ["bandpass", "bandstop"]:
                cutoffs = (
                    [self.cutoff[0], self.cutoff[1]]
                    if isinstance(self.cutoff, (tuple, list))
                    else [self.cutoff]
                )
            else:
                cutoffs = (
                    [self.cutoff]
                    if not isinstance(self.cutoff, (tuple, list))
                    else self.cutoff
                )

            for cutoff in cutoffs:
                for ax in axes:
                    ax.axvline(x=cutoff, color="r", linestyle="--", alpha=0.7)

        # Add grid
        if grid:
            for ax in axes:
                ax.grid(True, which="both", alpha=0.3)

        # Plot phase response
        if show_phase:
            ax_phase = axes[1] if num_plots > 1 else axes[0]
            angles = np.unwrap(np.angle(h))
            ax_phase.plot(w, angles, "g", linewidth=2)
            ax_phase.set_ylabel("Phase [rad]")

        # Plot group delay
        if show_group_delay:
            ax_gd = axes[-1]
            # Calculate group delay
            group_delay = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
            # Pad to match original length
            group_delay = np.concatenate([group_delay, [group_delay[-1]]])
            ax_gd.plot(w, group_delay, "m", linewidth=2)
            ax_gd.set_ylabel("Group Delay [s]")

        # Set x-axis label on bottom plot
        axes[-1].set_xlabel(
            "Frequency [Hz]" if plot_fs is not None else "Normalized Frequency"
        )

        # Set title
        if title is None:
            title_parts = []
            if self.filter_type == "lowpass":
                title_parts.append(f"Lowpass ({self.cutoff:.1f} Hz)")
            elif self.filter_type == "highpass":
                title_parts.append(f"Highpass ({self.cutoff:.1f} Hz)")
            elif self.filter_type == "bandpass":
                title_parts.append(
                    f"Bandpass ({self.cutoff[0]:.1f}-{self.cutoff[1]:.1f} Hz)"
                )
            elif self.filter_type == "bandstop":
                title_parts.append(
                    f"Bandstop ({self.cutoff[0]:.1f}-{self.cutoff[1]:.1f} Hz)"
                )

            title_parts.append(f"Order {self.order}")
            title = f"Butterworth {' '.join(title_parts)}"

        fig.suptitle(title)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def __str__(self) -> str:
        """String representation of the filter"""
        if self.filter_type in ["bandpass", "bandstop"]:
            cutoff_str = (
                f"{self.cutoff[0]}-{self.cutoff[1]} Hz"
                if self.fs
                else f"{self.cutoff[0]}-{self.cutoff[1]}"
            )
        else:
            cutoff_str = f"{self.cutoff} Hz" if self.fs else f"{self.cutoff}"

        return (
            f"ButterFilter({self.filter_type}, cutoff={cutoff_str}, "
            f"order={self.order}, fs={self.fs if self.fs else 'None'})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the filter"""
        return self.__str__()


# Convenient factory functions that use the ButterFilter class


def create_bandpass_filter(
    lowcut: float, highcut: float, order: int = 4
) -> ProcessingFunction:
    """
    Create a bandpass filter function.

    Args:
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        order: Filter order

    Returns:
        Filter function compatible with FilterProcessor
    """
    filter_obj = ButterFilter("bandpass", (lowcut, highcut), order)
    return filter_obj.get_filter_function()


def create_notch_filter(
    notch_freq: float, q: float = 30, order: int = 4
) -> ProcessingFunction:
    """
    Create a notch filter function.

    Args:
        notch_freq: Notch frequency in Hz
        q: Quality factor (higher means narrower notch)
        order: Filter order

    Returns:
        Filter function compatible with FilterProcessor
    """
    # Calculate bandwidth from Q factor
    bandwidth = notch_freq / q
    low = notch_freq - bandwidth / 2
    high = notch_freq + bandwidth / 2

    filter_obj = ButterFilter("bandstop", (low, high), order)
    return filter_obj.get_filter_function()


def create_lowpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    """
    Create a lowpass filter function.

    Args:
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Filter function compatible with FilterProcessor
    """
    filter_obj = ButterFilter("lowpass", cutoff, order)
    return filter_obj.get_filter_function()


def create_highpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    """
    Create a highpass filter function.

    Args:
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Filter function compatible with FilterProcessor
    """
    filter_obj = ButterFilter("highpass", cutoff, order)
    return filter_obj.get_filter_function()


def plot_filter_response(
    filter_type: str,
    cutoff: Union[float, Tuple[float, float]],
    fs: float,
    order: int = 4,
    q: float = 30,
    **kwargs,
) -> plt.Figure:
    """
    Plot the frequency response of a Butterworth filter.

    Args:
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
        cutoff: Cutoff frequency in Hz. For bandpass/bandstop, provide a tuple (low, high)
        fs: Sampling frequency in Hz
        order: Filter order
        q: Quality factor (for bandstop/notch filters)
        **kwargs: Additional keyword arguments for plot_frequency_response

    Returns:
        Matplotlib figure object
    """
    # Handle notch filter type
    if filter_type == "notch":
        notch_freq = cutoff if not isinstance(cutoff, (tuple, list)) else cutoff[0]
        bandwidth = notch_freq / q
        cutoff = (notch_freq - bandwidth / 2, notch_freq + bandwidth / 2)
        filter_type = "bandstop"

    # Create filter and plot
    filter_obj = ButterFilter(filter_type, cutoff, order, q, fs)
    return filter_obj.plot_frequency_response(fs=fs, **kwargs)
