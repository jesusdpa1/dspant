"""
IIR filter implementations with Rust acceleration.

This module provides high-performance implementations of various IIR filters
(Butterworth, Chebyshev, Elliptic, Bessel) with Rust acceleration for multi-channel data.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
from scipy import signal

from dspant.core.internals import public_api

from ...engine.base import BaseProcessor, ProcessingFunction

try:
    from dspant._rs import (
        apply_bessel_filter,
        apply_butter_filter,
        apply_cascaded_filters,
        apply_cheby_filter,
        apply_elliptic_filter,
        parallel_filter_channels,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )
    # Import the Python implementations
    from .butter_filters import parallel_filter_channels


class IIRFilter:
    """
    IIR filter implementation with flexible filter types and Rust acceleration.

    This class encapsulates various IIR filter types (Butterworth, Chebyshev,
    Elliptic, Bessel) and provides visualization capabilities.
    """

    def __init__(
        self,
        filter_type: Literal["butter", "cheby1", "cheby2", "ellip", "bessel"],
        btype: Literal["lowpass", "highpass", "bandpass", "bandstop"],
        cutoff: Union[float, Tuple[float, float]],
        order: int = 4,
        rs: float = 40,  # Minimum attenuation in the stop band (dB) for Chebyshev II and elliptic
        rp: float = 1,  # Maximum ripple in the passband (dB) for Chebyshev I and elliptic
        fs: Optional[float] = None,
    ):
        """
        Initialize an IIR filter.

        Args:
            filter_type: Filter family type
                "butter": Butterworth filter (maximally flat magnitude)
                "cheby1": Chebyshev type I filter (ripple in passband)
                "cheby2": Chebyshev type II filter (ripple in stopband)
                "ellip": Elliptic filter (ripple in both bands, sharper transition)
                "bessel": Bessel filter (maximally flat group delay)
            btype: Filter response type
                "lowpass": Low pass filter
                "highpass": High pass filter
                "bandpass": Band pass filter
                "bandstop": Band stop filter
            cutoff: Cutoff frequency(ies) in Hz
                For lowpass/highpass: single value
                For bandpass/bandstop: tuple of (low_cutoff, high_cutoff)
            order: Filter order (higher = steeper transition, more ringing)
            rs: Minimum attenuation in the stop band (dB) for Chebyshev II and elliptic
            rp: Maximum ripple in the passband (dB) for Chebyshev I and elliptic
            fs: Sampling frequency in Hz. If None, cutoff is treated as normalized frequency
        """
        self.filter_type = filter_type
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.rs = rs
        self.rp = rp
        self.fs = fs
        self._sos = None

        # Validate parameters
        self._validate_parameters()

        # Create filter coefficients if fs is provided
        if fs is not None:
            self._create_filter_coefficients()

    def _validate_parameters(self):
        """Validate filter parameters"""
        valid_filter_types = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
        if self.filter_type not in valid_filter_types:
            raise ValueError(
                f"Invalid filter type: {self.filter_type}. Must be one of {valid_filter_types}"
            )

        valid_btypes = ["lowpass", "highpass", "bandpass", "bandstop"]
        if self.btype not in valid_btypes:
            raise ValueError(
                f"Invalid btype: {self.btype}. Must be one of {valid_btypes}"
            )

        if self.btype in ["bandpass", "bandstop"]:
            if not isinstance(self.cutoff, (list, tuple)) or len(self.cutoff) != 2:
                raise ValueError(
                    f"For {self.btype} filter, cutoff must be a tuple of (low, high)"
                )
            if self.cutoff[0] >= self.cutoff[1]:
                raise ValueError(
                    f"For {self.btype} filter, low cutoff must be less than high cutoff"
                )
        else:
            if isinstance(self.cutoff, (list, tuple)):
                raise ValueError(
                    f"For {self.btype} filter, cutoff must be a single value"
                )

        if self.order < 1:
            raise ValueError("Filter order must be at least 1")

    def _create_filter_coefficients(self):
        """Create filter coefficients using SciPy's signal processing functions"""
        nyquist = 0.5 * self.fs if self.fs else 1.0

        # Normalize cutoff frequencies
        if self.btype in ["bandpass", "bandstop"]:
            cutoff_norm = (
                [self.cutoff[0] / nyquist, self.cutoff[1] / nyquist]
                if self.fs
                else self.cutoff
            )
        else:
            cutoff_norm = self.cutoff / nyquist if self.fs else self.cutoff

        # Create filter based on type
        if self.filter_type == "butter":
            self._sos = signal.butter(
                self.order, cutoff_norm, btype=self.btype, output="sos"
            )
        elif self.filter_type == "cheby1":
            self._sos = signal.cheby1(
                self.order, self.rp, cutoff_norm, btype=self.btype, output="sos"
            )
        elif self.filter_type == "cheby2":
            self._sos = signal.cheby2(
                self.order, self.rs, cutoff_norm, btype=self.btype, output="sos"
            )
        elif self.filter_type == "ellip":
            self._sos = signal.ellip(
                self.order,
                self.rp,
                self.rs,
                cutoff_norm,
                btype=self.btype,
                output="sos",
            )
        elif self.filter_type == "bessel":
            self._sos = signal.bessel(
                self.order, cutoff_norm, btype=self.btype, output="sos", norm="delay"
            )

    def filter(
        self,
        data: np.ndarray,
        fs: Optional[float] = None,
        filtfilt: bool = True,
        parallel: bool = True,
        use_rust: bool = True,
    ) -> np.ndarray:
        """
        Apply the filter to input data.

        Args:
            data: Input data array
            fs: Sampling frequency (required if not provided at initialization)
            filtfilt: Whether to apply zero-phase filtering (forward-backward)
            parallel: Whether to use parallel processing for multi-channel data
            use_rust: Whether to use Rust acceleration if available

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

        # Ensure data is at least 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            reshape_back = True
        else:
            reshape_back = False

        # Use Rust implementation if available and requested
        if _HAS_RUST and use_rust and parallel and data.shape[1] > 1:
            # Select the appropriate filter function based on filter type
            if self.filter_type == "butter":
                rust_filter_func = apply_butter_filter
            elif self.filter_type == "cheby1" or self.filter_type == "cheby2":
                rust_filter_func = apply_cheby_filter
            elif self.filter_type == "ellip":
                rust_filter_func = apply_elliptic_filter
            elif self.filter_type == "bessel":
                rust_filter_func = apply_bessel_filter
            else:
                # Fallback to generic filter
                rust_filter_func = parallel_filter_channels

            # Apply the filter (note: Rust functions always expect Option<bool>, not bool)
            result = rust_filter_func(
                data.astype(np.float32),
                self._sos.astype(np.float32),
                None
                if filtfilt
                else False,  # Use default (True) or explicitly set False
            )
        else:
            # Fall back to SciPy implementation
            if filtfilt:
                if parallel and data.ndim > 1 and data.shape[1] > 1 and _HAS_RUST:
                    # Use our optimized parallel implementation
                    result = parallel_filter_channels(
                        data, self._sos, None
                    )  # Use default (True)
                else:
                    result = signal.sosfiltfilt(self._sos, data, axis=0)
            else:
                # Forward-only filtering
                if parallel and data.ndim > 1 and data.shape[1] > 1 and _HAS_RUST:
                    result = parallel_filter_channels(
                        data, self._sos, False
                    )  # Pass False explicitly
                else:
                    result = signal.sosfilt(self._sos, data, axis=0)

        # Reshape back to original dimensions if needed
        if reshape_back:
            result = result.ravel()

        return result

    def get_filter_function(self, filtfilt: bool = True) -> ProcessingFunction:
        """
        Get a filter function compatible with FilterProcessor.

        Args:
            filtfilt: Whether to apply zero-phase filtering (forward-backward)

        Returns:
            Function that applies the filter
        """
        # Create a filter function with the parameters from this instance
        filter_args = {
            "type": self.filter_type,
            "btype": self.btype,
            "cutoff": self.cutoff,
            "order": self.order,
            "rs": self.rs,
            "rp": self.rp,
            "filtfilt": filtfilt,
        }

        def filter_function(
            chunk: np.ndarray,
            fs: float,
            parallel: bool = True,
            use_rust: bool = True,
            **kwargs,
        ) -> np.ndarray:
            # Update this instance with the provided sampling rate
            if fs != self.fs:
                self.fs = fs
                self._create_filter_coefficients()

            # Get filtfilt setting from kwargs or default
            filtfilt_param = kwargs.get("filtfilt", filtfilt)

            return self.filter(chunk, fs, filtfilt_param, parallel, use_rust)

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
        freq_scale: Literal["linear", "log"] = "log",
        cutoff_lines: bool = True,
        grid: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        y_min: float = -80,
        y_max: Optional[float] = None,
    ):
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
            y_min: Minimum value for magnitude y-axis in dB (default: -80 dB)
            y_max: Maximum value for magnitude y-axis in dB (default: auto)

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt

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

        # Use semilogx for log scale plotting of frequency axis
        if freq_scale == "log" and plot_fs is not None:
            ax_mag.semilogx(w, 20 * np.log10(abs(h)), "b", linewidth=2)
            # Avoid zero frequency for log scale
            min_freq = max(w[1], 0.1)
            ax_mag.set_xlim(min_freq, plot_fs / 2)
        else:
            ax_mag.plot(w, 20 * np.log10(abs(h)), "b", linewidth=2)

        # Set y-axis limits
        ax_mag.set_ylim(bottom=y_min, top=y_max)
        ax_mag.set_ylabel("Magnitude [dB]")

        # Add cutoff lines
        if cutoff_lines:
            if self.btype in ["bandpass", "bandstop"]:
                cutoffs = (
                    self.cutoff
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
                    # Add text label near the cutoff line
                    if ax is ax_mag:  # Only add text to magnitude plot
                        text_y = ax_mag.get_ylim()[0] + 0.1 * (
                            ax_mag.get_ylim()[1] - ax_mag.get_ylim()[0]
                        )
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
        if grid:
            for ax in axes:
                if freq_scale == "log":
                    ax.grid(True, which="major", linestyle="-", alpha=0.4)
                    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
                else:
                    ax.grid(True, alpha=0.3)

        # Plot phase response
        if show_phase:
            ax_phase = axes[1] if num_plots > 1 else axes[0]
            angles = np.unwrap(np.angle(h))

            if freq_scale == "log":
                ax_phase.semilogx(w, angles, "g", linewidth=2)
            else:
                ax_phase.plot(w, angles, "g", linewidth=2)

            ax_phase.set_ylabel("Phase [rad]")

        # Plot group delay
        if show_group_delay:
            ax_gd = axes[-1]
            # Calculate group delay
            group_delay = -np.diff(np.unwrap(np.angle(h))) / np.diff(w)
            # Pad to match original length
            group_delay = np.concatenate([group_delay, [group_delay[-1]]])

            if freq_scale == "log":
                ax_gd.semilogx(w, group_delay, "m", linewidth=2)
            else:
                ax_gd.plot(w, group_delay, "m", linewidth=2)

            ax_gd.set_ylabel("Group Delay [s]")

        # Set x-axis label on bottom plot
        axes[-1].set_xlabel(
            "Frequency [Hz]" if plot_fs is not None else "Normalized Frequency"
        )

        # Set title
        if title is None:
            filter_type_names = {
                "butter": "Butterworth",
                "cheby1": "Chebyshev I",
                "cheby2": "Chebyshev II",
                "ellip": "Elliptic",
                "bessel": "Bessel",
            }

            title_parts = []
            title_parts.append(
                filter_type_names.get(self.filter_type, self.filter_type)
            )

            if self.btype == "lowpass":
                title_parts.append(f"Lowpass ({self.cutoff:.1f} Hz)")
            elif self.btype == "highpass":
                title_parts.append(f"Highpass ({self.cutoff:.1f} Hz)")
            elif self.btype == "bandpass":
                title_parts.append(
                    f"Bandpass ({self.cutoff[0]:.1f}-{self.cutoff[1]:.1f} Hz)"
                )
            elif self.btype == "bandstop":
                title_parts.append(
                    f"Bandstop ({self.cutoff[0]:.1f}-{self.cutoff[1]:.1f} Hz)"
                )

            title_parts.append(f"Order {self.order}")

            if self.filter_type in ["cheby1", "ellip"]:
                title_parts.append(f"Rp={self.rp:.1f}dB")

            if self.filter_type in ["cheby2", "ellip"]:
                title_parts.append(f"Rs={self.rs:.1f}dB")

            if plot_fs is not None:
                title_parts.append(f"fs={plot_fs} Hz")

            title = " ".join(title_parts)

        fig.suptitle(title)
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


@public_api
class IIRFilterProcessor(BaseProcessor):
    """
    IIR Filter processor for use with Dask arrays.

    This processor applies IIR filtering to multi-channel data using Rust acceleration
    when available, with support for various filter types and configurations.
    """

    def __init__(
        self,
        filter_type: Literal[
            "butter", "cheby1", "cheby2", "ellip", "bessel"
        ] = "butter",
        btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
        cutoff: Union[float, Tuple[float, float]] = 100.0,
        order: int = 4,
        rs: float = 40,
        rp: float = 1,
        filtfilt: bool = True,
        overlap_scale: float = 3.0,
        use_rust: bool = True,
    ):
        """
        Initialize the IIR filter processor.

        Args:
            filter_type: Filter family type
            btype: Filter response type
            cutoff: Cutoff frequency in Hz (single value or tuple)
            order: Filter order
            rs: Minimum attenuation in stop band (dB)
            rp: Maximum ripple in passband (dB)
            filtfilt: Whether to use zero-phase (forward-backward) filtering
            overlap_scale: Scale factor for overlap samples (overlap = order * overlap_scale)
            use_rust: Whether to use Rust acceleration if available
        """
        self.filter_type = filter_type
        self.btype = btype
        self.cutoff = cutoff
        self.order = order
        self.rs = rs
        self.rp = rp
        self.filtfilt = filtfilt
        self.overlap_scale = overlap_scale
        self.use_rust = use_rust and _HAS_RUST

        # Create filter object
        self.filter = IIRFilter(
            filter_type=filter_type,
            btype=btype,
            cutoff=cutoff,
            order=order,
            rs=rs,
            rp=rp,
        )

        # Calculate overlap based on filter order
        self._overlap_samples = int(self.order * self.overlap_scale)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply IIR filtering to the input data using Dask's map_overlap.

        Args:
            data: Input Dask array
            fs: Sampling frequency in Hz (required)
            **kwargs: Additional keyword arguments
                filtfilt: Override the filtfilt setting
                use_rust: Override the use_rust setting
                parallel: Whether to use parallel processing

        Returns:
            Filtered Dask array
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Update filter's sampling rate
        if self.filter.fs != fs:
            self.filter.fs = fs
            self.filter._create_filter_coefficients()

        # Get parameters from kwargs with defaults
        filtfilt = kwargs.get("filtfilt", self.filtfilt)
        use_rust = kwargs.get("use_rust", self.use_rust)
        parallel = kwargs.get("parallel", True)

        # Define function to apply to each chunk
        def apply_filter(chunk: np.ndarray) -> np.ndarray:
            return self.filter.filter(
                chunk, fs=fs, filtfilt=filtfilt, parallel=parallel, use_rust=use_rust
            )

        # Use map_overlap to apply the filter with proper overlap
        return data.map_overlap(
            apply_filter,
            depth={-2: self._overlap_samples},  # Only need overlap in time dimension
            boundary="reflect",
            dtype=data.dtype,
        )

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for proper overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "filter_type": self.filter_type,
                "btype": self.btype,
                "cutoff": self.cutoff,
                "order": self.order,
                "rs": self.rs if self.filter_type in ["cheby2", "ellip"] else None,
                "rp": self.rp if self.filter_type in ["cheby1", "ellip"] else None,
                "filtfilt": self.filtfilt,
                "rust_acceleration": self.use_rust and _HAS_RUST,
            }
        )
        return base_summary


@public_api
class CascadedFilterProcessor(BaseProcessor):
    """
    Processor for applying multiple filters in cascade.

    This processor allows creating complex filter responses by cascading
    multiple IIR filters in sequence.
    """

    def __init__(
        self,
        filters: List[IIRFilter],
        filtfilt: bool = True,
        use_rust: bool = True,
    ):
        """
        Initialize the cascaded filter processor.

        Args:
            filters: List of IIRFilter objects to apply in sequence
            filtfilt: Whether to use zero-phase filtering
            use_rust: Whether to use Rust acceleration if available
        """
        self.filters = filters
        self.filtfilt = filtfilt
        self.use_rust = use_rust and _HAS_RUST

        # Calculate overlap based on the sum of all filter orders
        total_order = sum(f.order for f in filters)
        self._overlap_samples = total_order * 3  # Use 3x for safety

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply cascaded filtering to the input data.

        Args:
            data: Input Dask array
            fs: Sampling frequency in Hz (required)
            **kwargs: Additional keyword arguments

        Returns:
            Filtered Dask array
        """
        if fs is None:
            raise ValueError("Sampling frequency (fs) must be provided")

        # Update sampling rates and prepare filter coefficients
        for filter_obj in self.filters:
            if filter_obj.fs != fs:
                filter_obj.fs = fs
                filter_obj._create_filter_coefficients()

        # Get parameters from kwargs
        filtfilt = kwargs.get("filtfilt", self.filtfilt)
        use_rust = kwargs.get("use_rust", self.use_rust)

        # Define the cascaded filtering function
        def apply_cascaded_filters(chunk: np.ndarray) -> np.ndarray:
            # If Rust implementation is available and requested, use it for multiple filters
            if _HAS_RUST and use_rust and len(self.filters) > 1:
                # Prepare the SOS matrices
                sos_list = [
                    filter_obj._sos.astype(np.float32) for filter_obj in self.filters
                ]

                # Apply cascaded filters using Rust
                # Pass None as the filtfilt parameter to use default (True)
                return apply_cascaded_filters(
                    chunk.astype(np.float32),
                    sos_list,
                    None
                    if filtfilt
                    else False,  # Use default (True) or explicitly set False
                )
            else:
                # Apply filters sequentially using Python
                result = chunk
                for filter_obj in self.filters:
                    result = filter_obj.filter(
                        result, fs=fs, filtfilt=filtfilt, use_rust=use_rust
                    )
                return result

        # Use map_overlap to apply the filters with proper overlap
        return data.map_overlap(
            apply_cascaded_filters,
            depth={-2: self._overlap_samples},
            boundary="reflect",
            dtype=data.dtype,
        )

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for proper overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the processor configuration"""
        base_summary = super().summary

        # Create summaries for each filter
        filter_summaries = []
        for i, f in enumerate(self.filters):
            filter_summaries.append(
                {
                    "index": i,
                    "type": f.filter_type,
                    "btype": f.btype,
                    "cutoff": f.cutoff,
                    "order": f.order,
                }
            )

        base_summary.update(
            {
                "num_filters": len(self.filters),
                "filters": filter_summaries,
                "filtfilt": self.filtfilt,
                "rust_acceleration": self.use_rust and _HAS_RUST,
            }
        )
        return base_summary


## Factory functions

# Factory functions for easier creation of specific filter types


@public_api
def create_lowpass_filter(
    cutoff: float,
    fs: Optional[float] = None,
    filter_type: Literal["butter", "cheby1", "cheby2", "ellip", "bessel"] = "butter",
    order: int = 4,
    rs: float = 40,
    rp: float = 1,
) -> IIRFilter:
    """
    Create a lowpass IIR filter.

    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        filter_type: Filter family type
        order: Filter order
        rs: Minimum attenuation in stop band (dB), for Chebyshev II and elliptic
        rp: Maximum ripple in passband (dB), for Chebyshev I and elliptic

    Returns:
        Configured IIRFilter object
    """
    return IIRFilter(
        filter_type=filter_type,
        btype="lowpass",
        cutoff=cutoff,
        order=order,
        rs=rs,
        rp=rp,
        fs=fs,
    )


@public_api
def create_highpass_filter(
    cutoff: float,
    fs: Optional[float] = None,
    filter_type: Literal["butter", "cheby1", "cheby2", "ellip", "bessel"] = "butter",
    order: int = 4,
    rs: float = 40,
    rp: float = 1,
) -> IIRFilter:
    """
    Create a highpass IIR filter.

    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        filter_type: Filter family type
        order: Filter order
        rs: Minimum attenuation in stop band (dB), for Chebyshev II and elliptic
        rp: Maximum ripple in passband (dB), for Chebyshev I and elliptic

    Returns:
        Configured IIRFilter object
    """
    return IIRFilter(
        filter_type=filter_type,
        btype="highpass",
        cutoff=cutoff,
        order=order,
        rs=rs,
        rp=rp,
        fs=fs,
    )


@public_api
def create_bandpass_filter(
    lowcut: float,
    highcut: float,
    fs: Optional[float] = None,
    filter_type: Literal["butter", "cheby1", "cheby2", "ellip", "bessel"] = "butter",
    order: int = 4,
    rs: float = 40,
    rp: float = 1,
) -> IIRFilter:
    """
    Create a bandpass IIR filter.

    Args:
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        fs: Sampling frequency in Hz
        filter_type: Filter family type
        order: Filter order
        rs: Minimum attenuation in stop band (dB), for Chebyshev II and elliptic
        rp: Maximum ripple in passband (dB), for Chebyshev I and elliptic

    Returns:
        Configured IIRFilter object
    """
    return IIRFilter(
        filter_type=filter_type,
        btype="bandpass",
        cutoff=(lowcut, highcut),
        order=order,
        rs=rs,
        rp=rp,
        fs=fs,
    )


@public_api
def create_notch_filter(
    center_freq: float,
    q: float = 30,
    fs: Optional[float] = None,
    filter_type: Literal["butter", "cheby1", "cheby2", "ellip", "bessel"] = "butter",
    order: int = 4,
    rs: float = 40,
    rp: float = 1,
) -> IIRFilter:
    """
    Create a notch (band-stop) IIR filter.

    Args:
        center_freq: Center frequency to attenuate in Hz
        q: Quality factor (higher means narrower notch)
        fs: Sampling frequency in Hz
        filter_type: Filter family type
        order: Filter order
        rs: Minimum attenuation in stop band (dB), for Chebyshev II and elliptic
        rp: Maximum ripple in passband (dB), for Chebyshev I and elliptic

    Returns:
        Configured IIRFilter object
    """
    # Calculate bandwidth from Q factor
    bandwidth = center_freq / q
    lowcut = center_freq - bandwidth / 2
    highcut = center_freq + bandwidth / 2

    return IIRFilter(
        filter_type=filter_type,
        btype="bandstop",
        cutoff=(lowcut, highcut),
        order=order,
        rs=rs,
        rp=rp,
        fs=fs,
    )


@public_api
def create_filter_processor(
    filter_type: Literal["butter", "cheby1", "cheby2", "ellip", "bessel"] = "butter",
    btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass",
    cutoff: Union[float, Tuple[float, float]] = 100.0,
    order: int = 4,
    rs: float = 40,
    rp: float = 1,
    filtfilt: bool = True,
    use_rust: bool = True,
) -> IIRFilterProcessor:
    """
    Create an IIR filter processor for use in a processing pipeline.

    Args:
        filter_type: Filter family type
        btype: Filter response type
        cutoff: Cutoff frequency in Hz (single value or tuple)
        order: Filter order
        rs: Minimum attenuation in stop band (dB)
        rp: Maximum ripple in passband (dB)
        filtfilt: Whether to use zero-phase (forward-backward) filtering
        use_rust: Whether to use Rust acceleration if available

    Returns:
        Configured IIRFilterProcessor
    """
    return IIRFilterProcessor(
        filter_type=filter_type,
        btype=btype,
        cutoff=cutoff,
        order=order,
        rs=rs,
        rp=rp,
        filtfilt=filtfilt,
        use_rust=use_rust,
    )


@public_api
def create_cascaded_filter(
    filters: List[IIRFilter],
    filtfilt: bool = True,
    use_rust: bool = True,
) -> CascadedFilterProcessor:
    """
    Create a cascaded filter processor for complex filter responses.

    Args:
        filters: List of IIRFilter objects to apply in sequence
        filtfilt: Whether to use zero-phase filtering
        use_rust: Whether to use Rust acceleration if available

    Returns:
        Configured CascadedFilterProcessor
    """
    return CascadedFilterProcessor(
        filters=filters,
        filtfilt=filtfilt,
        use_rust=use_rust,
    )


# Specifically for traditional Butterworth filters to match the original API
@public_api
def create_butter_lowpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    """
    Create a Butterworth lowpass filter function compatible with FilterProcessor.

    Args:
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Filter function that can be passed to FilterProcessor
    """
    filter_obj = IIRFilter("butter", "lowpass", cutoff, order)
    return filter_obj.get_filter_function()


@public_api
def create_butter_highpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    """
    Create a Butterworth highpass filter function compatible with FilterProcessor.

    Args:
        cutoff: Cutoff frequency in Hz
        order: Filter order

    Returns:
        Filter function that can be passed to FilterProcessor
    """
    filter_obj = IIRFilter("butter", "highpass", cutoff, order)
    return filter_obj.get_filter_function()


@public_api
def create_butter_bandpass_filter(
    lowcut: float, highcut: float, order: int = 4
) -> ProcessingFunction:
    """
    Create a Butterworth bandpass filter function compatible with FilterProcessor.

    Args:
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        order: Filter order

    Returns:
        Filter function that can be passed to FilterProcessor
    """
    filter_obj = IIRFilter("butter", "bandpass", (lowcut, highcut), order)
    return filter_obj.get_filter_function()


@public_api
def create_butter_notch_filter(
    notch_freq: float, q: float = 30, order: int = 4
) -> ProcessingFunction:
    """
    Create a Butterworth notch filter function compatible with FilterProcessor.

    Args:
        notch_freq: Notch frequency in Hz
        q: Quality factor (higher means narrower notch)
        order: Filter order

    Returns:
        Filter function that can be passed to FilterProcessor
    """
    # Calculate bandwidth from Q factor
    bandwidth = notch_freq / q
    lowcut = notch_freq - bandwidth / 2
    highcut = notch_freq + bandwidth / 2

    filter_obj = IIRFilter("butter", "bandstop", (lowcut, highcut), order)
    return filter_obj.get_filter_function()
