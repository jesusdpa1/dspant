"""
Filter processors for dspant.

This module provides filter implementations for various types of filters:
- Butterworth filters (lowpass, highpass, bandpass, notch)
- Feed Forward Comb (FFC) filter
- Wavelet Packet Harmonic Interference Removal
- Base classes for custom filter implementations
- Filter visualization tools
"""

from .base import FilterProcessor, parallel_filter_channels
from .butter_filters import (
    ButterFilter,
    create_bandpass_filter,
    create_highpass_filter,
    create_lowpass_filter,
    create_notch_filter,
    plot_filter_response,
)
from .ffc_filter import (
    FFCFilter,
    create_ffc_enhancement,
    create_ffc_filter,
    create_ffc_notch,
)
from .wp_harmonic_removal import (
    WaveletPacketHarmonicRemoval,
    create_powerline_filter,
    create_wp_harmonic_removal,
)


# Factory function for creating a complete filter processor
def create_filter_processor(
    filter_type: str,
    cutoff_low: float = None,
    cutoff_high: float = None,
    order: int = 4,
    q: float = 30,
    parallel: bool = True,
):
    """
    Create a complete FilterProcessor with the specified filter type.

    Args:
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'notch')
        cutoff_low: Lower cutoff frequency in Hz
        cutoff_high: Upper cutoff frequency in Hz (for bandpass only)
        order: Filter order
        q: Quality factor (for notch filter)
        parallel: Whether to use parallel processing

    Returns:
        Configured FilterProcessor ready to use
    """
    if filter_type == "lowpass":
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for lowpass filter")
        filter_func = create_lowpass_filter(cutoff_low, order)
        overlap = order * 10
    elif filter_type == "highpass":
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for highpass filter")
        filter_func = create_highpass_filter(cutoff_low, order)
        overlap = order * 10
    elif filter_type == "bandpass":
        if cutoff_low is None or cutoff_high is None:
            raise ValueError(
                "Both cutoff_low and cutoff_high must be specified for bandpass filter"
            )
        filter_func = create_bandpass_filter(cutoff_low, cutoff_high, order)
        overlap = order * 10
    elif filter_type == "notch":
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for notch filter")
        filter_func = create_notch_filter(cutoff_low, q, order)
        overlap = order * 10
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return FilterProcessor(filter_func, overlap, parallel)


# Create a filter with visualization capabilities
def create_visualizable_filter(
    filter_type: str,
    cutoff_low: float = None,
    cutoff_high: float = None,
    order: int = 4,
    q: float = 30,
    fs: float = None,
) -> ButterFilter:
    """
    Create a filter object with visualization capabilities.

    Args:
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'notch')
        cutoff_low: Lower cutoff frequency in Hz
        cutoff_high: Upper cutoff frequency in Hz (for bandpass/notch)
        order: Filter order
        q: Quality factor (for notch filter)
        fs: Sampling frequency in Hz

    Returns:
        ButterFilter object with visualization methods
    """
    if filter_type == "lowpass":
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for lowpass filter")
        return ButterFilter("lowpass", cutoff_low, order=order, fs=fs)

    elif filter_type == "highpass":
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for highpass filter")
        return ButterFilter("highpass", cutoff_low, order=order, fs=fs)

    elif filter_type == "bandpass":
        if cutoff_low is None or cutoff_high is None:
            raise ValueError(
                "Both cutoff_low and cutoff_high must be specified for bandpass filter"
            )
        return ButterFilter("bandpass", (cutoff_low, cutoff_high), order=order, fs=fs)

    elif filter_type == "notch":
        if cutoff_low is None:
            raise ValueError("cutoff_low must be specified for notch filter")
        # Calculate bandstop parameters from notch frequency and Q
        bandwidth = cutoff_low / q
        low = cutoff_low - bandwidth / 2
        high = cutoff_low + bandwidth / 2
        return ButterFilter("bandstop", (low, high), order=order, fs=fs)

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


__all__ = [
    # Base classes
    "FilterProcessor",
    "parallel_filter_channels",
    # Filter class with visualization
    "ButterFilter",
    "plot_filter_response",
    # Filter creation functions
    "create_bandpass_filter",
    "create_highpass_filter",
    "create_lowpass_filter",
    "create_notch_filter",
    # Factory functions
    "create_filter_processor",
    "create_visualizable_filter",
    # FFC Filter
    "FFCFilter",
    "create_ffc_filter",
    "create_ffc_notch",
    "create_ffc_enhancement",
    # Wavelet Packet Harmonic Removal
    "WaveletPacketHarmonicRemoval",
    "create_wp_harmonic_removal",
    "create_powerline_filter",
]
