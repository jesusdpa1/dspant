"""
Spectral processors for dspant.

This module provides processors for spectral analysis of time-series data:
- Spectrogram for basic time-frequency analysis
- MFCC for mel-frequency cepstral coefficients
- LFCC for linear-frequency cepstral coefficients
"""

import torch

from .stft_base import (
    LFCCProcessor,
    MFCCProcessor,
    SpectrogramProcessor,
)

# Factory functions to create processors with common configurations


def create_spectrogram(
    n_fft: int = 400,
    hop_length: int = None,
    window: str = "hann",
    power: float = 2.0,
    **kwargs,
) -> SpectrogramProcessor:
    """
    Create a spectrogram processor with common parameters.

    Args:
        n_fft: FFT size
        hop_length: Hop length between frames (default: n_fft // 2)
        window: Window type ('hann', 'hamming', 'blackman')
        power: Exponent for the magnitude spectrogram
        **kwargs: Additional keyword arguments for SpectrogramProcessor

    Returns:
        Configured SpectrogramProcessor
    """
    # Map window name to function
    window_map = {
        "hann": torch.hann_window,
        "hamming": torch.hamming_window,
        "blackman": torch.blackman_window,
    }
    window_fn = window_map.get(window.lower(), torch.hann_window)

    return SpectrogramProcessor(
        n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, power=power, **kwargs
    )


def create_mfcc(
    n_mfcc: int = 13,
    n_fft: int = 400,
    hop_length: int = 200,
    n_mels: int = 128,
    **kwargs,
) -> MFCCProcessor:
    """
    Create an MFCC processor with common parameters.

    Args:
        n_mfcc: Number of MFCC to return
        n_fft: FFT size
        hop_length: Hop length between frames
        n_mels: Number of mel filterbanks
        **kwargs: Additional keyword arguments for MFCCProcessor

    Returns:
        Configured MFCCProcessor
    """
    melkwargs = {"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels}

    # Add any other mel kwargs from the function arguments
    for key, value in kwargs.pop("melkwargs", {}).items():
        melkwargs[key] = value

    return MFCCProcessor(n_mfcc=n_mfcc, melkwargs=melkwargs, **kwargs)


def create_lfcc(
    n_lfcc: int = 20,
    n_filter: int = 128,
    n_fft: int = 400,
    hop_length: int = 200,
    **kwargs,
) -> LFCCProcessor:
    """
    Create an LFCC processor with common parameters.

    Args:
        n_lfcc: Number of LFCC to return
        n_filter: Number of linear filterbanks
        n_fft: FFT size
        hop_length: Hop length between frames
        **kwargs: Additional keyword arguments for LFCCProcessor

    Returns:
        Configured LFCCProcessor
    """
    speckwargs = {
        "n_fft": n_fft,
        "hop_length": hop_length,
    }

    # Add any other spec kwargs from the function arguments
    for key, value in kwargs.pop("speckwargs", {}).items():
        speckwargs[key] = value

    return LFCCProcessor(
        n_lfcc=n_lfcc, n_filter=n_filter, speckwargs=speckwargs, **kwargs
    )


__all__ = [
    # Processor classes
    "SpectrogramProcessor",
    "MFCCProcessor",
    "LFCCProcessor",
    # Factory functions
    "create_spectrogram",
    "create_mfcc",
    "create_lfcc",
]
