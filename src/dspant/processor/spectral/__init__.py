"""
Spectral analysis processors for dspant.

This module provides processors for various time-frequency analysis techniques:
- Spectrograms (STFT-based)
- Mel-Frequency Cepstral Coefficients (MFCC)
- Linear Frequency Cepstral Coefficients (LFCC)
- Synchrosqueezing (CWT and STFT-based)
"""

from .stft_base import (
    LFCCProcessor,
    MFCCProcessor,
    SpectrogramProcessor,
    create_lfcc,
    create_mfcc,
    create_spectrogram,
)

# Import synchrosqueezing module with try/except to handle the case
# where ssqueezepy is not installed
try:
    from .ssqueeze_base import (
        SynchrosqueezingProcessor,
        create_cwt_processor,
        create_ssq_cwt_processor,
        create_ssq_stft_processor,
    )

    HAVE_SSQUEEZEPY = True
except ImportError:
    HAVE_SSQUEEZEPY = False

__all__ = [
    # STFT-based processors
    "SpectrogramProcessor",
    "MFCCProcessor",
    "LFCCProcessor",
    # Factory functions for STFT-based processors
    "create_spectrogram",
    "create_mfcc",
    "create_lfcc",
]

# Add synchrosqueezing exports if available
if HAVE_SSQUEEZEPY:
    __all__.extend(
        [
            # Synchrosqueezing processor
            "SynchrosqueezingProcessor",
            # Factory functions for synchrosqueezing
            "create_cwt_processor",
            "create_ssq_cwt_processor",
            "create_ssq_stft_processor",
        ]
    )
