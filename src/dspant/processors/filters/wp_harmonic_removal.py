"""
Wavelet Packet Harmonic Interference Removal implementation.

This module provides functionality to remove harmonic interference (such as power line noise)
from signals using wavelet packet decomposition. The algorithm resamples the signal to a
rate that facilitates identification and removal of harmonic components, performs wavelet
packet decomposition, removes baselines from detail subbands, and reconstructs the signal.

Lijun Xu. “Cancellation of Harmonic Interference by Baseline Shifting of Wavelet Packet Decomposition Coefficients.” IEEE Transactions on Signal Processing 53, no. 1 (January 2005): 222–30. https://doi.org/10.1109/TSP.2004.838954.
"""

from typing import Any, Dict, Optional, Tuple

import dask.array as da
import numpy as np
import pywt
from scipy import ndimage, signal

from ...engine.base import BaseProcessor


def determine_resampling_params(sample_rate: float, f0: float) -> Tuple[float, int]:
    """
    Determine optimal resampling parameters for harmonic interference removal.

    The function calculates a new sampling rate such that the fundamental frequency
    of the interference (f0) and its harmonics align well with the wavelet packet
    decomposition structure.

    Args:
        sample_rate: Original sampling rate in Hz
        f0: Fundamental frequency of the interference in Hz

    Returns:
        Tuple of (new_sample_rate, decomposition_level)
    """
    # Find a power of 2 multiple of f0 that's reasonably close to the sample rate
    # but not higher than the original sample rate
    j_max = int(np.floor(np.log2(sample_rate / f0)))
    new_sample_rate = f0 * 2**j_max

    # Ensure new sample rate is not higher than original
    if new_sample_rate > sample_rate:
        j_max -= 1
        new_sample_rate = f0 * 2**j_max

    return new_sample_rate, j_max


def resample_data(
    data: np.ndarray, original_rate: float, new_rate: float
) -> np.ndarray:
    """
    Resample data to a new sampling rate.

    Args:
        data: Input signal array
        original_rate: Original sampling rate in Hz
        new_rate: New sampling rate in Hz

    Returns:
        Resampled data array
    """
    # Calculate resampling ratio
    ratio = new_rate / original_rate

    # Determine number of samples in resampled signal
    n_samples = int(len(data) * ratio)

    # Resample using scipy's resample function for 1D arrays
    resampled_data = signal.resample(data, n_samples)

    return resampled_data


def estimate_baseline(
    coeffs: np.ndarray, window_size: Optional[int] = None
) -> np.ndarray:
    """
    Estimate the baseline of wavelet coefficients.

    This function identifies and extracts the low-frequency baseline component
    from wavelet detail coefficients, which often represents harmonic interference.

    Args:
        coeffs: Wavelet detail coefficients
        window_size: Size of the smoothing window. If None, an appropriate size is calculated.

    Returns:
        Baseline array with the same shape as coeffs
    """
    # Set default window size if not specified
    if window_size is None:
        window_size = max(5, len(coeffs) // 50)

    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Apply median filter to estimate baseline
    baseline = signal.medfilt(coeffs, kernel_size=window_size)

    # Further smooth with a Gaussian filter
    sigma = window_size / 6.0  # Heuristic: sigma = window_size/6
    baseline = ndimage.gaussian_filter1d(baseline, sigma)

    return baseline


class WaveletPacketHarmonicRemoval(BaseProcessor):
    """
    Wavelet Packet Harmonic Interference Removal processor implementation.

    This processor removes power line interference and other harmonic noise from signals
    using wavelet packet decomposition. The algorithm:
    1. Resamples the signal to optimize wavelet packet analysis for the target frequency
    2. Performs wavelet packet decomposition
    3. Estimates and removes baseline from detail subbands
    4. Reconstructs the cleaned signal
    5. Resamples back to the original sampling rate
    """

    def __init__(
        self,
        fundamental_freq: float = 60.0,
        wavelet: str = "db4",
        max_level: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        """
        Initialize the Wavelet Packet Harmonic Removal processor.

        Args:
            fundamental_freq: Fundamental frequency of the interference in Hz (e.g., 50 or 60 for power lines)
            wavelet: Wavelet type to use for decomposition (e.g., 'db4', 'sym5')
            max_level: Maximum decomposition level. If None, it will be determined automatically.
            window_size: Size of the window for baseline estimation. If None, it will be determined automatically.
        """
        self.fundamental_freq = fundamental_freq
        self.wavelet = wavelet
        self.max_level = max_level
        self.window_size = window_size
        self._overlap_samples = 0  # Will be set during processing

        # These will be set during processing
        self._new_sample_rate = None
        self._decomposition_level = None

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply wavelet packet harmonic interference removal to the input data.

        Args:
            data: Input dask array
            fs: Sampling frequency in Hz (required)
            **kwargs: Additional keyword arguments
                fundamental_freq: Override the fundamental frequency set during initialization
                wavelet: Override the wavelet type set during initialization
                max_level: Override the maximum decomposition level set during initialization
                window_size: Override the window size set during initialization

        Returns:
            Cleaned data array
        """
        if fs is None:
            raise ValueError(
                "Sampling frequency (fs) is required for wavelet packet harmonic removal"
            )

        # Get parameters from kwargs or use defaults from initialization
        f0 = kwargs.get("fundamental_freq", self.fundamental_freq)
        wavelet = kwargs.get("wavelet", self.wavelet)
        max_level = kwargs.get("max_level", self.max_level)
        window_size = kwargs.get("window_size", self.window_size)

        # Step 1 & 2: Determine resampling parameters
        new_sample_rate, decomp_level = determine_resampling_params(fs, f0)
        self._new_sample_rate = new_sample_rate
        self._decomposition_level = decomp_level

        # If max_level not specified, use calculated decomposition level
        if max_level is None:
            max_level = decomp_level

        # Set overlap samples based on wavelet filter length and decomposition level
        wavelet_obj = pywt.Wavelet(wavelet)
        filter_length = wavelet_obj.dec_len
        # Estimate maximum overlap based on filter length and decomposition level
        max_filter_spread = filter_length * (2**max_level - 1)
        self._overlap_samples = max_filter_spread

        # Define the processing function
        def process_chunk(chunk):
            # Ensure chunk is 1D
            original_shape = chunk.shape
            if chunk.ndim > 1:
                # Process each channel separately
                result = np.zeros_like(chunk)
                for c in range(chunk.shape[1]):
                    result[:, c] = process_chunk(chunk[:, c])
                return result

            # Handle 1D data
            data_1d = chunk.ravel()

            # Step 3: Resample data
            resampled_data = resample_data(data_1d, fs, new_sample_rate)

            # Step 4 & 5: Wavelet packet transform and baseline removal
            wp = pywt.WaveletPacket(
                data=resampled_data,
                wavelet=wavelet,
                mode="symmetric",
                maxlevel=max_level,
            )

            # Process each detail subband
            for level in range(
                1, max_level + 1
            ):  # Start from level 1 since level 0 is just the original signal
                for path in [node.path for node in wp.get_level(level, "natural")]:
                    node = wp[path]
                    # Only process detail subbands (containing high-frequency components)
                    if "d" in path:
                        coeffs = node.data
                        node_window_size = window_size
                        if node_window_size is None:
                            # Adapt window size based on level
                            node_window_size = max(5, len(coeffs) // (20 * level))
                        baseline = estimate_baseline(coeffs, node_window_size)
                        # Subtract baseline from coefficients
                        node.data = coeffs - baseline

            # Reconstruct signal
            cleaned_resampled = wp.reconstruct(update=True)

            # Step 6: Resample back to original rate
            cleaned_data = resample_data(cleaned_resampled, new_sample_rate, fs)

            # Ensure output length matches input length
            if len(cleaned_data) > len(data_1d):
                cleaned_data = cleaned_data[: len(data_1d)]
            elif len(cleaned_data) < len(data_1d):
                cleaned_data = np.pad(
                    cleaned_data, (0, len(data_1d) - len(cleaned_data))
                )

            # Reshape back to original shape if necessary
            if original_shape != chunk.shape:
                cleaned_data = cleaned_data.reshape(original_shape)

            return cleaned_data

        # Apply processing function to data with overlap
        return data.map_overlap(
            process_chunk,
            depth={0: self._overlap_samples},
            boundary="reflect",
            dtype=data.dtype,
        )

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "fundamental_freq": self.fundamental_freq,
                "wavelet": self.wavelet,
                "max_level": self.max_level,
                "window_size": self.window_size,
                "resampling_rate": self._new_sample_rate,
                "decomposition_level": self._decomposition_level,
            }
        )
        return base_summary


def create_wp_harmonic_removal(
    fundamental_freq: float = 50.0,
    wavelet: str = "db4",
    max_level: Optional[int] = None,
    window_size: Optional[int] = None,
) -> WaveletPacketHarmonicRemoval:
    """
    Create a Wavelet Packet Harmonic Interference Removal processor.

    Args:
        fundamental_freq: Fundamental frequency of the interference in Hz (e.g., 50 or 60 for power lines)
        wavelet: Wavelet type to use for decomposition (e.g., 'db4', 'sym5')
        max_level: Maximum decomposition level. If None, it will be determined automatically.
        window_size: Size of the window for baseline estimation. If None, it will be determined automatically.

    Returns:
        Configured WaveletPacketHarmonicRemoval processor
    """
    return WaveletPacketHarmonicRemoval(
        fundamental_freq=fundamental_freq,
        wavelet=wavelet,
        max_level=max_level,
        window_size=window_size,
    )


def create_powerline_filter(
    fundamental_freq: float = 50.0,
    wavelet: str = "db4",
) -> WaveletPacketHarmonicRemoval:
    """
    Create a Wavelet Packet processor configured for power line interference removal.

    Args:
        fundamental_freq: Power line frequency in Hz (50Hz in Europe/Asia, 60Hz in North America)
        wavelet: Wavelet type to use for decomposition

    Returns:
        Configured WaveletPacketHarmonicRemoval processor for power line removal
    """
    return WaveletPacketHarmonicRemoval(
        fundamental_freq=fundamental_freq,
        wavelet=wavelet,
        max_level=None,  # Auto-determine based on sampling rate and fundamental frequency
        window_size=None,  # Auto-determine based on signal length
    )
