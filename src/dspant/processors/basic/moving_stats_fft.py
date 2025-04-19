from typing import Any, Dict, Literal, Optional

import dask.array as da
import numpy as np
from numba import jit, prange
from scipy import signal

from dspant.core.internals import public_api

from ...engine.base import BaseProcessor


class FFTMovingAverageProcessor(BaseProcessor):
    """
    Moving average processor using FFT-based convolution for efficient computation.

    This processor calculates the moving average over a sliding window
    using FFT-based convolution, which is significantly faster than direct computation
    for large window sizes.
    """

    def __init__(
        self,
        window_size: int = 11,
        method: Literal["simple", "weighted", "exponential"] = "simple",
        weights: Optional[np.ndarray] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        use_numba_impl: bool = True,
    ):
        """
        Initialize the FFT-based moving average processor.

        Args:
            window_size: Size of the moving window
            method: Moving average type
                "simple": Equal weight to all samples (uniform window)
                "weighted": Custom weights or triangular window
                "exponential": Exponential weighting (causal only)
            weights: Optional custom weights for weighted moving average
            center: If True, window is centered on each point
                   If False, window includes only past samples
            pad_mode: Padding mode for edge handling ('constant', 'reflect', 'edge')
            use_numba_impl: Whether to use Numba acceleration (faster except for very large arrays)
        """
        self.window_size = window_size
        self.method = method
        self.center = center
        self.pad_mode = pad_mode
        self.use_numba_impl = use_numba_impl

        # Set up weights based on method
        if self.method == "weighted":
            if weights is not None:
                # Use provided weights
                self.weights = np.array(weights, dtype=np.float32)
                # Normalize weights to sum to 1
                self.weights = self.weights / np.sum(self.weights)
                # Update window size to match weights length
                self.window_size = len(self.weights)
            else:
                # Default to triangular weights
                self.weights = np.linspace(1, self.window_size, self.window_size)
                # Normalize weights to sum to 1
                self.weights = self.weights / np.sum(self.weights)
        elif self.method == "simple":
            # Equal weights (box filter)
            self.weights = (
                np.ones(self.window_size, dtype=np.float32) / self.window_size
            )
        else:  # exponential
            # Exponential weights implementation requires causal filtering
            self.center = False
            self.weights = None  # special handling in process method

        # Set overlap for map_overlap
        self._overlap_samples = window_size

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply FFT-based moving average to the input data.

        Args:
            data: Input dask array (samples × channels)
            **kwargs: Additional keyword arguments
                alpha: Smoothing factor for exponential moving average

        Returns:
            Data array with moving average applied
        """
        if self.method == "exponential":
            # Exponential moving average uses a different approach
            alpha = kwargs.get("alpha", 0.3)
            return data.map_blocks(
                _apply_exponential_moving_average_parallel
                if self.use_numba_impl
                else _apply_exponential_moving_average,
                alpha=alpha,
                dtype=data.dtype,
            )

        # For simple and weighted methods, use FFT convolution
        parallel = kwargs.get("parallel", True)

        # Select the appropriate implementation
        if self.use_numba_impl:
            ma_func = _apply_fft_ma_numba
        else:
            ma_func = _apply_fft_ma_scipy

        return data.map_overlap(
            ma_func,
            depth=(self._overlap_samples, 0),
            boundary=self.pad_mode,
            weights=self.weights,
            center=self.center,
            dtype=data.dtype,
        )

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "window_size": self.window_size,
                "method": self.method,
                "center": self.center,
                "pad_mode": self.pad_mode,
                "implementation": "fft-based convolution",
                "accelerated": self.use_numba_impl,
            }
        )
        return base_summary


@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def _apply_fft_ma_numba(
    data: np.ndarray, weights: np.ndarray, center: bool = True
) -> np.ndarray:
    """
    Apply moving average using FFT-based convolution with Numba acceleration.

    Args:
        data: Input signal array (samples × channels)
        weights: Window weights (normalized to sum to 1)
        center: Whether to center the window

    Returns:
        Moving average for each point and channel
    """
    # Handle different input shapes
    if data.ndim == 1:
        return _apply_fft_ma_single(data, weights, center)

    n_samples, n_channels = data.shape
    result = np.empty_like(data)

    # Process each channel in parallel
    for ch in prange(n_channels):
        channel_data = data[:, ch]
        result[:, ch] = _apply_fft_ma_single(channel_data, weights, center)

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _apply_fft_ma_single(
    data: np.ndarray, weights: np.ndarray, center: bool = True
) -> np.ndarray:
    """
    Apply moving average using FFT-based convolution for a single channel.

    Args:
        data: Input signal array
        weights: Window weights (normalized to sum to 1)
        center: Whether to center the window

    Returns:
        Moving average for each point
    """
    n_samples = len(data)
    window_size = len(weights)

    # Pad the data to avoid edge effects
    padded_len = n_samples + window_size - 1
    fft_size = 2 ** int(np.ceil(np.log2(padded_len)))

    # FFT of the data
    fft_data = np.fft.rfft(data, fft_size)

    # FFT of the window
    fft_window = np.fft.rfft(weights, fft_size)

    # Multiply in frequency domain
    fft_result = fft_data * fft_window

    # IFFT to get back to time domain
    conv_result = np.fft.irfft(fft_result, fft_size)

    # Extract the valid part of the convolution result
    if center:
        # For centered window, we need to shift the result
        start_idx = window_size // 2
        end_idx = start_idx + n_samples
        result = conv_result[start_idx:end_idx]
    else:
        # For causal window (past samples only)
        result = conv_result[:n_samples]

    return result


def _apply_fft_ma_scipy(
    data: np.ndarray, weights: np.ndarray, center: bool = True
) -> np.ndarray:
    """
    Apply moving average using SciPy's FFT-based convolution.

    Args:
        data: Input signal array
        weights: Window weights (normalized to sum to 1)
        center: Whether to center the window

    Returns:
        Moving average for each point
    """
    # Handle different input shapes
    if data.ndim == 1:
        # Perform fast convolution
        if center:
            # Mode 'same' centers the window
            return signal.fftconvolve(data, weights, mode="same")
        else:
            # For causal filtering (past samples only), use 'full' and shift
            conv_result = signal.fftconvolve(data, weights, mode="full")
            return conv_result[: len(data)]
    else:
        # For multi-channel data
        n_samples, n_channels = data.shape
        result = np.empty_like(data)

        for ch in range(n_channels):
            channel_data = data[:, ch]

            # Perform fast convolution
            if center:
                # Mode 'same' centers the window
                result[:, ch] = signal.fftconvolve(channel_data, weights, mode="same")
            else:
                # For causal filtering (past samples only), use 'full' and shift
                conv_result = signal.fftconvolve(channel_data, weights, mode="full")
                result[:, ch] = conv_result[:n_samples]

        return result


@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def _apply_exponential_moving_average_parallel(
    data: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Apply exponential moving average with parallel processing across channels.

    Args:
        data: Input signal array (samples × channels)
        alpha: Smoothing factor (0 < alpha < 1)

    Returns:
        Smoothed signal
    """
    # Handle different input shapes
    if data.ndim == 1:
        return _apply_exponential_moving_average(data, alpha)

    n_samples, n_channels = data.shape
    result = np.empty_like(data)

    # Process each channel in parallel
    for ch in prange(n_channels):
        channel_data = data[:, ch]
        result[:, ch] = _apply_exponential_moving_average(channel_data, alpha)

    return result


@jit(nopython=True, cache=True, fastmath=True)
def _apply_exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential moving average.

    Formula: y[t] = α * x[t] + (1-α) * y[t-1]

    Args:
        data: Input signal array
        alpha: Smoothing factor (0 < alpha < 1)

    Returns:
        Smoothed signal
    """
    n_samples = len(data)
    result = np.empty_like(data)

    # Initialize with first value
    result[0] = data[0]

    # Apply EMA formula
    for i in range(1, n_samples):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    return result


class FFTRMSProcessor(BaseProcessor):
    """
    Moving RMS processor using FFT-based convolution for efficient computation.

    This processor calculates the root mean square (RMS) value over a sliding window
    using FFT-based convolution, which is significantly faster than direct computation
    for large window sizes.
    """

    def __init__(
        self,
        window_size: int = 11,
        center: bool = True,
        pad_mode: str = "reflect",
        use_numpy_impl: bool = False,
    ):
        """
        Initialize the FFT-based RMS processor.

        Args:
            window_size: Size of the moving window
            center: If True, window is centered on each point
                   If False, window includes only past samples
            pad_mode: Padding mode for edge handling ('constant', 'reflect', 'edge')
            use_numpy_impl: Whether to use NumPy/SciPy implementation instead of Numba
        """
        self.window_size = window_size
        self.center = center
        self.pad_mode = pad_mode
        self.use_numpy_impl = use_numpy_impl

        # Set overlap for map_overlap
        self._overlap_samples = window_size

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply FFT-based RMS to the input data.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments

        Returns:
            Data array with moving RMS applied
        """
        parallel = kwargs.get("parallel", True)

        # Select the appropriate implementation
        if self.use_numpy_impl:
            rms_func = _apply_fft_rms_numpy
        else:
            rms_func = _apply_fft_rms_numba if parallel else _apply_fft_rms_numba_single

        return data.map_overlap(
            rms_func,
            depth=(self._overlap_samples, 0),
            boundary=self.pad_mode,
            window_size=self.window_size,
            center=self.center,
            dtype=data.dtype,
        )

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "window_size": self.window_size,
                "center": self.center,
                "pad_mode": self.pad_mode,
                "method": "fft-based convolution",
                "accelerated": not self.use_numpy_impl,
            }
        )
        return base_summary


@jit(nopython=True, cache=True, fastmath=True)
def _squared_signal(x):
    """Square the input signal."""
    return x * x


@jit(nopython=True, cache=True, fastmath=True)
def _apply_fft_rms_numba_single(
    data: np.ndarray, window_size: int, center: bool = True
) -> np.ndarray:
    """
    Apply moving RMS using FFT-based convolution for a single channel.

    Args:
        data: Input signal array
        window_size: Size of the moving window
        center: Whether to center the window

    Returns:
        RMS values for each point
    """
    # Ensure data is 1D
    if data.ndim > 1:
        if data.shape[1] == 1:
            data = data.ravel()
        else:
            # For multi-channel, process first channel only
            data = data[:, 0]

    n_samples = len(data)

    # Create the window (box filter)
    window = np.ones(window_size, dtype=np.float32) / window_size

    # Square the signal
    squared_data = data * data

    # Perform convolution using FFT
    # Note: We can't use scipy.signal.fftconvolve directly in Numba,
    # so we implement the FFT convolution explicitly

    # Pad the squared data to avoid edge effects
    padded_len = n_samples + window_size - 1
    fft_size = 2 ** int(np.ceil(np.log2(padded_len)))

    # FFT of the squared data
    fft_data = np.fft.rfft(squared_data, fft_size)

    # FFT of the window
    fft_window = np.fft.rfft(window, fft_size)

    # Multiply in frequency domain
    fft_result = fft_data * fft_window

    # IFFT to get back to time domain
    conv_result = np.fft.irfft(fft_result, fft_size)

    # Extract the valid part of the convolution result
    if center:
        # For centered window, we need to shift the result
        start_idx = window_size // 2
        end_idx = start_idx + n_samples
        result = conv_result[start_idx:end_idx]
    else:
        # For causal window (past samples only)
        result = conv_result[:n_samples]

    # Take square root to get RMS
    return np.sqrt(result)


@jit(nopython=True, parallel=False, cache=True, fastmath=True)
def _apply_fft_rms_numba(
    data: np.ndarray, window_size: int, center: bool = True
) -> np.ndarray:
    """
    Apply moving RMS using FFT-based convolution with parallelization.

    Args:
        data: Input signal array (samples x channels)
        window_size: Size of the moving window
        center: Whether to center the window

    Returns:
        RMS values for each point and channel
    """
    # Handle different input shapes
    if data.ndim == 1:
        return _apply_fft_rms_numba_single(data, window_size, center)

    n_samples, n_channels = data.shape
    result = np.empty_like(data)

    # Process each channel in parallel
    for ch in prange(n_channels):
        channel_data = data[:, ch]
        result[:, ch] = _apply_fft_rms_numba_single(channel_data, window_size, center)

    return result


def _apply_fft_rms_numpy(
    data: np.ndarray, window_size: int, center: bool = True
) -> np.ndarray:
    """
    Apply moving RMS using SciPy's FFT-based convolution.

    This implementation uses scipy.signal functions for maximum performance.

    Args:
        data: Input signal array
        window_size: Size of the moving window
        center: Whether to center the window

    Returns:
        RMS values for each point
    """
    # Handle different input shapes
    if data.ndim == 1:
        # Square the signal
        squared_data = data**2

        # Create box window for averaging
        window = np.ones(window_size) / window_size

        # Perform fast convolution
        if center:
            # Mode 'same' centers the window
            conv_result = signal.fftconvolve(squared_data, window, mode="same")
        else:
            # For causal filtering (past samples only), use 'full' and shift
            conv_result = signal.fftconvolve(squared_data, window, mode="full")
            conv_result = conv_result[: len(data)]

        # Take square root to get RMS
        return np.sqrt(conv_result)
    else:
        # For multi-channel data
        n_samples, n_channels = data.shape
        result = np.empty_like(data)

        for ch in range(n_channels):
            channel_data = data[:, ch]
            squared_data = channel_data**2

            # Create box window for averaging
            window = np.ones(window_size) / window_size

            # Perform fast convolution
            if center:
                # Mode 'same' centers the window
                conv_result = signal.fftconvolve(squared_data, window, mode="same")
            else:
                # For causal filtering (past samples only), use 'full' and shift
                conv_result = signal.fftconvolve(squared_data, window, mode="full")
                conv_result = conv_result[:n_samples]

            # Take square root to get RMS
            result[:, ch] = np.sqrt(conv_result)

        return result


@public_api
def create_fft_rms_processor(
    window_size: int = 11,
    center: bool = True,
    pad_mode: str = "reflect",
    use_numpy_impl: bool = False,
) -> FFTRMSProcessor:
    """
    Create an FFT-based RMS processor with the specified parameters.

    Args:
        window_size: Size of the moving window in samples
        center: Whether to center the window on each point
        pad_mode: Padding mode for edge handling
        use_numpy_impl: Whether to use NumPy/SciPy implementation instead of Numba

    Returns:
        Configured FFTRMSProcessor
    """
    return FFTRMSProcessor(
        window_size=window_size,
        center=center,
        pad_mode=pad_mode,
        use_numpy_impl=use_numpy_impl,
    )


@public_api
def create_fft_moving_average(
    window_size: int = 11,
    method: str = "simple",
    weights: Optional[np.ndarray] = None,
    center: bool = True,
    pad_mode: str = "reflect",
    use_numba_impl: bool = True,
) -> FFTMovingAverageProcessor:
    """
    Create an FFT-based moving average processor with the specified parameters.

    Args:
        window_size: Size of the moving window in samples
        method: Moving average type ('simple', 'weighted', 'exponential')
        weights: Optional custom weights for weighted moving average
        center: Whether to center the window on each point
        pad_mode: Padding mode for edge handling
        use_numba_impl: Whether to use Numba implementation

    Returns:
        Configured FFTMovingAverageProcessor
    """
    return FFTMovingAverageProcessor(
        window_size=window_size,
        method=method,
        weights=weights,
        center=center,
        pad_mode=pad_mode,
        use_numba_impl=use_numba_impl,
    )
