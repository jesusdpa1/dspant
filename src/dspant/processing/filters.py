from typing import Any, Dict, Optional

import dask.array as da
import numpy as np
from numba import jit, prange
from scipy.signal import butter, sosfiltfilt

# Import outside of JIT functions
from scipy.signal import sosfiltfilt as scipy_sosfiltfilt

from ..core.nodes.stream_processing import BaseProcessor, ProcessingFunction


# JIT-compiled parallel implementation for multi-channel data
@jit(nopython=True, parallel=True, cache=True)
def _apply_filter_parallel(chunk, sos):
    """Apply filter to each channel in parallel"""
    n_channels = chunk.shape[1]
    result = np.zeros_like(chunk)

    # Process each channel in parallel
    for i in prange(n_channels):
        # We can't call sosfiltfilt directly in numba, so we'll handle each channel separately
        # outside the JIT function
        channel = chunk[:, i]
        # Just copy the channel to result for now - actual filtering happens outside
        result[:, i] = channel

    return result


class FilterProcessor(BaseProcessor):
    """Filter processor implementation with parallel processing"""

    def __init__(
        self,
        filter_func: ProcessingFunction,
        overlap_samples: int,
        parallel: bool = True,
    ):
        self.filter_func = filter_func
        self._overlap_samples = overlap_samples
        self.parallel = parallel

        # Store filter parameters for optimization
        self.filter_args = getattr(filter_func, "filter_args", None)

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        # Pass the parallel flag to the filter function
        if self.parallel and "parallel" not in kwargs:
            kwargs["parallel"] = True

        return data.map_overlap(
            self.filter_func,
            depth=(self.overlap_samples, 0),
            boundary="reflect",
            fs=fs,
            dtype=data.dtype,
            **kwargs,
        )

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary
        base_summary.update(
            {
                "filter_function": self.filter_func.__name__,
                "args": self.filter_args,
                "parallel": self.parallel,
            }
        )
        return base_summary


# Alternative optimization approach using parallel processing without JIT
def parallel_filter_channels(chunk, sos):
    """Process multiple channels in parallel using Python's multiprocessing"""
    import concurrent.futures

    if chunk.ndim == 1:
        return sosfiltfilt(sos, chunk)

    n_channels = chunk.shape[1]
    result = np.zeros_like(chunk)

    # Define the worker function
    def filter_channel(i):
        return sosfiltfilt(sos, chunk[:, i])

    # Use a thread pool for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(filter_channel, i) for i in range(n_channels)]
        for i, future in enumerate(futures):
            result[:, i] = future.result()

    return result


# Updated filter functions
def create_bandpass_filter(
    lowcut: float, highcut: float, order: int = 4
) -> ProcessingFunction:
    filter_args = {"lowcut": lowcut, "highcut": highcut, "order": order}

    def bandpass_filter(
        chunk: np.ndarray, fs: float, parallel: bool = False, **kwargs
    ) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(
            order, [lowcut / nyquist, highcut / nyquist], btype="bandpass", output="sos"
        )

        if parallel and chunk.ndim > 1 and chunk.shape[1] > 1:
            # Use parallel implementation for multi-channel data
            return parallel_filter_channels(chunk, sos)
        else:
            # Use standard implementation
            return sosfiltfilt(sos, chunk, axis=0)

    # Attach parameters for introspection
    bandpass_filter.filter_args = filter_args
    return bandpass_filter


def create_notch_filter(
    notch_freq: float, q: float = 30, order: int = 4
) -> ProcessingFunction:
    filter_args = {"notch_freq": notch_freq, "q": q, "order": order}

    def notch_filter(
        chunk: np.ndarray, fs: float, parallel: bool = False, **kwargs
    ) -> np.ndarray:
        nyquist = 0.5 * fs
        low = (notch_freq - 1 / q) / nyquist
        high = (notch_freq + 1 / q) / nyquist
        sos = butter(order, [low, high], btype="bandstop", output="sos")

        if parallel and chunk.ndim > 1 and chunk.shape[1] > 1:
            return parallel_filter_channels(chunk, sos)
        else:
            return sosfiltfilt(sos, chunk, axis=0)

    notch_filter.filter_args = filter_args
    return notch_filter


def create_lowpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    filter_args = {"cutoff": cutoff, "order": order}

    def lowpass_filter(
        chunk: np.ndarray, fs: float, parallel: bool = False, **kwargs
    ) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(order, cutoff / nyquist, btype="lowpass", output="sos")

        if parallel and chunk.ndim > 1 and chunk.shape[1] > 1:
            return parallel_filter_channels(chunk, sos)
        else:
            return sosfiltfilt(sos, chunk, axis=0)

    lowpass_filter.filter_args = filter_args
    return lowpass_filter


def create_highpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    filter_args = {"cutoff": cutoff, "order": order}

    def highpass_filter(
        chunk: np.ndarray, fs: float, parallel: bool = False, **kwargs
    ) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(order, cutoff / nyquist, btype="highpass", output="sos")

        if parallel and chunk.ndim > 1 and chunk.shape[1] > 1:
            return parallel_filter_channels(chunk, sos)
        else:
            return sosfiltfilt(sos, chunk, axis=0)

    highpass_filter.filter_args = filter_args
    return highpass_filter
