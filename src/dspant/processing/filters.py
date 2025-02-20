from typing import Any, Dict, Optional

import dask.array as da
import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..core.nodes.stream_processing import BaseProcessor, ProcessingFunction


class FilterProcessor(BaseProcessor):
    """Filter processor implementation"""

    def __init__(self, filter_func: ProcessingFunction, overlap_samples: int):
        self.filter_func = filter_func
        self._overlap_samples = overlap_samples

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
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
                "args": getattr(self, "filter_args", None),
            }
        )
        return base_summary


# Example filter functions that can be used with FilterProcessor
def create_bandpass_filter(
    lowcut: float, highcut: float, order: int = 4
) -> ProcessingFunction:
    def bandpass_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(
            order, [lowcut / nyquist, highcut / nyquist], btype="bandpass", output="sos"
        )
        return sosfiltfilt(sos, chunk, axis=0)

    return bandpass_filter


def create_notch_filter(
    notch_freq: float, q: float = 30, order: int = 4
) -> ProcessingFunction:
    def notch_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        low = (notch_freq - 1 / q) / nyquist
        high = (notch_freq + 1 / q) / nyquist
        sos = butter(order, [low, high], btype="bandstop", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)

    return notch_filter


def create_lowpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    def lowpass_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(order, cutoff / nyquist, btype="lowpass", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)

    return lowpass_filter


def create_highpass_filter(cutoff: float, order: int = 4) -> ProcessingFunction:
    def highpass_filter(chunk: np.ndarray, fs: float) -> np.ndarray:
        nyquist = 0.5 * fs
        sos = butter(order, cutoff / nyquist, btype="highpass", output="sos")
        return sosfiltfilt(sos, chunk, axis=0)

    return highpass_filter
