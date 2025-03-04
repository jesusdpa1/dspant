from typing import Any, Dict, Optional

import dask.array as da
import numpy as np
from scipy.signal import sosfiltfilt

from ...engine.base import BaseProcessor, ProcessingFunction

# This function is only a placeholder - it can't actually apply SciPy's sosfiltfilt
# since Numba can't compile that. We should remove it since it's misleading.
# REMOVING: @jit(nopython=True, parallel=True, cache=True)
# REMOVING: def _apply_filter_parallel(chunk, sos): ...


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
