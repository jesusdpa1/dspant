"""
Rust-accelerated Teager-Kaiser Energy Operator (TKEO) implementations.

This module provides high-performance Rust implementations of the TKEO algorithms
with Python bindings, while preserving compatibility with Dask arrays for large datasets.
"""

from typing import Literal, Optional

import dask.array as da
import numpy as np

try:
    from dspant._rs import (
        compute_tkeo,
        compute_tkeo_classic,
        compute_tkeo_modified,
        compute_tkeo_parallel,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )
    from dspant.processors.basic.energy import _classic_tkeo, _modified_tkeo

from dspant.engine.streams.pipeline import StreamProcessingPipeline
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor, create_lowpass_filter


class TKEORustProcessor:
    """
    Rust-accelerated Teager-Kaiser Energy Operator (TKEO) processor.

    This class provides the same interface as the Python TKEOProcessor
    but uses the Rust implementation for better performance while maintaining
    compatibility with large Dask arrays.
    """

    def __init__(self, method: Literal["classic", "modified"] = "classic"):
        """
        Initialize the TKEO processor.

        Args:
            method: TKEO algorithm to use
                "classic": 3-point algorithm (Li et al., 2007)
                "modified": 4-point algorithm (Deburchgrave et al., 2008)
        """
        self.method = method
        self._overlap_samples = 2 if method == "classic" else 3

    def process(self, data: da.Array, **kwargs) -> da.Array:
        """
        Apply the TKEO operation to the input data using Rust implementation.

        Args:
            data: Input dask array
            **kwargs: Additional keyword arguments
                use_parallel: Whether to use the parallel optimized version (default: True)
                use_rust: Whether to use Rust implementation (default: True if available)

        Returns:
            Processed array with TKEO applied
        """
        use_parallel = kwargs.get("use_parallel", True)
        use_rust = kwargs.get("use_rust", _HAS_RUST)

        # Fall back to Python implementation if Rust is not available or not requested
        if not _HAS_RUST or not use_rust:
            # Use the same implementation as Python version
            if self.method == "classic":
                tkeo_func = _classic_tkeo
            else:
                tkeo_func = _modified_tkeo

            return data.map_overlap(
                tkeo_func,
                depth=(self.overlap_samples, 0),
                boundary="reflect",
                dtype=data.dtype,
            )

        # Ensure data is correct format for Rust processing
        needs_reshape = False
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            needs_reshape = True

        # Define Rust processing function based on method and parallel setting
        def process_chunk(x):
            # Ensure array is contiguous and float32
            x = np.ascontiguousarray(x, dtype=np.float32)

            if self.method == "classic":
                if use_parallel and x.ndim > 1 and x.shape[1] > 1:
                    # Parallel implementation for multi-channel data
                    return compute_tkeo_parallel(x)
                elif x.ndim == 1:
                    # 1D implementation for single channel data
                    return compute_tkeo_classic(x)
                else:
                    # Regular 2D implementation
                    return compute_tkeo(x)
            else:  # modified method
                if x.ndim > 1:
                    # Process each channel separately for modified method
                    result = np.zeros((x.shape[0], x.shape[1]), dtype=np.float32)
                    for c in range(x.shape[1]):
                        result[:, c] = compute_tkeo_modified(x[:, c])
                    return result
                else:
                    # Direct 1D processing
                    return compute_tkeo_modified(x)

        # Map the function across dask chunks
        result = data.map_overlap(
            process_chunk,
            depth=(self.overlap_samples, 0),
            boundary="reflect",
            dtype=data.dtype,
        )

        # Restore original shape if needed
        if needs_reshape:
            result = result.reshape(-1)

        return result

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> dict:
        """Get a summary of processor configuration"""
        return {
            "type": self.__class__.__name__,
            "method": self.method,
            "reference": "Li et al., 2007"
            if self.method == "classic"
            else "Deburchgrave et al., 2008",
            "accelerated": True,
            "rust_implementation": _HAS_RUST,
            "overlap": self._overlap_samples,
        }


def create_tkeo_envelope_rs(
    method: Literal["classic", "modified"] = "classic",
    rectify: bool = True,
    smooth: bool = True,
    cutoff_freq: Optional[float] = 10.0,
    fs: Optional[float] = None,
    filter_order: int = 2,
    use_rust: bool = True,
) -> StreamProcessingPipeline:
    """
    Create an envelope detector using the Rust-accelerated Teager-Kaiser Energy Operator (TKEO).

    TKEO is particularly effective for signals with both amplitude and frequency
    modulation, as it estimates the instantaneous energy of the signal.

    Args:
        method: TKEO algorithm to use: "classic" (3-point) or "modified" (4-point)
        rectify: Whether to apply rectification after TKEO (default: True)
        smooth: Whether to apply lowpass filtering after TKEO (default: True)
        cutoff_freq: Cutoff frequency for smoothing filter in Hz (default: 10.0)
            Only used if smooth=True
        fs: Sampling frequency (Hz). If None and smooth=True, it will be extracted
            during processing from the StreamNode or must be provided during manual processing.
        filter_order: Filter order for the smoothing filter (default: 2)
        use_rust: Whether to use Rust implementation if available (default: True)

    Returns:
        Processing pipeline containing the TKEO envelope detector
    """
    # Create a pipeline
    pipeline = StreamProcessingPipeline()

    # Create processor list
    processors = []

    # 1. Add TKEO processor (Rust-accelerated if available and requested)
    if _HAS_RUST and use_rust:
        tkeo = TKEORustProcessor(method=method)
    else:
        # Fall back to Python implementation
        from dspant.processors.basic.energy import TKEOProcessor

        tkeo = TKEOProcessor(method=method)

    processors.append(tkeo)

    # 2. Optional rectification
    if rectify:
        rect = RectificationProcessor(method="abs")
        processors.append(rect)

    # 3. Optional smoothing
    if smooth and cutoff_freq is not None:
        overlap = filter_order * 10
        filter_func = create_lowpass_filter(cutoff_freq, filter_order)
        smooth_filter = FilterProcessor(filter_func, overlap, parallel=True)
        processors.append(smooth_filter)

    # Add all processors to the pipeline
    pipeline.add_processor(processors, group="envelope")

    return pipeline


# Helper functions for direct use
def apply_tkeo_rs(signal, method="classic", use_parallel=True, use_rust=True):
    """
    Apply TKEO algorithm with Rust acceleration if available.

    Args:
        signal: Input signal (numpy array or dask array)
        method: TKEO algorithm ("classic" or "modified")
        use_parallel: Whether to use parallel optimized version
        use_rust: Whether to use Rust implementation if available

    Returns:
        Signal with TKEO applied
    """
    # For numpy arrays, directly use Rust functions if available
    if isinstance(signal, np.ndarray) and _HAS_RUST and use_rust:
        # Ensure array is contiguous and float32
        signal = np.ascontiguousarray(signal, dtype=np.float32)

        if signal.ndim == 1:
            # 1D processing
            if method == "classic":
                return compute_tkeo_classic(signal)
            else:
                return compute_tkeo_modified(signal)
        else:
            # 2D processing
            if method == "classic":
                if use_parallel and signal.shape[1] > 1:
                    return compute_tkeo_parallel(signal)
                else:
                    return compute_tkeo(signal)
            else:  # modified method
                # Process each channel separately for modified method
                result = np.zeros(
                    (signal.shape[0] - 3, signal.shape[1]), dtype=np.float32
                )
                for c in range(signal.shape[1]):
                    result[:, c] = compute_tkeo_modified(signal[:, c])
                return result

    # For dask arrays or fallback to Python
    processor = TKEORustProcessor(method=method)
    return processor.process(
        da.asarray(signal) if not isinstance(signal, da.Array) else signal,
        use_parallel=use_parallel,
        use_rust=use_rust,
    )
