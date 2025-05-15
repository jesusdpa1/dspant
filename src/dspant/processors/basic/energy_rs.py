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
            Processed array with TKEO applied, same shape as input
        """
        use_parallel = kwargs.get("use_parallel", True)
        use_rust = kwargs.get("use_rust", _HAS_RUST)

        # Ensure data is correct format for Rust processing
        needs_reshape = False
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            needs_reshape = True

        # Define processing function based on method
        def process_chunk(x):
            # Ensure array is contiguous and float32
            x = np.ascontiguousarray(x, dtype=np.float32)
            original_shape = x.shape

            if self.method == "classic":
                # Classic TKEO reduces by 2 samples
                if use_rust and _HAS_RUST:
                    if use_parallel and x.ndim > 1 and x.shape[1] > 1:
                        # Compute TKEO (returns N-2 samples)
                        tkeo_result = compute_tkeo_parallel(x)

                        # Create output array with original size
                        output = np.zeros(original_shape, dtype=np.float32)

                        # Copy TKEO result to the middle of the output
                        output[1:-1] = tkeo_result

                        # Handle boundaries - mirror adjacent values
                        output[0] = output[1]
                        output[-1] = output[-2]

                        return output

                    elif x.ndim == 1:
                        # 1D array processing
                        tkeo_result = compute_tkeo_classic(x)

                        # Create output array with original size
                        output = np.zeros(original_shape, dtype=np.float32)

                        # Copy TKEO result to the middle of the output
                        output[1:-1] = tkeo_result

                        # Handle boundaries - mirror adjacent values
                        output[0] = output[1]
                        output[-1] = output[-2]

                        return output

                    else:
                        # Multi-channel non-parallel processing
                        tkeo_result = compute_tkeo(x)

                        # Create output array with original size
                        output = np.zeros(original_shape, dtype=np.float32)

                        # Copy TKEO result to the middle of the output
                        output[1:-1] = tkeo_result

                        # Handle boundaries - mirror adjacent values
                        output[0] = output[1]
                        output[-1] = output[-2]

                        return output
                else:
                    # Fall back to Python implementation
                    output = np.zeros(original_shape, dtype=np.float32)
                    for c in range(x.shape[1]):
                        channel_result = _classic_tkeo(x[:, c])
                        output[1:-1, c] = channel_result
                        output[0, c] = output[1, c]
                        output[-1, c] = output[-2, c]
                    return output

            else:  # modified method
                # Modified TKEO reduces by 3 samples
                if use_rust and _HAS_RUST:
                    # Create output array with original size
                    output = np.zeros(original_shape, dtype=np.float32)

                    if x.ndim > 1:
                        # Process each channel separately
                        for c in range(x.shape[1]):
                            # Compute modified TKEO for this channel
                            tkeo_result = compute_tkeo_modified(x[:, c])

                            # Copy result to output (leaving 3 boundary samples)
                            output[1:-2, c] = tkeo_result

                            # Handle boundaries - mirror adjacent values
                            output[0, c] = output[1, c]
                            output[-2, c] = output[-3, c]
                            output[-1, c] = output[-3, c]
                    else:
                        # Direct 1D processing
                        tkeo_result = compute_tkeo_modified(x)

                        # Copy result to output (leaving boundary samples)
                        output[1:-2] = tkeo_result

                        # Handle boundaries
                        output[0] = output[1]
                        output[-2] = output[-3]
                        output[-1] = output[-3]

                    return output
                else:
                    # Fall back to Python implementation
                    output = np.zeros(original_shape, dtype=np.float32)
                    for c in range(x.shape[1]):
                        channel_result = _modified_tkeo(x[:, c])
                        output[1:-2, c] = channel_result
                        output[0, c] = output[1, c]
                        output[-2, c] = output[-3, c]
                        output[-1, c] = output[-3, c]
                    return output

        # Calculate proper overlap depth based on method
        if self.method == "classic":
            # For classic TKEO, we need 2 extra samples for output of the same size
            depth = {0: 2}
        else:  # modified method
            # For modified TKEO, we need 3 extra samples for output of the same size
            depth = {0: 3}

        # Map the function across chunks with proper overlap
        result = data.map_overlap(
            process_chunk,
            depth=depth,
            boundary="reflect",
            dtype=np.float32,
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
