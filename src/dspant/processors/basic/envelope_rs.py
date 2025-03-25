from typing import Literal, Optional

import dask.array as da
import numpy as np

from dspant.engine.streams.pipeline import StreamProcessingPipeline
from dspant.processors.basic.rectification import RectificationProcessor
from dspant.processors.filters import FilterProcessor, create_lowpass_filter
from dspant.processors.transforms.tkeo import TKEORustProcessor

try:
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )
    from dspant.processors.basic.energy import _classic_tkeo, _modified_tkeo


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
