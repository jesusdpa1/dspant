"""
Envelope detection utilities for neural signal processing.

This module provides utilities for extracting signal envelopes using various methods:
- Rectification + smoothing
- Hilbert transform
- RMS-based envelope
- TKEO-based envelope
- Peak envelope following
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np

from ...engine.streams.pipeline import StreamProcessingPipeline
from ..basic.energy import TKEOProcessor
from ..basic.moving_stats import MovingRMSProcessor

# Import directly from modules rather than through __init__.py to avoid circular imports
from ..basic.rectification import RectificationProcessor
from ..filters import FilterProcessor, create_lowpass_filter


def create_rectify_smooth_envelope(
    cutoff_freq: float,
    fs: Optional[float] = None,
    rect_method: Literal["abs", "square", "half"] = "abs",
    filter_order: int = 2,
    parallel: bool = True,
) -> StreamProcessingPipeline:
    """
    Create an envelope detector using rectification followed by lowpass filtering.

    This is a common method for extracting amplitude envelopes from signals.

    Args:
        cutoff_freq: Cutoff frequency (Hz) for the lowpass filter
        fs: Sampling frequency (Hz). If None, it will be extracted during processing
            from the StreamNode or must be provided during manual processing.
        rect_method: Rectification method: "abs" (full-wave), "square", or "half" (half-wave)
        filter_order: Filter order for the lowpass filter
        parallel: Whether to use parallel processing for filtering

    Returns:
        Processing pipeline containing the envelope detector
    """
    # Create a pipeline
    pipeline = StreamProcessingPipeline()

    # 1. Rectification processor
    rect = RectificationProcessor(method=rect_method)

    # 2. Low-pass filter for smoothing
    overlap = filter_order * 10
    filter_func = create_lowpass_filter(cutoff_freq, filter_order)
    smooth = FilterProcessor(filter_func, overlap, parallel)

    # Add both processors to the pipeline
    pipeline.add_processor([rect, smooth], group="envelope")

    return pipeline


def create_tkeo_envelope(
    method: Literal["classic", "modified"] = "classic",
    rectify: bool = True,
    smooth: bool = True,
    cutoff_freq: Optional[float] = 10.0,
    fs: Optional[float] = None,
    filter_order: int = 2,
) -> StreamProcessingPipeline:
    """
    Create an envelope detector using the Teager-Kaiser Energy Operator (TKEO).

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

    Returns:
        Processing pipeline containing the TKEO envelope detector
    """
    # Create a pipeline
    pipeline = StreamProcessingPipeline()

    # Create processor list
    processors = []

    # 1. Add TKEO processor
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


def create_rms_envelope(
    window_size: int = 101, center: bool = True
) -> StreamProcessingPipeline:
    """
    Create an envelope detector using the RMS method.

    RMS (Root Mean Square) is useful for extracting the energy envelope of signals.

    Args:
        window_size: Size of the moving window for RMS calculation
        center: Whether to center the window on each point

    Returns:
        Processing pipeline containing the RMS envelope detector
    """
    # Create a pipeline
    pipeline = StreamProcessingPipeline()

    # Add RMS processor
    rms = MovingRMSProcessor(window_size=window_size, center=center)

    # Add to pipeline
    pipeline.add_processor(rms, group="envelope")

    return pipeline


def create_hilbert_envelope(
    cutoff_freq: Optional[float] = 10.0,
    fs: Optional[float] = None,
    filter_order: int = 2,
    parallel: bool = True,
) -> StreamProcessingPipeline:
    """
    Create an envelope detector using the Hilbert transform followed by lowpass filtering.

    The Hilbert transform creates an analytic signal and extracts its magnitude
    which represents the instantaneous amplitude envelope. The lowpass filter
    smooths the result to remove high-frequency components.

    Args:
        cutoff_freq: Cutoff frequency for smoothing filter in Hz (default: 10.0)
            Set to None to skip the smoothing step
        fs: Sampling frequency (Hz). If None, it will be extracted during processing
            from the StreamNode or must be provided during manual processing.
        filter_order: Filter order for the smoothing filter (default: 2)
        parallel: Whether to use parallel processing

    Returns:
        Processing pipeline containing the Hilbert envelope detector
    """
    # Create a pipeline
    pipeline = StreamProcessingPipeline()

    # Create processor list
    processors = []

    # 1. Use Hilbert transform to get the envelope
    hilbert = RectificationProcessor(method="hilbert")
    processors.append(hilbert)

    # 2. Optional lowpass filter for smoothing
    if cutoff_freq is not None:
        overlap = filter_order * 10
        filter_func = create_lowpass_filter(cutoff_freq, filter_order)
        smooth_filter = FilterProcessor(filter_func, overlap, parallel=parallel)
        processors.append(smooth_filter)

    # Add all processors to the pipeline
    pipeline.add_processor(processors, group="envelope")

    return pipeline


def create_peak_envelope(
    attack_time: float,
    release_time: float,
    fs: Optional[float] = None,
) -> StreamProcessingPipeline:
    """
    Create a peak envelope follower with configurable attack and release times.

    Similar to audio compressors, this follows peaks in the signal with
    asymmetric attack and release characteristics.

    Args:
        attack_time: Attack time in seconds (how quickly envelope rises)
        release_time: Release time in seconds (how slowly envelope falls)
        fs: Sampling frequency (Hz). If None, it will be extracted during processing
            from the StreamNode or must be provided during manual processing.

    Returns:
        Processing pipeline containing the peak envelope follower
    """
    # Create a pipeline
    pipeline = StreamProcessingPipeline()

    # 1. Rectification (absolute value)
    rect = RectificationProcessor(method="abs")

    # Add to pipeline (placeholder until PeakFollowerProcessor is implemented)
    pipeline.add_processor(rect, group="envelope")

    # Note: A complete implementation would use the fs parameter to calculate
    # the appropriate coefficients for attack and release

    return pipeline
