"""
Cross-correlation based template subtraction.

This module implements template subtraction using correlation for optimal
alignment and scaling, which is effective for removing artifacts like ECG
from EMG or other electrophysiological signals.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from scipy.signal import correlate

from dspant.core.internals import public_api
from dspant.pattern.subtraction.base import BaseSubtractor


def apply_subtraction(
    data: np.ndarray,
    template: np.ndarray,
    indices: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """
    Subtract templates from data at specified indices using the exact algorithm
    from the working example.

    Args:
        data: Data array (1D)
        template: Template array (1D)
        indices: Indices where templates should be subtracted
        half_window: Half size of window around each index

    Returns:
        Data with templates subtracted
    """
    # Make a copy to avoid modifying input
    result = data.copy()

    # Handle 1D inputs (convert to 1D if needed)
    if data.ndim > 1:
        # Only support 1D for now to match working code
        result = result.ravel()

    # Make sure template is 1D
    if template.ndim > 1:
        template = template.ravel()

    # Get dimensions
    n_samples = len(result)
    window_samples = len(template)

    # Process each index
    for idx in indices:
        # Skip if index is too close to edges for extracting segment
        if idx < half_window or idx >= n_samples - half_window:
            continue

        # Extract segment for correlation
        segment = result[idx - half_window : idx + half_window]

        # Calculate correlation
        corr = correlate(segment, template, mode="valid")

        # Calculate shift by centering correlation at middle of window
        shift = np.argmax(corr) - (len(corr) // 2)

        # Calculate start and end for extraction with shift
        start = idx - half_window + shift
        end = start + window_samples

        # Safety check for boundaries
        if start >= 0 and end < n_samples:
            # Re-extract segment at aligned position
            aligned_segment = result[start:end]

            # Calculate scaling factor
            scale = np.dot(aligned_segment, template) / np.dot(template, template)

            # Subtract scaled template
            result[start:end] -= scale * template

    return result


@public_api
class CorrelationSubtractor(BaseSubtractor):
    """
    Template subtraction using cross-correlation for alignment.

    This subtractor aligns templates with signal segments using cross-correlation,
    scales them optimally, and subtracts them from the original signal.
    It's particularly effective for removing ECG artifacts from EMG or EEG.
    """

    def __init__(
        self,
        template: Optional[np.ndarray] = None,
        window_samples: Optional[int] = None,
    ):
        """
        Initialize the correlation-based subtractor.

        Args:
            template: Template array to subtract (1D)
            window_samples: Window size to use for subtraction
                            If None, determined from template size
        """
        super().__init__(template)

        # Initialize window size
        if template is not None:
            if template.ndim == 1:
                self._window_samples = len(template)
            else:
                self._window_samples = template.shape[0]
        else:
            self._window_samples = window_samples or 0

        # Calculate half window size
        self._half_window = self._window_samples // 2

    def set_template(self, template: np.ndarray) -> None:
        """
        Set or update the template.

        Args:
            template: New template array
        """
        super().set_template(template)

        # Update window size if not explicitly set
        if template is not None:
            if template.ndim == 1:
                # Single channel
                self._window_samples = len(template)
            else:
                # Multi-channel
                self._window_samples = template.shape[0]

        self._half_window = self._window_samples // 2

    def process(
        self,
        data: da.Array,
        indices: Optional[Union[np.ndarray, List[int]]] = None,
        fs: Optional[float] = None,
        **kwargs,
    ) -> da.Array:
        """
        Process data by subtracting templates at specified indices.

        Args:
            data: Input data array (1D)
            indices: Indices where templates should be subtracted
            fs: Sampling frequency (for metadata)
            **kwargs: Additional parameters

        Returns:
            Data with templates subtracted
        """
        # If indices not provided, can't do subtraction
        if indices is None:
            return data

        # Check for overrides
        window_samples = kwargs.get("window_samples", self._window_samples)
        half_window = window_samples // 2

        # Ensure we have a template
        if self.template is None:
            raise ValueError("No template provided for subtraction")

        # Ensure template is 1D or first channel of multi-channel
        template = self.template
        if template.ndim > 1:
            template = template[:, 0]  # Use first channel if multidimensional

        # Ensure indices are numpy array
        if isinstance(indices, list):
            indices = np.array(indices)

        # Define the subtraction function for chunks
        def subtract_templates_from_chunk(chunk, chunk_offset=0):
            """Process a single chunk of data"""
            # Filter indices to those within this chunk
            # Include extra buffer for window size
            chunk_indices = indices[
                (indices >= chunk_offset + half_window)
                & (indices < chunk_offset + len(chunk) - half_window)
            ]

            # Adjust indices to be relative to chunk
            if len(chunk_indices) > 0:
                chunk_indices = chunk_indices - chunk_offset

            # If no indices in this chunk, return unchanged
            if len(chunk_indices) == 0:
                return chunk

            # Ensure chunk is 1D
            if chunk.ndim > 1:
                chunk = chunk.ravel()

            # Apply subtraction using the simple algorithm
            return apply_subtraction(chunk, template, chunk_indices, half_window)

        # Apply subtraction with overlap
        processed_data = data.map_overlap(
            subtract_templates_from_chunk,
            depth=window_samples,
            boundary="reflect",
            dtype=data.dtype,
        )

        # Track statistics
        self._subtraction_stats.update(
            {
                "num_indices": len(indices),
                "window_samples": window_samples,
            }
        )

        return processed_data

    @property
    def window_samples(self) -> int:
        """Get the current window size in samples"""
        return self._window_samples

    @window_samples.setter
    def window_samples(self, value: int) -> None:
        """Set the window size in samples"""
        self._window_samples = value
        self._overlap_samples = value
        self._half_window = value // 2

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of subtractor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "window_samples": self._window_samples,
            }
        )
        return base_summary


@public_api
def create_correlation_subtractor(
    template: np.ndarray,
    window_samples: Optional[int] = None,
) -> CorrelationSubtractor:
    """
    Create a correlation-based template subtractor.

    Args:
        template: Template array to subtract
        window_samples: Window size (default: template length)

    Returns:
        Configured CorrelationSubtractor
    """
    return CorrelationSubtractor(
        template=template,
        window_samples=window_samples,
    )


@public_api
def create_ecg_subtractor(
    ecg_template: np.ndarray, window_ms: float = 60.0, fs: float = 1000.0
) -> CorrelationSubtractor:
    """
    Create a subtractor optimized for ECG artifact removal.

    Args:
        ecg_template: ECG template array
        window_ms: Window size in milliseconds
        fs: Sampling frequency in Hz

    Returns:
        Configured CorrelationSubtractor for ECG removal
    """
    # Calculate window size in samples
    window_samples = int((window_ms / 1000) * fs)

    # Ensure template is 1D for ECG
    if ecg_template.ndim > 1:
        ecg_template = ecg_template[:, 0]  # Use first channel if multidimensional

    return CorrelationSubtractor(
        template=ecg_template,
        window_samples=window_samples,
    )


@public_api
def subtract_templates(
    data: Union[np.ndarray, da.Array],
    template: np.ndarray,
    indices: Union[np.ndarray, List[int]],
    window_samples: Optional[int] = None,
) -> Union[np.ndarray, da.Array]:
    """
    Convenience function for one-off template subtraction.

    Args:
        data: Input data array
        template: Template to subtract
        indices: Indices where templates should be subtracted
        window_samples: Window size (default: template length)

    Returns:
        Data with templates subtracted
    """
    # Create subtractor
    subtractor = CorrelationSubtractor(
        template=template,
        window_samples=window_samples,
    )

    # Convert to dask array if numpy input
    if isinstance(data, np.ndarray):
        input_data = da.from_array(data)
        result = subtractor.process(input_data, indices)
        return result.compute()  # Convert back to numpy
    else:
        # Already dask array
        return subtractor.process(data, indices)
