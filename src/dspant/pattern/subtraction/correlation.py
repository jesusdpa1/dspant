"""
Cross-correlation based template subtraction.

This module implements template subtraction using correlation for optimal
alignment and scaling, which is effective for removing artifacts like ECG
from EMG or other electrophysiological signals.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange
from scipy.signal import correlate

from dspant.core.internals import public_api
from dspant.pattern.subtraction.base import BaseSubtractor


@jit(nopython=True, cache=True)
def _align_and_scale_template(
    segment: np.ndarray, template: np.ndarray, correlation_mode: str = "valid"
) -> Tuple[int, float]:
    """
    Find optimal alignment and scaling for template.

    Args:
        segment: Signal segment
        template: Template to align
        correlation_mode: Mode for correlation calculation
            "valid": Only compute where template and segment fully overlap
            "same": Output is same size as segment
            "full": Output is full correlation

    Returns:
        Tuple of (shift, scale_factor)
    """
    # Segment and template should be 1D arrays
    if segment.ndim > 1 or template.ndim > 1:
        # Flatten for single-channel operation
        segment = segment.ravel()
        template = template.ravel()

    # Compute correlation
    if correlation_mode == "valid":
        # Manual valid-mode correlation
        corr = np.zeros(len(segment) - len(template) + 1)
        for i in range(len(corr)):
            for j in range(len(template)):
                corr[i] += segment[i + j] * template[j]
        max_idx = np.argmax(corr)
        shift = max_idx
    elif correlation_mode == "same":
        # Manual same-mode correlation (centered)
        corr = np.zeros(len(segment))
        template_half = len(template) // 2
        for i in range(len(segment)):
            for j in range(len(template)):
                if i - template_half + j >= 0 and i - template_half + j < len(segment):
                    corr[i] += segment[i - template_half + j] * template[j]
        max_idx = np.argmax(corr)
        shift = max_idx - len(segment) // 2
    else:  # Full mode
        # Manual full-mode correlation
        corr = np.zeros(len(segment) + len(template) - 1)
        for i in range(len(corr)):
            for j in range(len(template)):
                if i - j >= 0 and i - j < len(segment):
                    corr[i] += segment[i - j] * template[j]
        max_idx = np.argmax(corr)
        shift = max_idx - (len(template) - 1)

    # Calculate optimal scaling using dot product
    # This minimizes the squared error between template and signal
    start = max(0, shift)
    end = min(len(segment), shift + len(template))

    # Get the overlapping part
    if end <= start:
        return shift, 0.0  # No overlap

    # Template portion that overlaps
    template_start = max(0, -shift)
    template_end = template_start + (end - start)

    # Calculate scaling factor
    template_portion = template[template_start:template_end]
    segment_portion = segment[start:end]

    # Dot products for scaling factor
    numerator = 0.0
    denominator = 0.0
    for i in range(len(template_portion)):
        numerator += segment_portion[i] * template_portion[i]
        denominator += template_portion[i] * template_portion[i]

    # Avoid division by zero
    if denominator <= 1e-10:
        scale = 0.0
    else:
        scale = numerator / denominator

    return shift, scale


@jit(nopython=True, parallel=True, cache=True)
def _apply_subtraction_numba(
    data: np.ndarray,
    template: np.ndarray,
    indices: np.ndarray,
    half_window: int,
    correlation_mode: str = "valid",
) -> np.ndarray:
    """
    Subtract templates from data at specified indices using Numba acceleration.

    Args:
        data: Data array (samples x channels)
        template: Template array (template_samples x channels)
        indices: Indices where templates should be subtracted
        half_window: Half size of window around each index
        correlation_mode: Mode for correlation calculation

    Returns:
        Data with templates subtracted
    """
    # Ensure data is contiguous for best performance
    data = np.ascontiguousarray(data)
    template = np.ascontiguousarray(template)

    # Get dimensions
    n_samples, n_channels = data.shape
    template_samples = template.shape[0]

    # Create a copy for subtraction
    output = data.copy()

    # Process each index
    for idx_pos in range(len(indices)):
        idx = indices[idx_pos]

        # Skip if index is too close to edges
        if idx < half_window or idx >= n_samples - half_window:
            continue

        # Extract segment for each channel
        start = idx - half_window
        end = idx + half_window

        # Process each channel separately
        for ch in prange(n_channels):
            # Get channel data
            segment = data[start:end, ch]
            template_ch = template[:, min(ch, template.shape[1] - 1)]

            # Align and scale template
            shift, scale = _align_and_scale_template(
                segment, template_ch, correlation_mode
            )

            # Apply subtraction
            # Shift is relative to the segment start
            t_start = max(0, shift)
            t_end = min(len(segment), shift + len(template_ch))

            # Skip if no overlap
            if t_end <= t_start:
                continue

            # Template portion that overlaps
            template_start = max(0, -shift)
            template_end = template_start + (t_end - t_start)

            # Apply subtraction to output
            for i in range(t_end - t_start):
                output[start + t_start + i, ch] -= (
                    scale * template_ch[template_start + i]
                )

    return output


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
        correlation_mode: str = "valid",
        use_numba: bool = True,
    ):
        """
        Initialize the correlation-based subtractor.

        Args:
            template: Template array to subtract (samples x channels)
            window_samples: Window size to use for subtraction
                            If None, determined from template size
            correlation_mode: Mode for correlation calculation
                "valid": Only compute where template and segment fully overlap
                "same": Output is same size as segment
                "full": Output is full correlation
            use_numba: Whether to use Numba acceleration
        """
        super().__init__(template)
        self.correlation_mode = correlation_mode
        self.use_numba = use_numba

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
                if self._window_samples == 0:
                    self._window_samples = len(template)
            else:
                # Multi-channel
                if self._window_samples == 0:
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
            data: Input data array
            indices: Indices where templates should be subtracted
            fs: Sampling frequency (for metadata)
            **kwargs: Additional parameters:
                window_samples: Override window size
                correlation_mode: Override correlation mode

        Returns:
            Data with templates subtracted
        """
        # If indices not provided, can't do subtraction
        if indices is None:
            return data

        # Check for overrides
        window_samples = kwargs.get("window_samples", self._window_samples)
        correlation_mode = kwargs.get("correlation_mode", self.correlation_mode)

        # Ensure we have a template
        if self.template is None:
            raise ValueError("No template provided for subtraction")

        # Calculate window parameters
        half_window = window_samples // 2

        # Ensure indices are numpy array
        if isinstance(indices, list):
            indices = np.array(indices)

        # Define the subtraction function for chunks
        def subtract_templates_from_chunk(chunk, chunk_offset=0):
            """Process a single chunk of data"""
            # Filter indices to those within this chunk
            # Include extra buffer for window size
            chunk_indices = (
                indices[
                    (indices >= chunk_offset + half_window)
                    & (indices < chunk_offset + len(chunk) - half_window)
                ]
                - chunk_offset
            )  # Adjust indices to chunk coordinates

            # If no indices in this chunk, return unchanged
            if len(chunk_indices) == 0:
                return chunk

            # Prepare data
            if chunk.ndim == 1:
                # Reshape 1D data to 2D for consistent processing
                chunk = chunk.reshape(-1, 1)

            # Prepare template
            template = self.template
            if template.ndim == 1:
                # Reshape 1D template to 2D
                template = template.reshape(-1, 1)

            # Apply subtraction
            if self.use_numba:
                # Use Numba-accelerated implementation
                result = _apply_subtraction_numba(
                    chunk, template, chunk_indices, half_window, correlation_mode
                )
            else:
                # Standard implementation
                result = self._apply_subtraction_standard(
                    chunk, template, chunk_indices, half_window, correlation_mode
                )

            # Return result with original shape
            if chunk.ndim != result.ndim:
                if chunk.shape[1] == 1:
                    # Reshape back to 1D if input was 1D
                    result = result.ravel()

            return result

        # Apply subtraction with overlap
        processed_data = data.map_overlap(
            subtract_templates_from_chunk,
            depth=self._window_samples,
            boundary="reflect",
            dtype=data.dtype,
        )

        # Track statistics
        self._subtraction_stats.update(
            {
                "num_indices": len(indices),
                "window_samples": window_samples,
                "correlation_mode": correlation_mode,
            }
        )

        return processed_data

    def _apply_subtraction_standard(
        self,
        data: np.ndarray,
        template: np.ndarray,
        indices: np.ndarray,
        half_window: int,
        correlation_mode: str,
    ) -> np.ndarray:
        """
        Standard (non-Numba) implementation of template subtraction.

        Args:
            data: Data array
            template: Template array
            indices: Indices where templates should be subtracted
            half_window: Half size of window around each index
            correlation_mode: Mode for correlation calculation

        Returns:
            Data with templates subtracted
        """
        # Create a copy to avoid modifying the input
        result = data.copy()

        # Get dimensions
        n_samples, n_channels = data.shape

        # Process each index
        for idx in indices:
            # Skip if index is too close to edges
            if idx < half_window or idx >= n_samples - half_window:
                continue

            # Extract segment
            start = idx - half_window
            end = idx + half_window
            segment = data[start:end, :]

            # Process each channel
            for ch in range(n_channels):
                # Get channel data
                segment_ch = segment[:, ch]
                template_ch = template[:, min(ch, template.shape[1] - 1)]

                # Find optimal alignment using correlation
                corr = correlate(segment_ch, template_ch, mode=correlation_mode)
                max_idx = np.argmax(corr)

                # Calculate shift based on correlation mode
                if correlation_mode == "valid":
                    shift = max_idx
                elif correlation_mode == "same":
                    shift = max_idx - len(segment_ch) // 2
                else:  # "full"
                    shift = max_idx - (len(template_ch) - 1)

                # Apply shift to get proper alignment
                aligned_start = start + max(0, shift)
                aligned_end = min(n_samples, aligned_start + len(template_ch))

                # Skip if no overlap after alignment
                if aligned_end <= aligned_start:
                    continue

                # Get template portion that overlaps
                template_start = max(0, -shift)
                template_end = template_start + (aligned_end - aligned_start)
                template_portion = template_ch[template_start:template_end]

                # Get data portion
                data_portion = data[aligned_start:aligned_end, ch]

                # Calculate scaling factor
                numerator = np.dot(data_portion, template_portion)
                denominator = np.dot(template_portion, template_portion)

                # Avoid division by zero
                if denominator <= 1e-10:
                    scale = 0.0
                else:
                    scale = numerator / denominator

                # Subtract scaled template
                result[aligned_start:aligned_end, ch] -= scale * template_portion

        return result

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
                "correlation_mode": self.correlation_mode,
                "use_numba": self.use_numba,
            }
        )
        return base_summary


@public_api
def create_correlation_subtractor(
    template: np.ndarray,
    window_samples: Optional[int] = None,
    correlation_mode: str = "valid",
) -> CorrelationSubtractor:
    """
    Create a correlation-based template subtractor.

    Args:
        template: Template array to subtract
        window_samples: Window size (default: template length)
        correlation_mode: Mode for correlation calculation

    Returns:
        Configured CorrelationSubtractor
    """
    return CorrelationSubtractor(
        template=template,
        window_samples=window_samples,
        correlation_mode=correlation_mode,
        use_numba=True,
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

    return CorrelationSubtractor(
        template=ecg_template,
        window_samples=window_samples,
        correlation_mode="valid",
        use_numba=True,
    )


@public_api
def subtract_templates(
    data: Union[np.ndarray, da.Array],
    template: np.ndarray,
    indices: Union[np.ndarray, List[int]],
    window_samples: Optional[int] = None,
    correlation_mode: str = "valid",
) -> Union[np.ndarray, da.Array]:
    """
    Convenience function for one-off template subtraction.

    Args:
        data: Input data array
        template: Template to subtract
        indices: Indices where templates should be subtracted
        window_samples: Window size (default: template length)
        correlation_mode: Mode for correlation calculation

    Returns:
        Data with templates subtracted
    """
    # Create subtractor
    subtractor = CorrelationSubtractor(
        template=template,
        window_samples=window_samples,
        correlation_mode=correlation_mode,
    )

    # Convert to dask array if numpy input
    if isinstance(data, np.ndarray):
        input_data = da.from_array(data)
        result = subtractor.process(input_data, indices)
        return result.compute()  # Convert back to numpy
    else:
        # Already dask array
        return subtractor.process(data, indices)
