"""
Cross-correlation based template subtraction.

This module implements template subtraction using correlation for optimal
alignment and scaling. It's particularly effective for removing artifacts
like ECG from EMG or other electrophysiological signals.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import dask.array as da
import numpy as np
from scipy.signal import correlate

from dspant.core.internals import public_api
from dspant.pattern.subtraction.base import BaseSubtractor


@public_api
class CorrelationSubtractor(BaseSubtractor):
    """
    Template subtraction using cross-correlation for alignment.

    This subtractor aligns templates with signal segments using cross-correlation,
    scales them optimally, and subtracts them from the original signal.
    Useful for removing artifacts like ECG from EMG or EEG signals.
    """

    def __init__(
        self,
    ):
        """
        Initialize the correlation subtractor.
        """
        super().__init__()
        self.processed_indices = []

    def process(
        self,
        data: da.Array,
        indices: Optional[Union[np.ndarray, List[int]]] = None,
        fs: Optional[float] = None,
        template: Optional[Union[np.ndarray, da.Array]] = None,
        mode: Optional[Literal["global", None]] = None,
        half_window: Optional[int] = None,
        window_factor: Optional[float] = 2.0,
        **kwargs,
    ) -> da.Array:
        """
        Subtract templates from data at specified indices.

        Args:
            data: Dask array of shape (samples, channels).
            indices: List or array of sample indices where templates are subtracted.
            fs: Sampling frequency in Hz (optional).
            template: Template array of shape (samples, channels) or (samples,).
            mode: 'global' applies the first channel of template to all channels;
                  None uses per-channel subtraction.
            half_window: Half-window size in samples (defaults to half template length).
            window_factor: Multiplier for window size relative to template length.
            **kwargs: Reserved for future options.

        Returns:
            Data array with template artifacts subtracted.
        """
        # Reset processed indices
        self.processed_indices = []
        self.window_factor = window_factor

        if template is None:
            raise ValueError("Template must be provided for subtraction")

        if indices is None:
            raise ValueError("Indices must be provided for template subtraction.")

        # Ensure template is contiguous numpy array
        if isinstance(template, da.Array):
            template = np.ascontiguousarray(template.compute())
        else:
            template = np.ascontiguousarray(template)

        if template.ndim == 1:
            template = template[:, None]

        # Validate dimensions
        if mode is None and template.shape[1] != data.shape[1]:
            raise ValueError(
                f"Template channels ({template.shape[1]}) must match data channels ({data.shape[1]}) "
                f"when mode is None. Use mode='global' for single-channel templates."
            )

        # Set up window parameters
        template_len = template.shape[0]
        half_window = half_window or template_len // 2

        # Critical fix: Use a much larger overlap to ensure proper boundary handling
        # This is the key change that should fix the boundary issues
        window_size = int(template_len * window_factor)
        self._overlap_samples = window_size * 3  # Triple the window size for overlap

        # Ensure indices are sorted contiguous array for efficiency
        indices = np.sort(np.asarray(indices))

        # Adjust indices for sliced arrays
        data_offset = 0
        if hasattr(data, "_key"):
            try:
                key_info = data._key
                if isinstance(key_info, tuple) and len(key_info) >= 2:
                    slice_info = key_info[1]
                    if isinstance(slice_info, tuple) and len(slice_info) > 0:
                        if isinstance(slice_info[0], slice):
                            start = slice_info[0].start
                            if start is not None:
                                data_offset = start
                                # Check if indices need adjustment
                                if np.any(indices >= data.shape[0] + start):
                                    indices = indices - start
            except Exception as e:
                pass

        # Track processed indices across chunks
        all_processed_indices = []

        def subtract_chunk(chunk, block_info=None):
            """Process a single chunk with proper boundary handling"""
            # Ensure chunk is 2D contiguous
            if chunk.ndim == 1:
                chunk = np.ascontiguousarray(chunk[:, None])
            else:
                chunk = np.ascontiguousarray(chunk)

            # Get chunk position
            chunk_start = 0
            if block_info and 0 in block_info:
                chunk_start = block_info[0]["array-location"][0][0]

            # Get chunk end
            chunk_end = chunk_start + chunk.shape[0]

            # Find relevant indices - use a wider margin to ensure boundary indices are processed
            margin = window_size * 4
            mask = (indices >= chunk_start - margin) & (indices < chunk_end + margin)
            relevant_indices = indices[mask]

            # If no relevant indices, return chunk unchanged
            if len(relevant_indices) == 0:
                return chunk

            # Create output array
            result = chunk.copy()

            # Create array of local indices within this chunk
            local_indices = relevant_indices - chunk_start

            # List to collect processed indices for this chunk
            chunk_processed_indices = []

            # Process each channel
            if mode == "global":
                # Global mode: use same template channel for all data channels
                template_ch = np.ascontiguousarray(template[:, 0])
                for ch in range(result.shape[1]):
                    channel_data = np.ascontiguousarray(result[:, ch])
                    result[:, ch], processed = self._subtract_from_channel(
                        channel_data,
                        template_ch,
                        local_indices,
                        half_window,
                    )
                    # Only collect indices from the first channel to avoid duplicates
                    if ch == 0:
                        # Convert local indices back to global indices
                        global_indices = [idx + chunk_start for idx in processed]
                        chunk_processed_indices.extend(global_indices)
            else:
                # Per-channel mode: use matching template channel
                for ch in range(result.shape[1]):
                    # Get template channel (use last one if out of bounds)
                    t_ch = min(ch, template.shape[1] - 1)
                    template_ch = np.ascontiguousarray(template[:, t_ch])

                    # Process channel
                    channel_data = np.ascontiguousarray(result[:, ch])
                    result[:, ch], processed = self._subtract_from_channel(
                        channel_data,
                        template_ch,
                        local_indices,
                        half_window,
                    )
                    # For multi-channel, collect indices from each channel
                    if ch == 0:  # Only collect once per template position
                        # Convert local indices back to global indices
                        global_indices = [idx + chunk_start for idx in processed]
                        chunk_processed_indices.extend(global_indices)

            # Collect the global indices that were processed in this chunk
            all_processed_indices.extend(chunk_processed_indices)

            return result

        # Process data with overlap
        result = data.map_overlap(
            subtract_chunk,
            depth={-2: self._overlap_samples},  # Increased overlap depth
            boundary="reflect",
            dtype=data.dtype,
            block_info=True,
        )

        # Store the processed indices
        if all_processed_indices:
            # Remove duplicates and sort
            self.processed_indices = sorted(set(all_processed_indices))
        else:
            self.processed_indices = []

        # Save stats
        self._subtraction_stats.update(
            {
                "num_input_indices": len(indices),
                "num_processed_indices": len(self.processed_indices),
                "window_size": window_size,
                "template_length": template_len,
                "fs": fs,
                "mode": mode,
                "data_offset": data_offset,
            }
        )

        return result

    def _subtract_from_channel(
        self,
        data: np.ndarray,
        template: np.ndarray,
        indices: np.ndarray,
        half_window: int,
    ) -> tuple:
        """
        Subtract template from a single channel at specified indices.

        Args:
            data: Channel data array
            template: Template to subtract
            indices: Indices (in local coordinates) where templates should be subtracted
            half_window: Half-window size for correlation

        Returns:
            Tuple of (processed_data, processed_indices)
        """
        # Make a copy and ensure it's contiguous
        result = np.ascontiguousarray(data.copy())
        n_samples = len(result)

        # Keep track of successfully processed indices
        processed_indices = []

        # Process each index
        for idx in indices:
            # Skip if index is out of range or too close to edge
            if idx < 0 or idx >= n_samples:
                continue

            # Extract segment for correlation
            segment = result[idx - half_window : idx + half_window]

            # Skip if segment is too short
            if len(segment) < half_window:
                continue

            # Calculate correlation with full mode for better alignment
            corr = correlate(segment, template, mode="full")

            # Find optimal lag (shift) to align template with signal
            lag = np.argmax(corr) - (len(corr) // 2)

            # Calculate start and end positions with shift
            start = idx - half_window + lag
            end = start + len(template)

            # Skip if adjustment puts us out of bounds
            if start < 0 or end > n_samples:
                continue

            # Extract the aligned segment
            aligned = result[start:end]

            # Skip if lengths don't match
            if len(aligned) != len(template):
                continue

            # Calculate optimal scaling factor to match template amplitude
            energy = np.dot(template, template)
            if energy > 1e-12:  # Avoid division by zero
                scale = np.dot(aligned, template) / energy

                # Subtract scaled template
                result[start:end] -= scale * template
                processed_indices.append(idx)

        return result, processed_indices

    def get_processed_indices(self) -> list:
        """Get the indices that were successfully processed."""
        return self.processed_indices

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the subtractor configuration"""
        base_summary = super().summary
        if hasattr(self, "window_factor"):
            base_summary.update(
                {
                    "window_factor": self.window_factor,
                }
            )

        # Include information about processed indices
        if hasattr(self, "processed_indices") and self.processed_indices:
            base_summary.update(
                {
                    "num_processed_indices": len(self.processed_indices),
                    "processed_indices_range": (
                        min(self.processed_indices),
                        max(self.processed_indices),
                    )
                    if self.processed_indices
                    else None,
                }
            )

        return base_summary


@public_api
def subtract_templates(
    data: Union[np.ndarray, da.Array],
    template: np.ndarray,
    indices: Union[np.ndarray, List[int]],
    half_window: Optional[int] = None,
    mode: Optional[Literal["global", None]] = None,
) -> Union[np.ndarray, da.Array]:
    """
    Convenience wrapper for one-off template subtraction.

    Args:
        data: Input signal array (samples × channels).
        template: Template to subtract (samples × channels or 1D).
        indices: Where to subtract template.
        half_window: Half window around each index (optional).
        mode: 'global' for shared template; None for per-channel.

    Returns:
        Signal with artifacts removed.
    """
    subtractor = CorrelationSubtractor()
    result = subtractor.process(
        data=data,
        indices=indices,
        template=template,
        half_window=half_window,
        mode=mode,
    )
    return result


@public_api
def create_correlation_subtractor() -> CorrelationSubtractor:
    """
    Create a correlation-based template subtractor.

    Returns:
        Configured CorrelationSubtractor
    """
    return CorrelationSubtractor()
