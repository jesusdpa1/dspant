"""
Segment extraction module for time-series data.

This module provides functionality to extract segments from continuous data
based on onset markers, with various methods for determining segment boundaries.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np

from ...engine.base import BaseProcessor


class SegmentExtractionProcessor(BaseProcessor):
    """
    Processor for extracting segments from time-series data based on onset markers.

    This processor extracts segments of data around specified onsets, with options
    for defining segment boundaries using fixed windows, offset markers, or
    centering around onsets.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        method: Literal["fixed", "offset", "reflect"] = "fixed",
        reflect_direction: Literal["left", "right", "both"] = "both",
        overlap_allowed: bool = True,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        equalize_lengths: bool = False,
    ):
        """
        Initialize the segment extraction processor.

        Args:
            window_size: Size of the window in samples (required for 'fixed' and 'reflect' methods)
            method: Method for determining segment boundaries
                "fixed": Extract fixed-size windows after each onset
                "offset": Use offset array to determine segment end
                "reflect": Center window around onset point
            reflect_direction: Direction for the reflect method
                "left": Window extends before the onset
                "right": Window extends after the onset
                "both": Window is centered on the onset
            overlap_allowed: Whether segments are allowed to overlap
            pad_mode: Padding mode for segments near boundaries ('constant', 'edge', 'reflect')
            pad_value: Value to use for constant padding
            equalize_lengths: Whether to pad variable-length segments to make them all equal length
                              Only relevant for 'offset' method, ignored for others
        """
        self.window_size = window_size
        self.method = method
        self.reflect_direction = reflect_direction
        self.overlap_allowed = overlap_allowed
        self.pad_mode = pad_mode
        self.pad_value = pad_value
        self.equalize_lengths = equalize_lengths

        # Validate parameters
        self._validate_parameters()

        # No overlap needed for this operation as we'll be extracting discontinuous segments
        self._overlap_samples = 0

    def _validate_parameters(self):
        """Validate processor parameters."""
        if self.method in ["fixed", "reflect"] and self.window_size is None:
            raise ValueError(
                f"window_size must be specified for method '{self.method}'"
            )

        if self.method not in ["fixed", "offset", "reflect"]:
            raise ValueError(f"Unknown method: {self.method}")

        if self.reflect_direction not in ["left", "right", "both"]:
            raise ValueError(f"Unknown reflect_direction: {self.reflect_direction}")

    def process(
        self,
        data: da.Array,
        onsets: Union[np.ndarray, List[int]],
        offsets: Optional[Union[np.ndarray, List[int]]] = None,
        **kwargs,
    ) -> da.Array:
        """
        Extract segments from the input data based on onsets.

        Args:
            data: Input dask array (samples × channels)
            onsets: Array of onset indices
            offsets: Optional array of offset indices (required for 'offset' method)
            **kwargs: Additional keyword arguments
                window_size: Override the window size

        Returns:
            Dask array of extracted segments (segments × samples × channels)
        """
        # Override window_size if provided
        window_size = kwargs.get("window_size", self.window_size)

        # Convert onsets and offsets to numpy arrays
        onsets = np.asarray(onsets, dtype=np.int64)

        # Sort onsets if needed
        if kwargs.get("sort_onsets", True):
            onsets = np.sort(onsets)

        # Ensure input is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Get data shape
        n_samples, n_channels = data.shape

        # Handle different segmentation methods
        if self.method == "fixed":
            if window_size is None:
                raise ValueError("window_size must be specified for 'fixed' method")

            # Create segment boundaries for fixed windows
            segment_boundaries = self._compute_fixed_boundaries(
                onsets, window_size, n_samples
            )

        elif self.method == "offset":
            if offsets is None:
                raise ValueError("offsets must be provided for 'offset' method")

            offsets = np.asarray(offsets, dtype=np.int64)

            if len(onsets) != len(offsets):
                raise ValueError("onsets and offsets must have the same length")

            # Create segment boundaries from onset-offset pairs
            segment_boundaries = self._compute_offset_boundaries(onsets, offsets)

        elif self.method == "reflect":
            if window_size is None:
                raise ValueError("window_size must be specified for 'reflect' method")

            # Create segment boundaries for reflected windows
            segment_boundaries = self._compute_reflect_boundaries(
                onsets, window_size, n_samples, self.reflect_direction
            )

        # Filter out invalid boundaries
        valid_indices = [
            i
            for i, (start, end) in enumerate(segment_boundaries)
            if start < n_samples and end > 0 and end > start
        ]

        if not valid_indices:
            # Return empty result if no valid segments
            return da.zeros((0, 1, n_channels), chunks=(1, 1, n_channels))

        valid_boundaries = [segment_boundaries[i] for i in valid_indices]

        # Check for overlapping segments if not allowed
        if not self.overlap_allowed:
            valid_boundaries = self._filter_overlapping_segments(valid_boundaries)

        if not valid_boundaries:
            # Return empty result if no valid segments after filtering
            return da.zeros((0, 1, n_channels), chunks=(1, 1, n_channels))

        # Extract segments using dask array indexing
        segment_lengths = [end - start for start, end in valid_boundaries]
        max_length = max(segment_lengths)
        n_segments = len(valid_boundaries)

        # Pre-allocate result array with appropriate chunking
        # Chunk along the segments dimension for better parallelism
        result = da.zeros(
            (n_segments, max_length, n_channels),
            chunks=(min(100, n_segments), max_length, n_channels),
            dtype=data.dtype,
        )

        # Extract each segment and handle padding if needed
        for i, (start, end) in enumerate(valid_boundaries):
            segment_length = end - start

            # Handle boundary cases where padding is needed
            if start < 0 or end > n_samples:
                # Create padded extract
                pad_left = max(0, -start)
                pad_right = max(0, end - n_samples)
                valid_start = max(0, start)
                valid_end = min(n_samples, end)

                # Extract the valid portion
                valid_segment = data[valid_start:valid_end]

                # Pad as needed for boundary handling
                if pad_left > 0 or pad_right > 0:
                    if self.pad_mode == "constant":
                        # Use constant padding
                        padded_segment = da.pad(
                            valid_segment,
                            ((pad_left, pad_right), (0, 0)),
                            mode="constant",
                            constant_values=self.pad_value,
                        )
                    else:
                        # Use specified padding mode
                        padded_segment = da.pad(
                            valid_segment,
                            ((pad_left, pad_right), (0, 0)),
                            mode=self.pad_mode,
                        )

                    # Store in result array
                    if self.equalize_lengths and segment_length < max_length:
                        # Pad the segment to max_length to equalize lengths
                        pad_right_eq = max_length - segment_length
                        if self.pad_mode == "constant":
                            result[i] = da.pad(
                                padded_segment,
                                ((0, pad_right_eq), (0, 0)),
                                mode="constant",
                                constant_values=self.pad_value,
                            )
                        else:
                            result[i] = da.pad(
                                padded_segment,
                                ((0, pad_right_eq), (0, 0)),
                                mode=self.pad_mode,
                            )
                    else:
                        # Store without additional padding
                        result[i, :segment_length] = padded_segment
                else:
                    # Handle equalizing lengths if needed
                    if self.equalize_lengths and segment_length < max_length:
                        pad_right_eq = max_length - segment_length
                        if self.pad_mode == "constant":
                            result[i] = da.pad(
                                valid_segment,
                                ((0, pad_right_eq), (0, 0)),
                                mode="constant",
                                constant_values=self.pad_value,
                            )
                        else:
                            result[i] = da.pad(
                                valid_segment,
                                ((0, pad_right_eq), (0, 0)),
                                mode=self.pad_mode,
                            )
                    else:
                        # Store without additional padding
                        result[i, :segment_length] = valid_segment
            else:
                # Direct extraction (no boundary padding needed)
                segment = data[start:end]

                # Handle equalizing lengths if needed
                if self.equalize_lengths and segment_length < max_length:
                    pad_right_eq = max_length - segment_length
                    if self.pad_mode == "constant":
                        result[i] = da.pad(
                            segment,
                            ((0, pad_right_eq), (0, 0)),
                            mode="constant",
                            constant_values=self.pad_value,
                        )
                    else:
                        result[i] = da.pad(
                            segment,
                            ((0, pad_right_eq), (0, 0)),
                            mode=self.pad_mode,
                        )
                else:
                    # Store without additional padding
                    result[i, :segment_length] = segment

        return result

    def _compute_fixed_boundaries(
        self, onsets: np.ndarray, window_size: int, n_samples: int
    ) -> List[Tuple[int, int]]:
        """
        Compute segment boundaries for fixed-size windows after onsets.

        Args:
            onsets: Array of onset indices
            window_size: Size of the window
            n_samples: Total number of samples in the data

        Returns:
            List of (start, end) segment boundary tuples
        """
        return [(int(onset), int(onset + window_size)) for onset in onsets]

    def _compute_offset_boundaries(
        self,
        onsets: np.ndarray,
        offsets: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Compute segment boundaries from onset-offset pairs.

        Args:
            onsets: Array of onset indices
            offsets: Array of offset indices

        Returns:
            List of (start, end) segment boundary tuples
        """
        return [(int(onset), int(offset)) for onset, offset in zip(onsets, offsets)]

    def _compute_reflect_boundaries(
        self,
        onsets: np.ndarray,
        window_size: int,
        n_samples: int,
        direction: str,
    ) -> List[Tuple[int, int]]:
        """
        Compute segment boundaries for windows centered on or around onsets.

        Args:
            onsets: Array of onset indices
            window_size: Size of the window
            n_samples: Total number of samples in the data
            direction: Direction for window ("left", "right", or "both")

        Returns:
            List of (start, end) segment boundary tuples
        """
        if direction == "left":
            # Windows extend before onsets
            return [(int(onset - window_size), int(onset)) for onset in onsets]
        elif direction == "right":
            # Windows extend after onsets
            return [(int(onset), int(onset + window_size)) for onset in onsets]
        else:  # "both"
            # Windows centered on onsets
            half_window = window_size // 2
            return [
                (int(onset - half_window), int(onset + window_size - half_window))
                for onset in onsets
            ]

    def _filter_overlapping_segments(
        self, boundaries: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Filter out overlapping segments, keeping earlier ones.

        Args:
            boundaries: List of (start, end) segment boundary tuples

        Returns:
            Filtered list of non-overlapping segment boundaries
        """
        if not boundaries:
            return []

        # Sort by start time
        sorted_boundaries = sorted(boundaries, key=lambda x: x[0])

        filtered = [sorted_boundaries[0]]
        for start, end in sorted_boundaries[1:]:
            prev_start, prev_end = filtered[-1]

            # If current segment starts after previous ends, no overlap
            if start >= prev_end:
                filtered.append((start, end))

        return filtered

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap processing."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "method": self.method,
                "window_size": self.window_size,
                "reflect_direction": self.reflect_direction
                if self.method == "reflect"
                else None,
                "overlap_allowed": self.overlap_allowed,
                "pad_mode": self.pad_mode,
                "equalize_lengths": self.equalize_lengths,
            }
        )
        return base_summary


# Convenience factory functions


def create_fixed_window_extractor(
    window_size: int,
    overlap_allowed: bool = True,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> SegmentExtractionProcessor:
    """
    Create a segment extractor that uses fixed-size windows after onsets.

    Args:
        window_size: Size of the window in samples
        overlap_allowed: Whether segments are allowed to overlap
        pad_mode: Padding mode for segments near boundaries
        pad_value: Value to use for constant padding

    Returns:
        Configured SegmentExtractionProcessor
    """
    return SegmentExtractionProcessor(
        window_size=window_size,
        method="fixed",
        overlap_allowed=overlap_allowed,
        pad_mode=pad_mode,
        pad_value=pad_value,
        # Fixed windows are already equal length by definition
    )


def create_onset_offset_extractor(
    overlap_allowed: bool = True,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
    equalize_lengths: bool = False,
) -> SegmentExtractionProcessor:
    """
    Create a segment extractor that uses onset-offset pairs to define segments.

    Args:
        overlap_allowed: Whether segments are allowed to overlap
        pad_mode: Padding mode for segments near boundaries
        pad_value: Value to use for constant padding
        equalize_lengths: Whether to pad segments to make them all equal length

    Returns:
        Configured SegmentExtractionProcessor
    """
    return SegmentExtractionProcessor(
        method="offset",
        overlap_allowed=overlap_allowed,
        pad_mode=pad_mode,
        pad_value=pad_value,
        equalize_lengths=equalize_lengths,
    )


def create_centered_extractor(
    window_size: int,
    direction: Literal["left", "right", "both"] = "both",
    overlap_allowed: bool = True,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> SegmentExtractionProcessor:
    """
    Create a segment extractor with windows centered on or around onsets.

    Args:
        window_size: Size of the window in samples
        direction: Direction for window placement
            "left": Window extends before onsets
            "right": Window extends after onsets
            "both": Window is centered on onsets
        overlap_allowed: Whether segments are allowed to overlap
        pad_mode: Padding mode for segments near boundaries
        pad_value: Value to use for constant padding

    Returns:
        Configured SegmentExtractionProcessor
    """
    return SegmentExtractionProcessor(
        window_size=window_size,
        method="reflect",
        reflect_direction=direction,
        overlap_allowed=overlap_allowed,
        pad_mode=pad_mode,
        pad_value=pad_value,
        # Reflect windows are already equal length by definition
    )
