"""
Dynamic Time Warping (DTW) based template subtraction.

This module implements template subtraction using Dynamic Time Warping for alignment,
which can handle time-stretched or compressed versions of the template pattern.
This is particularly useful for removing artifacts with variable morphology
or duration, such as ECG artifacts with changing heart rates.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from dspant.core.internals import public_api
from dspant.pattern.subtraction.base import BaseSubtractor


@jit(nopython=True, cache=True)
def _dtw_distance_matrix(
    x: np.ndarray,
    y: np.ndarray,
    window: Optional[int] = None,
    distance_only: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Calculate the DTW distance matrix between two sequences.

    Args:
        x: First sequence
        y: Second sequence
        window: Sakoe-Chiba constraint window size (None for no constraint)
        distance_only: Whether to only compute the final distance

    Returns:
        Tuple of (cumulative_distance_matrix, final_distance)
        If distance_only=True, returns (None, final_distance)
    """
    nx = len(x)
    ny = len(y)

    # Use squared Euclidean for performance
    def dist(a, b):
        return (a - b) ** 2

    # Apply window constraint if specified
    if window is not None:
        window = max(
            window, abs(nx - ny)
        )  # Window must be at least the length difference
    else:
        window = max(nx, ny)

    # Initialize distance matrix with infinity
    if not distance_only:
        dtw_matrix = np.full((nx + 1, ny + 1), np.inf)
        dtw_matrix[0, 0] = 0
    else:
        # Only store two rows for memory efficiency
        dtw_matrix = np.full((2, ny + 1), np.inf)
        dtw_matrix[0, 0] = 0

    # Fill distance matrix
    for i in range(1, nx + 1):
        # Determine columns to fill based on window constraint
        j_start = max(1, i - window)
        j_end = min(ny + 1, i + window + 1)

        # For memory-efficient version, use alternating rows
        if distance_only:
            curr_row = i % 2
            prev_row = (i - 1) % 2

            # Reset current row
            for j in range(ny + 1):
                dtw_matrix[curr_row, j] = np.inf
        else:
            curr_row = i
            prev_row = i - 1

        # Fill current row
        for j in range(j_start, j_end):
            cost = dist(x[i - 1], y[j - 1])

            # Find minimum of three possible previous steps
            min_prev = min(
                dtw_matrix[prev_row, j - 1],  # Diagonal
                dtw_matrix[prev_row, j],  # Vertical
                dtw_matrix[curr_row, j - 1],  # Horizontal
            )

            # Update current cell
            dtw_matrix[curr_row, j] = cost + min_prev

    # Return final distance value
    if distance_only:
        final_distance = dtw_matrix[(nx % 2), ny]
        return None, final_distance
    else:
        final_distance = dtw_matrix[nx, ny]
        return dtw_matrix, final_distance


@jit(nopython=True, cache=True)
def _dtw_warping_path(dtw_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the optimal warping path from a DTW matrix.

    Args:
        dtw_matrix: DTW cumulative distance matrix

    Returns:
        Array of index pairs (i, j) representing the warping path
    """
    n, m = dtw_matrix.shape
    i, j = n - 1, m - 1

    # Pre-allocate path with maximum possible length
    path = np.zeros((n + m - 1, 2), dtype=np.int32)
    path_idx = 0

    # Trace path from bottom-right to top-left
    while i > 0 or j > 0:
        path[path_idx, 0] = i - 1 if i > 0 else 0
        path[path_idx, 1] = j - 1 if j > 0 else 0
        path_idx += 1

        # Decide which direction to move
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Find the minimum of the three options
            min_val = min(
                dtw_matrix[i - 1, j - 1],  # Diagonal
                dtw_matrix[i - 1, j],  # Up
                dtw_matrix[i, j - 1],  # Left
            )

            if min_val == dtw_matrix[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif min_val == dtw_matrix[i - 1, j]:
                i -= 1
            else:
                j -= 1

    # Reverse path and trim to actual size
    path = path[:path_idx]
    path = path[::-1]

    return path


@jit(nopython=True, cache=True)
def _warp_template(
    template: np.ndarray, path: np.ndarray, target_length: int
) -> np.ndarray:
    """
    Warp a template according to a DTW path.

    Args:
        template: Template to warp
        path: DTW warping path indices
        target_length: Length of the target signal

    Returns:
        Warped template aligned to the target signal
    """
    # Extract template indices from the path
    template_indices = path[:, 0]
    target_indices = path[:, 1]

    # Create mapping from target indices to template values
    mapping = np.zeros(target_length)
    values = np.zeros(target_length)
    counts = np.zeros(target_length)

    # Fill in the mapping
    for i in range(len(path)):
        template_idx = template_indices[i]
        target_idx = target_indices[i]

        if target_idx < target_length:
            mapping[target_idx] = template_idx
            values[target_idx] += template[template_idx]
            counts[target_idx] += 1

    # Average values for indices that map to multiple template points
    for i in range(target_length):
        if counts[i] > 0:
            values[i] /= counts[i]

    # Interpolate missing values
    valid_indices = np.where(counts > 0)[0]
    if len(valid_indices) < 2:
        # Not enough points for interpolation
        return np.zeros(target_length)

    # Create interpolating function
    valid_values = values[valid_indices]

    # Linear interpolation for missing points
    warped_template = np.zeros(target_length)
    last_valid = -1

    for i in range(target_length):
        if counts[i] > 0:
            warped_template[i] = values[i]

            # Fill gap from last valid point
            if last_valid >= 0 and last_valid < i - 1:
                # Linear interpolation
                for j in range(last_valid + 1, i):
                    alpha = (j - last_valid) / (i - last_valid)
                    warped_template[j] = (1 - alpha) * warped_template[
                        last_valid
                    ] + alpha * values[i]

            last_valid = i

    # Fill in any trailing values
    if last_valid >= 0 and last_valid < target_length - 1:
        for i in range(last_valid + 1, target_length):
            warped_template[i] = warped_template[last_valid]

    return warped_template


@jit(nopython=True, parallel=True, cache=True)
def _apply_dtw_subtraction_numba(
    data: np.ndarray,
    template: np.ndarray,
    indices: np.ndarray,
    half_window: int,
    window_constraint: Optional[int] = None,
) -> np.ndarray:
    """
    Subtract templates from data using DTW alignment and Numba acceleration.

    Args:
        data: Data array (samples x channels)
        template: Template array (template_samples x channels)
        indices: Indices where templates should be subtracted
        half_window: Half size of window around each index
        window_constraint: Sakoe-Chiba window constraint size

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
        window_size = end - start

        # Process each channel separately
        for ch in prange(n_channels):
            # Get channel data
            segment = data[start:end, ch]
            template_ch = template[:, min(ch, template.shape[1] - 1)]

            # Calculate DTW matrix
            dtw_matrix, _ = _dtw_distance_matrix(
                template_ch, segment, window=window_constraint, distance_only=False
            )

            # Compute warping path
            path = _dtw_warping_path(dtw_matrix)

            # Warp template to match segment timing
            warped_template = _warp_template(template_ch, path, window_size)

            # Calculate scaling factor
            numerator = np.dot(segment, warped_template)
            denominator = np.dot(warped_template, warped_template)

            # Avoid division by zero
            if denominator <= 1e-10:
                scale = 0.0
            else:
                scale = numerator / denominator

            # Subtract scaled warped template
            output[start:end, ch] -= scale * warped_template

    return output


@public_api
class DTWSubtractor(BaseSubtractor):
    """
    Template subtraction using Dynamic Time Warping for alignment.

    This subtractor uses DTW to align templates with signal segments,
    allowing it to handle time-stretched or compressed versions of the template.
    It's particularly effective for removing artifacts with variable duration
    or morphology, such as ECG artifacts with varying heart rates.
    """

    def __init__(
        self,
        template: Optional[np.ndarray] = None,
        window_samples: Optional[int] = None,
        window_constraint: Optional[int] = None,
        use_numba: bool = True,
    ):
        """
        Initialize the DTW-based subtractor.

        Args:
            template: Template array to subtract (samples x channels)
                     If None, must be set later
            window_samples: Window size to use for subtraction
                            If None, determined from template size
            window_constraint: Sakoe-Chiba bandwidth constraint
                               If None, no constraint is applied
            use_numba: Whether to use Numba acceleration
        """
        super().__init__(template)
        self.window_constraint = window_constraint
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
        Process data by subtracting templates using DTW alignment.

        Args:
            data: Input data array
            indices: Indices where templates should be subtracted
            fs: Sampling frequency (for metadata)
            **kwargs: Additional parameters:
                window_samples: Override window size
                window_constraint: Override DTW constraint window

        Returns:
            Data with templates subtracted
        """
        # If indices not provided, can't do subtraction
        if indices is None:
            return data

        # Check for overrides
        window_samples = kwargs.get("window_samples", self._window_samples)
        window_constraint = kwargs.get("window_constraint", self.window_constraint)

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
                result = _apply_dtw_subtraction_numba(
                    chunk, template, chunk_indices, half_window, window_constraint
                )
            else:
                # Standard implementation
                result = self._apply_subtraction_standard(
                    chunk, template, chunk_indices, half_window, window_constraint
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
                "window_constraint": window_constraint,
            }
        )

        return processed_data

    def _apply_subtraction_standard(
        self,
        data: np.ndarray,
        template: np.ndarray,
        indices: np.ndarray,
        half_window: int,
        window_constraint: Optional[int],
    ) -> np.ndarray:
        """
        Standard (non-Numba) implementation of DTW-based subtraction.

        Args:
            data: Data array
            template: Template array
            indices: Indices where templates should be subtracted
            half_window: Half size of window around each index
            window_constraint: Sakoe-Chiba window constraint size

        Returns:
            Data with templates subtracted
        """
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean

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

                # Apply DTW (using fastdtw library for non-Numba version)
                # This is faster than pure Python implementation
                distance, path = fastdtw(
                    template_ch,
                    segment_ch,
                    dist=euclidean,
                    radius=window_constraint if window_constraint else None,
                )

                # Convert path to numpy array for warping
                path = np.array(path)

                # Warp template to match segment
                window_size = end - start
                warped_template = np.zeros(window_size)

                # Extract mapping
                template_indices = path[:, 0]
                segment_indices = path[:, 1]

                # Create a mapping from segment to template
                for i in range(len(path)):
                    template_idx = template_indices[i]
                    segment_idx = segment_indices[i]
                    if segment_idx < window_size:
                        warped_template[segment_idx] = template_ch[template_idx]

                # Calculate scaling factor
                numerator = np.dot(segment_ch, warped_template)
                denominator = np.dot(warped_template, warped_template)

                # Avoid division by zero
                if denominator <= 1e-10:
                    scale = 0.0
                else:
                    scale = numerator / denominator

                # Subtract scaled warped template
                result[start:end, ch] -= scale * warped_template

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
                "window_constraint": self.window_constraint,
                "use_numba": self.use_numba,
            }
        )
        return base_summary


@public_api
def create_dtw_subtractor(
    template: np.ndarray,
    window_samples: Optional[int] = None,
    window_constraint: Optional[int] = None,
) -> DTWSubtractor:
    """
    Create a DTW-based template subtractor.

    Args:
        template: Template array to subtract
        window_samples: Window size (default: template length)
        window_constraint: DTW Sakoe-Chiba constraint window

    Returns:
        Configured DTWSubtractor
    """
    return DTWSubtractor(
        template=template,
        window_samples=window_samples,
        window_constraint=window_constraint,
        use_numba=True,
    )


@public_api
def create_variable_ecg_subtractor(
    ecg_template: np.ndarray,
    window_ms: float = 120.0,
    constraint_ms: float = 40.0,
    fs: float = 1000.0,
) -> DTWSubtractor:
    """
    Create a subtractor optimized for ECG artifact removal with variable heart rates.

    Args:
        ecg_template: ECG template array
        window_ms: Window size in milliseconds
        constraint_ms: DTW constraint window in milliseconds
        fs: Sampling frequency in Hz

    Returns:
        Configured DTWSubtractor for variable ECG removal
    """
    # Calculate window and constraint sizes in samples
    window_samples = int((window_ms / 1000) * fs)
    constraint_samples = int((constraint_ms / 1000) * fs)

    return DTWSubtractor(
        template=ecg_template,
        window_samples=window_samples,
        window_constraint=constraint_samples,
        use_numba=True,
    )


@public_api
def subtract_templates_dtw(
    data: Union[np.ndarray, da.Array],
    template: np.ndarray,
    indices: Union[np.ndarray, List[int]],
    window_samples: Optional[int] = None,
    window_constraint: Optional[int] = None,
) -> Union[np.ndarray, da.Array]:
    """
    Convenience function for one-off DTW-based template subtraction.

    Args:
        data: Input data array
        template: Template to subtract
        indices: Indices where templates should be subtracted
        window_samples: Window size (default: template length)
        window_constraint: DTW constraint window

    Returns:
        Data with templates subtracted
    """
    # Create subtractor
    subtractor = DTWSubtractor(
        template=template,
        window_samples=window_samples,
        window_constraint=window_constraint,
    )

    # Convert to dask array if numpy input
    if isinstance(data, np.ndarray):
        input_data = da.from_array(data)
        result = subtractor.process(input_data, indices)
        return result.compute()  # Convert back to numpy
    else:
        # Already dask array
        return subtractor.process(data, indices)
