from typing import Any, Dict, List, Literal, Optional, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from ...engine.base import BaseProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_channel_median(chunk):
    """Compute median across channels (axis=1) with parallelization"""
    n_samples = chunk.shape[0]
    n_channels = chunk.shape[1]
    result = np.zeros(n_samples, dtype=np.float32)

    # Parallelize across samples
    for i in prange(n_samples):
        result[i] = np.median(chunk[i, :])

    return result.reshape(-1, 1)  # Return as column vector


@jit(nopython=True)
def _compute_channel_mean(chunk):
    """Compute mean across channels (axis=1)"""
    return np.mean(chunk, axis=1, keepdims=True)


@jit(nopython=True)
def _apply_global_reference(chunk, shift):
    """Apply global reference by subtracting shift from all channels"""
    return chunk - shift


@jit(nopython=True, parallel=True, cache=True)
def _apply_group_median_reference(chunk, group_channels):
    """Apply median reference to a specific group of channels with parallelization"""
    # Extract the channels for this group
    group_data = chunk[:, group_channels]

    # Compute median for this group
    n_samples = group_data.shape[0]
    median = np.zeros(n_samples, dtype=np.float32)

    # Parallelize the median computation across samples
    for i in prange(n_samples):
        median[i] = np.median(group_data[i, :])

    # Reshape median to column vector
    median = median.reshape(-1, 1)

    # Apply shift to group channels
    result = group_data - median

    return result, median


@jit(nopython=True)
def _apply_group_mean_reference(chunk, group_channels):
    """Apply mean reference to a specific group of channels"""
    # Extract the channels for this group
    group_data = chunk[:, group_channels]

    # Compute mean for this group
    mean = np.mean(group_data, axis=1, keepdims=True)

    # Apply shift to group channels
    result = group_data - mean

    return result, mean


class CommonReferenceProcessor(BaseProcessor):
    """
    Common Reference Processor implementation with JIT acceleration

    Re-references the signal traces by shifting values to a new reference.
    This can be useful for removing common noise across channels.

    Two referencing methods are supported:
        - "global": subtracts the median/average of all channels from each channel
        - "single": subtracts a single channel or the median/average of a group of channels
    """

    def __init__(
        self,
        reference: Literal["global", "single"] = "global",
        operator: Literal["median", "average"] = "median",
        reference_channels: Optional[Union[List[int], int]] = None,
        groups: Optional[List[List[int]]] = None,
        use_jit: bool = True,
    ):
        # Validate arguments
        if reference not in ("global", "single"):
            raise ValueError("'reference' must be either 'global' or 'single'")
        if operator not in ("median", "average"):
            raise ValueError("'operator' must be either 'median' or 'average'")

        self.reference = reference
        self.operator = operator
        self.reference_channels = reference_channels
        self.groups = groups
        self.use_jit = use_jit

        # Set operator function
        if use_jit:
            self.operator_func = (
                _compute_channel_mean
                if operator == "average"
                else _compute_channel_median
            )
        else:
            self.operator_func = np.mean if operator == "average" else np.median

        # Additional checks based on reference type
        if reference == "single":
            if reference_channels is None:
                raise ValueError(
                    "With 'single' reference, 'reference_channels' must be provided"
                )

            # Convert scalar to list
            if np.isscalar(reference_channels):
                self.reference_channels = [reference_channels]

            # Check groups and reference_channels length
            if groups is not None and len(self.reference_channels) != len(groups):
                raise ValueError(
                    "'reference_channels' and 'groups' must have the same length"
                )

        # Set overlap samples (no overlap needed for this operation)
        self._overlap_samples = 0

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Apply common referencing to the data lazily with optional JIT acceleration
        """
        use_jit = kwargs.get("use_jit", self.use_jit)

        # Define the referencing function to apply to each chunk
        def apply_reference(chunk: np.ndarray) -> np.ndarray:
            # If data is 1D, expand to 2D
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)

            # Convert data to float32 for computation if needed
            if chunk.dtype.kind in ["u", "i"]:
                chunk = chunk.astype(np.float32)

            # Ensure data is contiguous for better performance with Numba
            if not chunk.flags.c_contiguous:
                chunk = np.ascontiguousarray(chunk)

            # Apply the appropriate reference method
            if self.groups is None:
                # No groups - apply reference to all channels
                if self.reference == "global":
                    # Global reference
                    if self.reference_channels is None:
                        # Use all channels
                        if use_jit:
                            if self.operator == "median":
                                shift = _compute_channel_median(chunk)
                            else:  # average
                                shift = _compute_channel_mean(chunk)
                            return _apply_global_reference(chunk, shift)
                        else:
                            shift = self.operator_func(chunk, axis=1, keepdims=True)
                            return chunk - shift
                    else:
                        # Use specified channels
                        ref_channels = np.array(self.reference_channels, dtype=np.int32)
                        if use_jit:
                            if self.operator == "median":
                                shift = _compute_channel_median(chunk[:, ref_channels])
                            else:  # average
                                shift = _compute_channel_mean(chunk[:, ref_channels])
                            return _apply_global_reference(chunk, shift)
                        else:
                            shift = self.operator_func(
                                chunk[:, ref_channels], axis=1, keepdims=True
                            )
                            return chunk - shift
                else:  # single reference
                    # Single channel reference
                    ref_channels = np.array(self.reference_channels, dtype=np.int32)
                    shift = chunk[:, ref_channels].mean(axis=1, keepdims=True)
                    return chunk - shift
            else:
                # Apply reference group-wise
                n_samples, n_channels = chunk.shape
                re_referenced = np.zeros_like(chunk, dtype=np.float32)

                # Apply reference to each group separately
                for group_idx, group_channels in enumerate(self.groups):
                    # Ensure group channels are within range
                    valid_channels = np.array(
                        [ch for ch in group_channels if ch < n_channels], dtype=np.int32
                    )

                    if len(valid_channels) == 0:
                        continue

                    if self.reference == "global":
                        # Compute shift from all channels in this group
                        if use_jit:
                            if self.operator == "median":
                                group_result, shift = _apply_group_median_reference(
                                    chunk, valid_channels
                                )
                            else:  # average
                                group_result, shift = _apply_group_mean_reference(
                                    chunk, valid_channels
                                )
                            re_referenced[:, valid_channels] = group_result
                        else:
                            shift = self.operator_func(
                                chunk[:, valid_channels], axis=1, keepdims=True
                            )
                            re_referenced[:, valid_channels] = (
                                chunk[:, valid_channels] - shift
                            )
                    elif self.reference == "single":
                        # Get reference channel for this group
                        ref_idx = (
                            self.reference_channels[group_idx]
                            if group_idx < len(self.reference_channels)
                            else self.reference_channels[0]
                        )
                        # Ensure reference channel is valid
                        if ref_idx >= n_channels:
                            continue
                        # Compute shift from the reference channel
                        shift = chunk[:, ref_idx].reshape(-1, 1)
                        # Apply shift to all channels in the group
                        re_referenced[:, valid_channels] = (
                            chunk[:, valid_channels] - shift
                        )

                return re_referenced

        # Use map_blocks without explicitly specifying chunks to maintain laziness
        return data.map_blocks(apply_reference, dtype=np.float32)

    @property
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Return a dictionary containing processor configuration details"""
        base_summary = super().summary
        base_summary.update(
            {
                "reference": self.reference,
                "operator": self.operator,
                "reference_channels": self.reference_channels,
                "groups": f"{len(self.groups)} groups"
                if self.groups is not None
                else None,
                "jit_acceleration": self.use_jit,
            }
        )
        return base_summary


def create_car_processor(
    reference_channels: Optional[List[int]] = None,
    groups: Optional[List[List[int]]] = None,
    use_jit: bool = True,
) -> CommonReferenceProcessor:
    """
    Create a Common Average Reference (CAR) processor with optional JIT acceleration

    Args:
        reference_channels: Specific channels to use as reference
        groups: Channel groups for group-wise referencing
        use_jit: Whether to use JIT acceleration

    Returns:
        Configured CommonReferenceProcessor
    """
    return CommonReferenceProcessor(
        reference="global",
        operator="average",
        reference_channels=reference_channels,
        groups=groups,
        use_jit=use_jit,
    )


def create_cmr_processor(
    reference_channels: Optional[List[int]] = None,
    groups: Optional[List[List[int]]] = None,
    use_jit: bool = True,
) -> CommonReferenceProcessor:
    """
    Create a Common Median Reference (CMR) processor with optional JIT acceleration

    Args:
        reference_channels: Specific channels to use as reference
        groups: Channel groups for group-wise referencing
        use_jit: Whether to use JIT acceleration

    Returns:
        Configured CommonReferenceProcessor
    """
    return CommonReferenceProcessor(
        reference="global",
        operator="median",
        reference_channels=reference_channels,
        groups=groups,
        use_jit=use_jit,
    )
