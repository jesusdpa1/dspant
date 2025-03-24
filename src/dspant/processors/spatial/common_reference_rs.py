"""
Rust-accelerated common reference implementation.

This module provides high-performance Rust implementations of common reference
methods like CAR (Common Average Reference) and CMR (Common Median Reference).
"""

from typing import Any, Dict, List, Literal, Optional, Union

import dask.array as da
import numpy as np

try:
    from dspant._rs import (
        apply_channel_reference,
        apply_global_reference,
        apply_global_reference_parallel,
        apply_group_reference,
        compute_channel_mean,
        compute_channel_mean_parallel,
        # Import Rust functions
        compute_channel_median,
        compute_channel_median_parallel,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    print(
        "Warning: Rust extension not available, falling back to Python implementation."
    )

from ...engine.base import BaseProcessor


class CommonReferenceRustProcessor(BaseProcessor):
    """
    Rust-accelerated Common Reference Processor implementation.

    Re-references the signal traces by shifting values to a new reference
    using Rust implementations for better performance.

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
        use_parallel: bool = True,
    ):
        """Initialize the Rust-accelerated common reference processor."""
        if not _HAS_RUST:
            raise ImportError(
                "Rust extension not available. Install with 'pip install dspant[rust]'"
            )

        # Validate arguments
        if reference not in ("global", "single"):
            raise ValueError("'reference' must be either 'global' or 'single'")
        if operator not in ("median", "average"):
            raise ValueError("'operator' must be either 'median' or 'average'")

        self.reference = reference
        self.operator = operator
        self.reference_channels = reference_channels
        self.groups = groups
        self.use_parallel = use_parallel

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
        Apply common referencing to the data lazily with Rust acceleration.

        Parameters
        ----------
        data : da.Array
            Input data as a Dask array
        fs : float, optional
            Sampling frequency (not used but required by interface)
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        referenced_data : da.Array
            Re-referenced data as a Dask array
        """
        # Override parallel setting if specified in kwargs
        use_parallel = kwargs.get("use_parallel", self.use_parallel)

        # Select appropriate Rust functions based on settings
        if self.operator == "median":
            compute_func = (
                compute_channel_median_parallel
                if use_parallel
                else compute_channel_median
            )
        else:  # average
            compute_func = (
                compute_channel_mean_parallel if use_parallel else compute_channel_mean
            )

        apply_func = (
            apply_global_reference_parallel if use_parallel else apply_global_reference
        )

        # Define the referencing function to apply to each chunk
        def apply_reference(chunk: np.ndarray) -> np.ndarray:
            # If data is 1D, expand to 2D
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)

            # Convert data to float32 for computation if needed
            if chunk.dtype.kind in ["u", "i"]:
                chunk = chunk.astype(np.float32)

            # Ensure data is contiguous for better performance
            if not chunk.flags.c_contiguous:
                chunk = np.ascontiguousarray(chunk)

            # Apply the appropriate reference method
            if self.groups is None:
                # No groups - apply reference to all channels
                if self.reference == "global":
                    # Global reference
                    if self.reference_channels is None:
                        # Use all channels
                        shift = compute_func(chunk)
                        return apply_func(chunk, shift)
                    else:
                        # Use specified channels
                        return apply_channel_reference(
                            chunk, self.reference_channels, self.operator
                        )
                else:  # single reference
                    # Single channel reference
                    return apply_channel_reference(
                        chunk, self.reference_channels, "single"
                    )
            else:
                # Apply reference group-wise
                return apply_group_reference(chunk, self.groups, self.operator)

        # Use map_blocks to maintain laziness
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
                "rust_acceleration": True,
                "parallel": self.use_parallel,
            }
        )
        return base_summary


def create_car_processor_rs(
    reference_channels: Optional[List[int]] = None,
    groups: Optional[List[List[int]]] = None,
    use_parallel: bool = True,
) -> BaseProcessor:
    """
    Create a Rust-accelerated Common Average Reference (CAR) processor.

    Parameters
    ----------
    reference_channels : list of int, optional
        Specific channels to use as reference
    groups : list of lists of int, optional
        Channel groups for group-wise referencing
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    processor : BaseProcessor
        Configured common reference processor
    """
    if not _HAS_RUST:
        from ..spatial.common_reference import create_car_processor

        print(
            "Falling back to Python implementation as Rust extension is not available"
        )
        return create_car_processor(
            reference_channels=reference_channels,
            groups=groups,
            use_jit=True,
        )

    return CommonReferenceRustProcessor(
        reference="global",
        operator="average",
        reference_channels=reference_channels,
        groups=groups,
        use_parallel=use_parallel,
    )


def create_cmr_processor_rs(
    reference_channels: Optional[List[int]] = None,
    groups: Optional[List[List[int]]] = None,
    use_parallel: bool = True,
) -> BaseProcessor:
    """
    Create a Rust-accelerated Common Median Reference (CMR) processor.

    Parameters
    ----------
    reference_channels : list of int, optional
        Specific channels to use as reference
    groups : list of lists of int, optional
        Channel groups for group-wise referencing
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    processor : BaseProcessor
        Configured common reference processor
    """
    if not _HAS_RUST:
        from ..spatial.common_reference import create_cmr_processor

        print(
            "Falling back to Python implementation as Rust extension is not available"
        )
        return create_cmr_processor(
            reference_channels=reference_channels,
            groups=groups,
            use_jit=True,
        )

    return CommonReferenceRustProcessor(
        reference="global",
        operator="median",
        reference_channels=reference_channels,
        groups=groups,
        use_parallel=use_parallel,
    )


# Direct functions for immediate use
def apply_car_rs(
    data: np.ndarray,
    reference_channels: Optional[List[int]] = None,
    use_parallel: bool = True,
) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) directly using Rust implementation.

    Parameters
    ----------
    data : np.ndarray
        Input data (samples × channels)
    reference_channels : list of int, optional
        Specific channels to use as reference
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    np.ndarray
        Re-referenced data
    """
    if not _HAS_RUST:
        # Fallback to Numba implementation
        from ..spatial.common_reference import (
            _apply_global_reference,
            _compute_channel_mean,
        )

        print(
            "Falling back to Python implementation as Rust extension is not available"
        )
        shift = _compute_channel_mean(data)
        return _apply_global_reference(data, shift)

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        reshape_back = True
    else:
        reshape_back = False

    # Ensure float32 type
    data = data.astype(np.float32)

    # Select functions based on parallel setting
    compute_func = (
        compute_channel_mean_parallel if use_parallel else compute_channel_mean
    )
    apply_func = (
        apply_global_reference_parallel if use_parallel else apply_global_reference
    )

    if reference_channels is None:
        # Apply to all channels
        shift = compute_func(data)
        result = apply_func(data, shift)
    else:
        # Apply to specific channels
        result = apply_channel_reference(data, reference_channels, "mean")

    # Reshape back if needed
    if reshape_back:
        result = result.ravel()

    return result


def apply_cmr_rs(
    data: np.ndarray,
    reference_channels: Optional[List[int]] = None,
    use_parallel: bool = True,
) -> np.ndarray:
    """
    Apply Common Median Reference (CMR) directly using Rust implementation.

    Parameters
    ----------
    data : np.ndarray
        Input data (samples × channels)
    reference_channels : list of int, optional
        Specific channels to use as reference
    use_parallel : bool, default: True
        Whether to use parallel processing

    Returns
    -------
    np.ndarray
        Re-referenced data
    """
    if not _HAS_RUST:
        # Fallback to Numba implementation
        from ..spatial.common_reference import (
            _apply_global_reference,
            _compute_channel_median,
        )

        print(
            "Falling back to Python implementation as Rust extension is not available"
        )
        shift = _compute_channel_median(data)
        return _apply_global_reference(data, shift)

    # Ensure data is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        reshape_back = True
    else:
        reshape_back = False

    # Ensure float32 type
    data = data.astype(np.float32)

    # Select functions based on parallel setting
    compute_func = (
        compute_channel_median_parallel if use_parallel else compute_channel_median
    )
    apply_func = (
        apply_global_reference_parallel if use_parallel else apply_global_reference
    )

    if reference_channels is None:
        # Apply to all channels
        shift = compute_func(data)
        result = apply_func(data, shift)
    else:
        # Apply to specific channels
        result = apply_channel_reference(data, reference_channels, "median")

    # Reshape back if needed
    if reshape_back:
        result = result.ravel()

    return result
