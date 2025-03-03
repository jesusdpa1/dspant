from typing import Any, Dict, List, Literal, Optional, Union

import dask.array as da
import numpy as np

from ..core.nodes.stream_processing import BaseProcessor


class CommonReferenceProcessor(BaseProcessor):
    """
    Common Reference Processor implementation for dspAnt framework

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

        # Set operator function
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
        Apply common referencing to the data lazily
        """

        # Define the referencing function to apply to each chunk
        def apply_reference(chunk: np.ndarray) -> np.ndarray:
            # If data is 1D, expand to 2D
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)

            # Convert data to float32 for computation if needed
            if chunk.dtype.kind in ["u", "i"]:
                chunk = chunk.astype(np.float32)

            # Apply the appropriate reference method
            if self.groups is None:
                # No groups - apply reference to all channels
                if self.reference == "global":
                    # Global reference
                    if self.reference_channels is None:
                        # Use all channels
                        shift = self.operator_func(chunk, axis=1, keepdims=True)
                    else:
                        # Use specified channels
                        shift = self.operator_func(
                            chunk[:, self.reference_channels], axis=1, keepdims=True
                        )
                    # Apply shift to all channels
                    return chunk - shift
                else:  # single reference
                    # Single channel reference
                    shift = chunk[:, self.reference_channels].mean(
                        axis=1, keepdims=True
                    )
                    return chunk - shift
            else:
                # Apply reference group-wise
                n_samples, n_channels = chunk.shape
                re_referenced = np.zeros_like(chunk, dtype=np.float32)

                # Apply reference to each group separately
                for group_idx, group_channels in enumerate(self.groups):
                    # Ensure group channels are within range
                    valid_channels = [ch for ch in group_channels if ch < n_channels]

                    if not valid_channels:
                        continue

                    if self.reference == "global":
                        # Compute shift from all channels in this group
                        shift = self.operator_func(
                            chunk[:, valid_channels], axis=1, keepdims=True
                        )
                        # Apply shift to all channels in the group
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
        # This avoids accessing data.chunks which might trigger computation
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
            }
        )
        return base_summary


def create_car_processor(
    reference_channels: Optional[List[int]] = None,
    groups: Optional[List[List[int]]] = None,
) -> CommonReferenceProcessor:
    """
    Create a Common Average Reference (CAR) processor
    """
    return CommonReferenceProcessor(
        reference="global",
        operator="average",
        reference_channels=reference_channels,
        groups=groups,
    )


def create_cmr_processor(
    reference_channels: Optional[List[int]] = None,
    groups: Optional[List[List[int]]] = None,
) -> CommonReferenceProcessor:
    """
    Create a Common Median Reference (CMR) processor
    """
    return CommonReferenceProcessor(
        reference="global",
        operator="median",
        reference_channels=reference_channels,
        groups=groups,
    )
