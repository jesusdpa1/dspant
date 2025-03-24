# src/dspant/pipeline/stream/pipeline.py
from typing import Dict, List, Optional, Union

import dask.array as da

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor


@public_api
class StreamProcessingPipeline:
    """Class for managing a sequence of processors with group support"""

    def __init__(self):
        self.processors: Dict[str, List[BaseProcessor]] = {}

    def add_processor(
        self,
        processor: Union[BaseProcessor, List[BaseProcessor]],
        group: str = "default",
        position: Optional[int] = None,
    ) -> None:
        """
        Add a processor or list of processors to a specified group.

        Args:
            processor: Processor or list of processors to add
            group: Group name to add processors to (default is 'default')
            position: Optional position to insert processors.
                      If None, appends to the end of the group.
        """
        # Ensure the group exists
        if group not in self.processors:
            self.processors[group] = []

        # Convert single processor to list for consistent handling
        processors = processor if isinstance(processor, list) else [processor]

        # Insert or append processors
        if position is not None:
            if position < 0 or position > len(self.processors[group]):
                raise ValueError(f"Invalid position: {position}")

            for proc in reversed(processors):
                self.processors[group].insert(position, proc)
        else:
            self.processors[group].extend(processors)

    def process(
        self,
        data: da.Array,
        processors: Optional[List[BaseProcessor]] = None,
        fs: Optional[float] = None,
        start: Optional[Union[int, float]] = None,  # New parameter
        end: Optional[Union[int, float]] = None,  # New parameter
        unit: str = "seconds",  # New parameter
        **kwargs,
    ) -> da.Array:
        """
        Apply specified processors in sequence

        Args:
            data: Input data array
            processors: List of specific processors to apply.
                        If None, applies all processors in all groups
            fs: Sampling frequency
            start: Start position for processing. If None, starts from the beginning.
                Can be time (in seconds) or sample index based on the unit parameter.
            end: End position for processing. If None, processes to the end.
                Can be time (in seconds) or sample index based on the unit parameter.
            unit: Unit for start/end values. Either "seconds" or "samples".
            **kwargs: Additional keyword arguments passed to processors

        Returns:
            Processed data array
        """
        # Validate unit parameter
        if unit not in ["seconds", "samples"]:
            raise ValueError("Unit must be either 'seconds' or 'samples'")

        # Calculate start and end indices based on unit
        start_idx = None
        end_idx = None
        data_length = data.shape[0]

        # Convert start/end to sample indices if needed
        if start is not None:
            if unit == "seconds":
                # Convert time to samples, require fs
                if fs is None:
                    raise ValueError(
                        "Sampling rate (fs) is required when using time units"
                    )
                start_idx = int(start * fs)
            else:  # unit == "samples"
                start_idx = int(start)

            # Validate start index
            if start_idx < 0:
                raise ValueError(f"Start index ({start_idx}) cannot be negative")
            if start_idx >= data_length:
                raise ValueError(
                    f"Start index ({start_idx}) exceeds data length ({data_length})"
                )

        if end is not None:
            if unit == "seconds":
                # Convert time to samples, require fs
                if fs is None:
                    raise ValueError(
                        "Sampling rate (fs) is required when using time units"
                    )
                end_idx = int(end * fs)
            else:  # unit == "samples"
                end_idx = int(end)

            # Validate end index
            if end_idx <= 0:
                raise ValueError(f"End index ({end_idx}) must be positive")
            if end_idx > data_length:
                end_idx = data_length  # Clamp to data length

        # Apply segment selection
        if start_idx is not None or end_idx is not None:
            # Default values if not specified
            start_idx = start_idx if start_idx is not None else 0
            end_idx = end_idx if end_idx is not None else data_length

            # Ensure proper ordering
            if start_idx >= end_idx:
                raise ValueError(
                    f"Start index ({start_idx}) must be less than end index ({end_idx})"
                )

            # Apply slicing to data
            data = data[start_idx:end_idx]

        # If no specific processors provided, flatten all processors from all groups
        if processors is None:
            processors = []
            for group_processors in self.processors.values():
                processors.extend(group_processors)

        # Apply processors in sequence
        result = data
        for processor in processors:
            result = processor.process(result, fs=fs, **kwargs)
        return result

    def get_group_processors(self, group: str) -> List[BaseProcessor]:
        """
        Retrieve processors from a specific group

        Args:
            group: Name of the processor group

        Returns:
            List of processors in the specified group
        """
        return self.processors.get(group, [])

    def clear_group(self, group: str) -> None:
        """
        Clear all processors from a specific group

        Args:
            group: Name of the processor group to clear
        """
        if group in self.processors:
            self.processors[group].clear()

    def remove_processor(self, group: str, index: int) -> Optional[BaseProcessor]:
        """
        Remove a processor from a specific group at the given index

        Args:
            group: Name of the processor group
            index: Index of the processor to remove

        Returns:
            Removed processor or None if removal fails
        """
        if group in self.processors and 0 <= index < len(self.processors[group]):
            return self.processors[group].pop(index)
        return None
