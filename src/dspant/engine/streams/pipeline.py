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
        **kwargs,
    ) -> da.Array:
        """
        Apply specified processors in sequence

        Args:
            data: Input data array
            processors: List of specific processors to apply.
                        If None, applies all processors in all groups
            fs: Sampling frequency
            **kwargs: Additional keyword arguments passed to processors

        Returns:
            Processed data array
        """
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
