from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import dask.array as da
import numpy as np
import pendulum  # Replace datetime import
from pendulum.datetime import DateTime  # This is the correct import
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .data import StreamNode


class BaseProcessor(ABC):
    """Abstract base class for all processors"""

    @abstractmethod
    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """Process the input data"""
        pass

    @property
    @abstractmethod
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        pass

    @property
    def summary(self) -> Dict[str, Any]:
        """
        Return a dictionary containing processor configuration details
        Override this in derived classes to provide specific details
        """
        return {"type": self.__class__.__name__, "overlap": self.overlap_samples}


class ProcessingFunction(Protocol):
    """Protocol defining the interface for processing functions"""

    def __call__(
        self, data: np.ndarray, fs: Optional[float] = None, **kwargs
    ) -> np.ndarray: ...


class ProcessingPipeline:
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


class ProcessingNode:
    """Class for applying processing to a StreamNode"""

    def __init__(self, stream_node: StreamNode, name: Optional[str] = None):
        self.stream_node = stream_node
        self.pipeline = ProcessingPipeline()
        self.name = name or f"processing_node_{id(self)}"
        self._is_active = True
        self._last_modified = pendulum.now()
        self._processor_history: List[str] = []

    @property
    def is_active(self) -> bool:
        """Check if the processing node is active and available"""
        return self._is_active

    @property
    def last_modified(self) -> DateTime:
        """Get the last modification timestamp"""
        return self._last_modified

    def check_availability(self) -> Dict[str, Any]:
        """
        Check if the processing node is available for use or modification
        Returns a dictionary with status information
        """
        status = {
            "available": self.is_active,
            "name": self.name,
            "last_modified": self.last_modified,
            "processor_count": sum(len(p) for p in self.pipeline.processors.values()),
            "stream_node_path": str(self.stream_node.data_path),
        }

        if not self.is_active:
            status["reason"] = "Node has been deactivated"

        return status

    def add_processor(
        self,
        processor: Union[BaseProcessor, List[BaseProcessor]],
        group: str,
        position: Optional[int] = None,
    ) -> None:
        """
        Add a processor or list of processors to a specified group.
        If position is None, append to the end of the group.
        """
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        if group not in self.pipeline.processors:
            self.pipeline.processors[group] = []

        processors = processor if isinstance(processor, list) else [processor]

        for proc in processors:
            if position is not None:
                if position < 0 or position > len(self.pipeline.processors[group]):
                    raise ValueError(f"Invalid position: {position}")
                self.pipeline.processors[group].insert(position, proc)
                position += 1
            else:
                self.pipeline.processors[group].append(proc)

            self._processor_history.insert(
                0,
                f"Added {proc.__class__.__name__} to {group} at {pendulum.now().to_datetime_string()}",
            )

        self._last_modified = pendulum.now()

    def remove_processor(self, index: int) -> Optional[BaseProcessor]:
        """Remove a processor at the specified index"""
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        if 0 <= index < len(self.pipeline.processors):
            processor = self.pipeline.processors.pop(index)
            self._last_modified = pendulum.now()
            self._processor_history.insert(
                0,
                f"Removed {processor.__class__.__name__} at {self._last_modified.to_datetime_string()}",
            )
            return processor
        return None

    def clear_processors(self) -> None:
        """Remove all processors from the pipeline"""
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        self.pipeline.processors.clear()
        self._last_modified = pendulum.now()
        self._processor_history.insert(
            0, f"Cleared all processors at {self._last_modified.to_datetime_string()}"
        )

    def deactivate(self) -> None:
        """Deactivate the processing node"""
        self._is_active = False
        self._last_modified = pendulum.now()
        self._processor_history.insert(
            0, f"Node deactivated at {self._last_modified.to_datetime_string()}"
        )

    def reactivate(self) -> None:
        """Reactivate the processing node"""
        self._is_active = True
        self._last_modified = pendulum.now()
        self._processor_history.insert(
            0, f"Node reactivated at {self._last_modified.to_datetime_string()}"
        )

    def can_overwrite(self) -> Tuple[bool, str]:
        """
        Check if the processing node can be overwritten
        Returns a tuple of (can_overwrite, reason)
        """
        if not self.is_active:
            return False, f"Node '{self.name}' is not active"

        if len(self.pipeline.processors) > 0:
            return True, "Node has existing processors that will be replaced"

        return True, "Node is empty and ready for use"

    def process(
        self,
        group: Optional[List[str]] = None,
        node: Optional[List[int]] = None,
        return_info: bool = False,
        persist_intermediates: bool = False,  # New parameter
        optimize_chunks: bool = True,  # New parameter
        num_workers: Optional[int] = None,  # New parameter
        **kwargs,
    ) -> Union[da.Array, Tuple[da.Array, Dict[str, Any]]]:
        """
        Process the stream data through specified processor groups and/or nodes with optimized Dask execution.

        Args:
            group: List of processor groups to apply sequentially.
                Example: ["filters", "le"] applies "filters" first, then "le".
            node: List of processor nodes to apply in order.
                Example: [0,1] applies only the first and second processor steps.
            return_info: If True, returns processing information along with data.
            persist_intermediates: Whether to persist intermediate results to speed up computation.
            optimize_chunks: Whether to optimize chunk sizes for better performance.
            num_workers: Override the number of Dask workers for this computation.
            **kwargs: Additional keyword arguments passed to processors.

        Returns:
            Processed data array, optionally with processing information.
        """
        if not self.is_active:
            raise RuntimeError(f"Processing node '{self.name}' is not active")

        if not hasattr(self.stream_node, "fs") or not self.stream_node.fs:
            raise ValueError("Sampling rate (fs) must be defined in stream node")

        if not self.pipeline.processors:
            raise ValueError("No processors configured in the pipeline")

        # Use existing data if already loaded, otherwise load it
        if hasattr(self.stream_node, "data") and self.stream_node.data is not None:
            data = self.stream_node.data
        else:
            data = self.stream_node.load_data()

        # Configure dask compute parameters
        compute_kwargs = {}
        if num_workers is not None:
            compute_kwargs["num_workers"] = num_workers

        # Optimize chunk size if requested
        if optimize_chunks and hasattr(data, "chunks"):
            # Find maximum overlap required by any processor
            max_overlap = 0
            all_processors = []

            if group:
                for grp in group:
                    if grp in self.pipeline.processors:
                        all_processors.extend(self.pipeline.processors[grp])
            elif node is not None:
                for i, processors in enumerate(self.pipeline.processors.values()):
                    if i in node:
                        all_processors.extend(processors)
            else:
                for processors in self.pipeline.processors.values():
                    all_processors.extend(processors)

            for proc in all_processors:
                if hasattr(proc, "overlap_samples"):
                    max_overlap = max(max_overlap, proc.overlap_samples)

            # If significant overlap is needed, optimize chunk size
            if max_overlap > 0:
                current_chunks = data.chunks[0]
                avg_chunk_size = sum(current_chunks) / len(current_chunks)

                # If chunks are too small or too numerous, rechunk
                if avg_chunk_size < max_overlap * 4 or len(current_chunks) > 100:
                    optimal_size = max(max_overlap * 4, 10000)  # Rule of thumb
                    data = data.rechunk({0: optimal_size, 1: -1})

                    # Persist the rechunked data if requested
                    if persist_intermediates:
                        data = data.persist(**compute_kwargs)

        # Track which processors are applied
        applied_processors = []

        # Processing by group
        if group:
            # For each group in sequence
            for grp_idx, grp in enumerate(group):
                if grp not in self.pipeline.processors:
                    raise ValueError(f"Processor group '{grp}' not found")

                # Get processors for this group
                processors = self.pipeline.processors[grp]

                # Apply processors in this group
                for proc_idx, proc in enumerate(processors):
                    # Apply the processor
                    data = proc.process(data, fs=self.stream_node.fs, **kwargs)
                    applied_processors.append(proc.__class__.__name__)

                    # Persist intermediate results if requested and not the final processor
                    is_final = (grp_idx == len(group) - 1) and (
                        proc_idx == len(processors) - 1
                    )
                    if persist_intermediates and not is_final:
                        # Find optimal task fusion with dask optimization
                        data = data.persist(**compute_kwargs)

        # Processing by node index
        elif node is not None:
            ordered_processors = []
            # Get processors from selected nodes
            for i, processors in enumerate(self.pipeline.processors.values()):
                if i in node:
                    ordered_processors.extend(processors)

            # Apply each processor
            for proc_idx, proc in enumerate(ordered_processors):
                data = proc.process(data, fs=self.stream_node.fs, **kwargs)
                applied_processors.append(proc.__class__.__name__)

                # Persist intermediate results if requested and not the final processor
                if persist_intermediates and proc_idx < len(ordered_processors) - 1:
                    data = data.persist(**compute_kwargs)

        # Default: process all groups in sequence
        else:
            group_count = len(self.pipeline.processors)
            for grp_idx, (_, processors) in enumerate(self.pipeline.processors.items()):
                # Apply each processor in this group
                for proc_idx, proc in enumerate(processors):
                    data = proc.process(data, fs=self.stream_node.fs, **kwargs)
                    applied_processors.append(proc.__class__.__name__)

                    # Persist intermediate results if requested and not the final processor
                    is_final = (grp_idx == group_count - 1) and (
                        proc_idx == len(processors) - 1
                    )
                    if persist_intermediates and not is_final:
                        data = data.persist(**compute_kwargs)

        # Return results
        if return_info:
            info = {
                "applied_processors": applied_processors,
                "history": self._processor_history[:5],
                "optimized_chunks": data.chunks if hasattr(data, "chunks") else None,
            }
            return data, info

        return data

    def get_history(self) -> List[str]:
        """Get the processing history in chronological order"""
        return self._processor_history.copy()

    def summarize(self, show_history: bool = True, max_history: int = 5):
        try:
            console = Console()

            table = Table(
                title=f"Processing Node Summary: {self.name}",
                show_header=True,
                header_style="bold magenta",
            )

            table.add_column("Component", justify="right", style="cyan", no_wrap=True)
            table.add_column("Details", justify="left", style="green")

            # Node status
            table.add_section()
            status_style = "green" if self.is_active else "red"
            table.add_row(
                "Status",
                Text("Active" if self.is_active else "Inactive", style=status_style),
            )
            table.add_row("Name", self.name)
            table.add_row("Last Modified", self._last_modified.to_datetime_string())

            # Stream Node information
            table.add_section()
            table.add_row("Stream Node Path", str(self.stream_node.data_path))
            table.add_row(
                "Sampling Rate",
                f"{self.stream_node.fs} Hz"
                if self.stream_node.fs
                else "[red]Not set[/red]",
            )

            if hasattr(self.stream_node, "data_shape") and self.stream_node.data_shape:
                table.add_row("Data Shape", str(self.stream_node.data_shape))

            # Processing Pipeline Information
            if self.pipeline.processors:
                table.add_section()
                total_processors = sum(
                    len(group) for group in self.pipeline.processors.values()
                )
                table.add_row(
                    "Pipeline Status",
                    f"Active with {total_processors} processor(s)",
                )

                # Create a nested table for processors
                processor_table = Table(show_header=True)
                processor_table.add_column("Group", justify="center", style="cyan")
                processor_table.add_column("ID", justify="center", style="cyan")
                processor_table.add_column("Type", style="green")
                processor_table.add_column("Configuration", style="yellow")
                processor_table.add_column("Overlap", justify="right")

                # Iterate through processor groups
                for group_name, group_processors in self.pipeline.processors.items():
                    for i, processor in enumerate(group_processors, 1):
                        # Ensure processor has a summary method
                        try:
                            proc_summary = (
                                processor.summary
                                if hasattr(processor, "summary")
                                else {}
                            )

                            # Extract configuration details
                            config_details = [
                                f"{key}: {value}"
                                for key, value in proc_summary.items()
                                if key not in ["type", "overlap"] and value is not None
                            ]

                            processor_table.add_row(
                                group_name,
                                str(i),
                                proc_summary.get("type", processor.__class__.__name__),
                                "\n".join(config_details)
                                if config_details
                                else "Default config",
                                f"{proc_summary.get('overlap', 0)} samples",
                            )
                        except Exception as e:
                            processor_table.add_row(
                                group_name,
                                str(i),
                                processor.__class__.__name__,
                                f"[red]Error generating summary: {str(e)}[/red]",
                                "N/A",
                            )

                table.add_row("Processors", processor_table)
            else:
                table.add_row(
                    "Pipeline Status", Text("No processors configured", style="red")
                )

            # Processing Chain (formerly Recent History)
            if show_history and self._processor_history:
                table.add_section()
                history_entries = self._processor_history[
                    :max_history
                ]  # Take from start
                formatted_history = "\n".join(
                    f"[dim]{i + 1}.[/dim] {entry}"
                    for i, entry in enumerate(history_entries)
                )
                table.add_row("Processing Chain", formatted_history)

            # System resources
            if hasattr(self.stream_node, "data"):
                table.add_section()
                table.add_row(
                    "Memory Usage",
                    f"Chunks: {self.stream_node.data.chunks}\n"
                    f"Type: {self.stream_node.data.dtype}",
                )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error generating summary: {str(e)}[/red]")
