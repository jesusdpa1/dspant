# Processing Pipeline

The processing pipeline is a central concept in dspant that provides a flexible framework for building and executing complex signal processing workflows. It allows you to chain together multiple processing steps, manage their execution, and optimize performance for large datasets.

## Architecture

The processing pipeline architecture consists of three main components:

1. **BaseProcessor**: Abstract interface that all processors must implement
2. **StreamProcessingPipeline**: Manages sequences of processors and their execution
3. **StreamProcessingNode**: Connects a data node to a processing pipeline

This design provides a clean separation between:
- Data access (handled by nodes)
- Processing algorithm implementation (handled by processors)
- Processing workflow management (handled by pipelines)

## BaseProcessor

The `BaseProcessor` abstract class defines the interface for all signal processing components:

```python
class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """Process the input data"""
        pass

    @property
    @abstractmethod
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        pass
```

All processors must implement:
- A `process` method that takes a Dask array and returns a processed Dask array
- An `overlap_samples` property that defines how many samples are needed at chunk boundaries

## StreamProcessingPipeline

The `StreamProcessingPipeline` class manages a sequence of processor instances:

```python
pipeline = StreamProcessingPipeline()

# Add processors to different groups
pipeline.add_processor(filter_processor, group="filters")
pipeline.add_processor(envelope_processor, group="features")
pipeline.add_processor(threshold_processor, group="detection")

# Process data with all processors in sequence
result = pipeline.process(data, fs=sampling_rate)

# Or process with specific groups
result = pipeline.process(
    data, 
    processors=pipeline.get_group_processors("filters"),
    fs=sampling_rate
)
```

Key features include:
- Organizing processors into named groups
- Adding, removing, or replacing processors dynamically
- Processing data with specific processor groups or sequences

## StreamProcessingNode

The `StreamProcessingNode` connects a data node to a processing pipeline:

```python
from dspant.nodes import StreamNode
from dspant.engine import StreamProcessingNode, create_pipeline

# Create a data node
stream_node = StreamNode(data_path="path/to/recording.ant").load_data()

# Create a processing node connected to the data node
proc_node = StreamProcessingNode(stream_node, name="my_processing")

# Add processors to the pipeline
proc_node.add_processor(filter_processor, group="filters")
proc_node.add_processor([detector1, detector2], group="detection")

# Process the data
result = proc_node.process(group=["filters", "detection"])
```

Key features include:
- Direct connection to a data node
- Managing processor execution and data flow
- Tracking processing history and state
- Handling memory optimization and chunk management

## Performance Optimization

The processing pipeline includes several features for optimizing performance with large datasets:

### Chunk Optimization

```python
# Process with chunk optimization
result = proc_node.process(
    optimize_chunks=True,
    persist_intermediates=True,
    num_workers=4
)
```

Chunk optimization features include:
- Automatic adjustment of chunk sizes based on processor requirements
- Persisting intermediate results for faster recomputation
- Controlling the number of parallel workers

### Processor Overlap

The pipeline handles the complexity of ensuring that chunk boundaries have sufficient context for accurate processing, using the `overlap_samples` property from each processor.

### Stream Processing

Rather than loading all data into memory, the pipeline processes data in streaming chunks, enabling the analysis of very large datasets that wouldn't fit in memory.

## Creating Custom Processors

You can create custom processors by implementing the `BaseProcessor` interface:

```python
from dspant.engine import BaseProcessor
import dask.array as da
import numpy as np

class CustomFilter(BaseProcessor):
    def __init__(self, cutoff_hz):
        self.cutoff_hz = cutoff_hz
        self._overlap_samples = 100  # Samples needed at chunk boundaries
        
    def process(self, data: da.Array, fs: float = None, **kwargs) -> da.Array:
        """Apply custom filtering to the input data"""
        # Implement your processing logic here
        # This is typically a map of a numpy function over dask chunks
        return data.map_blocks(
            self._filter_chunk, 
            fs=fs, 
            cutoff=self.cutoff_hz,
            dtype=data.dtype
        )
    
    def _filter_chunk(self, chunk, fs, cutoff):
        # Process a single chunk using numpy
        # Add your NumPy-based filtering code here
        return filtered_chunk
    
    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples
        
    @property
    def summary(self) -> Dict[str, Any]:
        """Return processor configuration details"""
        return {
            "type": "CustomFilter",
            "cutoff_hz": self.cutoff_hz,
            "overlap": self._overlap_samples
        }
```

## Factory Functions

For common processor combinations, dspant provides factory functions:

```python
from dspant.engine import create_pipeline
from dspant.processor.filters import create_bandpass_filter
from dspant.processor.feature import create_envelope_extractor

# Create common processors
pipeline = create_pipeline()
bandpass = create_bandpass_filter(low_hz=300, high_hz=3000)
envelope = create_envelope_extractor(smoothing_ms=5)

# Add to pipeline
pipeline.add_processor([bandpass, envelope], group="preprocessing")
```

These factory functions provide a simpler interface for creating commonly used processor configurations.