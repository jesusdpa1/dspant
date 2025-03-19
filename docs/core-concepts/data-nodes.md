# Data Nodes

Data nodes are a core concept in dspant and serve as the primary interface for accessing and manipulating different types of neural data. They handle data loading, validation, metadata management, and provide a consistent API for working with various data types.

## Node Types

dspant provides several node types for different kinds of data:

- **BaseNode**: The foundation class that provides common functionality for all node types
- **StreamNode**: For handling time-series data (continuous recordings)
- **EpocNode**: For handling event-based data (epochs or timestamps)

## Base Node

All node types inherit from `BaseNode`, which provides:

- Path management for data and metadata files
- File validation and discovery
- Metadata loading and parsing

## Stream Node

The `StreamNode` class is designed for working with continuous time-series data like neural recordings. It provides:

- Efficient loading of large datasets using Dask arrays
- Automatic chunking for memory-efficient processing
- Sampling rate and channel information management
- Data shape and type validation

## Epoch Node

The `EpocNode` class is specialized for handling event-based data like spike timestamps, stimulus events, or behavioral markers. It provides:

- Loading event data into Polars DataFrames for efficient manipulation
- Event timing and relationship analysis
- Integration with time-series data for event-locked analyses

## Example Usage

### Working with Stream Nodes

```python
from dspant.nodes import StreamNode

# Create a stream node from a data path
stream_node = StreamNode(data_path="path/to/recording.ant")

# Load the data (lazy loading with Dask)
data = stream_node.load_data()

# Access node properties
print(f"Sampling rate: {stream_node.fs} Hz")
print(f"Duration: {stream_node.number_of_samples / stream_node.fs:.2f} seconds")
print(f"Channels: {stream_node.channel_numbers}")

# Print a summary of the node
stream_node.summarize()
```

### Working with Epoch Nodes

```python
from dspant.nodes import EpocNode

# Create an epoch node from a data path
epoc_node = EpocNode(data_path="path/to/events.ant")

# Load the event data
events = epoc_node.load_data()

# Basic operations with events
event_count = len(events)
first_event_time = events[0, "onset"]
event_intervals = events.with_columns(
    (pl.col("onset").shift(-1) - pl.col("onset")).alias("interval")
)

# Print a summary of the node
epoc_node.summarize()
```

## Under the Hood

Data nodes use a consistent file structure:

- Each node corresponds to a directory with an `.ant` extension
- Inside that directory are `data_*.parquet` files containing the actual data
- Accompanying `metadata_*.json` files contain metadata like sampling rate, channel info, etc.

This structure enables efficient data sharing, versioning, and access patterns for different analysis needs.

## Integration with Processing Pipeline

Data nodes are designed to integrate seamlessly with the processing pipeline components:

```python
from dspant.nodes import StreamNode
from dspant.engine import StreamProcessingNode

# Create a stream node and load data
stream_node = StreamNode(data_path="path/to/recording.ant").load_data()

# Create a processing node that references the stream node
proc_node = StreamProcessingNode(stream_node)

# Now the processing node can apply operations to the stream node's data
filtered_data = proc_node.process()
```

This separation of data nodes and processing nodes creates a clean architecture where:

1. Data nodes focus on data access, validation, and metadata management
2. Processing nodes focus on applying algorithms and transformations to the data