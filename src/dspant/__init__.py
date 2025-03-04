"""
dspant: Digital Signal Processing for Analysis of Neural Time-series

A Python package for processing and analyzing neural time-series data,
with a focus on efficient computation and scalability.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("dspant")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# Import the hello_from_bin function from the Rust extension
from dspant._core import hello_from_bin
from dspant.engine import BaseProcessor

# Import key components for easier access
from dspant.nodes import EpocNode, StreamNode


# Node creation helpers
def create_stream_node(data_path, **kwargs):
    """Create a StreamNode for time-series data"""
    from dspant.nodes import StreamNode

    return StreamNode(data_path, **kwargs)


def create_epoch_node(data_path, **kwargs):
    """Create an EpocNode for event-based data"""
    from dspant.nodes import EpocNode

    return EpocNode(data_path, **kwargs)


# Process creation helper
def create_processor_node(stream_node, name=None):
    """Create a processing node for applying processors to data"""
    from dspant.engine.stream import StreamProcessingNode

    return StreamProcessingNode(stream_node, name=name)


def main() -> None:
    """Main entry point for CLI"""
    print(hello_from_bin())


__all__ = [
    # Version info
    "__version__",
    # Main entry point
    "main",
    # Core classes
    "StreamNode",
    "EpocNode",
    "BaseProcessor",
    # Helper functions
    "create_stream_node",
    "create_epoch_node",
    "create_processor_node",
]
