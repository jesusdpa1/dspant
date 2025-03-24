"""
dspant: Digital Signal Processing for Analysis of Neural Time-series

A Python package for processing and analyzing neural time-series data,
with a focus on efficient computation and scalability.
"""

"""
DSPANT: Digital Signal Processing for Analysis Toolkit
"""

__version__ = "0.1.0"


import importlib.metadata

from dspant.engine import BaseProcessor
from dspant.nodes import EpocNode, StreamNode

try:
    __version__ = importlib.metadata.version("dspant")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


# Node creation helpers
def create_stream_node(data_path, **kwargs):
    """Create a StreamNode for time-series data"""
    from dspant.nodes import StreamNode

    return StreamNode(data_path, **kwargs)


def create_epoch_node(data_path, **kwargs):
    """Create an EpocNode for event-based data"""
    from dspant.nodes import EpocNode

    return EpocNode(data_path, **kwargs)


# Optionally try to import extension modules
try:
    import dspant_emgproc

    __emgproc_available__ = True
except ImportError:
    __emgproc_available__ = False

try:
    import dspant_neuralproc

    __neuralproc_available__ = False
except ImportError:
    __neuralproc_available__ = False

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
