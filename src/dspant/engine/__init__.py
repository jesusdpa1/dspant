"""
Engine module for dspant.

This module provides the core processing engine components:
- Base processor interfaces
- Pipeline management
- Processing nodes for different data types
"""

from .base import BaseProcessor, ProcessingFunction
from .streams import StreamProcessingNode, StreamProcessingPipeline

# Convenience functions for creating processing components


def create_pipeline():
    """
    Create a new stream processing pipeline.

    Returns:
        A new StreamProcessingPipeline instance
    """
    return StreamProcessingPipeline()


def create_processing_node(stream_node, name=None):
    """
    Create a processing node for a stream node.

    Args:
        stream_node: Stream node to process
        name: Optional name for the processing node

    Returns:
        A new StreamProcessingNode instance
    """
    return StreamProcessingNode(stream_node, name=name)


__all__ = [
    # Base abstractions
    "BaseProcessor",
    "ProcessingFunction",
    # Stream processing
    "StreamProcessingNode",
    "StreamProcessingPipeline",
    # Factory functions
    "create_pipeline",
    "create_processing_node",
]
