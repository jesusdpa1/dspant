"""
Stream processing components for dspant engine.

This module provides specialized components for processing streaming data:
- StreamProcessingPipeline for managing processor sequences
- StreamProcessingNode for applying processors to stream nodes
"""

from .node import StreamProcessingNode
from .pipeline import StreamProcessingPipeline


# Optional: Add any stream-specific utility functions here
def create_standard_pipeline(processors=None, group_name="default"):
    """
    Create a standard processing pipeline with optional initial processors.

    Args:
        processors: Optional list of processors to add initially
        group_name: Group name for the initial processors

    Returns:
        Configured StreamProcessingPipeline instance
    """
    pipeline = StreamProcessingPipeline()

    if processors:
        pipeline.add_processor(processors, group=group_name)

    return pipeline


__all__ = [
    # Core classes
    "StreamProcessingPipeline",
    "StreamProcessingNode",
    # Helper functions
    "create_standard_pipeline",
]
