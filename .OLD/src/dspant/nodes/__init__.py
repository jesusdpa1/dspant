"""
Node definitions for data access in dspant.

This module provides node classes for working with different types of data:
- StreamNode: For time-series data with multiple channels
- EpocNode: For event-based data
"""

from .base import BaseNode
from .epoch import BaseEpocNode, EpocNode
from .stream import BaseStreamNode, StreamNode

__all__ = [
    # Base classes
    "BaseNode",
    # Stream nodes
    "BaseStreamNode",
    "StreamNode",
    # Epoch nodes
    "BaseEpocNode",
    "EpocNode",
]
