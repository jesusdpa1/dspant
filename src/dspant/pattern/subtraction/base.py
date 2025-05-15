"""
Base classes and interfaces for template subtraction algorithms.

This module provides base classes and shared functionality for different 
template subtraction strategies in electrophysiological data, following
a consistent design pattern that aligns with the filter module architecture.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor


@public_api
class BaseSubtractor(BaseProcessor):
    """
    Base class for all template subtraction processors.
    
    Template subtraction algorithms remove specific patterns (like ECG artifacts)
    from electrophysiological signals. This base class establishes a common interface
    and shared functionality for all subtraction implementations.
    """

    def __init__(self):
        """
        Initialize the base subtractor with no required parameters.
        """
        super().__init__()
        self._overlap_samples = 0
        self._subtraction_stats = {}
    
    @abstractmethod
    def process(
        self, 
        data: da.Array,
        template: Union[np.ndarray, da.Array],
        fs: Optional[float] = None,
        mode: Optional[Literal["global", None]] = None,
        **kwargs
    ) -> da.Array:
        """
        Process input data by subtracting templates.
        
        This abstract method must be implemented by subclasses to define
        their specific template subtraction algorithm.
        
        Args:
            data: Input data array (samples × channels)
            template: Template array (samples × channels)
            fs: Sampling frequency in Hz (can be None)
            mode: Subtraction mode:
                  - "global": Apply global template subtraction across all channels
                  - None: Apply per-channel template subtraction (default)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Data with templates subtracted
        """
        pass
    
    def reset_stats(self) -> None:
        """Reset the subtraction statistics."""
        self._subtraction_stats = {}
    
    @property
    def subtraction_stats(self) -> Dict[str, Any]:
        """Get statistics from the last subtraction operation."""
        return self._subtraction_stats.copy()  # Return a copy for safety

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of subtractor configuration."""
        base_summary = super().summary
        base_summary.update({
            "overlap_samples": self._overlap_samples,
        })
        return base_summary