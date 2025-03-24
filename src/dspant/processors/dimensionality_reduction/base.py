# src/dspant/processor/dimensionality_reduction/base.py
"""
Base classes for dimensionality reduction in signal processing.

This module provides abstract base classes and utilities for
dimensionality reduction techniques that can be applied to both
neural and EMG data processing.
"""

from typing import Any, Dict, Optional, Union

import dask.array as da
import numpy as np

from ...engine.base import BaseProcessor


class BaseDimensionalityReductionProcessor(BaseProcessor):
    """
    Abstract base class for all dimensionality reduction processors.

    This class defines the common interface for dimensionality reduction
    techniques, ensuring consistency across implementations.
    """

    def __init__(self):
        """Initialize the base dimensionality reduction processor."""
        self._is_fitted = False
        self._overlap_samples = 0  # No overlap needed for dimensionality reduction
        self._components = None

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    @property
    def components(self) -> Optional[np.ndarray]:
        """Get the learned components if available."""
        return self._components

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    def fit_transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Fit the model and transform the data in one step.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Transformed data
        """
        self.fit(data, **kwargs)
        return self.transform(data, **kwargs)

    def fit(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> "BaseDimensionalityReductionProcessor":
        """
        Fit the model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data using the fitted model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Transformed data
        """
        raise NotImplementedError("Subclasses must implement transform()")

    def inverse_transform(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> np.ndarray:
        """
        Transform data back to its original space if possible.

        Args:
            data: Input data array in reduced dimensions
            **kwargs: Additional keyword arguments

        Returns:
            Data in original space
        """
        raise NotImplementedError(
            "Subclasses must implement inverse_transform() if supported"
        )
