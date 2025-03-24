# src/dspant/processor/clustering/base.py
"""
Base classes for clustering algorithms.

This module provides abstract base classes and utilities for
clustering techniques that can be applied to both neural and EMG data.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from ...engine.base import BaseProcessor


class BaseClusteringProcessor(BaseProcessor):
    """
    Abstract base class for all clustering processors.

    This class defines the common interface for clustering techniques,
    ensuring consistency across implementations.
    """

    def __init__(self):
        """Initialize the base clustering processor."""
        self._is_fitted = False
        self._overlap_samples = 0  # No overlap needed for clustering
        self._cluster_labels = None
        self._cluster_centers = None

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    @property
    def cluster_labels(self) -> Optional[np.ndarray]:
        """Get the cluster labels from the last clustering run."""
        return self._cluster_labels

    @property
    def cluster_centers(self) -> Optional[np.ndarray]:
        """Get the cluster centers if available."""
        return self._cluster_centers

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    @abstractmethod
    def fit(
        self, data: Union[np.ndarray, da.Array], **kwargs
    ) -> "BaseClusteringProcessor":
        """
        Fit the clustering model to the data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict cluster assignments for new data points.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Array of cluster assignments
        """
        pass

    def fit_predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Fit the model and predict cluster assignments in one step.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Array of cluster assignments
        """
        self.fit(data, **kwargs)
        return self.predict(data, **kwargs)

    def to_dataframe(
        self,
        labels: Union[np.ndarray, da.Array],
        indices: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> pl.DataFrame:
        """
        Convert cluster labels to a Polars DataFrame.

        Args:
            labels: Array of cluster labels
            indices: Optional array of indices to include in the DataFrame

        Returns:
            Polars DataFrame with cluster assignments
        """
        # Convert dask array to numpy if needed
        if isinstance(labels, da.Array):
            labels = labels.compute()

        # Create DataFrame
        if indices is not None:
            if isinstance(indices, da.Array):
                indices = indices.compute()

            # Validate indices and labels have same length
            if len(indices) != len(labels):
                raise ValueError(
                    f"Indices length ({len(indices)}) does not match labels length ({len(labels)})"
                )

            df = pl.DataFrame({"index": indices, "cluster": labels})
        else:
            df = pl.DataFrame({"cluster": labels})

        return df

    def get_cluster_counts(
        self, labels: Optional[Union[np.ndarray, da.Array]] = None
    ) -> Dict[int, int]:
        """
        Get counts of data points per cluster.

        Args:
            labels: Array of cluster labels (if None, uses stored labels)

        Returns:
            Dictionary mapping cluster IDs to counts
        """
        # Use provided labels or stored labels
        if labels is None:
            if self._cluster_labels is None:
                return {}
            labels = self._cluster_labels

        # Convert dask array to numpy if needed
        if isinstance(labels, da.Array):
            labels = labels.compute()

        # Count occurrences of each label
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create dictionary mapping labels to counts
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform clustering.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used in clustering, but required by BaseProcessor interface)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dask array with cluster assignments
        """
        # Get processing parameters
        compute_now = kwargs.pop("compute_now", False)

        if compute_now:
            # Convert to numpy, fit, and convert back to dask
            data_np = data.compute()
            labels = self.fit_predict(data_np, **kwargs)
            return da.from_array(labels)
        else:
            # Define function to apply to chunks
            def apply_clustering(chunk, block_info=None):
                # If already fitted, just predict
                if self._is_fitted:
                    return self.predict(chunk, **kwargs)
                else:
                    # Fit on this chunk and predict
                    return self.fit_predict(chunk, **kwargs)

            # Apply to dask array
            result = data.map_blocks(
                apply_clustering,
                drop_axis=list(range(1, data.ndim)),  # Output has one dimension fewer
                dtype=np.int32,
            )

            return result
