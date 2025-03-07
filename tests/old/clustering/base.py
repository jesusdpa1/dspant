"""
Base classes for neural signal clustering.

This module provides the abstract base classes and common
utilities for clustering neural spike waveforms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl

from ...engine.base import BaseProcessor


class BaseClusteringProcessor(BaseProcessor):
    """
    Abstract base class for all clustering processors.

    This class defines the interface that all clustering implementations
    should follow, providing methods for clustering, visualization, and
    result analysis.
    """

    def __init__(self):
        """Initialize the base clustering processor."""
        self._clustering_stats = {}
        self._overlap_samples = 0  # Subclasses should set appropriate overlap
        self._cluster_labels = None
        self._is_fitted = False

    @abstractmethod
    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process input data to perform clustering.

        Args:
            data: Input dask array (typically waveforms)
            fs: Sampling frequency (usually not used in clustering, but kept for BaseProcessor interface)
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Dask array with cluster assignments
        """
        pass

    @abstractmethod
    def predict(self, data: Union[np.ndarray, da.Array]) -> np.ndarray:
        """
        Predict cluster assignments for new data points.

        Args:
            data: Input data points (typically waveforms)

        Returns:
            Array of cluster assignments
        """
        pass

    def to_dataframe(
        self,
        labels: Union[np.ndarray, da.Array],
        indices: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> pl.DataFrame:
        """
        Convert cluster labels to a Polars DataFrame.

        Args:
            labels: Array of cluster labels
            indices: Optional array of indices (e.g., spike indices) to include in the DataFrame

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

    @property
    def is_fitted(self) -> bool:
        """Check if the clustering model has been fitted."""
        return self._is_fitted

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    @property
    def cluster_labels(self) -> Optional[np.ndarray]:
        """Get the cluster labels from the last clustering run."""
        return self._cluster_labels

    @property
    def clustering_stats(self) -> Dict[str, Any]:
        """Get statistics from the last clustering run."""
        return self._clustering_stats

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of clustering processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "is_fitted": self.is_fitted,
                "cluster_counts": self.get_cluster_counts() if self.is_fitted else None,
            }
        )
        # Subclasses should extend this with processor-specific information
        return base_summary


class FeatureExtractionMixin:
    """
    Mixin class that provides feature extraction utilities for clustering processors.

    This mixin adds methods for extracting various features from waveforms
    that can be used for clustering.
    """

    @staticmethod
    def extract_time_domain_features(
        waveforms: np.ndarray, n_features: int = 3
    ) -> np.ndarray:
        """
        Extract simple time-domain features from waveforms.

        Args:
            waveforms: Input waveforms array (n_waveforms × n_samples × n_channels)
            n_features: Number of features to extract

        Returns:
            Feature matrix (n_waveforms × n_features)
        """
        # Get waveform dimensions
        n_waveforms, n_samples, n_channels = waveforms.shape

        # Initialize feature matrix
        features = np.zeros((n_waveforms, n_features * n_channels), dtype=np.float32)

        # Extract features
        for i in range(n_waveforms):
            for c in range(n_channels):
                waveform = waveforms[i, :, c]

                # Basic features (adjust based on specific needs)
                # Feature 1: Peak amplitude
                peak_idx = np.argmax(np.abs(waveform))
                features[i, c * n_features] = waveform[peak_idx]

                # Feature 2: Peak time
                features[i, c * n_features + 1] = peak_idx / n_samples

                # Feature 3: Energy
                features[i, c * n_features + 2] = np.sum(waveform**2)

        return features

    @staticmethod
    def normalize_waveforms(waveforms: np.ndarray) -> np.ndarray:
        """
        Normalize waveforms to improve clustering.

        Args:
            waveforms: Input waveforms array (n_waveforms × n_samples × n_channels)

        Returns:
            Normalized waveforms
        """
        # Get dimensions
        n_waveforms, n_samples, n_channels = waveforms.shape

        # Initialize normalized array
        normalized = np.zeros_like(waveforms, dtype=np.float32)

        # Normalize each waveform
        for i in range(n_waveforms):
            for c in range(n_channels):
                wf = waveforms[i, :, c]

                # Normalize to [0, 1] range
                wf_min = np.min(wf)
                wf_max = np.max(wf)

                if wf_max - wf_min > 1e-10:
                    normalized[i, :, c] = (wf - wf_min) / (wf_max - wf_min)
                else:
                    # If waveform is flat, just center it
                    normalized[i, :, c] = wf - np.mean(wf)

        return normalized


# Example of a concrete clustering class (simplified for illustration)
class SimpleKMeansClustering(BaseClusteringProcessor, FeatureExtractionMixin):
    """
    Simple KMeans clustering implementation for demonstration.

    This class provides a minimal example of how to implement the
    BaseClusteringProcessor abstract class.
    """

    def __init__(self, n_clusters: int = 3, normalize: bool = True):
        """
        Initialize the simple KMeans clustering processor.

        Args:
            n_clusters: Number of clusters for KMeans
            normalize: Whether to normalize waveforms before clustering
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.normalize = normalize
        self._kmeans_model = None

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process waveforms using KMeans clustering.

        Args:
            data: Input dask array of waveforms
            fs: Sampling frequency (not used here)
            **kwargs: Additional parameters

        Returns:
            Dask array with cluster labels
        """
        # For this simplified example, we just compute and flatten the data
        waveforms = data.compute()

        # Normalize if requested
        if self.normalize:
            waveforms = self.normalize_waveforms(waveforms)

        # Flatten waveforms for clustering
        n_waveforms = waveforms.shape[0]
        waveforms_flat = waveforms.reshape(n_waveforms, -1)

        # Import KMeans here to avoid circular import
        from sklearn.cluster import KMeans

        # Create and fit KMeans model
        self._kmeans_model = KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init="auto"
        )
        labels = self._kmeans_model.fit_predict(waveforms_flat)

        # Store results
        self._cluster_labels = labels
        self._is_fitted = True

        # Return as dask array
        return da.from_array(labels)

    def predict(self, data: Union[np.ndarray, da.Array]) -> np.ndarray:
        """
        Predict cluster assignments for new waveforms.

        Args:
            data: Input waveforms array

        Returns:
            Array of cluster assignments
        """
        if not self._is_fitted or self._kmeans_model is None:
            raise ValueError("Model not fitted. Call process() first.")

        # Convert to numpy if needed
        if isinstance(data, da.Array):
            data = data.compute()

        # Normalize if needed
        if self.normalize:
            data = self.normalize_waveforms(data)

        # Flatten waveforms
        n_waveforms = data.shape[0]
        data_flat = data.reshape(n_waveforms, -1)

        # Predict labels
        return self._kmeans_model.predict(data_flat)
