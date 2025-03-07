"""
PCA-KMeans clustering processor combining dimensionality reduction and clustering.

This module provides a modular implementation that combines the PCA processor
from the dimensionality_reduction module with the KMeans processor from the
clustering module, optimized for neural spike waveform analysis.
"""

from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from sklearn.metrics import silhouette_score

from ...engine.base import BaseProcessor
from ...processor.clustering.kmeans import KMeansProcessor
from ...processor.dimensionality_reduction.pca import PCAProcessor


class ComposedPCAKMeansProcessor(BaseProcessor):
    """
    Composed PCA-KMeans processor that combines PCA dimensionality reduction
    with KMeans clustering for neural data analysis.

    This processor uses composition to combine a PCAProcessor and KMeansProcessor,
    allowing for better separation of concerns and more flexible configuration.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_components: int = 10,
        normalize: bool = True,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
        whiten: bool = False,
        use_numba: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """
        Initialize the composed PCA-KMeans processor.

        Args:
            n_clusters: Number of clusters for KMeans
            n_components: Number of PCA components to use
            normalize: Whether to normalize data before processing
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations for KMeans
            n_init: Number of initializations for KMeans
            whiten: Whether to apply whitening in PCA
            use_numba: Whether to use Numba acceleration for PCA
            chunk_size: Size of chunks for Dask processing
        """
        # Create the PCA processor
        self.pca_processor = PCAProcessor(
            n_components=n_components,
            normalize=normalize,
            random_state=random_state,
            whiten=whiten,
            use_numba=use_numba,
        )

        # Create the KMeans processor
        self.kmeans_processor = KMeansProcessor(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            chunk_size=chunk_size,
        )

        # Store configuration
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.normalize = normalize
        self.random_state = random_state
        self.chunk_size = chunk_size
        self._original_waveforms = None
        # Additional components for visualization
        self._silhouette_score = None

        # No overlap needed for this operation
        self._overlap_samples = 0

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Process the input data by applying PCA followed by KMeans clustering.

        Args:
            data: Input dask array
            fs: Sampling frequency (not used, but required by BaseProcessor interface)
            **kwargs: Additional keyword arguments
                compute_now: Whether to compute immediately (default: True)
                subsample: Maximum number of samples to use for fitting
                chunk_size: Size of chunks for processing

        Returns:
            Dask array with cluster labels
        """
        # Extract processing parameters
        compute_now = kwargs.get("compute_now", True)
        chunk_size = kwargs.get("chunk_size", self.chunk_size)

        def process_chunk(chunk: np.ndarray) -> np.ndarray:
            """Process a chunk of data with PCA and KMeans."""
            # Ensure input is float32
            chunk = chunk.astype(np.float32)

            # Apply PCA - use process instead of fit_transform for better Numba compatibility
            pca_result = (
                self.pca_processor.process(da.from_array(chunk), compute_now=True)
                .compute()
                .astype(np.float32)
            )

            # Ensure float32 type for KMeans
            pca_result = np.ascontiguousarray(pca_result, dtype=np.float32)

            # Use fit_predict directly instead of process
            cluster_labels = self.kmeans_processor.fit_predict(pca_result)

            # Compute silhouette score if we have enough samples and clusters
            if len(pca_result) > self.n_clusters and self.n_clusters > 1:
                try:
                    self._silhouette_score = silhouette_score(
                        pca_result, cluster_labels
                    )
                except Exception:
                    self._silhouette_score = None

            return cluster_labels.astype(np.int32)

        # Process based on compute_now parameter
        if compute_now:
            # Apply custom chunk size if provided
            if chunk_size is not None and isinstance(data, da.Array):
                data = data.rechunk({0: chunk_size})

            data_np = data.compute()
            labels = process_chunk(data_np)
            return da.from_array(labels)
        else:
            # Apply to dask array
            output_dtype = np.int32

            # Determine output shape by dropping dimensions
            output_chunks = tuple([data.chunks[0]])  # Keep first dimension chunks

            # Apply custom chunk size if provided
            if chunk_size is not None:
                data = data.rechunk({0: chunk_size})

            result = data.map_blocks(
                process_chunk,
                drop_axis=list(range(1, data.ndim)),
                dtype=output_dtype,
                chunks=output_chunks,
            )
            return result

    def transform(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Transform data using the fitted PCA model.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Transformed data in PCA space
        """
        # Simply delegate to PCA processor
        return self.pca_processor.transform(data, **kwargs)

    def predict(self, data: Union[np.ndarray, da.Array], **kwargs) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            data: Input data array
            **kwargs: Additional keyword arguments

        Returns:
            Array of cluster labels
        """
        # Convert to dask array if it's a numpy array
        if isinstance(data, np.ndarray):
            data = da.from_array(data)

        # Apply PCA transformation using process method
        pca_result = self.pca_processor.process(data, compute_now=True)

        # Apply KMeans prediction using process method
        return self.kmeans_processor.process(pca_result, compute_now=True).compute()

    def plot_clusters(
        self,
        max_points: int = 5000,
        figsize: Tuple[int, int] = (15, 10),
        alpha: float = 0.7,
        s: int = 15,
        cmap: str = "tab10",
        title: Optional[str] = None,
        fig: Optional[Figure] = None,
        include_silhouette: bool = True,
        plot_waveforms: bool = True,
        n_waveforms_per_cluster: int = 50,
        waveforms=None,
    ) -> Figure:
        """
        Visualize clustering results with PCA.

        Args:
            max_points: Maximum number of points to plot
            figsize: Figure size as (width, height)
            alpha: Alpha value for scatter points
            s: Size of scatter points
            cmap: Colormap to use
            title: Plot title (if None, a default title is generated)
            fig: Existing figure to use (if None, creates a new figure)
            include_silhouette: Whether to include silhouette score in title
            plot_waveforms: Whether to plot mean waveforms for each cluster
            n_waveforms_per_cluster: Number of waveforms to sample per cluster

        Returns:
            Matplotlib figure
        """
        # Check if models have been fitted
        if not self.pca_processor.is_fitted or not self.kmeans_processor.is_fitted:
            raise ValueError("Models not fitted. Call process() first.")

        # Transform original waveforms to PCA space
        pca_components = self.pca_processor.transform(waveforms)
        cluster_labels = self.kmeans_processor.cluster_labels
        explained_variance_ratio = self.pca_processor.get_explained_variance()

        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=figsize)

        # Determine number of subplots based on whether to plot waveforms
        if plot_waveforms:
            gs = GridSpec(2, 2, height_ratios=[2, 1], figure=fig)
            ax_scatter = fig.add_subplot(gs[0, 0])
            ax_variance = fig.add_subplot(gs[0, 1])
            ax_waveforms = fig.add_subplot(gs[1, :])
        else:
            gs = GridSpec(1, 2, figure=fig)
            ax_scatter = fig.add_subplot(gs[0, 0])
            ax_variance = fig.add_subplot(gs[0, 1])

        # Handle limiting the number of points
        if len(cluster_labels) > max_points:
            # Stratified sampling
            indices = []
            for label in np.unique(cluster_labels):
                label_indices = np.where(cluster_labels == label)[0]
                points_per_cluster = max(
                    1, max_points // len(np.unique(cluster_labels))
                )
                if len(label_indices) > points_per_cluster:
                    sampled = np.random.choice(
                        label_indices, points_per_cluster, replace=False
                    )
                    indices.extend(sampled)
                else:
                    indices.extend(label_indices)

            pca_data = pca_components[indices]
            labels = cluster_labels[indices]
        else:
            pca_data = pca_components
            labels = cluster_labels
            indices = np.arange(len(labels))

        # Get the colormap
        cmap_obj = plt.get_cmap(cmap)

        # Generate cluster colors
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        colors = [cmap_obj(i / max(1, n_clusters - 1)) for i in range(n_clusters)]

        # 2D PCA plot
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax_scatter.scatter(
                pca_data[mask, 0],
                pca_data[mask, 1],
                c=[colors[i]],
                label=f"Cluster {label}",
                alpha=alpha,
                s=s,
            )
        ax_scatter.set_xlabel("PC1")
        ax_scatter.set_ylabel("PC2")
        ax_scatter.set_title(f"PCA Visualization of {n_clusters} Clusters")
        ax_scatter.legend(loc="best")
        ax_scatter.grid(True, linestyle="--", alpha=0.7)

        # Plot explained variance
        if explained_variance_ratio is not None:
            variance_ratio = explained_variance_ratio
            cumulative_variance = np.cumsum(variance_ratio)
            components = np.arange(1, len(variance_ratio) + 1)

            ax_variance.bar(components, variance_ratio, color="skyblue", alpha=0.7)
            ax_variance.plot(
                components, cumulative_variance, "r-", marker="o", markersize=4
            )
            ax_variance.set_xlabel("Principal Component")
            ax_variance.set_ylabel("Explained Variance Ratio")
            ax_variance.set_title("PCA Explained Variance")
            ax_variance.grid(True, linestyle="--", alpha=0.7)

            # Add second y-axis for cumulative variance
            ax2 = ax_variance.twinx()
            ax2.set_ylabel("Cumulative Variance Ratio")
            ax2.set_ylim(0, 1.05)

            # Add 80% and 90% variance thresholds
            ax_variance.axhline(y=0.8, color="r", linestyle="--", alpha=0.3)
            ax_variance.axhline(y=0.9, color="g", linestyle="--", alpha=0.3)

        # Plot waveforms if requested and available
        if plot_waveforms:
            if waveforms is not None:
                # Sample waveforms for each cluster
                cluster_waveforms = []
                for label in unique_labels:
                    cluster_indices = np.where(cluster_labels == label)[0]
                    if len(cluster_indices) > n_waveforms_per_cluster:
                        sampled_indices = np.random.choice(
                            cluster_indices, n_waveforms_per_cluster, replace=False
                        )
                    else:
                        sampled_indices = cluster_indices

                    # Get mean waveform for this cluster
                    sampled_waveforms = waveforms[sampled_indices]
                    mean_waveform = np.mean(sampled_waveforms, axis=0)
                    std_waveform = np.std(sampled_waveforms, axis=0)

                    cluster_waveforms.append((mean_waveform, std_waveform, label))

                # Plot each cluster's mean waveform
                x = np.arange(cluster_waveforms[0][0].shape[0])
                for mean_wf, std_wf, label in cluster_waveforms:
                    color = cmap_obj(int(label) / max(1, n_clusters - 1))
                    ax_waveforms.plot(
                        x, mean_wf, color=color, label=f"Cluster {label}", linewidth=2
                    )
                    ax_waveforms.fill_between(
                        x, mean_wf - std_wf, mean_wf + std_wf, color=color, alpha=0.2
                    )

                ax_waveforms.set_xlabel("Sample")
                ax_waveforms.set_ylabel("Amplitude")
                ax_waveforms.set_title("Mean Waveforms by Cluster")
                ax_waveforms.legend(loc="best")
                ax_waveforms.grid(True, linestyle="--", alpha=0.7)
            else:
                ax_waveforms.set_xlabel("Sample")
                ax_waveforms.set_ylabel("Amplitude")
                ax_waveforms.set_title("Waveform plotting disabled")
                ax_waveforms.text(
                    0.5,
                    0.5,
                    "Waveform plotting disabled\n(no waveforms available)",
                    ha="center",
                    va="center",
                    transform=ax_waveforms.transAxes,
                )
                ax_waveforms.grid(True, linestyle="--", alpha=0.7)

        # Set figure title if provided
        if title is not None:
            fig_title = title
        else:
            fig_title = f"PCA-KMeans Clustering Results with {n_clusters} Clusters"
            if include_silhouette and self._silhouette_score is not None:
                fig_title += f" (Silhouette Score: {self._silhouette_score:.3f})"

        fig.suptitle(fig_title, fontsize=16)
        fig.tight_layout()

        return fig

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = {
            "type": self.__class__.__name__,
            "overlap": self._overlap_samples,
        }

        # Add configuration details
        base_summary.update(
            {
                "n_clusters": self.n_clusters,
                "n_components": self.n_components,
                "normalize": self.normalize,
                "random_state": self.random_state,
                "chunk_size": self.chunk_size,
                "is_fitted": self.pca_processor.is_fitted
                and self.kmeans_processor.is_fitted,
                "explained_variance_sum": (
                    np.sum(self.pca_processor.get_explained_variance())
                    if self.pca_processor.is_fitted
                    else None
                ),
                "silhouette_score": self._silhouette_score,
            }
        )
        return base_summary


# Factory functions for easy creation


def create_pca_kmeans(
    n_clusters: int = 3,
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> ComposedPCAKMeansProcessor:
    """
    Create a standard PCA-KMeans processor with default parameters.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components: Number of PCA components to use
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured ComposedPCAKMeansProcessor
    """
    return ComposedPCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
        chunk_size=chunk_size,
    )


def create_adaptive_clustering(
    normalize: bool = True,
    max_clusters: int = 10,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> ComposedPCAKMeansProcessor:
    """
    Create a PCA-KMeans processor configured for adaptive clustering.

    This processor will attempt to automatically determine the optimal
    number of clusters during processing, up to max_clusters.

    Args:
        normalize: Whether to normalize waveforms
        max_clusters: Maximum number of clusters to consider
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured ComposedPCAKMeansProcessor
    """
    # For adaptive clustering, we start with a high number of clusters
    processor = ComposedPCAKMeansProcessor(
        n_clusters=max_clusters,
        n_components=min(20, max_clusters * 2),  # More components for better separation
        normalize=normalize,
        random_state=random_state,
        max_iter=500,  # More iterations for better convergence
        n_init=15,  # More initializations to find optimal clusters
        chunk_size=chunk_size,
    )

    # Set a flag to indicate this is for adaptive clustering
    processor._adaptive = True

    return processor


def create_high_quality_clustering(
    n_clusters: int = 3,
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> ComposedPCAKMeansProcessor:
    """
    Create a PCA-KMeans processor optimized for high-quality clustering.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components: Number of PCA components to use
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured ComposedPCAKMeansProcessor
    """
    return ComposedPCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
        max_iter=500,  # More iterations for better convergence
        n_init=20,  # More initialization attempts
        whiten=True,  # Apply whitening to decorrelate features
        chunk_size=chunk_size,
    )


def create_incremental_clustering(
    initial_clusters: int = 3,
    normalize: bool = True,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> ComposedPCAKMeansProcessor:
    """
    Create a PCA-KMeans processor for incremental learning.

    This configuration is suitable for scenarios where data is processed in batches
    and the clustering model needs to be updated incrementally.

    Args:
        initial_clusters: Initial number of clusters to start with
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured ComposedPCAKMeansProcessor
    """
    processor = ComposedPCAKMeansProcessor(
        n_clusters=initial_clusters,
        n_components=max(
            10, initial_clusters * 3
        ),  # More components to allow for growth
        normalize=normalize,
        random_state=random_state,
        max_iter=300,
        chunk_size=chunk_size,
    )

    # Set a flag to indicate this is for incremental learning
    processor._incremental = True

    return processor


def create_quality_metrics_clustering(
    n_clusters: int = 3,
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> ComposedPCAKMeansProcessor:
    """
    Create a PCA-KMeans processor with additional quality metrics.

    This configuration computes additional cluster quality metrics beyond
    the standard silhouette score, helpful for evaluating clustering performance.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components: Number of PCA components to use
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured ComposedPCAKMeansProcessor
    """
    processor = ComposedPCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    # Set a flag to indicate computing additional quality metrics
    processor._compute_extra_metrics = True

    return processor


def create_multichannel_clustering(
    n_clusters: int = 3,
    n_components_per_channel: int = 3,
    n_channels: int = 4,
    normalize: bool = True,
    random_state: int = 42,
    chunk_size: Optional[int] = None,
) -> ComposedPCAKMeansProcessor:
    """
    Create a PCA-KMeans processor optimized for multi-channel neural recordings.

    This configuration allocates a specific number of PCA components per channel,
    which can improve clustering for multi-channel data where channels carry
    different information.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components_per_channel: Number of PCA components to use per channel
        n_channels: Number of recording channels
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility
        chunk_size: Size of chunks for Dask processing

    Returns:
        Configured ComposedPCAKMeansProcessor
    """
    total_components = min(
        n_components_per_channel * n_channels, 30
    )  # Cap at 30 components

    processor = ComposedPCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=total_components,
        normalize=normalize,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    # Set multichannel specific attributes
    processor._multichannel = True
    processor._n_channels = n_channels
    processor._components_per_channel = n_components_per_channel

    return processor
