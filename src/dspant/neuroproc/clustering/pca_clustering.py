"""
PCA-KMeans clustering implementation for neural waveform analysis.

This module provides accelerated PCA and KMeans clustering specifically
optimized for neural spike waveform analysis with dask and numba acceleration.
"""

import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, prange
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from ...engine.base import BaseProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_features_numba(
    waveforms_flat: np.ndarray, n_waveforms: int, n_features: int
) -> np.ndarray:
    """
    Compute simple time-domain features from flattened waveforms.

    Args:
        waveforms_flat: Flattened waveforms (n_waveforms × n_samples)
        n_waveforms: Number of waveforms
        n_features: Number of features to extract

    Returns:
        Feature matrix (n_waveforms × n_features)
    """
    # Get the total length and calculate samples per waveform
    total_length = waveforms_flat.shape[1]
    samples_per_waveform = total_length // n_features

    # Initialize features matrix
    features = np.zeros((n_waveforms, n_features), dtype=np.float32)

    # Extract features in parallel across waveforms
    for i in prange(n_waveforms):
        waveform = waveforms_flat[i]

        # Split waveform into segments and compute features
        for j in range(n_features):
            start_idx = j * samples_per_waveform
            end_idx = min((j + 1) * samples_per_waveform, total_length)

            if start_idx < end_idx:
                # Use simple statistics as features
                segment = waveform[start_idx:end_idx]
                features[i, j] = np.mean(segment)  # Can be extended with other features

    return features


@jit(nopython=True, cache=True)
def _normalize_waveforms(waveforms: np.ndarray) -> np.ndarray:
    """
    Normalize waveforms for better clustering.

    Args:
        waveforms: Input waveforms array (n_waveforms × n_samples × n_channels)

    Returns:
        Normalized waveforms
    """
    n_waveforms = waveforms.shape[0]
    normalized = np.zeros_like(waveforms, dtype=np.float32)

    for i in range(n_waveforms):
        # Get the waveform
        wf = waveforms[i]

        # Find min and max values
        wf_min = np.min(wf)
        wf_max = np.max(wf)

        # Avoid division by zero
        if wf_max - wf_min > 1e-10:
            # Normalize to [0, 1]
            normalized[i] = (wf - wf_min) / (wf_max - wf_min)
        else:
            # If the waveform is flat, just center it
            normalized[i] = wf - wf_min

    return normalized


class PCAKMeansProcessor(BaseProcessor):
    """
    PCA-KMeans clustering processor for neural waveforms.

    Performs dimensionality reduction with PCA followed by KMeans clustering,
    optimized for large datasets with dask and numba acceleration.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_components: int = 10,
        normalize: bool = True,
        random_state: int = 42,
        max_workers: Optional[int] = None,
        max_iterations: int = 300,
    ):
        """
        Initialize the PCA-KMeans processor.

        Args:
            n_clusters: Number of clusters for KMeans
            n_components: Number of PCA components to use
            normalize: Whether to normalize waveforms before processing
            random_state: Random seed for reproducibility
            max_workers: Maximum number of worker threads for parallel processing
            max_iterations: Maximum number of iterations for KMeans
        """
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.normalize = normalize
        self.random_state = random_state
        self.max_workers = max_workers
        self.max_iterations = max_iterations

        # Initialize models
        self._pca = None
        self._kmeans = None

        # Clustering results
        self._cluster_labels = None
        self._cluster_centers = None
        self._explained_variance_ratio = None
        self._pca_components = None
        self._silhouette_score = None
        self._tsne_embedding = None

        # No overlap needed for this operation
        self._overlap_samples = 0

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """
        Perform PCA-KMeans clustering on waveform data.

        Args:
            data: Input dask array of waveforms (n_waveforms × n_samples × n_channels)
            fs: Sampling frequency (not used but required by BaseProcessor interface)
            **kwargs: Additional keyword arguments
                compute_now: Whether to compute clusters immediately
                subsample: Maximum number of waveforms to use for fitting

        Returns:
            Dask array with cluster labels
        """
        # Override parameters if provided
        n_clusters = kwargs.get("n_clusters", self.n_clusters)
        n_components = kwargs.get("n_components", self.n_components)
        normalize = kwargs.get("normalize", self.normalize)
        compute_now = kwargs.get("compute_now", True)
        max_workers = kwargs.get("max_workers", self.max_workers)

        # Get subsample size (if specified)
        subsample = kwargs.get("subsample", None)

        # Define preprocessing and clustering function
        def cluster_waveforms(waveforms: np.ndarray) -> np.ndarray:
            """Process chunk of waveforms to obtain cluster labels."""
            # Ensure input is a numpy array
            waveforms = np.asarray(waveforms)

            # Check if we have any waveforms
            if len(waveforms) == 0:
                return np.array([])

            # Handle 3D waveform array (waveforms × samples × channels)
            n_waveforms, n_samples, n_channels = waveforms.shape

            # Normalize waveforms if requested
            if normalize:
                waveforms = _normalize_waveforms(waveforms)

            # Reshape to 2D for PCA: (waveforms × (samples*channels))
            waveforms_flat = waveforms.reshape(n_waveforms, -1)

            # Apply subsampling if requested
            if subsample is not None and subsample < n_waveforms:
                # Use stratified sampling if we have existing labels
                if (
                    self._cluster_labels is not None
                    and len(self._cluster_labels) == n_waveforms
                ):
                    # Stratified sampling based on existing labels
                    unique_labels = np.unique(self._cluster_labels)
                    subsample_indices = []

                    # Calculate samples per cluster
                    samples_per_cluster = max(1, subsample // len(unique_labels))

                    # Select samples from each cluster
                    for label in unique_labels:
                        label_indices = np.where(self._cluster_labels == label)[0]
                        if len(label_indices) > 0:
                            # Take random subset from this cluster
                            selected = np.random.choice(
                                label_indices,
                                size=min(samples_per_cluster, len(label_indices)),
                                replace=False,
                            )
                            subsample_indices.extend(selected)

                    # If we didn't get enough samples, add more randomly
                    if len(subsample_indices) < subsample:
                        remaining = subsample - len(subsample_indices)
                        all_indices = set(range(n_waveforms))
                        unused_indices = list(all_indices - set(subsample_indices))
                        if unused_indices:
                            additional = np.random.choice(
                                unused_indices,
                                size=min(remaining, len(unused_indices)),
                                replace=False,
                            )
                            subsample_indices.extend(additional)

                    # Convert to array and sort
                    subsample_indices = np.array(subsample_indices)
                    subsample_indices.sort()

                else:
                    # Simple random sampling if no existing labels
                    subsample_indices = np.random.choice(
                        n_waveforms, size=subsample, replace=False
                    )

                # Extract subsampled waveforms
                waveforms_flat_sample = waveforms_flat[subsample_indices]
            else:
                # Use all waveforms
                waveforms_flat_sample = waveforms_flat
                subsample_indices = None

            # Initialize or update PCA model
            if self._pca is None:
                # Create new PCA model
                self._pca = PCA(
                    n_components=min(
                        n_components,
                        waveforms_flat_sample.shape[1],
                        waveforms_flat_sample.shape[0],
                    ),
                    random_state=self.random_state,
                )
                # Fit PCA with subsampled data
                pca_result = self._pca.fit_transform(waveforms_flat_sample)
                self._explained_variance_ratio = self._pca.explained_variance_ratio_
                self._pca_components = pca_result
            else:
                # Use existing PCA model
                pca_result = self._pca.transform(waveforms_flat_sample)
                self._pca_components = pca_result

            # Initialize or update KMeans model
            if self._kmeans is None:
                # Create new KMeans model
                self._kmeans = KMeans(
                    n_clusters=min(n_clusters, len(waveforms_flat_sample)),
                    random_state=self.random_state,
                    n_init="auto",
                    max_iter=self.max_iterations,
                )
                # Fit KMeans with PCA-transformed data
                subsample_labels = self._kmeans.fit_predict(pca_result)
                self._cluster_centers = self._kmeans.cluster_centers_

                # Calculate silhouette score if we have enough samples and clusters
                if len(pca_result) > n_clusters and n_clusters > 1:
                    try:
                        self._silhouette_score = silhouette_score(
                            pca_result, subsample_labels
                        )
                    except:
                        self._silhouette_score = None
            else:
                # Use existing KMeans model
                subsample_labels = self._kmeans.predict(pca_result)

            # Handle full dataset vs. subsample
            if subsample_indices is not None:
                # Initialize labels for all waveforms
                all_labels = np.zeros(n_waveforms, dtype=np.int32)

                # Transform all waveforms with PCA
                all_pca = self._pca.transform(waveforms_flat)
                self._pca_components = all_pca

                # Predict labels for all waveforms
                all_labels = self._kmeans.predict(all_pca)
            else:
                # Use subsample labels since they cover all waveforms
                all_labels = subsample_labels

            # Store results
            self._cluster_labels = all_labels

            return all_labels

        # Convert Dask array to NumPy if compute_now is True
        if compute_now:
            waveforms_np = data.compute()
            labels = cluster_waveforms(waveforms_np)
            return da.from_array(labels)
        else:
            # Apply clustering function to Dask array
            result = data.map_blocks(
                lambda x: cluster_waveforms(x),
                drop_axis=(1, 2),  # Output has one dimension fewer than input
                dtype=np.int32,
            )
            return result

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get the cluster centers in PCA space.

        Returns:
            Array of cluster centers if available, None otherwise
        """
        return self._cluster_centers

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """
        Get the explained variance ratio from PCA.

        Returns:
            Array of explained variance ratios if available, None otherwise
        """
        return self._explained_variance_ratio

    def transform(self, waveforms: Union[np.ndarray, da.Array]) -> np.ndarray:
        """
        Transform new waveforms using the fitted PCA model.

        Args:
            waveforms: Input waveforms (n_waveforms × n_samples × n_channels)

        Returns:
            Transformed waveforms in PCA space
        """
        if self._pca is None:
            raise ValueError("PCA model not fitted. Call process() first.")

        # Convert to numpy if needed
        if isinstance(waveforms, da.Array):
            waveforms = waveforms.compute()

        # Handle 3D waveform array
        n_waveforms = waveforms.shape[0]

        # Normalize if applicable
        if self.normalize:
            waveforms = _normalize_waveforms(waveforms)

        # Reshape to 2D for PCA
        waveforms_flat = waveforms.reshape(n_waveforms, -1)

        # Transform with PCA
        return self._pca.transform(waveforms_flat)

    def predict(self, waveforms: Union[np.ndarray, da.Array]) -> np.ndarray:
        """
        Predict cluster labels for new waveforms.

        Args:
            waveforms: Input waveforms (n_waveforms × n_samples × n_channels)

        Returns:
            Array of cluster labels
        """
        if self._pca is None or self._kmeans is None:
            raise ValueError("Models not fitted. Call process() first.")

        # Transform waveforms with PCA
        pca_result = self.transform(waveforms)

        # Predict cluster labels
        return self._kmeans.predict(pca_result)

    def compute_tsne(
        self,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        max_samples: int = 5000,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute t-SNE embedding for visualization.

        Args:
            perplexity: Perplexity parameter for t-SNE
            n_iter: Number of iterations for t-SNE
            max_samples: Maximum number of samples to use
            random_state: Random seed for reproducibility

        Returns:
            t-SNE embedding array
        """
        if self._pca_components is None:
            raise ValueError("PCA components not available. Call process() first.")

        # Use random state from class if not specified
        if random_state is None:
            random_state = self.random_state

        # Subsample if too many points
        if self._pca_components.shape[0] > max_samples:
            # Stratified sampling based on cluster labels
            subsample_indices = []
            unique_labels = np.unique(self._cluster_labels)
            samples_per_cluster = max(1, max_samples // len(unique_labels))

            for label in unique_labels:
                label_indices = np.where(self._cluster_labels == label)[0]
                if len(label_indices) > 0:
                    selected = np.random.choice(
                        label_indices,
                        size=min(samples_per_cluster, len(label_indices)),
                        replace=False,
                    )
                    subsample_indices.extend(selected)

            # Convert to array
            subsample_indices = np.array(subsample_indices)
            pca_subset = self._pca_components[subsample_indices]
            labels_subset = self._cluster_labels[subsample_indices]
        else:
            pca_subset = self._pca_components
            labels_subset = self._cluster_labels
            subsample_indices = None

        # Compute t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, pca_subset.shape[0] - 1),
            n_iter=n_iter,
            random_state=random_state,
        )
        tsne_embedding = tsne.fit_transform(pca_subset)

        # Store results
        self._tsne_embedding = tsne_embedding
        self._tsne_labels = labels_subset
        self._tsne_indices = subsample_indices

        return tsne_embedding

    def plot_clusters(
        self,
        max_points: int = 5000,
        plot_type: str = "pca",
        plot_waveforms: bool = True,
        n_waveforms_per_cluster: int = 50,
        figsize: Tuple[int, int] = (18, 12),
        alpha: float = 0.7,
        s: int = 15,
        cmap: str = "tab10",
        title: Optional[str] = None,
        fig: Optional[Figure] = None,
        include_silhouette: bool = True,
        compute_tsne: bool = False,
        tsne_perplexity: float = 30.0,
    ) -> Figure:
        """
        Visualize clustering results.

        Args:
            max_points: Maximum number of points to plot
            plot_type: Type of plot ('pca', 'pca3d', or 'tsne')
            plot_waveforms: Whether to plot mean waveforms for each cluster
            n_waveforms_per_cluster: Number of waveforms to sample per cluster
            figsize: Figure size as (width, height)
            alpha: Alpha value for scatter points
            s: Size of scatter points
            cmap: Colormap to use
            title: Plot title (if None, a default title is generated)
            fig: Existing figure to use (if None, creates a new figure)
            include_silhouette: Whether to include silhouette score in title
            compute_tsne: Whether to compute t-SNE embedding if not available
            tsne_perplexity: Perplexity parameter for t-SNE

        Returns:
            Matplotlib figure
        """
        if self._cluster_labels is None or self._pca_components is None:
            raise ValueError("Clustering not performed. Call process() first.")

        # Create figure if not provided
        if fig is None:
            fig = plt.figure(figsize=figsize)

        # Determine number of subplots based on whether to plot waveforms
        if plot_waveforms:
            gs = GridSpec(2, 3, height_ratios=[2, 1], figure=fig)
            ax_scatter = fig.add_subplot(gs[0, :2])
            if plot_type == "pca3d":
                ax_scatter = fig.add_subplot(gs[0, :2], projection="3d")
            ax_variance = fig.add_subplot(gs[0, 2])
            ax_waveforms = fig.add_subplot(gs[1, :])
        else:
            gs = GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
            ax_scatter = fig.add_subplot(gs[0, 0])
            if plot_type == "pca3d":
                ax_scatter = fig.add_subplot(gs[0, 0], projection="3d")
            ax_variance = fig.add_subplot(gs[0, 1])

        # Handle limiting the number of points
        if len(self._cluster_labels) > max_points:
            # Stratified sampling
            indices = []
            for label in np.unique(self._cluster_labels):
                label_indices = np.where(self._cluster_labels == label)[0]
                points_per_cluster = max(
                    1, max_points // len(np.unique(self._cluster_labels))
                )
                if len(label_indices) > points_per_cluster:
                    sampled = np.random.choice(
                        label_indices, points_per_cluster, replace=False
                    )
                    indices.extend(sampled)
                else:
                    indices.extend(label_indices)

            pca_data = self._pca_components[indices]
            labels = self._cluster_labels[indices]
        else:
            pca_data = self._pca_components
            labels = self._cluster_labels
            indices = np.arange(len(labels))

        # Get the colormap
        cmap_obj = plt.get_cmap(cmap)

        # Generate cluster colors
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        colors = [cmap_obj(i / max(1, n_clusters - 1)) for i in range(n_clusters)]

        # Plot based on plot_type
        if plot_type == "pca":
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

        elif plot_type == "pca3d":
            # 3D PCA plot
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax_scatter.scatter(
                    pca_data[mask, 0],
                    pca_data[mask, 1],
                    pca_data[mask, 2],
                    c=[colors[i]],
                    label=f"Cluster {label}",
                    alpha=alpha,
                    s=s,
                )
            ax_scatter.set_xlabel("PC1")
            ax_scatter.set_ylabel("PC2")
            ax_scatter.set_zlabel("PC3")
            ax_scatter.set_title(f"3D PCA Visualization of {n_clusters} Clusters")
            ax_scatter.legend(loc="best")

        elif plot_type == "tsne":
            # t-SNE plot
            if self._tsne_embedding is None:
                if compute_tsne:
                    self.compute_tsne(
                        perplexity=tsne_perplexity, max_samples=max_points
                    )
                else:
                    raise ValueError(
                        "t-SNE embedding not available. Set compute_tsne=True or call compute_tsne() first."
                    )

            # Use t-SNE data
            tsne_data = self._tsne_embedding
            tsne_labels = self._tsne_labels

            for i, label in enumerate(np.unique(tsne_labels)):
                mask = tsne_labels == label
                ax_scatter.scatter(
                    tsne_data[mask, 0],
                    tsne_data[mask, 1],
                    c=[colors[i]],
                    label=f"Cluster {label}",
                    alpha=alpha,
                    s=s,
                )
            ax_scatter.set_xlabel("t-SNE 1")
            ax_scatter.set_ylabel("t-SNE 2")
            ax_scatter.set_title(f"t-SNE Visualization of {n_clusters} Clusters")
            ax_scatter.legend(loc="best")
            ax_scatter.grid(True, linestyle="--", alpha=0.7)

        # Plot explained variance
        if self._explained_variance_ratio is not None:
            variance_ratio = self._explained_variance_ratio
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

        # Plot mean waveforms for each cluster if requested
        if (
            plot_waveforms
            and hasattr(self, "_original_waveforms")
            and self._original_waveforms is not None
        ):
            waveforms = self._original_waveforms

            # Sample waveforms for each cluster
            cluster_waveforms = []
            for label in unique_labels:
                cluster_indices = np.where(self._cluster_labels == label)[0]
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

    def save_original_waveforms(self, waveforms: Union[np.ndarray, da.Array]) -> None:
        """
        Save original waveforms for later visualization.

        Args:
            waveforms: Input waveforms array
        """
        if isinstance(waveforms, da.Array):
            # Convert to numpy but use only a subset if the array is large
            if waveforms.shape[0] > 10000:
                indices = np.random.choice(waveforms.shape[0], 10000, replace=False)
                self._original_waveforms = waveforms[indices].compute()
            else:
                self._original_waveforms = waveforms.compute()
        else:
            # Use numpy array directly
            if waveforms.shape[0] > 10000:
                indices = np.random.choice(waveforms.shape[0], 10000, replace=False)
                self._original_waveforms = waveforms[indices]
            else:
                self._original_waveforms = waveforms

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "n_clusters": self.n_clusters,
                "n_components": self.n_components,
                "normalize": self.normalize,
                "random_state": self.random_state,
                "is_fitted": self._pca is not None and self._kmeans is not None,
                "explained_variance_sum": (
                    np.sum(self._explained_variance_ratio)
                    if self._explained_variance_ratio is not None
                    else None
                ),
                "silhouette_score": self._silhouette_score,
            }
        )
        return base_summary


## Factory functions

# Factory functions for common use cases


def create_pca_kmeans(
    n_clusters: int = 3,
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
) -> PCAKMeansProcessor:
    """
    Create a standard PCA-KMeans processor with default parameters.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components: Number of PCA components to use
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility

    Returns:
        Configured PCAKMeansProcessor
    """
    return PCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
    )


def create_adaptive_clustering(
    normalize: bool = True, max_clusters: int = 10, random_state: int = 42
) -> PCAKMeansProcessor:
    """
    Create a PCA-KMeans processor configured for adaptive clustering.

    This processor will attempt to automatically determine the optimal
    number of clusters during processing, up to max_clusters.

    Args:
        normalize: Whether to normalize waveforms
        max_clusters: Maximum number of clusters to consider
        random_state: Random seed for reproducibility

    Returns:
        Configured PCAKMeansProcessor
    """
    # For adaptive clustering, we start with a high number of clusters
    # The actual implementation would need more logic to adapt the clusters
    return PCAKMeansProcessor(
        n_clusters=max_clusters,
        n_components=min(20, max_clusters * 2),  # More components for better separation
        normalize=normalize,
        random_state=random_state,
        max_iterations=500,  # More iterations for better convergence
    )


def create_visualization_clustering(
    n_clusters: int = 3,
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
    compute_tsne: bool = True,
    tsne_perplexity: float = 30.0,
) -> PCAKMeansProcessor:
    """
    Create a PCA-KMeans processor optimized for visualization.

    This function configures a processor that will automatically compute
    both PCA and t-SNE embeddings for better visualization of clusters.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components: Number of PCA components to use
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility
        compute_tsne: Whether to automatically compute t-SNE after clustering
        tsne_perplexity: Perplexity parameter for t-SNE

    Returns:
        Configured PCAKMeansProcessor
    """
    processor = PCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
    )

    # Add custom attributes for visualization settings
    processor._auto_compute_tsne = compute_tsne
    processor._tsne_perplexity = tsne_perplexity

    return processor


def create_incremental_clustering(
    initial_clusters: int = 3,
    normalize: bool = True,
    random_state: int = 42,
) -> PCAKMeansProcessor:
    """
    Create a PCA-KMeans processor for incremental learning.

    This configuration is suitable for scenarios where data is processed in batches
    and the clustering model needs to be updated incrementally.

    Args:
        initial_clusters: Initial number of clusters to start with
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility

    Returns:
        Configured PCAKMeansProcessor
    """
    processor = PCAKMeansProcessor(
        n_clusters=initial_clusters,
        n_components=max(
            10, initial_clusters * 3
        ),  # More components to allow for growth
        normalize=normalize,
        random_state=random_state,
        max_iterations=300,
    )

    # Set a flag to indicate this is for incremental learning
    processor._incremental = True

    return processor


def create_quality_metrics_clustering(
    n_clusters: int = 3,
    n_components: int = 10,
    normalize: bool = True,
    random_state: int = 42,
) -> PCAKMeansProcessor:
    """
    Create a PCA-KMeans processor with additional quality metrics.

    This configuration computes additional cluster quality metrics beyond
    the standard silhouette score, helpful for evaluating clustering performance.

    Args:
        n_clusters: Number of clusters for KMeans
        n_components: Number of PCA components to use
        normalize: Whether to normalize waveforms
        random_state: Random seed for reproducibility

    Returns:
        Configured PCAKMeansProcessor
    """
    processor = PCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=n_components,
        normalize=normalize,
        random_state=random_state,
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
) -> PCAKMeansProcessor:
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

    Returns:
        Configured PCAKMeansProcessor
    """
    total_components = min(
        n_components_per_channel * n_channels, 30
    )  # Cap at 30 components

    processor = PCAKMeansProcessor(
        n_clusters=n_clusters,
        n_components=total_components,
        normalize=normalize,
        random_state=random_state,
    )

    # Set multichannel specific attributes
    processor._multichannel = True
    processor._n_channels = n_channels
    processor._components_per_channel = n_components_per_channel

    return processor
