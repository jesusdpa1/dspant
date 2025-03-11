"""
Template utilities for neural spike sorting with Numba acceleration.

This module provides optimized functions for computing spike templates
and template similarities using Numba to achieve significant performance improvements.
These functions are particularly useful for spike sorting workflows and
template matching algorithms in neural signal processing.
"""

from typing import Tuple, Union

import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, cache=True)
def compute_templates(
    waveforms: np.ndarray, cluster_labels: np.ndarray, unique_clusters: np.ndarray
) -> np.ndarray:
    """
    Compute average templates for each cluster with Numba acceleration.

    Parameters
    ----------
    waveforms : np.ndarray
        Spike waveforms with shape (n_spikes, n_samples, n_channels) or (n_spikes, n_samples)
    cluster_labels : np.ndarray
        Cluster assignments for each spike with shape (n_spikes,)
    unique_clusters : np.ndarray
        Array of unique cluster IDs

    Returns
    -------
    np.ndarray
        Templates array with shape (n_clusters, n_samples, n_channels) or (n_clusters, n_samples)
    """
    n_clusters = len(unique_clusters)

    # Handle different input dimensions
    if waveforms.ndim == 3:
        n_samples_waveform = waveforms.shape[1]
        n_channels = waveforms.shape[2]
        templates = np.zeros(
            (n_clusters, n_samples_waveform, n_channels), dtype=np.float32
        )

        # Compute templates in parallel
        for i in prange(n_clusters):
            cluster_id = unique_clusters[i]

            # Count spikes in this cluster
            n_spikes_in_cluster = 0
            for k in range(len(cluster_labels)):
                if cluster_labels[k] == cluster_id:
                    n_spikes_in_cluster += 1

            if n_spikes_in_cluster > 0:
                # Compute sum for this cluster
                for k in range(len(cluster_labels)):
                    if cluster_labels[k] == cluster_id:
                        for t in range(n_samples_waveform):
                            for c in range(n_channels):
                                templates[i, t, c] += waveforms[k, t, c]

                # Compute mean
                for t in range(n_samples_waveform):
                    for c in range(n_channels):
                        templates[i, t, c] /= n_spikes_in_cluster

    else:  # 2D case (n_spikes, n_samples)
        n_samples_waveform = waveforms.shape[1]
        templates = np.zeros((n_clusters, n_samples_waveform), dtype=np.float32)

        # Compute templates in parallel
        for i in prange(n_clusters):
            cluster_id = unique_clusters[i]

            # Count spikes in this cluster
            n_spikes_in_cluster = 0
            for k in range(len(cluster_labels)):
                if cluster_labels[k] == cluster_id:
                    n_spikes_in_cluster += 1

            if n_spikes_in_cluster > 0:
                # Compute sum for this cluster
                for k in range(len(cluster_labels)):
                    if cluster_labels[k] == cluster_id:
                        for t in range(n_samples_waveform):
                            templates[i, t] += waveforms[k, t]

                # Compute mean
                for t in range(n_samples_waveform):
                    templates[i, t] /= n_spikes_in_cluster

    return templates


@jit(nopython=True, parallel=True, cache=True)
def compute_template_similarity(templates: np.ndarray) -> np.ndarray:
    """
    Compute normalized similarity matrix between templates with Numba acceleration.

    Parameters
    ----------
    templates : np.ndarray
        Templates array with shape (n_templates, n_samples, n_channels) or (n_templates, n_samples)

    Returns
    -------
    np.ndarray
        Similarity matrix with shape (n_templates, n_templates)
    """
    n_templates = templates.shape[0]
    similarity = np.zeros((n_templates, n_templates), dtype=np.float32)

    # Reshape templates to 2D for easier computation
    if templates.ndim == 3:
        n_samples = templates.shape[1]
        n_channels = templates.shape[2]
        templates_flat = np.zeros(
            (n_templates, n_samples * n_channels), dtype=np.float32
        )

        # Flatten each template
        for i in range(n_templates):
            idx = 0
            for t in range(n_samples):
                for c in range(n_channels):
                    templates_flat[i, idx] = templates[i, t, c]
                    idx += 1
    else:
        templates_flat = templates  # Already 2D

    # Compute template norms
    templates_norm = np.zeros(n_templates, dtype=np.float32)
    for i in range(n_templates):
        norm_sq = 0.0
        for j in range(templates_flat.shape[1]):
            norm_sq += templates_flat[i, j] * templates_flat[i, j]
        templates_norm[i] = np.sqrt(norm_sq)

    # Compute similarity matrix in parallel
    for i in prange(n_templates):
        for j in range(n_templates):
            if templates_norm[i] > 0 and templates_norm[j] > 0:
                # Compute dot product
                dot_product = 0.0
                for k in range(templates_flat.shape[1]):
                    dot_product += templates_flat[i, k] * templates_flat[j, k]

                # Normalize
                similarity[i, j] = dot_product / (templates_norm[i] * templates_norm[j])
            else:
                similarity[i, j] = 0.0

    return similarity


@jit(nopython=True, cache=True)
def align_template(
    template: np.ndarray, reference: np.ndarray, max_shift: int = 5
) -> Tuple[np.ndarray, int]:
    """
    Align a template to a reference template by finding the optimal shift.

    Parameters
    ----------
    template : np.ndarray
        Template to align with shape (n_samples, n_channels) or (n_samples,)
    reference : np.ndarray
        Reference template with shape (n_samples, n_channels) or (n_samples,)
    max_shift : int, default: 5
        Maximum shift in samples to try in each direction

    Returns
    -------
    Tuple[np.ndarray, int]
        Aligned template and the optimal shift amount
    """
    n_samples = template.shape[0]

    if template.ndim == 2 and reference.ndim == 2:
        # Multi-channel case
        n_channels = template.shape[1]

        # Compute correlation at different shifts
        best_corr = -np.inf
        best_shift = 0

        for shift in range(-max_shift, max_shift + 1):
            # Skip if shift would go out of bounds
            if shift <= -n_samples or shift >= n_samples:
                continue

            # Compute correlation for this shift
            corr_sum = 0.0
            for ch in range(n_channels):
                # Create shifted template
                if shift > 0:
                    # Shift right
                    t1 = template[:-shift, ch] if shift < n_samples else np.zeros(1)
                    t2 = reference[shift:, ch] if shift < n_samples else np.zeros(1)
                elif shift < 0:
                    # Shift left
                    t1 = template[-shift:, ch] if -shift < n_samples else np.zeros(1)
                    t2 = reference[:shift, ch] if -shift < n_samples else np.zeros(1)
                else:
                    # No shift
                    t1 = template[:, ch]
                    t2 = reference[:, ch]

                # Skip if arrays have different lengths
                if len(t1) != len(t2) or len(t1) == 0:
                    continue

                # Compute correlation for this channel and shift
                corr = 0.0
                for i in range(len(t1)):
                    corr += t1[i] * t2[i]
                corr_sum += corr

            # Update best shift if this correlation is better
            if corr_sum > best_corr:
                best_corr = corr_sum
                best_shift = shift

    else:
        # Single-channel case
        # Compute correlation at different shifts
        best_corr = -np.inf
        best_shift = 0

        for shift in range(-max_shift, max_shift + 1):
            # Skip if shift would go out of bounds
            if shift <= -n_samples or shift >= n_samples:
                continue

            # Compute correlation for this shift
            if shift > 0:
                # Shift right
                t1 = template[:-shift] if shift < n_samples else np.zeros(1)
                t2 = reference[shift:] if shift < n_samples else np.zeros(1)
            elif shift < 0:
                # Shift left
                t1 = template[-shift:] if -shift < n_samples else np.zeros(1)
                t2 = reference[:shift] if -shift < n_samples else np.zeros(1)
            else:
                # No shift
                t1 = template
                t2 = reference

            # Skip if arrays have different lengths
            if len(t1) != len(t2) or len(t1) == 0:
                continue

            # Compute correlation for this shift
            corr = 0.0
            for i in range(len(t1)):
                corr += t1[i] * t2[i]

            # Update best shift if this correlation is better
            if corr > best_corr:
                best_corr = corr
                best_shift = shift

    # Apply the best shift to create the aligned template
    if template.ndim == 2:
        n_channels = template.shape[1]
        aligned = np.zeros_like(template)

        if best_shift > 0:
            # Shift right
            aligned[best_shift:, :] = (
                template[:-best_shift, :]
                if best_shift < n_samples
                else np.zeros((1, n_channels))
            )
        elif best_shift < 0:
            # Shift left
            aligned[:best_shift, :] = (
                template[-best_shift:, :]
                if -best_shift < n_samples
                else np.zeros((1, n_channels))
            )
        else:
            # No shift
            aligned = template.copy()
    else:
        aligned = np.zeros_like(template)

        if best_shift > 0:
            # Shift right
            aligned[best_shift:] = (
                template[:-best_shift] if best_shift < n_samples else np.zeros(1)
            )
        elif best_shift < 0:
            # Shift left
            aligned[:best_shift] = (
                template[-best_shift:] if -best_shift < n_samples else np.zeros(1)
            )
        else:
            # No shift
            aligned = template.copy()

    return aligned, best_shift


@jit(nopython=True, cache=True)
def compute_template_metrics(templates: np.ndarray) -> dict:
    """
    Compute basic metrics for each template.

    Parameters
    ----------
    templates : np.ndarray
        Templates array with shape (n_templates, n_samples, n_channels) or (n_templates, n_samples)

    Returns
    -------
    dict
        Dictionary containing metrics for each template:
        - peak_amplitude: Peak amplitude (minimum or maximum value)
        - peak_channel: Channel with highest amplitude (if multi-channel)
        - peak_time: Sample index of peak
    """
    n_templates = templates.shape[0]

    # Initialize arrays for metrics
    peak_amplitude = np.zeros(n_templates, dtype=np.float32)
    peak_time = np.zeros(n_templates, dtype=np.int32)

    if templates.ndim == 3:
        # Multi-channel case
        n_samples = templates.shape[1]
        n_channels = templates.shape[2]
        peak_channel = np.zeros(n_templates, dtype=np.int32)

        for i in range(n_templates):
            # Find peak amplitude for this template
            max_amp = 0.0
            max_amp_abs = 0.0
            max_t = 0
            max_c = 0

            for t in range(n_samples):
                for c in range(n_channels):
                    amp = templates[i, t, c]
                    amp_abs = abs(amp)
                    if amp_abs > max_amp_abs:
                        max_amp = amp
                        max_amp_abs = amp_abs
                        max_t = t
                        max_c = c

            peak_amplitude[i] = max_amp
            peak_time[i] = max_t
            peak_channel[i] = max_c

        # Return as dictionary-like arrays
        return {
            "peak_amplitude": peak_amplitude,
            "peak_time": peak_time,
            "peak_channel": peak_channel,
        }
    else:
        # Single-channel case
        n_samples = templates.shape[1]

        for i in range(n_templates):
            # Find peak amplitude for this template
            max_amp = 0.0
            max_amp_abs = 0.0
            max_t = 0

            for t in range(n_samples):
                amp = templates[i, t]
                amp_abs = abs(amp)
                if amp_abs > max_amp_abs:
                    max_amp = amp
                    max_amp_abs = amp_abs
                    max_t = t

            peak_amplitude[i] = max_amp
            peak_time[i] = max_t

        # Return as dictionary-like arrays
        return {"peak_amplitude": peak_amplitude, "peak_time": peak_time}


@jit(nopython=True, parallel=True, cache=True)
def compute_template_pca(
    templates: np.ndarray, n_components: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal components of templates using power iteration method.
    This is a simplified PCA for template dimensionality reduction.

    Parameters
    ----------
    templates : np.ndarray
        Templates array with shape (n_templates, n_samples, n_channels) or (n_templates, n_samples)
    n_components : int, default: 3
        Number of principal components to compute

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Principal components and explained variance
    """
    # Reshape templates to 2D
    n_templates = templates.shape[0]
    if templates.ndim == 3:
        n_samples = templates.shape[1]
        n_channels = templates.shape[2]
        templates_flat = np.zeros(
            (n_templates, n_samples * n_channels), dtype=np.float32
        )

        # Flatten each template
        for i in range(n_templates):
            idx = 0
            for t in range(n_samples):
                for c in range(n_channels):
                    templates_flat[i, idx] = templates[i, t, c]
                    idx += 1
    else:
        templates_flat = templates.copy()

    # Center data
    mean = np.zeros(templates_flat.shape[1], dtype=np.float32)
    for j in range(templates_flat.shape[1]):
        for i in range(n_templates):
            mean[j] += templates_flat[i, j]
        mean[j] /= n_templates

    for i in range(n_templates):
        for j in range(templates_flat.shape[1]):
            templates_flat[i, j] -= mean[j]

    # Compute covariance matrix (this is memory-intensive for large templates)
    n_features = templates_flat.shape[1]
    cov = np.zeros((n_features, n_features), dtype=np.float32)

    for i in range(n_features):
        for j in range(i, n_features):
            cov_val = 0.0
            for k in range(n_templates):
                cov_val += templates_flat[k, i] * templates_flat[k, j]
            cov_val /= n_templates - 1
            cov[i, j] = cov_val
            cov[j, i] = cov_val  # Symmetric

    # Power iteration method for top eigenvectors (simple PCA approach)
    components = np.zeros((n_components, n_features), dtype=np.float32)
    explained_variance = np.zeros(n_components, dtype=np.float32)

    for c in range(n_components):
        # Initialize random vector
        v = np.zeros(n_features, dtype=np.float32)
        for i in range(n_features):
            v[i] = np.random.random()

        # Normalize
        norm = 0.0
        for i in range(n_features):
            norm += v[i] * v[i]
        norm = np.sqrt(norm)
        for i in range(n_features):
            v[i] /= norm

        # Power iteration (simplified)
        for _ in range(10):  # Usually converges quickly
            # Matrix-vector product
            u = np.zeros(n_features, dtype=np.float32)
            for i in range(n_features):
                for j in range(n_features):
                    u[i] += cov[i, j] * v[j]

            # Normalize
            norm = 0.0
            for i in range(n_features):
                norm += u[i] * u[i]
            norm = np.sqrt(norm)
            for i in range(n_features):
                v[i] = u[i] / norm

        # Store component
        for i in range(n_features):
            components[c, i] = v[i]

        # Calculate explained variance
        variance = 0.0
        for i in range(n_features):
            for j in range(n_features):
                variance += v[i] * cov[i, j] * v[j]
        explained_variance[c] = variance

        # Deflate covariance matrix for next component
        for i in range(n_features):
            for j in range(n_features):
                cov[i, j] -= variance * v[i] * v[j]

    return components, explained_variance
