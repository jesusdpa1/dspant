"""
Exporter for spike sorting results to PHY template-GUI format.
This module provides functionality to export sorted waveforms and their
properties to the PHY template-GUI format for manual curation.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm


class PhyExporter:
    """
    Exports spike sorting results to the PHY template-GUI format.

    This class handles the export of spike times, templates, amplitudes,
    and other metadata required by PHY for manual curation of spike sorting results.
    """

    def __init__(self, output_folder: Union[str, Path]):
        """
        Initialize the PHY exporter.

        Parameters
        ----------
        output_folder : str or Path
            The output folder where PHY files will be saved
        """
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def export_to_template_gui(
        self,
        raw_data: da.Array,
        waveforms: np.ndarray,
        spike_times: np.ndarray,
        cluster_labels: np.ndarray,
        sampling_frequency: float,
        channel_ids: Optional[np.ndarray] = None,
        amplitudes: Optional[np.ndarray] = None,
        channel_positions: Optional[np.ndarray] = None,
        channel_groups: Optional[np.ndarray] = None,
        compute_pc_features: bool = True,
        n_pc_components: int = 3,
        chunk_size: int = 100000,
        dtype: Optional[np.dtype] = None,
        remove_if_exists: bool = False,
        copy_binary: bool = True,
        verbose: bool = True,
    ):
        """
        Export spike sorting results to PHY template-GUI format.

        Parameters
        ----------
        raw_data : da.Array
            The raw data as a dask array (samples x channels)
        waveforms : np.ndarray
            Extracted spike waveforms (spikes x samples x channels)
        spike_times : np.ndarray
            Spike times in samples
        cluster_labels : np.ndarray
            Cluster assignments for each spike
        sampling_frequency : float
            Sampling frequency in Hz
        channel_ids : np.ndarray, optional
            Channel IDs, defaults to range(n_channels)
        amplitudes : np.ndarray, optional
            Spike amplitudes, computed if not provided
        channel_positions : np.ndarray, optional
            Channel positions in 2D space, created as grid if not provided
        channel_groups : np.ndarray, optional
            Channel group assignments, defaults to zeros
        compute_pc_features : bool, default: True
            Whether to compute principal component features
        n_pc_components : int, default: 3
            Number of principal components to compute
        chunk_size : int, default: 100000
            Number of samples to process at once for recording.dat
        dtype : np.dtype, optional
            Data type for recording.dat, defaults to raw_data.dtype
        remove_if_exists : bool, default: False
            Whether to remove the output folder if it exists
        copy_binary : bool, default: True
            Whether to copy the raw data to recording.dat
        verbose : bool, default: True
            Whether to print progress information

        Returns
        -------
        dict
            Dictionary with paths to created files
        """
        if verbose:
            print(f"Exporting to PHY template-GUI format in {self.output_folder}")

        # Check if folder exists
        if self.output_folder.exists():
            if remove_if_exists:
                import shutil

                shutil.rmtree(self.output_folder)
                self.output_folder.mkdir(parents=True)
            else:
                raise FileExistsError(
                    f"{self.output_folder} already exists. Use remove_if_exists=True to overwrite."
                )

        # Get dimensions
        n_spikes = len(spike_times)
        n_samples_waveform = waveforms.shape[1]
        n_channels = raw_data.shape[1]

        if dtype is None:
            dtype = raw_data.dtype

        # Set up channel information
        if channel_ids is None:
            channel_ids = np.arange(n_channels)

        if channel_positions is None:
            # Create a grid of channel positions
            n_cols = int(np.ceil(np.sqrt(n_channels)))
            n_rows = int(np.ceil(n_channels / n_cols))
            channel_positions = np.zeros((n_channels, 2), dtype=np.float32)
            for i in range(n_channels):
                channel_positions[i, 0] = i % n_cols  # x-coordinate
                channel_positions[i, 1] = i // n_cols  # y-coordinate

        if channel_groups is None:
            channel_groups = np.zeros(n_channels, dtype=np.int32)

        # Get unique clusters
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)

        if verbose:
            print(f"Found {n_spikes} spikes in {n_clusters} clusters")

        # Compute amplitudes if not provided
        if amplitudes is None:
            if verbose:
                print("Computing spike amplitudes...")
            amplitudes = np.zeros(n_spikes, dtype=np.float32)
            for i in range(n_spikes):
                # Take minimum amplitude across all channels for each spike
                amplitudes[i] = np.min(waveforms[i])

        # Export recording.dat if requested
        if copy_binary:
            if verbose:
                print("Exporting recording.dat...")
            self._export_recording_dat(raw_data, chunk_size, dtype, verbose)

        # Export spike times and clusters
        if verbose:
            print("Exporting spike times and clusters...")
        np.save(str(self.output_folder / "spike_times.npy"), spike_times[:, np.newaxis])
        np.save(
            str(self.output_folder / "spike_clusters.npy"),
            cluster_labels[:, np.newaxis],
        )
        np.save(
            str(self.output_folder / "spike_templates.npy"),
            cluster_labels[:, np.newaxis],
        )

        # Export amplitudes
        np.save(str(self.output_folder / "amplitudes.npy"), amplitudes[:, np.newaxis])

        # Compute and export templates
        if verbose:
            print("Computing templates...")
        templates = self._compute_templates(waveforms, cluster_labels, unique_clusters)
        np.save(str(self.output_folder / "templates.npy"), templates)

        # Compute and export similar templates
        similar_templates = self._compute_template_similarity(templates)
        np.save(str(self.output_folder / "similar_templates.npy"), similar_templates)

        # Export channel information
        np.save(
            str(self.output_folder / "channel_map.npy"),
            np.arange(n_channels, dtype=np.int32),
        )
        np.save(str(self.output_folder / "channel_map_si.npy"), channel_ids)
        np.save(str(self.output_folder / "channel_positions.npy"), channel_positions)
        np.save(str(self.output_folder / "channel_groups.npy"), channel_groups)

        # Compute and export PC features if requested
        if compute_pc_features:
            if verbose:
                print("Computing PC features...")
            pc_features, pc_feature_ind = self._compute_pc_features(
                waveforms, cluster_labels, unique_clusters, n_pc_components
            )
            np.save(str(self.output_folder / "pc_features.npy"), pc_features)
            np.save(str(self.output_folder / "pc_feature_ind.npy"), pc_feature_ind)

        # Export metadata
        cluster_group = pd.DataFrame(
            {"cluster_id": np.arange(n_clusters), "group": ["unsorted"] * n_clusters}
        )
        cluster_group.to_csv(
            self.output_folder / "cluster_group.tsv", sep="\t", index=False
        )

        # Write params.py file
        self._write_params_file(sampling_frequency, n_channels, dtype)

        if verbose:
            print(f"PHY template-GUI export complete. To open, run:")
            print(f"phy template-gui {self.output_folder}/params.py")

        return {
            "output_folder": self.output_folder,
            "params_file": self.output_folder / "params.py",
        }

    def _export_recording_dat(self, raw_data, chunk_size, dtype, verbose):
        """Export raw data to recording.dat in chunks."""
        data_file = self.output_folder / "recording.dat"

        # Get dimensions
        n_samples = raw_data.shape[0]
        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        try:
            from tqdm.auto import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False

        # Open file and write in chunks
        with open(data_file, "wb") as f:
            if has_tqdm and verbose:
                chunk_iterator = tqdm(range(n_chunks), desc="Exporting data chunks")
            else:
                chunk_iterator = range(n_chunks)
                if verbose:
                    print(f"Processing {n_chunks} chunks...")

            for i in chunk_iterator:
                # Get chunk start and end
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n_samples)

                # Get chunk data and write to file
                chunk_data = raw_data[start:end].compute()
                chunk_data.astype(dtype).tofile(f)

        return data_file

    def _compute_templates(self, waveforms, cluster_labels, unique_clusters):
        """Compute average templates for each cluster."""
        n_clusters = len(unique_clusters)
        n_samples_waveform = waveforms.shape[1]
        n_channels = waveforms.shape[2] if waveforms.ndim > 2 else 1

        templates = np.zeros(
            (n_clusters, n_samples_waveform, n_channels), dtype=np.float32
        )

        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            if np.sum(mask) > 0:
                templates[i] = np.mean(waveforms[mask], axis=0)

        return templates

    def _compute_template_similarity(self, templates):
        """Compute similarity between templates."""
        n_templates = templates.shape[0]
        similarity = np.zeros((n_templates, n_templates), dtype=np.float32)

        # Reshape templates to 2D for easier computation
        templates_flat = templates.reshape(n_templates, -1)

        # Compute normalized dot product between all pairs of templates
        templates_norm = np.linalg.norm(templates_flat, axis=1)
        for i in range(n_templates):
            for j in range(n_templates):
                if templates_norm[i] > 0 and templates_norm[j] > 0:
                    similarity[i, j] = np.dot(templates_flat[i], templates_flat[j]) / (
                        templates_norm[i] * templates_norm[j]
                    )
                else:
                    similarity[i, j] = 0

        return similarity

    def _compute_pc_features(
        self, waveforms, cluster_labels, unique_clusters, n_components
    ):
        """Compute PCA features for spike waveforms."""
        from sklearn.decomposition import PCA

        n_spikes, n_samples_waveform, n_channels = waveforms.shape
        n_clusters = len(unique_clusters)

        # Compute PC features for each channel
        pc_features = np.zeros((n_spikes, n_components, n_channels), dtype=np.float32)
        pc_feature_ind = np.zeros((n_clusters, n_channels), dtype=np.int32)

        # Fill pc_feature_ind with channel indices
        for i in range(n_clusters):
            pc_feature_ind[i] = np.arange(n_channels)

        # Compute PCA for each channel
        for ch in range(n_channels):
            channel_waveforms = waveforms[:, :, ch]

            if np.std(channel_waveforms.flatten()) > 0:
                pca = PCA(n_components=n_components)
                try:
                    pc_features[:, :, ch] = pca.fit_transform(channel_waveforms)[
                        :, :n_components
                    ]
                except:
                    # If PCA fails, fill with zeros
                    pc_features[:, :, ch] = 0
            else:
                pc_features[:, :, ch] = 0

        return pc_features, pc_feature_ind

    def _write_params_file(self, sampling_frequency, n_channels, dtype):
        """Write params.py file for PHY."""
        params_file = self.output_folder / "params.py"

        with open(params_file, "w") as f:
            f.write(f"dat_path = r'recording.dat'\n")
            f.write(f"n_channels_dat = {n_channels}\n")
            f.write(f"dtype = '{dtype.name}'\n")
            f.write(f"offset = 0\n")
            f.write(f"sample_rate = {sampling_frequency}\n")
            f.write(f"hp_filtered = True\n")

        return params_file


def export_to_phy(
    raw_data,
    waveforms,
    spike_times,
    cluster_labels,
    sampling_frequency,
    output_folder,
    **kwargs,
):
    """
    Convenience function to export spike sorting results to PHY template-GUI format.

    Parameters
    ----------
    raw_data : da.Array
        The raw data as a dask array (samples x channels)
    waveforms : np.ndarray
        Extracted spike waveforms (spikes x samples x channels)
    spike_times : np.ndarray
        Spike times in samples
    cluster_labels : np.ndarray
        Cluster assignments for each spike
    sampling_frequency : float
        Sampling frequency in Hz
    output_folder : str or Path
        The output folder where PHY files will be saved
    **kwargs : dict
        Additional arguments to pass to PhyExporter.export_to_template_gui

    Returns
    -------
    dict
        Dictionary with paths to created files
    """
    exporter = PhyExporter(output_folder)
    return exporter.export_to_template_gui(
        raw_data=raw_data,
        waveforms=waveforms,
        spike_times=spike_times,
        cluster_labels=cluster_labels,
        sampling_frequency=sampling_frequency,
        **kwargs,
    )
