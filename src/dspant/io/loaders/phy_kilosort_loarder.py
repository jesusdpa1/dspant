"""
Loaders for Phy/Kilosort spike sorting output.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dspant.core.internals import public_api
from dspant.nodes.sorter import SorterNode


@public_api
def read_params_file(params_file: Union[str, Path]) -> Dict:
    """
    Read a params.py file and extract parameters.

    Args:
        params_file: Path to params.py file

    Returns:
        Dictionary of parameters
    """
    params = {}

    # Convert to Path object
    params_path = Path(params_file)

    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    # Read the file and parse parameters
    with open(params_path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                try:
                    # Split at the first equals sign
                    key, value = [x.strip() for x in line.split("=", 1)]

                    # Strip any trailing comments
                    if "#" in value:
                        value = value.split("#", 1)[0].strip()

                    # Remove any quotes
                    if value.startswith('"') or value.startswith("'"):
                        value = value[1:-1]

                    # Try to convert to appropriate type
                    try:
                        if value.lower() == "true":
                            params[key] = True
                        elif value.lower() == "false":
                            params[key] = False
                        elif value.lower() == "none":
                            params[key] = None
                        elif "." in value or "e" in value.lower():
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string
                        params[key] = value
                except Exception as e:
                    print(f"Error parsing line: {line}, Error: {e}")

    return params


@public_api
class PhyKilosortLoader:
    """Loader for Phy/Kilosort spike sorting output."""

    def __init__(self, folder_path: Union[str, Path]):
        """
        Initialize the loader.

        Args:
            folder_path: Path to Phy/Kilosort output folder
        """
        self.folder_path = Path(folder_path)

        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")

        # Check for required files
        req_files = ["params.py", "spike_times.npy", "spike_clusters.npy"]
        missing = [f for f in req_files if not (self.folder_path / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files: {', '.join(missing)}")

    def load_into_node(
        self,
        exclude_cluster_groups: Optional[Union[List[str], str]] = None,
        keep_good_only: bool = False,
        load_all_cluster_properties: bool = True,
        load_templates: bool = False,
    ) -> SorterNode:
        """
        Load Phy/Kilosort data into a SorterNode.

        Args:
            exclude_cluster_groups: Cluster groups to exclude (e.g., 'noise', ['noise', 'mua'])
            keep_good_only: Whether to only include 'good' units
            load_all_cluster_properties: Whether to load all cluster properties
            load_templates: Whether to load template information

        Returns:
            SorterNode containing the loaded data
        """
        # Create a SorterNode with the folder path
        node = SorterNode(str(self.folder_path))

        # Load params file
        params = read_params_file(self.folder_path / "params.py")

        # Get sampling frequency
        sampling_frequency = params.get("sample_rate", None)

        # Load spike times and clusters
        spike_times = np.load(self.folder_path / "spike_times.npy").astype(int)

        # Choose spike_clusters if available, fall back to spike_templates
        clusters_file = (
            "spike_clusters.npy"
            if (self.folder_path / "spike_clusters.npy").exists()
            else "spike_templates.npy"
        )
        spike_clusters = np.load(self.folder_path / clusters_file)

        # Ensure 1D arrays
        spike_times = np.atleast_1d(spike_times.squeeze())
        spike_clusters = np.atleast_1d(spike_clusters.squeeze())

        # Get cluster info from CSV/TSV files
        cluster_info = self._load_cluster_info(load_all_cluster_properties)

        # Filter clusters based on criteria
        if exclude_cluster_groups is not None:
            if isinstance(exclude_cluster_groups, str):
                cluster_info = cluster_info.query(
                    f"group != '{exclude_cluster_groups}'"
                )
            elif isinstance(exclude_cluster_groups, list):
                for exclude_group in exclude_cluster_groups:
                    cluster_info = cluster_info.query(f"group != '{exclude_group}'")

        if keep_good_only and "KSLabel" in cluster_info.columns:
            cluster_info = cluster_info.query("KSLabel == 'good'")

        # Get final list of unit IDs from cluster info
        unit_ids = cluster_info["cluster_id"].values.astype(int)

        # Filter spike data to only include selected units
        mask = np.isin(spike_clusters, unit_ids)
        spike_times_clean = spike_times[mask]
        spike_clusters_clean = spike_clusters[mask]

        # Set node attributes
        node.sampling_frequency = sampling_frequency
        node.unit_ids = unit_ids.tolist()
        node.spike_times = spike_times_clean
        node.spike_clusters = spike_clusters_clean

        # Add unit properties
        for col in cluster_info.columns:
            if col == "cluster_id":
                continue

            # Create a dictionary mapping unit ID to property value
            prop_dict = {
                uid: val
                for uid, val in zip(cluster_info["cluster_id"], cluster_info[col])
            }
            node.unit_properties[col] = prop_dict

        # Load template information if requested
        if load_templates:
            templates_data = self._load_template_data()
            node.templates_data = templates_data

        # Add metadata from params file
        node.metadata = {
            "source": "phy_kilosort",
            "base": {
                "name": "KiloSort output",
                "sampling_frequency": sampling_frequency,
                "unit_ids": unit_ids.tolist(),
                "data_shape": (len(spike_times_clean),),
            },
            "other": {
                "params": params,
                "exclude_cluster_groups": exclude_cluster_groups,
                "keep_good_only": keep_good_only,
                "templates_loaded": load_templates,
                "available_templates": list(templates_data.keys())
                if load_templates
                else [],
            },
        }

        # Build index for efficient spike access
        for uid in unit_ids:
            node._spike_indices[uid] = np.where(spike_clusters_clean == uid)[0]

        return node

    def _load_template_data(self) -> Dict[str, np.ndarray]:
        """
        Load all available template-related data files.

        Returns:
            Dictionary containing template data arrays
        """
        template_data = {}

        # Define template-related files to load with optional status
        template_files = {
            "templates": True,  # Required
            "templates_ind": False,  # Optional
            "similar_templates": False,  # Optional
            "spike_templates": True,  # Required
            "channel_map": True,  # Required
            "channel_positions": False,  # Optional
            "pc_features": False,  # Optional
            "pc_feature_ind": False,  # Optional
            "amplitudes": False,  # Optional
        }

        for file_name, is_required in template_files.items():
            file_path = self.folder_path / f"{file_name}.npy"

            if file_path.exists():
                try:
                    data = np.load(file_path)
                    template_data[file_name] = data
                    print(
                        f"✅ Loaded {file_name}: shape {data.shape}, dtype {data.dtype}"
                    )
                except Exception as e:
                    print(f"❌ Error loading {file_name}: {e}")
            else:
                if is_required:
                    print(f"⚠️ Required template file not found: {file_name}")
                else:
                    print(f"ℹ️ Optional template file not found: {file_name}")

        # Load channel metadata if available
        for file_name in ["channel_shanks"]:
            file_path = self.folder_path / f"{file_name}.npy"
            if file_path.exists():
                try:
                    data = np.load(file_path)
                    template_data[file_name] = data
                    print(f"✅ Loaded {file_name}: shape {data.shape}")
                except Exception as e:
                    print(f"❌ Error loading {file_name}: {e}")

        return template_data

    def _load_cluster_info(self, load_all_properties: bool = True) -> pd.DataFrame:
        """
        Load cluster information from tsv/csv files.

        Args:
            load_all_properties: Whether to load all cluster properties

        Returns:
            DataFrame containing cluster information
        """
        # Look for cluster_info file first
        cluster_info_files = [
            p
            for p in self.folder_path.iterdir()
            if p.suffix in [".csv", ".tsv"] and "cluster_info" in p.name
        ]

        if len(cluster_info_files) == 1:
            # Load from cluster_info file
            cluster_info_file = cluster_info_files[0]
            delimiter = "\t" if cluster_info_file.suffix == ".tsv" else ","
            cluster_info = pd.read_csv(cluster_info_file, delimiter=delimiter)
        else:
            # Load from individual property files
            all_property_files = [
                p for p in self.folder_path.iterdir() if p.suffix in [".csv", ".tsv"]
            ]

            # Start with cluster_group.tsv if available
            cluster_group_file = self.folder_path / "cluster_group.tsv"
            if cluster_group_file.exists():
                cluster_info = pd.read_csv(cluster_group_file, delimiter="\t")
            else:
                # Fall back to other files or create minimal info
                cluster_ids = np.unique(
                    np.load(self.folder_path / "spike_clusters.npy")
                )
                cluster_info = pd.DataFrame({"cluster_id": cluster_ids})

            # Add data from other property files if requested
            if load_all_properties:
                for file in all_property_files:
                    if file == cluster_group_file:
                        continue

                    delimiter = "\t" if file.suffix == ".tsv" else ","

                    try:
                        new_property = pd.read_csv(file, delimiter=delimiter)
                        if "cluster_id" in new_property.columns:
                            cluster_info = pd.merge(
                                cluster_info,
                                new_property,
                                on="cluster_id",
                                suffixes=[None, "_repeat"],
                                how="left",
                            )
                    except Exception as e:
                        print(f"Error loading property file {file}: {e}")

        # Ensure we have a 'group' column for compatibility
        if "group" not in cluster_info.columns:
            cluster_info["group"] = "unsorted"

        # Standardize column names if needed
        if "id" in cluster_info.columns and "cluster_id" not in cluster_info.columns:
            cluster_info["cluster_id"] = cluster_info["id"]

        return cluster_info


@public_api
def load_kilosort(
    folder_path: Union[str, Path],
    exclude_cluster_groups: Optional[Union[List[str], str]] = None,
    keep_good_only: bool = False,
    load_templates: bool = False,
) -> SorterNode:
    """
    Load Kilosort spike sorting output into a SorterNode.

    Args:
        folder_path: Path to Kilosort output folder
        exclude_cluster_groups: Cluster groups to exclude (e.g., 'noise', ['noise', 'mua'])
        keep_good_only: Whether to only include 'good' units
        load_templates: Whether to load template information

    Returns:
        SorterNode containing the loaded data
    """
    loader = PhyKilosortLoader(folder_path)
    return loader.load_into_node(
        exclude_cluster_groups=exclude_cluster_groups,
        keep_good_only=keep_good_only,
        load_templates=load_templates,
    )


@public_api
def load_phy(
    folder_path: Union[str, Path],
    exclude_cluster_groups: Optional[Union[List[str], str]] = None,
    load_templates: bool = False,
) -> SorterNode:
    """
    Load Phy spike sorting output into a SorterNode.

    Args:
        folder_path: Path to Phy output folder
        exclude_cluster_groups: Cluster groups to exclude (e.g., 'noise', ['noise', 'mua'])
        load_templates: Whether to load template information

    Returns:
        SorterNode containing the loaded data
    """
    loader = PhyKilosortLoader(folder_path)
    return loader.load_into_node(
        exclude_cluster_groups=exclude_cluster_groups,
        keep_good_only=False,
        load_templates=load_templates,
    )
