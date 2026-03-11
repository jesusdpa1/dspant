"""
Clean unified Phy/Kilosort loader with integrated spike analysis functionality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from rich.console import Console
from rich.table import Table

from dspant.core.internals import public_api


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
    """
    Unified loader for Phy/Kilosort spike sorting output with integrated analysis functionality.
    
    This class combines data loading with spike train analysis methods in a single, clean interface.
    """

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

        # Data attributes
        self.name: Optional[str] = None
        self.sampling_frequency: Optional[float] = None
        self.unit_ids: Optional[List[int]] = None
        self.spike_times: Optional[np.ndarray] = None
        self.spike_clusters: Optional[np.ndarray] = None
        self.unit_properties: Dict = {}
        self.metadata: Optional[Dict] = None
        self.templates_data: Optional[Dict] = None
        
        # Private attributes for efficient access
        self._spike_indices: Dict[int, np.ndarray] = {}
        self._channel_map: Optional[np.ndarray] = None
        self._templates: Optional[np.ndarray] = None
        self._is_loaded: bool = False

    def load_data(
        self,
        exclude_cluster_groups: Optional[Union[List[str], str]] = None,
        keep_good_only: bool = False,
        load_all_cluster_properties: bool = True,
        load_templates: bool = False,
        force_reload: bool = False,
    ) -> bool:
        """
        Load Phy/Kilosort data.

        Args:
            exclude_cluster_groups: Cluster groups to exclude (e.g., 'noise', ['noise', 'mua'])
            keep_good_only: Whether to only include 'good' units
            load_all_cluster_properties: Whether to load all cluster properties
            load_templates: Whether to load template information
            force_reload: Whether to reload even if data is already loaded

        Returns:
            True if data was loaded successfully
        """
        if self._is_loaded and not force_reload:
            return True

        try:
            # Load params file
            params = read_params_file(self.folder_path / "params.py")

            # Get sampling frequency
            self.sampling_frequency = params.get("sample_rate", None)

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

            # Set attributes
            self.unit_ids = unit_ids.tolist()
            self.spike_times = spike_times_clean
            self.spike_clusters = spike_clusters_clean

            # Add unit properties
            for col in cluster_info.columns:
                if col == "cluster_id":
                    continue

                # Create a dictionary mapping unit ID to property value
                prop_dict = {
                    uid: val
                    for uid, val in zip(cluster_info["cluster_id"], cluster_info[col])
                }
                self.unit_properties[col] = prop_dict

            # Load template information if requested
            if load_templates:
                self.templates_data = self._load_template_data()

            # Add metadata
            self.metadata = {
                "source": "phy_kilosort",
                "base": {
                    "name": "KiloSort output",
                    "sampling_frequency": self.sampling_frequency,
                    "unit_ids": self.unit_ids,
                    "data_shape": (len(spike_times_clean),),
                },
                "other": {
                    "params": params,
                    "exclude_cluster_groups": exclude_cluster_groups,
                    "keep_good_only": keep_good_only,
                    "templates_loaded": load_templates,
                    "available_templates": list(self.templates_data.keys())
                    if load_templates and self.templates_data
                    else [],
                },
            }

            # Build index for efficient spike access
            for uid in self.unit_ids:
                self._spike_indices[uid] = np.where(spike_clusters_clean == uid)[0]

            # Load channel information if templates are loaded
            if load_templates:
                self._load_channel_info()

            self._is_loaded = True
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}") from e

    def get_unit_spike_train(
        self,
        unit_id: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get spike times for a specific unit.

        Args:
            unit_id: Unit ID to retrieve
            start_frame: Optional start frame for filtering spikes
            end_frame: Optional end frame for filtering spikes

        Returns:
            Array of spike times in samples
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        if unit_id not in self.unit_ids:
            raise ValueError(f"Unit ID {unit_id} not found")

        # Use cached indices if available
        if unit_id not in self._spike_indices:
            self._spike_indices[unit_id] = np.where(self.spike_clusters == unit_id)[0]

        unit_indices = self._spike_indices[unit_id]
        unit_spikes = self.spike_times[unit_indices]

        # Apply time filter if requested
        if start_frame is not None or end_frame is not None:
            start = 0 if start_frame is None else start_frame
            end = np.inf if end_frame is None else end_frame

            mask = (unit_spikes >= start) & (unit_spikes < end)
            unit_spikes = unit_spikes[mask]

        return unit_spikes

    def get_unit_summary(self) -> pl.DataFrame:
        """
        Get a comprehensive summary of all units.

        Returns:
            Polars DataFrame with unit-level statistics
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Calculate basic statistics for each unit
        unit_stats = []

        for unit_id in self.unit_ids:
            # Get spike train for this unit
            spike_train = self.get_unit_spike_train(unit_id)

            # Calculate firing rate
            if self.sampling_frequency and len(spike_train) > 0:
                duration_s = np.max(self.spike_times) / self.sampling_frequency
                firing_rate = len(spike_train) / duration_s
            else:
                firing_rate = 0.0

            # Calculate ISI statistics
            if len(spike_train) > 1:
                isis = np.diff(spike_train) / self.sampling_frequency  # Convert to seconds
                mean_isi = np.mean(isis)
                median_isi = np.median(isis)
                cv_isi = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else 0

                # Calculate refractory period violations (ISI < 2ms)
                refrac_violations = np.sum(isis < 0.002)
                refrac_violation_rate = refrac_violations / len(isis) * 100
            else:
                mean_isi = median_isi = cv_isi = 0.0
                refrac_violations = 0
                refrac_violation_rate = 0.0

            # Get unit properties
            unit_props = {
                "unit_id": unit_id,
                "n_spikes": len(spike_train),
                "firing_rate_hz": firing_rate,
                "mean_isi_s": mean_isi,
                "median_isi_s": median_isi,
                "cv_isi": cv_isi,
                "refrac_violations": refrac_violations,
                "refrac_violation_rate_pct": refrac_violation_rate,
            }

            # Add unit properties from the loader
            for prop_name, prop_dict in self.unit_properties.items():
                if unit_id in prop_dict:
                    unit_props[prop_name] = prop_dict[unit_id]

            unit_stats.append(unit_props)

        return pl.DataFrame(unit_stats)

    def get_channel_summary(self) -> pl.DataFrame:
        """
        Get a summary of units per channel.

        Returns:
            Polars DataFrame with channel-level statistics
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Load channel map and templates if not already loaded
        if self._channel_map is None:
            self._load_channel_info()

        # Get unit summary first
        unit_df = self.get_unit_summary()

        # If we have template data, we can determine primary channels
        if self._templates is not None and self._channel_map is not None:
            # Find primary channel for each unit (channel with max amplitude)
            unit_channels = []

            for unit_id in self.unit_ids:
                if unit_id < len(self._templates):
                    # Get template for this unit
                    template = self._templates[unit_id]

                    # Find channel with maximum absolute amplitude
                    max_amp_per_channel = np.max(np.abs(template), axis=0)
                    primary_channel = np.argmax(max_amp_per_channel)

                    unit_channels.append(
                        {
                            "unit_id": unit_id,
                            "primary_channel": int(self._channel_map[primary_channel])
                            if self._channel_map is not None
                            else primary_channel,
                            "max_amplitude": float(max_amp_per_channel[primary_channel]),
                        }
                    )

            # Convert to DataFrame and join with unit summary
            channel_df = pl.DataFrame(unit_channels)
            unit_df = unit_df.join(channel_df, on="unit_id", how="left")

        # Group by channel to get channel-level statistics
        if "primary_channel" in unit_df.columns:
            channel_summary = (
                unit_df.group_by("primary_channel")
                .agg(
                    [
                        pl.count("unit_id").alias("n_units"),
                        pl.col("KSLabel")
                        .filter(pl.col("KSLabel") == "good")
                        .count()
                        .alias("n_good_units"),
                        pl.col("KSLabel")
                        .filter(pl.col("KSLabel") == "mua")
                        .count()
                        .alias("n_mua_units"),
                        pl.col("KSLabel")
                        .filter(pl.col("KSLabel").is_null())
                        .count()
                        .alias("n_unknown_units"),
                        pl.col("firing_rate_hz").mean().alias("mean_firing_rate_hz"),
                        pl.col("Amplitude").mean().alias("mean_amplitude")
                        if "Amplitude" in unit_df.columns
                        else pl.lit(None).alias("mean_amplitude"),
                        pl.col("ContamPct").mean().alias("mean_contamination_pct")
                        if "ContamPct" in unit_df.columns
                        else pl.lit(None).alias("mean_contamination_pct"),
                        pl.col("n_spikes").sum().alias("total_spikes"),
                    ]
                )
                .sort("primary_channel")
            )
        else:
            # If no channel information available, create a summary without channel grouping
            channel_summary = pl.DataFrame(
                {
                    "primary_channel": [None],
                    "n_units": [len(unit_df)],
                    "n_good_units": [len(unit_df.filter(pl.col("KSLabel") == "good"))],
                    "n_mua_units": [len(unit_df.filter(pl.col("KSLabel") == "mua"))],
                    "n_unknown_units": [len(unit_df.filter(pl.col("KSLabel").is_null()))],
                    "mean_firing_rate_hz": [unit_df["firing_rate_hz"].mean()],
                    "mean_amplitude": [
                        unit_df["Amplitude"].mean()
                        if "Amplitude" in unit_df.columns
                        else None
                    ],
                    "mean_contamination_pct": [
                        unit_df["ContamPct"].mean()
                        if "ContamPct" in unit_df.columns
                        else None
                    ],
                    "total_spikes": [unit_df["n_spikes"].sum()],
                }
            )

        return channel_summary

    def get_quality_summary(self) -> pl.DataFrame:
        """
        Get a summary of unit quality metrics.

        Returns:
            Polars DataFrame with quality-based statistics
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        unit_df = self.get_unit_summary()

        # Group by quality label
        if "KSLabel" in unit_df.columns:
            quality_summary = (
                unit_df.group_by("KSLabel")
                .agg(
                    [
                        pl.count("unit_id").alias("n_units"),
                        pl.col("firing_rate_hz").mean().alias("mean_firing_rate_hz"),
                        pl.col("firing_rate_hz").std().alias("std_firing_rate_hz"),
                        pl.col("firing_rate_hz").min().alias("min_firing_rate_hz"),
                        pl.col("firing_rate_hz").max().alias("max_firing_rate_hz"),
                        pl.col("Amplitude").mean().alias("mean_amplitude")
                        if "Amplitude" in unit_df.columns
                        else pl.lit(None).alias("mean_amplitude"),
                        pl.col("Amplitude").std().alias("std_amplitude")
                        if "Amplitude" in unit_df.columns
                        else pl.lit(None).alias("std_amplitude"),
                        pl.col("ContamPct").mean().alias("mean_contamination_pct")
                        if "ContamPct" in unit_df.columns
                        else pl.lit(None).alias("mean_contamination_pct"),
                        pl.col("ContamPct").std().alias("std_contamination_pct")
                        if "ContamPct" in unit_df.columns
                        else pl.lit(None).alias("std_contamination_pct"),
                        pl.col("cv_isi").mean().alias("mean_cv_isi"),
                        pl.col("refrac_violation_rate_pct")
                        .mean()
                        .alias("mean_refrac_violation_rate_pct"),
                        pl.col("n_spikes").sum().alias("total_spikes"),
                        pl.col("n_spikes").mean().alias("mean_spikes_per_unit"),
                    ]
                )
                .sort("KSLabel")
            )
        else:
            # If no quality labels, create overall summary
            quality_summary = pl.DataFrame(
                {
                    "KSLabel": ["all"],
                    "n_units": [len(unit_df)],
                    "mean_firing_rate_hz": [unit_df["firing_rate_hz"].mean()],
                    "std_firing_rate_hz": [unit_df["firing_rate_hz"].std()],
                    "min_firing_rate_hz": [unit_df["firing_rate_hz"].min()],
                    "max_firing_rate_hz": [unit_df["firing_rate_hz"].max()],
                    "mean_amplitude": [
                        unit_df["Amplitude"].mean()
                        if "Amplitude" in unit_df.columns
                        else None
                    ],
                    "std_amplitude": [
                        unit_df["Amplitude"].std()
                        if "Amplitude" in unit_df.columns
                        else None
                    ],
                    "mean_contamination_pct": [
                        unit_df["ContamPct"].mean()
                        if "ContamPct" in unit_df.columns
                        else None
                    ],
                    "std_contamination_pct": [
                        unit_df["ContamPct"].std()
                        if "ContamPct" in unit_df.columns
                        else None
                    ],
                    "mean_cv_isi": [unit_df["cv_isi"].mean()],
                    "mean_refrac_violation_rate_pct": [
                        unit_df["refrac_violation_rate_pct"].mean()
                    ],
                    "total_spikes": [unit_df["n_spikes"].sum()],
                    "mean_spikes_per_unit": [unit_df["n_spikes"].mean()],
                }
            )

        return quality_summary

    def get_recording_summary(self) -> pl.DataFrame:
        """
        Get overall recording summary statistics.

        Returns:
            Polars DataFrame with recording-level statistics
        """
        if not self._is_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        # Calculate recording duration
        if self.sampling_frequency and len(self.spike_times) > 0:
            duration_s = np.max(self.spike_times) / self.sampling_frequency
            duration_min = duration_s / 60
        else:
            duration_s = duration_min = 0.0

        # Get unit summary for aggregation
        unit_df = self.get_unit_summary()

        # Calculate overall statistics
        recording_stats = {
            "recording_duration_s": duration_s,
            "recording_duration_min": duration_min,
            "sampling_frequency_hz": self.sampling_frequency,
            "total_units": len(self.unit_ids),
            "total_spikes": len(self.spike_times),
            "mean_firing_rate_hz": unit_df["firing_rate_hz"].mean(),
            "std_firing_rate_hz": unit_df["firing_rate_hz"].std(),
            "spikes_per_second": len(self.spike_times) / duration_s if duration_s > 0 else 0,
        }

        # Add quality-based counts if available
        if "KSLabel" in unit_df.columns:
            recording_stats.update(
                {
                    "n_good_units": len(unit_df.filter(pl.col("KSLabel") == "good")),
                    "n_mua_units": len(unit_df.filter(pl.col("KSLabel") == "mua")),
                    "n_noise_units": len(unit_df.filter(pl.col("KSLabel") == "noise")),
                    "pct_good_units": len(unit_df.filter(pl.col("KSLabel") == "good"))
                    / len(unit_df)
                    * 100,
                }
            )

        # Add amplitude and contamination stats if available
        if "Amplitude" in unit_df.columns:
            recording_stats.update(
                {
                    "mean_amplitude": unit_df["Amplitude"].mean(),
                    "std_amplitude": unit_df["Amplitude"].std(),
                }
            )

        if "ContamPct" in unit_df.columns:
            recording_stats.update(
                {
                    "mean_contamination_pct": unit_df["ContamPct"].mean(),
                    "std_contamination_pct": unit_df["ContamPct"].std(),
                }
            )

        return pl.DataFrame([recording_stats])

    def summarize(self):
        """Print a summary of the loader configuration and metadata."""
        console = Console()

        # Create main table
        table = Table(title="PhyKilosort Loader Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.folder_path))
        table.add_row("Data Loaded", "Yes" if self._is_loaded else "No")

        # Add metadata information if loaded
        if self._is_loaded and self.metadata:
            table.add_section()
            table.add_row("Name", str(self.name) if self.name else "KiloSort output")
            table.add_row(
                "Sampling Rate",
                f"{self.sampling_frequency} Hz"
                if self.sampling_frequency
                else "Not set",
            )

            if self.unit_ids:
                table.add_row("Number of Units", str(len(self.unit_ids)))

            # Add unit properties summary if available
            if self.unit_properties:
                props_str = ", ".join(list(self.unit_properties.keys()))
                table.add_row("Unit Properties", props_str)

        # Add data information if loaded
        if self._is_loaded and self.spike_times is not None and self.spike_clusters is not None:
            table.add_section()
            table.add_row("Number of Spikes", str(len(self.spike_times)))

            # Calculate mean firing rates if possible
            if self.sampling_frequency:
                duration_s = np.max(self.spike_times) / self.sampling_frequency
                spikes_per_unit = {
                    uid: np.sum(self.spike_clusters == uid) for uid in self.unit_ids
                }

                mean_fr = np.mean(
                    [spikes / duration_s for spikes in spikes_per_unit.values()]
                )
                max_fr = np.max(
                    [spikes / duration_s for spikes in spikes_per_unit.values()]
                )

                table.add_row("Recording Duration", f"{duration_s:.2f} s")
                table.add_row("Mean Firing Rate", f"{mean_fr:.2f} Hz")
                table.add_row("Max Firing Rate", f"{max_fr:.2f} Hz")

        console.print(table)

    def _load_channel_info(self):
        """Load channel map and template information for channel-based analysis."""
        # Load channel map
        channel_map_file = self.folder_path / "channel_map.npy"
        if channel_map_file.exists():
            self._channel_map = np.load(channel_map_file)

        # Load templates
        templates_file = self.folder_path / "templates.npy"
        if templates_file.exists():
            self._templates = np.load(templates_file)

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
) -> PhyKilosortLoader:
    """
    Load Kilosort spike sorting output.

    Args:
        folder_path: Path to Kilosort output folder
        exclude_cluster_groups: Cluster groups to exclude (e.g., 'noise', ['noise', 'mua'])
        keep_good_only: Whether to only include 'good' units
        load_templates: Whether to load template information

    Returns:
        PhyKilosortLoader with loaded data
    """
    loader = PhyKilosortLoader(folder_path)
    loader.load_data(
        exclude_cluster_groups=exclude_cluster_groups,
        keep_good_only=keep_good_only,
        load_templates=load_templates,
    )
    return loader


@public_api
def load_phy(
    folder_path: Union[str, Path],
    exclude_cluster_groups: Optional[Union[List[str], str]] = None,
    load_templates: bool = False,
) -> PhyKilosortLoader:
    """
    Load Phy spike sorting output.

    Args:
        folder_path: Path to Phy output folder
        exclude_cluster_groups: Cluster groups to exclude (e.g., 'noise', ['noise', 'mua'])
        load_templates: Whether to load template information

    Returns:
        PhyKilosortLoader with loaded data
    """
    loader = PhyKilosortLoader(folder_path)
    loader.load_data(
        exclude_cluster_groups=exclude_cluster_groups,
        keep_good_only=False,
        load_templates=load_templates,
    )
    return loader