"""
CEBRA implementation for neural data from multiple diaphragm regions
"""

import os
from pathlib import Path

# Import for CEBRA
import cebra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cebra.models import CEBRA
from dotenv import load_dotenv
from matplotlib.colors import ListedColormap
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Import your data processing modules
from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)

# Set style
sns.set_theme(style="darkgrid")
load_dotenv()

# Step 1: Load data
print("Loading data...")
data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/RawG.ant")
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)

# Load EMG and spike data
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
sorter_data = load_kilosort(sorter_path)
fs = stream_emg.fs

# Step 2: Process EMG data
print("Processing EMG data...")
# Create a processing node
processor_emg = create_processing_node(stream_emg)

# Create and apply filters
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)

notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
filtered_emg = processor_emg.process(group=["filters"]).persist()

# Apply TKEO for envelope extraction
print("Extracting EMG envelope...")
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=15)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()

# Normalize TKEO
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()

# Step 3: Prepare spike data with spatial information
print("Preparing spike data with spatial regions...")

# Set up parameters
max_time_s = 300  # 5 minutes for faster processing
bin_size_ms = 10  # 10ms bins
bin_size_s = bin_size_ms / 1000.0


# Define diaphragm regions (example - modify based on your actual electrode arrangement)
# For this example, we'll simulate 3 regions: ventral, lateral, and dorsal
# You should replace this with your actual regional classification of neurons
def assign_regions_to_units(unit_ids, sorter_data):
    """
    Assign diaphragm regions to units based on your knowledge of electrode placement.
    This is a placeholder function - modify to match your actual experiment setup.

    Returns:
        unit_regions: dict mapping unit_id to region name
        unit_coordinates: dict mapping unit_id to (x,y) coordinates
    """
    # In a real implementation, you would use actual regional classifications
    # Here we're simulating by assigning regions randomly for demonstration
    np.random.seed(42)  # For reproducibility

    regions = ["ventral", "lateral", "dorsal"]
    unit_regions = {}
    unit_coordinates = {}

    # Example: Assign regions based on unit ID ranges
    # You would replace this with actual knowledge about your electrode array
    for unit_id in unit_ids:
        # Simulate regional assignment - replace with your actual assignments
        if unit_id % 3 == 0:
            region = "ventral"
            # Simulate ventral coordinates (lower diaphragm)
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 3)
        elif unit_id % 3 == 1:
            region = "lateral"
            # Simulate lateral coordinates (side of diaphragm)
            x = np.random.uniform(8, 15)
            y = np.random.uniform(3, 8)
        else:
            region = "dorsal"
            # Simulate dorsal coordinates (upper diaphragm)
            x = np.random.uniform(0, 8)
            y = np.random.uniform(6, 10)

        unit_regions[unit_id] = region
        unit_coordinates[unit_id] = (x, y)

    return unit_regions, unit_coordinates


# Function to create binned spikes with regional information
def create_regional_binned_spikes(sorter_data, bin_size_ms=10, max_time_s=None):
    """Create time-binned spike counts from Kilosort data, organized by diaphragm region."""
    # Convert spike times to seconds
    spike_times = sorter_data["spike_times"] / sorter_data["sampling_frequency"]
    spike_clusters = sorter_data["spike_clusters"]

    # Get unique units
    unit_ids = np.unique(spike_clusters)

    # Assign regions to units
    unit_regions, unit_coordinates = assign_regions_to_units(unit_ids, sorter_data)

    # Determine max time
    if max_time_s is None:
        max_time_s = np.ceil(spike_times.max())

    # Create time bins
    bin_size_s = bin_size_ms / 1000
    n_bins = int(np.ceil(max_time_s / bin_size_s))
    bin_edges = np.arange(0, max_time_s + bin_size_s, bin_size_s)

    # Get unique regions
    all_regions = sorted(set(unit_regions.values()))

    # Create dictionary to hold binned spikes by region
    region_spikes = {region: [] for region in all_regions}
    region_unit_ids = {region: [] for region in all_regions}

    # Create binned spike counts for all units
    all_binned_spikes = np.zeros((n_bins, len(unit_ids)))

    for i, unit_id in enumerate(unit_ids):
        unit_spikes = spike_times[spike_clusters == unit_id]
        hist, _ = np.histogram(unit_spikes, bins=bin_edges)
        all_binned_spikes[: len(hist), i] = hist

    # Group units by region
    for i, unit_id in enumerate(unit_ids):
        region = unit_regions[unit_id]
        region_spikes[region].append(all_binned_spikes[:, i])
        region_unit_ids[region].append(unit_id)

    # Convert lists to arrays for each region
    for region in all_regions:
        if region_spikes[region]:
            region_spikes[region] = np.column_stack(region_spikes[region])
        else:
            # Empty array if no units in this region
            region_spikes[region] = np.zeros((n_bins, 0))

    return (
        all_binned_spikes,
        unit_ids,
        unit_regions,
        unit_coordinates,
        region_spikes,
        region_unit_ids,
        bin_size_s,
    )


# Create regional binned spike data
(
    all_neural_data,
    unit_ids,
    unit_regions,
    unit_coordinates,
    region_spikes,
    region_unit_ids,
    bin_size_s,
) = create_regional_binned_spikes(
    sorter_data, bin_size_ms=bin_size_ms, max_time_s=max_time_s
)

# Get unique regions
all_regions = sorted(set(unit_regions.values()))
print(f"Identified {len(all_regions)} diaphragm regions: {', '.join(all_regions)}")

for region in all_regions:
    n_units = region_spikes[region].shape[1] if region_spikes[region].size > 0 else 0
    print(f"  Region '{region}': {n_units} units")

# Step 4: Align EMG data to neural data timebase
print("Aligning EMG data...")
# Get EMG data (first channel)
emg_raw = zscore_tkeo[: int(max_time_s * fs), 0].compute()

# Create time points
emg_times = np.arange(0, len(emg_raw) / fs, 1 / fs)[: len(emg_raw)]
neural_times = np.arange(0, max_time_s, bin_size_s)[: all_neural_data.shape[0]]

# Resample EMG to match neural data timebase
f = interpolate.interp1d(
    emg_times, emg_raw, bounds_error=False, fill_value="extrapolate"
)
emg_aligned = f(neural_times)

# Step 5: Train CEBRA models incorporating regional information
print("Training CEBRA models with regional information...")

# Approach 1: Train a single CEBRA model on all units
# We'll include region labels as a categorical variable in our visualization
model_all = CEBRA(
    model_architecture="offset10-model",
    batch_size=512,
    max_iterations=1000,
    learning_rate=0.001,
    temperature=1.0,
    output_dimension=3,
    verbose=True,
)

embedding_all = model_all.fit_transform(all_neural_data, emg_aligned.reshape(-1, 1))
print(f"All-units embedding shape: {embedding_all.shape}")

# Approach 2: Train separate CEBRA models for each region
embeddings_by_region = {}
models_by_region = {}

for region in all_regions:
    if region_spikes[region].shape[1] > 0:  # If we have units in this region
        print(f"Training CEBRA model for {region} region...")
        model_region = CEBRA(
            model_architecture="offset10-model",
            batch_size=512,
            max_iterations=1000,
            learning_rate=0.001,
            temperature=1.0,
            output_dimension=3,
            verbose=True,
        )

        embedding_region = model_region.fit_transform(
            region_spikes[region], emg_aligned.reshape(-1, 1)
        )

        embeddings_by_region[region] = embedding_region
        models_by_region[region] = model_region
        print(f"  {region} embedding shape: {embedding_region.shape}")

# Step 6: Visualize embeddings with regional information
print("Creating visualizations...")

# Assign colors to regions
region_colors = {"ventral": "blue", "lateral": "green", "dorsal": "red"}

# Create a plot for the combined model with units colored by region
plt.figure(figsize=(15, 5))

# 2D embedding with region coloring
ax1 = plt.subplot(1, 3, 1)
region_scatter_handles = []

# Need to map unit indices in all_neural_data to their regions
unit_region_list = []
for unit_id in unit_ids:
    unit_region_list.append(unit_regions[unit_id])

# Create a categorical colormap
unique_regions = sorted(set(unit_region_list))
cmap = ListedColormap([region_colors[r] for r in unique_regions])
region_indices = np.array([unique_regions.index(r) for r in unit_region_list])

# Create custom embedding that combines multiple neurons' activity
# This creates a "population embedding" for visualization
pop_embedding = np.zeros((all_neural_data.shape[0], len(unique_regions)))
for t in range(all_neural_data.shape[0]):
    for i, region_idx in enumerate(region_indices):
        pop_embedding[t, region_idx] += all_neural_data[t, i]

# Plot first 2 dimensions of embedding with points colored by region
for i, region in enumerate(unique_regions):
    # Get indices of units from this region
    region_unit_indices = [j for j, r in enumerate(unit_region_list) if r == region]

    # Calculate mean activity for each region at each time point
    region_activity = np.mean(all_neural_data[:, region_unit_indices], axis=1)

    # Use size of point to represent activity level
    sizes = 10 + 20 * (
        region_activity / np.max(region_activity) if np.max(region_activity) > 0 else 0
    )

    # Create scatter plot
    scatter = ax1.scatter(
        embedding_all[:, 0],
        embedding_all[:, 1],
        c=region_colors[region],
        alpha=0.7,
        s=sizes,
        label=region,
    )
    region_scatter_handles.append(scatter)

ax1.set_title("CEBRA Embedding - All Units\nColored by Region")
ax1.set_xlabel("Dimension 1")
ax1.set_ylabel("Dimension 2")
ax1.legend()

# 2D embedding with EMG amplitude coloring
ax2 = plt.subplot(1, 3, 2)
scatter = ax2.scatter(
    embedding_all[:, 0],
    embedding_all[:, 1],
    c=emg_aligned,
    cmap="viridis",
    alpha=0.7,
    s=3,
)
plt.colorbar(scatter, ax=ax2, label="EMG Amplitude")
ax2.set_title("CEBRA Embedding - All Units\nColored by EMG Amplitude")
ax2.set_xlabel("Dimension 1")
ax2.set_ylabel("Dimension 2")

# 3D embedding with region coloring
ax3 = plt.subplot(1, 3, 3, projection="3d")

for i, region in enumerate(unique_regions):
    region_unit_indices = [j for j, r in enumerate(unit_region_list) if r == region]
    region_activity = np.mean(all_neural_data[:, region_unit_indices], axis=1)
    sizes = 10 + 20 * (
        region_activity / np.max(region_activity) if np.max(region_activity) > 0 else 0
    )

    scatter = ax3.scatter(
        embedding_all[:, 0],
        embedding_all[:, 1],
        embedding_all[:, 2],
        c=region_colors[region],
        alpha=0.7,
        s=sizes,
        label=region,
    )

ax3.set_title("CEBRA 3D Embedding\nColored by Region")
ax3.set_xlabel("Dimension 1")
ax3.set_ylabel("Dimension 2")
ax3.set_zlabel("Dimension 3")
ax3.legend()

plt.tight_layout()
plt.savefig("cebra_regional_analysis.png", dpi=300)
plt.show()

# Plot separate regional models comparison
if len(embeddings_by_region) > 1:  # Only if we have multiple regions with embeddings
    plt.figure(figsize=(15, 5 * len(embeddings_by_region)))

    for i, (region, embedding) in enumerate(embeddings_by_region.items()):
        # EMG colored embedding for this region
        ax1 = plt.subplot(len(embeddings_by_region), 3, i * 3 + 1)
        scatter = ax1.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=emg_aligned,
            cmap="viridis",
            alpha=0.7,
            s=3,
        )
        plt.colorbar(scatter, ax=ax1, label="EMG Amplitude")
        ax1.set_title(f"{region.capitalize()} Region Embedding\nEMG Colored")
        ax1.set_xlabel("Dimension 1")
        ax1.set_ylabel("Dimension 2")

        # Time colored embedding
        ax2 = plt.subplot(len(embeddings_by_region), 3, i * 3 + 2)
        scatter = ax2.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=np.arange(len(embedding)),
            cmap="plasma",
            alpha=0.7,
            s=3,
        )
        plt.colorbar(scatter, ax=ax2, label="Time")
        ax2.set_title(f"{region.capitalize()} Region Embedding\nTime Colored")
        ax2.set_xlabel("Dimension 1")
        ax2.set_ylabel("Dimension 2")

        # 3D embedding
        ax3 = plt.subplot(len(embeddings_by_region), 3, i * 3 + 3, projection="3d")
        scatter = ax3.scatter(
            embedding[:, 0],
            embedding[:, 1],
            embedding[:, 2],
            c=emg_aligned,
            cmap="viridis",
            alpha=0.7,
            s=3,
        )
        plt.colorbar(scatter, ax=ax3, label="EMG Amplitude")
        ax3.set_title(f"{region.capitalize()} Region 3D Embedding")
        ax3.set_xlabel("Dimension 1")
        ax3.set_ylabel("Dimension 2")
        ax3.set_zlabel("Dimension 3")

    plt.tight_layout()
    plt.savefig("cebra_regional_models.png", dpi=300)
    plt.show()

# Step 7: Evaluate prediction performance by region
print("Evaluating regional prediction performance...")


# Function to evaluate EMG prediction from embedding
def evaluate_emg_prediction(embedding, emg_data, test_size=0.3):
    """Evaluate how well an embedding predicts EMG signal."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        embedding, emg_data, test_size=test_size, random_state=42
    )

    # Train KNN regressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = knn.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return r2, y_pred, y_test


# Evaluate combined model
r2_all, y_pred_all, y_test_all = evaluate_emg_prediction(embedding_all, emg_aligned)
print(f"EMG prediction R² from all units: {r2_all:.3f}")

# Evaluate each regional model
if embeddings_by_region:
    print("Regional EMG prediction performance:")
    for region, embedding in embeddings_by_region.items():
        r2, _, _ = evaluate_emg_prediction(embedding, emg_aligned)
        print(f"  {region.capitalize()} region: R² = {r2:.3f}")

# Step 8: Cross-prediction between regions
if len(embeddings_by_region) > 1:
    print("\nCross-region prediction analysis:")

    # Create a cross-prediction matrix
    regions = list(embeddings_by_region.keys())
    cross_r2 = np.zeros((len(regions), len(regions)))

    for i, region_i in enumerate(regions):
        for j, region_j in enumerate(regions):
            # Train on region i, predict region j
            r2, _, _ = evaluate_emg_prediction(
                embeddings_by_region[region_i], emg_aligned
            )
            cross_r2[i, j] = r2

    # Plot cross-prediction matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cross_r2, cmap="viridis", vmin=0, vmax=max(1.0, np.max(cross_r2)))
    plt.colorbar(label="R² Score")
    plt.xticks(np.arange(len(regions)), [r.capitalize() for r in regions])
    plt.yticks(np.arange(len(regions)), [r.capitalize() for r in regions])
    plt.xlabel("Region for Prediction")
    plt.ylabel("Region for Training")
    plt.title("Cross-Region EMG Prediction Performance")

    # Add text annotations
    for i in range(len(regions)):
        for j in range(len(regions)):
            plt.text(
                j,
                i,
                f"{cross_r2[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if cross_r2[i, j] < 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig("regional_cross_prediction.png", dpi=300)
    plt.show()

# Optional: Plot diaphragm unit positions with region coloring
plt.figure(figsize=(10, 8))

# Extract x, y coordinates
x_coords = [unit_coordinates[uid][0] for uid in unit_ids]
y_coords = [unit_coordinates[uid][1] for uid in unit_ids]
region_names = [unit_regions[uid] for uid in unit_ids]

# Create scatter plot
for region in unique_regions:
    # Get indices for this region
    indices = [i for i, r in enumerate(region_names) if r == region]

    # Plot points
    plt.scatter(
        [x_coords[i] for i in indices],
        [y_coords[i] for i in indices],
        color=region_colors[region],
        alpha=0.8,
        s=100,
        label=region.capitalize(),
    )

plt.title(
    "Simulated Diaphragm Unit Positions\n(Replace with actual electrode positions)"
)
plt.xlabel("X Position (mm)")
plt.ylabel("Y Position (mm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("diaphragm_regions.png", dpi=300)
plt.show()

print("Regional CEBRA analysis complete!")

# Optional: Save results
import pickle

with open("cebra_regional_results.pkl", "wb") as f:
    pickle.dump(
        {
            "model_all": model_all,
            "embedding_all": embedding_all,
            "models_by_region": models_by_region,
            "embeddings_by_region": embeddings_by_region,
            "emg_aligned": emg_aligned,
            "unit_regions": unit_regions,
            "unit_coordinates": unit_coordinates,
            "region_spikes": region_spikes,
            "r2_all": r2_all,
        },
        f,
    )
print("Results saved to cebra_regional_results.pkl")
