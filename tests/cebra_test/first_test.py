"""
Simple implementation of CEBRA for neural data and EMG analysis
"""

import os
from pathlib import Path

# Import for CEBRA
import cebra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from scipy import interpolate

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

# Step 2: Process EMG to get envelope
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

# Step 3: Prepare neural data
print("Preparing neural data...")


def create_binned_spikes(sorter_data, bin_size_ms=10, max_time_s=None):
    """Create time-binned spike counts from Kilosort data."""
    # Convert spike times to seconds
    spike_times = sorter_data["spike_times"] / sorter_data["sampling_frequency"]
    spike_clusters = sorter_data["spike_clusters"]

    # Get unique units
    unit_ids = np.unique(spike_clusters)

    # Determine max time
    if max_time_s is None:
        max_time_s = np.ceil(spike_times.max())

    # Create time bins
    bin_size_s = bin_size_ms / 1000
    n_bins = int(np.ceil(max_time_s / bin_size_s))
    bin_edges = np.arange(0, max_time_s + bin_size_s, bin_size_s)

    # Create binned spike counts
    binned_spikes = np.zeros((n_bins, len(unit_ids)))

    for i, unit_id in enumerate(unit_ids):
        unit_spikes = spike_times[spike_clusters == unit_id]
        hist, _ = np.histogram(unit_spikes, bins=bin_edges)
        binned_spikes[: len(hist), i] = hist

    return binned_spikes, unit_ids, bin_size_s


# Set a reasonable duration for faster processing
max_time_s = 300  # 5 minutes
bin_size_ms = 10  # 10ms bins

# Create binned spike data
neural_data, unit_ids, bin_size_s = create_binned_spikes(
    sorter_data, bin_size_ms=bin_size_ms, max_time_s=max_time_s
)
print(f"Neural data shape: {neural_data.shape}")

# Step 4: Align EMG data to neural data timebase
print("Aligning EMG data...")
# Get EMG data
emg_raw = zscore_tkeo[: int(max_time_s * fs), 0].compute()

# Create time points
emg_times = np.arange(0, len(emg_raw) / fs, 1 / fs)[: len(emg_raw)]
neural_times = np.arange(0, max_time_s, bin_size_s)[: neural_data.shape[0]]

# Resample EMG to match neural data timebase
f = interpolate.interp1d(
    emg_times, emg_raw, bounds_error=False, fill_value="extrapolate"
)
emg_aligned = f(neural_times)

# Step 5: Train CEBRA
print("Training CEBRA model...")
# Create and train CEBRA model
model = cebra.CEBRA(
    model_architecture="offset10-model",  # Default architecture with time offsets
    batch_size=512,
    max_iterations=1000,  # Reduced for faster training
    learning_rate=0.001,
    temperature=1.0,
    output_dimension=3,  # 3D embedding for visualization
    verbose=True,
)

# Fit CEBRA model to neural data with EMG as behavioral variable
embedding = model.fit_transform(
    neural_data,
    emg_aligned.reshape(-1, 1),  # Reshape to (n_samples, 1)
)
print(f"Embedding shape: {embedding.shape}")

# Step 6: Visualize results
print("Creating visualizations...")
plt.figure(figsize=(12, 10))

# 2D embedding with EMG amplitude coloring
plt.subplot(2, 2, 1)
scatter = plt.scatter(
    embedding[:, 0], embedding[:, 1], c=emg_aligned, cmap="viridis", alpha=0.7, s=3
)
plt.colorbar(scatter, label="EMG Amplitude")
plt.title("CEBRA 2D Embedding - EMG Amplitude")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# 3D embedding with EMG amplitude coloring
ax = plt.subplot(2, 2, 2, projection="3d")
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    embedding[:, 2],
    c=emg_aligned,
    cmap="viridis",
    alpha=0.7,
    s=3,
)
plt.colorbar(scatter, label="EMG Amplitude")
plt.title("CEBRA 3D Embedding - EMG Amplitude")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")

# Create a time-colored plot to see dynamics
plt.subplot(2, 2, 3)
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=np.arange(len(embedding)),
    cmap="plasma",
    alpha=0.7,
    s=3,
)
plt.colorbar(scatter, label="Time")
plt.title("CEBRA 2D Embedding - Temporal Dynamics")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# Compare with simple PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca_result = pca.fit_transform(neural_data)

plt.subplot(2, 2, 4)
scatter = plt.scatter(
    pca_result[:, 0], pca_result[:, 1], c=emg_aligned, cmap="viridis", alpha=0.7, s=3
)
plt.colorbar(scatter, label="EMG Amplitude")
plt.title("PCA 2D - For Comparison")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.savefig("cebra_simple_analysis.png", dpi=300)
plt.show()

# Step 7: Evaluate how well CEBRA captures EMG information
print("Evaluating CEBRA performance...")
# Simple decoding test using KNN
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    embedding, emg_aligned, test_size=0.3, random_state=42
)

# Train KNN regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"EMG prediction R² score: {r2:.3f}")

# Do the same with PCA for comparison
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    pca_result, emg_aligned, test_size=0.3, random_state=42
)

knn_pca = KNeighborsRegressor(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = knn_pca.predict(X_test_pca)
r2_pca = r2_score(y_test_pca, y_pred_pca)
print(f"PCA EMG prediction R² score: {r2_pca:.3f}")

# Improvement percentage
improvement = ((r2 - r2_pca) / abs(r2_pca)) * 100 if r2_pca != 0 else float("inf")
print(f"CEBRA improvement over PCA: {improvement:.1f}%")

# Step 8: Plot a segment of actual vs predicted EMG
plt.figure(figsize=(12, 5))
segment_length = 1000  # Plot 1000 time points

plt.plot(y_test[:segment_length], "b-", label="Actual EMG", alpha=0.7)
plt.plot(y_pred[:segment_length], "r-", label="Predicted from CEBRA", alpha=0.7)
plt.plot(y_pred_pca[:segment_length], "g-", label="Predicted from PCA", alpha=0.5)
plt.title(f"EMG Reconstruction (CEBRA R² = {r2:.3f}, PCA R² = {r2_pca:.3f})")
plt.xlabel("Time (bins)")
plt.ylabel("EMG Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("emg_reconstruction_comparison.png", dpi=300)
plt.show()

print("CEBRA analysis complete!")

# Optional: Save results
import pickle

with open("cebra_simple_results.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "embedding": embedding,
            "emg_aligned": emg_aligned,
            "neural_data": neural_data,
            "unit_ids": unit_ids,
            "bin_size_s": bin_size_s,
            "r2_score": r2,
            "r2_pca": r2_pca,
        },
        f,
    )
print("Results saved to cebra_simple_results.pkl")
