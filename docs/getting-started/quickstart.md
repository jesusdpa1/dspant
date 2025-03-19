# Quickstart Guide

This guide will walk you through the basics of using dspant for neural data analysis, including loading data, building processing pipelines, detecting spikes, and visualizing results.

## Installation

First, make sure you have dspant installed:

```bash
pip install dspant
```

For development or the latest features, you can install from the repository:

```bash
git clone https://github.com/yourusername/dspant.git
cd dspant
pip install -e .
```

## Loading Data

dspant uses data nodes to load and manage different types of neural data. Let's start by loading some time-series data:

```python
from dspant.nodes import StreamNode

# Load from existing ANT format data
stream_node = StreamNode(data_path="path/to/recording.ant")
stream_node.load_data()  # Loads the data lazily

# Print a summary of the data
stream_node.summarize()
```

If you have data in TDT format, you can convert it to ANT format first:

```python
from dspant.io import convert_tdt_to_ant

# Convert TDT block to ANT format
ant_path = convert_tdt_to_ant(
    tdt_block_path="path/to/tdt_block",
    stream_names=["Wave"]  # Which streams to convert
)

# Now load the converted data
stream_node = StreamNode(data_path=f"{ant_path}/Wave.ant")
stream_node.load_data()
```

## Basic Signal Processing

Let's create a simple processing pipeline to filter the data:

```python
from dspant.engine import create_processing_node
from dspant.processor.filters import create_bandpass_filter
import matplotlib.pyplot as plt

# Create a processing node connected to our data
proc_node = create_processing_node(stream_node)

# Create a bandpass filter (300-3000 Hz)
bandpass = create_bandpass_filter(low_hz=300, high_hz=3000)

# Add the filter to the processing pipeline
proc_node.add_processor(bandpass, group="filtering")

# Process the data
filtered_data = proc_node.process()

# Plot a segment of the original and filtered data
t = np.arange(10000) / stream_node.fs
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, stream_node.data[:10000, 0].compute())
plt.title("Original Data")
plt.subplot(2, 1, 2)
plt.plot(t, filtered_data[:10000].compute())
plt.title("Filtered Data (300-3000 Hz)")
plt.tight_layout()
plt.show()
```

## Spike Detection

Now let's detect spikes in our filtered data:

```python
from dspant.neuroproc.detection import create_negative_peak_detector

# Create a spike detector
detector = create_negative_peak_detector(
    threshold=4.0,  # 4x MAD threshold
    refractory_period=0.001  # 1 ms refractory period
)

# Clear any existing processors and add our filter and detector
proc_node.clear_processors()
proc_node.add_processor(bandpass, group="preprocessing")
proc_node.add_processor(detector, group="detection")

# Run spike detection
spike_df = proc_node.process()

# Display spike detection results
print(f"Detected {len(spike_df)} spikes")
print(spike_df.head())
```

## Visualizing Spikes

Let's visualize the detected spikes:

```python
from dspant.neuroproc.visualization import plot_spike_events

# Create visualization
fig_waveforms, fig_raster = plot_spike_events(
    stream_node.data,  # Original data
    spike_df,          # Spike detection results
    stream_node.fs,    # Sampling rate
    time_window=(10, 15),  # 5-second window from 10-15 seconds
    window_ms=2.0      # 2 ms window around each spike
)

# Display plots
fig_waveforms.savefig("spike_waveforms.png")
fig_raster.savefig("spike_raster.png")
```

## Extracting Spike Waveforms

Now let's extract the waveforms around each spike for further analysis:

```python
from dspant.neuroproc.extraction import extract_spike_waveforms

# Extract waveforms around spikes
waveforms, spike_times, metadata = extract_spike_waveforms(
    stream_node.data,  # Raw data
    spike_df,          # Spike detection results
    pre_samples=10,    # Samples before spike peak
    post_samples=30,   # Samples after spike peak
    align_to_min=True, # Align to negative peak
    compute_result=True # Compute result immediately
)

print(f"Extracted {len(waveforms)} waveforms")
print(f"Waveform shape: {waveforms.shape}")
```

## Spike Sorting

Let's perform PCA-KMeans spike sorting on the extracted waveforms:

```python
from dspant.neuroproc.sorters import create_numba_pca_kmeans
import matplotlib.pyplot as plt

# Create a PCA-KMeans processor
sorter = create_numba_pca_kmeans(
    n_clusters=3,       # Number of clusters to find
    n_components=10,    # Number of PCA components
    normalize=True      # Normalize waveforms
)

# Process the waveforms
cluster_labels = sorter.process(waveforms)

# Visualize the clustering results
fig = sorter.plot_clusters(
    waveforms=waveforms,
    max_points=5000,     # Maximum points to plot
    alpha=0.7,           # Transparency of points
    s=15,                # Size of points
    plot_waveforms=True  # Show waveform shapes
)

plt.savefig("spike_clusters.png")
```

## Computing Quality Metrics

Finally, let's compute quality metrics for our sorted units:

```python
from dspant.neuroproc.metrics import compute_quality_metrics
from dspant.neuroproc.utils import prepare_spike_data_for_metrics

# Convert spike times to the format expected by the metrics module
spike_times_by_cluster = {}
for cluster in np.unique(cluster_labels):
    mask = cluster_labels == cluster
    spike_times_by_cluster[f"unit_{cluster}"] = [spike_times[mask]]

# Compute quality metrics
metrics = compute_quality_metrics(
    spike_times=spike_times_by_cluster,
    sampling_frequency=stream_node.fs,
    total_duration=len(stream_node.data) / stream_node.fs,
    metrics=["num_spikes", "firing_rate", "presence_ratio", "isi_violation"]
)

# Print metrics
for metric_name, metric_values in metrics.items():
    print(f"\n{metric_name}:")
    for unit_id, value in metric_values.items():
        print(f"  {unit_id}: {value}")