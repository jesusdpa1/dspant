# Neural Processing Overview

The `neuroproc` module is a specialized collection of tools for neural data analysis, focusing particularly on spike detection, waveform extraction, spike sorting, and quality assessment. This module builds on dspant's core architecture to provide optimized algorithms for neural data analysis workflows.

## Module Structure

The neural processing module is organized into several submodules:

- **Detection**: Algorithms for identifying neural action potentials (spikes)
- **Extraction**: Tools for extracting waveforms around detected spikes
- **Visualization**: Plotting and visualization tools for neural data and analysis results
- **Sorters**: Spike sorting algorithms for clustering neural units
- **Metrics**: Quality metrics and validation tools for spike sorting results
- **Utils**: Utility functions supporting various neural processing tasks

## Key Features

### High-Performance Processing

Neural processing algorithms are optimized for performance using:

- **Numba**: Just-in-time compilation for CPU-intensive operations
- **Dask**: Parallel and distributed computing for large datasets
- **Chunked Processing**: Processing data in manageable chunks to handle large recordings

### Comprehensive Workflow

The module supports a complete neural data analysis workflow:

1. **Spike Detection**: Identify spike events in continuous recordings
2. **Waveform Extraction**: Extract and align spike waveforms
3. **Feature Extraction**: Compute features for clustering
4. **Spike Sorting**: Cluster waveforms into putative neural units
5. **Quality Assessment**: Evaluate the quality of sorted units
6. **Visualization**: Visualize results at each stage of the workflow

### Modularity and Extensibility

Each component follows dspant's modular design principles:

- Processors implement the `BaseProcessor` interface for pipeline integration
- Factory functions provide simplified interfaces for common use cases
- Clear separation between algorithms and data structures

## Common Workflow

A typical neural data analysis workflow with dspant might look like:

```python
# Import components
from dspant.nodes import StreamNode
from dspant.engine import create_processing_node
from dspant.neuroproc.detection import create_negative_peak_detector
from dspant.neuroproc.extraction import extract_spike_waveforms
from dspant.neuroproc.sorters import create_numba_pca_kmeans
from dspant.neuroproc.metrics import compute_quality_metrics
from dspant.neuroproc.visualization import plot_spike_events, plot_waveform_clusters

# 1. Load data
node = StreamNode("path/to/data.ant").load_data()

# 2. Set up processing pipeline with spike detector
proc_node = create_processing_node(node)
detector = create_negative_peak_detector(threshold=4.0)
proc_node.add_processor(detector, group="detection")

# 3. Detect spikes
spikes_df = proc_node.process()

# 4. Extract waveforms
waveforms, spike_times, metadata = extract_spike_waveforms(
    node.data, spikes_df, pre_samples=10, post_samples=30
)

# 5. Sort spikes with PCA-KMeans
sorter = create_numba_pca_kmeans(n_clusters=3, n_components=10)
cluster_labels = sorter.process(waveforms)

# 6. Visualize results
cluster_fig = sorter.plot_clusters(waveforms=waveforms)
spike_fig, raster_fig = plot_spike_events(
    node.data, spikes_df, node.fs, cluster_column="cluster"
)

# 7. Compute quality metrics
from dspant.neuroproc.utils import prepare_spike_data_for_metrics
metrics_data = prepare_spike_data_for_metrics(
    spike_times=spike_times,
    cluster_labels=cluster_labels,
    sampling_frequency=node.fs
)
metrics = compute_quality_metrics(**metrics_data)
```

## Submodule Details

### Detection

The detection submodule provides spike detection algorithms:

- **Threshold-based detection**: Simple threshold crossing detection
- **Peak detection**: Identify local extrema (peaks) with thresholding
- **Template matching**: Detect spikes based on template similarity

### Extraction

The extraction submodule handles spike waveform extraction:

- **Waveform extraction**: Extract time windows around detected spikes
- **Waveform alignment**: Align spikes to peaks for better clustering
- **Batch processing**: Efficiently process large datasets

### Visualization

The visualization submodule offers specialized plotting functions:

- **Spike visualization**: Display detected spikes overlaid on raw data
- **Raster plots**: Show spike timing across channels or units
- **Cluster visualization**: Visualize spike sorting results with PCA projections
- **Quality metrics visualization**: Plot unit quality metrics

### Sorters

The sorters submodule implements spike sorting algorithms:

- **PCA-KMeans**: Dimensionality reduction with PCA followed by KMeans clustering
- **Numba-accelerated versions**: High-performance implementations

### Metrics

The metrics submodule provides quality assessment tools:

- **ISI violations**: Measure refractory period violations
- **Presence ratio**: Assess unit stability over time
- **Amplitude cutoff**: Estimate fraction of missing spikes
- **Signal-to-noise ratio**: Measure unit isolation quality

### Utils

The utils submodule contains supporting utilities:

- **Spike utilities**: Helper functions for spike data handling
- **Template utilities**: Functions for working with templates
- **PHY exporter**: Tools for exporting to the PHY template-GUI format