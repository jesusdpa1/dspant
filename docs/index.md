# 🚧⚠ Site under construction!!

# dspant: Digital Signal Processing for Neural Data

**dspant** is a comprehensive Python library for digital signal processing with a focus on neural data analysis. It provides a modular and highly optimized set of tools for loading, processing, analyzing, and visualizing neural time-series data with a strong emphasis on spike detection and sorting.

🐜

## Key Features

- **Efficient data loading and conversion**: Support for common neural data formats including TDT (Tucker-Davis Technologies)
- **Scalable processing pipelines**: Processing large datasets efficiently with Dask arrays and Numba acceleration
- **Comprehensive spike detection and sorting**: Multiple algorithms for detecting and classifying neural action potentials
- **Advanced visualization**: Rich plotting capabilities for neural data, spike waveforms, and sorting results
- **Quality metrics**: Tools for assessing the quality of spike sorting results
- **Modular architecture**: Easily extend functionality with custom processors and analysis methods

## Example

```python
from dspant.nodes import StreamNode
from dspant.engine import create_processing_node
from dspant.neuroproc.detection import create_negative_peak_detector
from dspant.neuroproc.extraction import extract_spike_waveforms

# Load neural data
node = StreamNode(data_path="my_recording").load_data()

# Create processing pipeline with a spike detector
proc_node = create_processing_node(node)
detector = create_negative_peak_detector(threshold=4.0)
proc_node.add_processor(detector, group="detection")

# Run the detection
spikes_df = proc_node.process()

# Extract spike waveforms
waveforms, spike_times, metadata = extract_spike_waveforms(
    node.data, spikes_df, pre_samples=10, post_samples=30
)

# Visualize results
from dspant.neuroproc.visualization import plot_spike_events
fig_waveforms, fig_raster = plot_spike_events(
    node.data, spikes_df, node.fs, time_window=(10, 15)
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
