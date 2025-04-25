# dspant_viz API Guide

This comprehensive guide demonstrates how to use the dspant_viz API for visualizing electrophysiology data, focusing on time series and spike data visualization with multiple backend renderers.

## Available Components

### Time Series Visualizations
- **TimeSeriesPlot**: Efficient visualization for multi-channel time series
- **TimeSeriesAreaPlot**: Area plots for envelope-like data
- **TimeSeriesRasterPlot**: Raster view of continuous spiking activity

### Spike Visualizations
- **RasterPlot**: Trial-based spike raster
- **PSTHPlot**: Peristimulus time histogram
- **WaveformPlot**: Neural waveform visualization
- **CorrelogramPlot**: Auto/cross-correlogram visualization

### Composite Visualizations
- **RasterPSTHComposite**: Combined raster and PSTH view

### Interactive Widgets
- **PSTHRasterInspector**: Interactive exploration of multiple units
- **CorrelogramInspector**: Interactive correlation analysis
- **WaveformInspector**: Interactive waveform exploration

## Basic Usage Examples

### Time Series Visualization

```python
import numpy as np
import dask.array as da
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot

# Create sample data
x = np.linspace(0, 10, 1000)
data = np.column_stack([np.sin(x), np.cos(x), np.sin(2*x)])
dask_data = da.from_array(data)

# Create TimeSeriesPlot instance
ts_plot = TimeSeriesPlot(
    data=dask_data,
    sampling_rate=100.0,
    title="Multi-Channel Time Series"
)

# Render with matplotlib
fig_mpl, ax = ts_plot.plot(backend="mpl")
fig_mpl.savefig("time_series_mpl.png")

# Render with plotly
fig_plotly = ts_plot.plot(backend="plotly")
fig_plotly.write_html("time_series_plotly.html")
```

### Spike Data Visualization

```python
import numpy as np
from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.spike.raster import RasterPlot

# Create spike data
spike_data = SpikeData(
    spikes={
        "trial_1": [0.1, 0.5, 1.0],
        "trial_2": [0.2, 0.6, 1.2, 1.5],
        "trial_3": [0.15, 0.7, 1.1],
    },
    unit_id=42,
)

# Create event times
event_times = np.array([0.0, 1.0, 2.0])

# Create RasterPlot
raster_plot = RasterPlot(
    data=spike_data,
    event_times=event_times,
    pre_time=0.5,  # 500ms before events
    post_time=0.5,  # 500ms after events
    marker_size=6,
    marker_color="teal"
)

# Plot with matplotlib
fig_mpl, ax = raster_plot.plot(backend="mpl")

# Plot with plotly
fig_plotly = raster_plot.plot(backend="plotly")
```

## Combining Multiple Visualizations

### Example: Time Series with Event Annotations

```python
import numpy as np
import matplotlib.pyplot as plt
import dask.array as da
import polars as pl

from dspant_viz.visualization.stream.time_series import TimeSeriesPlot
from dspant_viz.visualization.events.events_test import EventAnnotator

# 1. Generate sample data
time = np.linspace(0, 10, 1000)
data = np.column_stack([np.sin(time), np.cos(time)])
dask_data = da.from_array(data)

# 2. Create event data
events = pl.DataFrame({
    "start": [1.5, 3.2, 7.5],
    "end": [2.0, 3.7, 8.0],
    "label": ["Event A", "Event B", "Event C"]
})

# 3. Create visualizations with matplotlib
# ---------------------------------------
# Create time series plot
ts_plot_mpl = TimeSeriesPlot(
    data=dask_data,
    sampling_rate=100.0,
    title="Time Series with Events (Matplotlib)"
)

# Plot with matplotlib
fig_mpl, ax_mpl = ts_plot_mpl.plot(backend="mpl")

# Create and add event annotations
event_annotator_mpl = EventAnnotator(
    events,
    highlight_color="red",
    marker_style="span"
)
event_annotator_mpl.plot(backend="mpl", ax=ax_mpl)

# 4. Create visualizations with plotly
# -----------------------------------
# Create time series plot
ts_plot_plotly = TimeSeriesPlot(
    data=dask_data,
    sampling_rate=100.0,
    title="Time Series with Events (Plotly)"
)

# Plot with plotly
fig_plotly = ts_plot_plotly.plot(backend="plotly")

# Create and add event annotations
event_annotator_plotly = EventAnnotator(
    events,
    highlight_color="red",
    marker_style="span"
)
event_annotator_plotly.plot(backend="plotly", ax=fig_plotly)
```

### Example: Raster and PSTH Visualization

```python
import numpy as np
from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.composites.raster_psth import RasterPSTHComposite

# Create spike data dictionary
spike_times = {
    0: np.sort(np.random.uniform(0, 30, 100)),  # Unit 0 spikes
    1: np.sort(np.random.uniform(0, 30, 150)),  # Unit 1 spikes
}

# Create SpikeData object
spike_data = SpikeData(spikes=spike_times)

# Create event times
event_times = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

# Create RasterPSTHComposite for unit 0
composite = RasterPSTHComposite(
    spike_data=spike_data,
    event_times=event_times,
    pre_time=1.0,       # 1 second before event
    post_time=1.0,      # 1 second after event
    bin_width=0.05,     # 50 ms bins
    unit_id=0,
    title="Unit 0 Neural Response"
)

# Plot with matplotlib
fig_mpl, axes_mpl = composite.plot(backend="mpl")

# Plot with plotly
fig_plotly = composite.plot(backend="plotly")
```

## Using Interactive Widgets

### Example: PSTH Raster Inspector

```python
import numpy as np
from dspant_viz.core.data_models import SpikeData
from dspant_viz.widgets.psth_raster_inspector import PSTHRasterInspector

# Create spike data for multiple units
spike_times = {
    0: np.sort(np.random.uniform(0, 30, 100)),  # Unit 0
    1: np.sort(np.random.uniform(0, 30, 150)),  # Unit 1
    2: np.sort(np.random.uniform(0, 30, 120)),  # Unit 2
}
spike_data = SpikeData(spikes=spike_times)

# Create event times
event_times = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

# Create inspector widget with Plotly backend
inspector = PSTHRasterInspector(
    spike_data=spike_data,
    event_times=event_times,
    pre_time=1.0,
    post_time=1.0,
    bin_width=0.05,
    backend="plotly"
)

# Display the widget (run in Jupyter notebook)
inspector.display()
```

## Tips and Best Practices

1. **Large Datasets**: 
   - Use Dask arrays with appropriate chunking
   - Enable dynamic resampling with Plotly (`use_resampler=True`)
   - Use WebGL acceleration with Plotly (`use_webgl=True`)

2. **Publication Figures**:
   - Use the Matplotlib backend with the "publication" theme
   - Apply themes: `apply_matplotlib_theme("publication")`

3. **Interactive Exploration**:
   - Use the Plotly backend with interactive widgets
   - Set appropriate max_points for resampling

4. **Combining Components**:
   - Create individual components first
   - Share axes for synchronized views
   - Update time windows consistently

5. **Custom Styling**:
   - Use the `ThemeManager` for consistent styling
   - Create custom themes for specific visualization needs

## Further Resources

For more detailed examples, check out the test files in the repository:
- `tests/time_series_test.py`
- `tests/test_event_annotator.py`
- `tests/raster_psth_test.py`
- `tests/waveforms_test.py`
- `tests/correlogram_test.py`