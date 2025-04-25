# dspant_viz API Guide

This comprehensive guide demonstrates how to use the dspant_viz API for visualizing electrophysiology data, focusing on time series and spike data visualization with multiple backend renderers.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Data Models](#data-models)
- [Visualization Components](#visualization-components)
  - [Time Series Visualizations](#time-series-visualizations)
  - [Spike Visualizations](#spike-visualizations)
  - [Composite Visualizations](#composite-visualizations)
- [Interactive Widgets](#interactive-widgets)
- [Backend Renderers](#backend-renderers)
- [Themes and Styling](#themes-and-styling)
- [Performance Optimization](#performance-optimization)
- [Extending the Library](#extending-the-library)

## Core Concepts

### Architecture Overview

dspant_viz follows a layered architecture that separates data organization, visualization logic, and rendering:

```
┌───────────────────────────────────────────────┐
│                                               │
│    Visualization Components (High-Level API)  │
│    - TimeSeriesPlot                           │
│    - RasterPlot                               │
│    - PSTHPlot                                 │
│                                               │
└──────────────────────┬────────────────────────┘
                       │
┌──────────────────────┼────────────────────────┐
│                      │                         │
│    Core Components   │                         │
│    - Data Models     │                         │
│    - Base Classes    │                         │
│                      │                         │
└──────────────────────┼────────────────────────┘
                       │
          ┌────────────┴─────────────┐
          │                          │
┌─────────▼──────────┐    ┌──────────▼─────────┐
│                    │    │                     │
│  Matplotlib Backend│    │   Plotly Backend    │
│                    │    │                     │
└────────────────────┘    └─────────────────────┘
```

### Base Classes

The foundation of dspant_viz is the `VisualizationComponent` abstract base class, which defines the interface for all visualization components:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class VisualizationComponent(ABC):
    def __init__(self, data, **kwargs):
        self.data = data
        self.config = kwargs

    @abstractmethod
    def plot(self, backend="mpl", **kwargs):
        """Generate visualization using specified backend"""
        pass
        
    @abstractmethod
    def get_data(self) -> Dict:
        """Prepare data for rendering"""
        pass
        
    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update component parameters"""
        pass
```

For composite visualizations that combine multiple plots, there's the `CompositeVisualization` class:

```python
class CompositeVisualization(ABC):
    def __init__(self, components, **kwargs):
        self.components = components
        self.config = kwargs
        
    @abstractmethod
    def plot(self, backend="mpl", **kwargs):
        """Generate composite visualization"""
        pass
        
    @abstractmethod
    def get_data(self) -> Dict:
        """Prepare combined data for rendering"""
        pass
        
    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update all components"""
        pass
```

## Data Models

dspant_viz provides structured data models for different types of neurophysiological data:

### SpikeData

For spike train data:

```python
from dspant_viz.core.data_models import SpikeData

# Create spike data
spike_data = SpikeData(
    spikes={
        0: np.array([0.1, 0.5, 1.0]),  # Unit 0 spike times
        1: np.array([0.2, 0.6, 1.2]),  # Unit 1 spike times
    },
    unit_labels={0: "Pyramidal Cell", 1: "Interneuron"}  # Optional
)

# Access spike data
unit_ids = spike_data.get_unit_ids()
unit0_spikes = spike_data.get_unit_spikes(0)
```

### TimeSeriesData

For continuous time series data:

```python
from dspant_viz.core.data_models import TimeSeriesData

# Create time series data
ts_data = TimeSeriesData(
    times=np.linspace(0, 10, 1000),
    values=np.sin(np.linspace(0, 10, 1000)),
    sampling_rate=100.0,
    channel_id=0,
    channel_name="LFP"
)
```

### MultiChannelData

For multi-channel continuous data:

```python
from dspant_viz.core.data_models import MultiChannelData

# Create multi-channel data
mc_data = MultiChannelData(
    times=np.linspace(0, 10, 1000),
    channels={
        0: np.sin(np.linspace(0, 10, 1000)),
        1: np.cos(np.linspace(0, 10, 1000))
    },
    sampling_rate=100.0
)
```

## Visualization Components

### Time Series Visualizations

#### TimeSeriesPlot

Visualize multi-channel time series data:

```python
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot

# Create TimeSeriesPlot instance
ts_plot = TimeSeriesPlot(
    data=dask_data,           # Dask array (samples × channels)
    sampling_rate=1000.0,     # 1 kHz sampling rate
    channels=[0, 1, 2],       # Optional: specific channels to display
    time_window=(5.0, 15.0),  # Optional: time window to display
    color_mode="colormap",    # "colormap" or "single"
    colormap="viridis",       # Colormap for multi-channel display
    normalize=True,           # Normalize each channel
    y_spread=1.0,             # Vertical spacing between channels
    grid=True,                # Show grid
)

# Plot with different backends
fig_mpl, ax = ts_plot.plot(backend="mpl")
fig_plotly = ts_plot.plot(backend="plotly")

# Update parameters
ts_plot.update(time_window=(10.0, 20.0), colormap="plasma")
```

#### TimeSeriesAreaPlot

Create area plots for envelope-like data:

```python
from dspant_viz.visualization.stream.ts_area import TimeSeriesAreaPlot

# Create TimeSeriesAreaPlot instance
ts_area = TimeSeriesAreaPlot(
    data=dask_data,          # Dask array (samples × channels)
    sampling_rate=1000.0,    # 1 kHz sampling rate
    fill_to="zero",          # Fill method: "zero", "min", or float value
    fill_alpha=0.3,          # Transparency of fill
    color_mode="colormap",   # "colormap" or "single"
    colormap="plasma",       # Colormap for channels
)

# Plot with different backends
fig_mpl, ax = ts_area.plot(backend="mpl")
fig_plotly = ts_area.plot(backend="plotly")
```

### Spike Visualizations

#### RasterPlot

Visualize spike trains in a raster format:

```python
from dspant_viz.visualization.spike.raster import RasterPlot

# Create RasterPlot instance
raster_plot = RasterPlot(
    data=spike_data,          # SpikeData object
    event_times=event_times,  # Event/trigger times (optional)
    pre_time=1.0,             # Time before each event (if event_times provided)
    post_time=1.0,            # Time after each event (if event_times provided)
    marker_size=6,            # Size of spike markers
    marker_color="teal",      # Color of spike markers
    marker_alpha=0.8,         # Transparency of markers
    marker_type="|",          # Marker style
    unit_id=0,                # Unit to display
)

# Plot with different backends
fig_mpl, ax = raster_plot.plot(backend="mpl")
fig_plotly = raster_plot.plot(backend="plotly")

# Switch to a different unit
raster_plot.update(unit_id=1, marker_color="crimson")
```

#### PSTHPlot

Compute and visualize peristimulus time histograms:

```python
from dspant_viz.visualization.spike.psth import PSTHPlot

# Create PSTHPlot instance
psth_plot = PSTHPlot(
    data=spike_data,          # SpikeData object
    event_times=event_times,  # Event/trigger times
    pre_time=1.0,             # Time before each event
    post_time=1.0,            # Time after each event
    bin_width=0.05,           # Bin width in seconds (50 ms)
    line_color="orange",      # Line color
    line_width=2,             # Line width
    show_sem=True,            # Show standard error
    unit_id=0,                # Unit to analyze
    sigma=0.1,                # Gaussian smoothing sigma (optional)
)

# Plot with different backends
fig_mpl, ax = psth_plot.plot(backend="mpl")
fig_plotly = psth_plot.plot(backend="plotly")
```

#### CorrelogramPlot

Compute and visualize auto/cross-correlograms:

```python
from dspant_viz.visualization.spike.correlogram import CorrelogramPlot

# Create autocorrelogram
auto_corr = CorrelogramPlot(
    data=spike_data,       # SpikeData object
    unit01=0,              # Reference unit
    unit02=None,           # None for autocorrelogram
    bin_width=0.01,        # 10 ms bins
    window_size=0.5,       # ±500 ms window
    normalize=True,        # Normalize by firing rate
)

# Create cross-correlogram
cross_corr = CorrelogramPlot(
    data=spike_data,       # SpikeData object
    unit01=0,              # Reference unit
    unit02=1,              # Target unit
    bin_width=0.01,        # 10 ms bins
    window_size=0.5,       # ±500 ms window
    normalize=True,        # Normalize by firing rate
)

# Plot with different backends
fig_mpl, ax = cross_corr.plot(backend="mpl")
fig_plotly = cross_corr.plot(backend="plotly")
```

#### WaveformPlot

Visualize spike waveforms:

```python
from dspant_viz.visualization.spike.waveforms import WaveformPlot

# Create WaveformPlot instance
waveform_plot = WaveformPlot(
    waveforms=waveform_data,  # 3D array (neurons, samples, waveforms)
    unit_id=0,                # Unit to display
    sampling_rate=30000.0,    # 30 kHz sampling rate
    template=False,           # Show individual waveforms (not template)
    normalization="zscore",   # Normalization method (None, "zscore", "minmax")
    color_mode="colormap",    # "colormap" or "single"
    colormap="colorblind",    # Colormap for waveforms
)

# Plot with different backends
fig_mpl, ax = waveform_plot.plot(backend="mpl")
fig_plotly = waveform_plot.plot(backend="plotly")

# Switch to template view
waveform_plot.update(template=True, normalization="minmax")
```

### Composite Visualizations

#### RasterPSTHComposite

Combine raster and PSTH plots:

```python
from dspant_viz.visualization.composites.raster_psth import RasterPSTHComposite

# Create composite visualization
composite = RasterPSTHComposite(
    spike_data=spike_data,     # SpikeData object
    event_times=event_times,   # Event/trigger times
    pre_time=1.0,              # Time before each event
    post_time=1.0,             # Time after each event
    bin_width=0.05,            # PSTH bin width
    raster_color="navy",       # Raster marker color
    psth_color="crimson",      # PSTH line color
    show_sem=True,             # Show PSTH standard error
    raster_height_ratio=2.0,   # Ratio of raster height to PSTH height
    unit_id=0,                 # Unit to display
)

# Plot with different backends
fig_mpl, axes = composite.plot(backend="mpl")
fig_plotly = composite.plot(backend="plotly")

# Switch to a different unit
composite.update(unit_id=1, title="Unit 1 Response")
```

## Interactive Widgets

### PSTHRasterInspector

Interactive widget for exploring multiple units:

```python
from dspant_viz.widgets.psth_raster_inspector import PSTHRasterInspector

# Create inspector widget
inspector = PSTHRasterInspector(
    spike_data=spike_data,    # SpikeData object with multiple units
    event_times=event_times,  # Event/trigger times
    pre_time=1.0,             # Time before each event
    post_time=1.0,            # Time after each event
    bin_width=0.05,           # PSTH bin width
    backend="plotly",         # Backend to use ("mpl" or "plotly")
)

# Display the widget (in Jupyter notebook)
inspector.display()
```

### CorrelogramInspector

Interactive widget for exploring correlations between units:

```python
from dspant_viz.widgets.correlogram_inspector import CorrelogramInspector

# Create inspector widget
corr_inspector = CorrelogramInspector(
    spike_data=spike_data,    # SpikeData object with multiple units
    backend="plotly",         # Backend to use ("mpl" or "plotly")
)

# Display the widget (in Jupyter notebook)
corr_inspector.display()
```

### WaveformInspector

Interactive widget for exploring waveforms:

```python
from dspant_viz.widgets.waveform_inspector import WaveformInspector

# Create inspector widget
waveform_inspector = WaveformInspector(
    waveforms=waveform_data,  # 3D array (neurons, samples, waveforms)
    sampling_rate=30000.0,    # 30 kHz sampling rate
    backend="plotly",         # Backend to use ("mpl" or "plotly")
)

# Display the widget (in Jupyter notebook)
waveform_inspector.display()
```

## Backend Renderers

The backend renderers are responsible for transforming the data prepared by visualization components into actual visualizations. You typically won't call these directly, but it's useful to understand how they work.

### Matplotlib Renderers

Matplotlib renderers create static, publication-quality figures:

```python
from dspant_viz.backends.mpl.time_series import render_time_series

# Direct use of a renderer (uncommon)
fig, ax = render_time_series(
    data=ts_plot.get_data(),  # Data dictionary from a component
    ax=None,                  # Optional existing axes (creates new figure if None)
)
```

### Plotly Renderers

Plotly renderers create interactive, web-based visualizations:

```python
from dspant_viz.backends.plotly.time_series import render_time_series

# Direct use of a renderer (uncommon)
fig = render_time_series(
    data=ts_plot.get_data(),  # Data dictionary from a component
    use_resampler=True,       # Enable dynamic resampling
    use_webgl=True,           # Enable WebGL acceleration
)
```

## Themes and Styling

dspant_viz provides a theming system to ensure consistent styling across visualizations:

```python
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme

# Apply themes
apply_matplotlib_theme("publication")  # For publication-quality figures
apply_plotly_theme("seaborn")          # For interactive visualizations

# List available themes
from dspant_viz.core.themes_manager import list_available_themes
themes = list_available_themes()
print(themes)

# Create a custom theme
from dspant_viz.core.themes_manager import create_custom_theme
create_custom_theme("my_theme", {
    "palette": {
        "primary": ["#1F77B4", "#FF7F0E", "#2CA02C"],
    },
    "typography": {
        "font_family": "Arial",
        "sizes": {"title": 16, "axis_label": 12},
    },
    "visualization": {
        "grid_style": "darkgrid",
        "line_width": 1.0,
    }
})
```

## Performance Optimization

### Working with Large Datasets

dspant_viz provides several strategies for handling large datasets efficiently:

#### Dask Arrays

Use Dask arrays for lazy computation and memory-efficient processing:

```python
import dask.array as da
import numpy as np

# Create large dataset (100M samples × 8 channels)
data = da.random.normal(0, 1, size=(100_000_000, 8), chunks=(10000, 8))

# Create TimeSeriesPlot with dask array
ts_plot = TimeSeriesPlot(
    data=data,
    sampling_rate=30000.0,  # 30 kHz sampling rate
    downsample=False,       # Let plotly-resampler handle downsampling
)
```

#### Dynamic Resampling

Use plotly-resampler for interactive visualization of large datasets:

```python
# Render with plotly-resampler
fig = ts_plot.plot(
    backend="plotly",
    use_resampler=True,       # Enable dynamic resampling
    max_n_samples=10000,      # Maximum samples to display at once
)
```

#### WebGL Acceleration

Use WebGL for faster rendering with Plotly:

```python
# Render with WebGL acceleration
fig = ts_plot.plot(
    backend="plotly",
    use_webgl=True,  # Enable WebGL acceleration
)
```

#### Chunking

Optimize Dask array chunk sizes for performance:

```python
# Rule of thumb: 10K-30K samples per chunk
optimal_chunk_size = 20000
data = da.random.normal(0, 1, size=(1_000_000, 8), chunks=(optimal_chunk_size, 8))
```

## Extending the Library

### Creating a New Visualization Component

To create a new visualization component:

1. Subclass `VisualizationComponent` from `core/base.py`
2. Implement the required methods: `plot()`, `get_data()`, and `update()`
3. Create renderer implementations for each backend

```python
# visualization/my_component.py
from dspant_viz.core.base import VisualizationComponent
from typing import Dict, Any

class MyComponent(VisualizationComponent):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        # Initialize component-specific attributes
    
    def get_data(self) -> Dict[str, Any]:
        # Prepare data for rendering
        return {
            "data": {...},    # Data for rendering
            "params": {...}   # Parameters for rendering
        }
    
    def update(self, **kwargs) -> None:
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value
    
    def plot(self, backend="mpl", **kwargs):
        # Import appropriate renderer based on backend
        if backend == "mpl":
            from dspant_viz.backends.mpl.my_component import render_my_component
        elif backend == "plotly":
            from dspant_viz.backends.plotly.my_component import render_my_component
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        return render_my_component(self.get_data(), **kwargs)
```

### Creating Backend Renderers

For each visualization component, create renderers for each supported backend:

```python
# backends/mpl/my_component.py
def render_my_component(data: Dict[str, Any], ax=None, **kwargs):
    """Matplotlib renderer for MyComponent"""
    # Extract data
    plot_data = data["data"]
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Create visualization using matplotlib
    # ... implementation ...
    
    return fig, ax

# backends/plotly/my_component.py
def render_my_component(data: Dict[str, Any], **kwargs):
    """Plotly renderer for MyComponent"""
    # Extract data
    plot_data = data["data"]
    params = data["params"]
    params.update(kwargs)  # Override with provided kwargs
    
    # Create visualization using plotly
    fig = go.Figure()
    
    # ... implementation ...
    
    return fig
```

### Best Practices

1. **Backend Agnostic Components**: Keep visualization logic separate from rendering details
2. **Efficient Data Processing**: Use dask for large dataset handling
3. **Consistent Parameter Naming**: Follow existing parameter naming conventions
4. **Documentation**: Document parameters and behavior clearly
5. **Optimization**: Consider performance for large datasets