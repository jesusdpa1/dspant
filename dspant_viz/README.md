# dspant_viz

multi-backend visualization library for electrophysiology data, with special focus on efficient rendering of large time series datasets.

## Overview

dspant_viz extends the dspant library with specialized visualization capabilities. It's designed to render large electrophysiology datasets efficiently using various backends (matplotlib, plotly) and includes optimization strategies for handling datasets with millions of data points.

Key features:

- **Multi-backend architecture**: Create visualizations that work with both Matplotlib and Plotly
- **Efficient large dataset handling**: Optimized for time series with millions of data points
- **Dask integration**: Lazy loading and efficient processing of large datasets
- **Plotly-resampler support**: Dynamic resampling of large datasets for interactive visualization
- **Consistent styling**: Theming systems for publication-quality figures

## Installation

```bash
# Basic installation
pip install dspant-viz

# With plotly-resampler for large dataset visualization
pip install dspant-viz plotly-resampler
```

## Architecture Overview

dspant_viz follows a layered, multi-backend architecture:

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

### Core Components

- **`core/`**: Base classes and core functionality
  - `base.py`: Contains `VisualizationComponent` abstract base class
  - `data_models.py`: Pydantic models for structured data representation
  - `internals.py`: Internal utilities and decorators

### Backend Renderers

- **`backends/mpl/`**: Matplotlib rendering implementation
- **`backends/plotly/`**: Plotly rendering implementation with plotly-resampler integration

### Visualization Components

- **`visualization/`**: Specialized visualization components
  - `stream/`: Time series visualization components
  - `spike/`: Spike train visualization components
  - Common visualization functionality

## Key Components

### VisualizationComponent

The base class for all visualization components with a common interface for multi-backend rendering:

```python
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

### TimeSeriesPlot

Efficient multi-channel time series visualization with support for large datasets:

```python
class TimeSeriesPlot(VisualizationComponent):
    """Component for time series visualization optimized for dask arrays"""
    
    def __init__(self, data, sampling_rate, channels=None, ...):
        # Initialize with data and parameters
        
    def plot(self, backend="mpl", **kwargs):
        # Render using specified backend
        
    def get_data(self) -> Dict:
        # Prepare data for rendering
        
    def update(self, **kwargs) -> None:
        # Update parameters
```

### Backend Renderers

Backend renderers transform the data prepared by visualization components into actual visualizations:

```python
# Matplotlib implementation
def render_time_series(data: Dict, ax=None, **kwargs) -> Tuple[Figure, Axes]:
    # Create Matplotlib figure with data
    
# Plotly implementation
def render_time_series(data: Dict, use_resampler=True, **kwargs) -> Union[go.Figure, FigureResampler]:
    # Create Plotly figure with data, optionally using plotly-resampler
```

## Using the Components

### Basic Usage

```python
import numpy as np
import dask.array as da
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot

# Create sample data (or load real data)
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

### Using with Large Datasets

For large datasets, use dask arrays and plotly-resampler:

```python
import numpy as np
import dask.array as da
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot

# Create or load large dataset
data = da.random.normal(0, 1, size=(1000000, 8), chunks=(10000, 8))

# Create TimeSeriesPlot with plotly-resampler
ts_plot = TimeSeriesPlot(
    data=data,
    sampling_rate=30000.0,  # 30 kHz sampling rate
    downsample=False,  # Let plotly-resampler handle downsampling
)

# Render with plotly-resampler
fig = ts_plot.plot(
    backend="plotly",
    use_resampler=True,
    max_n_samples=5000  # Samples shown at each zoom level
)

# Save interactive visualization
fig.write_html("large_dataset.html")
```

## Extending the Library

### Creating a New Visualization Component

1. Subclass `VisualizationComponent` from `core/base.py`
2. Implement the required methods: `plot()`, `get_data()`, and `update()`
3. Create renderer implementations for each backend in `backends/mpl/` and `backends/plotly/`

Example skeleton for a new component:

```python
# visualization/my_component.py
from dspant_viz.core.base import VisualizationComponent

class MyComponent(VisualizationComponent):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        # Initialize component-specific attributes
    
    def get_data(self) -> Dict:
        # Prepare data for rendering
        return {
            "data": {...},
            "params": {...}
        }
    
    def update(self, **kwargs) -> None:
        # Update parameters
        
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

Create corresponding renderer functions in each backend directory:

```python
# backends/mpl/my_component.py
def render_my_component(data: Dict, ax=None, **kwargs):
    # Matplotlib implementation
    
# backends/plotly/my_component.py
def render_my_component(data: Dict, **kwargs):
    # Plotly implementation
```

### Best Practices

1. **Backend Agnostic Components**: Keep visualization logic separate from rendering details
2. **Efficient Data Processing**: Use dask for large dataset handling
3. **Consistent Parameter Naming**: Follow existing parameter naming conventions
4. **Documentation**: Document parameters and behavior clearly
5. **Optimization**: Consider performance for large datasets

## Performance Considerations

- **Chunking**: Use appropriate chunk sizes for dask arrays (typically 10000-30000 samples)
- **Downsampling**: For large datasets:
  - Use `plotly-resampler` with Plotly backend
  - Use the `downsample` parameter with Matplotlib backend
- **Memory Management**: Monitor memory usage with large datasets

## Debugging Tips

1. **Check Data Shape**: Ensure data has correct dimensions (samples × channels)
2. **Backend Errors**: Check specific backend renderer implementation
3. **Plotly-Resampler Issues**: Verify installation and compatibility
4. **Memory Errors**: Adjust chunk sizes for dask arrays

## Contributing

Contributions are welcome! To add a new feature:

1. Fork the repository
2. Create a feature branch
3. Add your functionality following the architecture principles
4. Add appropriate tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.