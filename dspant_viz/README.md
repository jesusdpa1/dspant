# dspant_viz

Multi-backend visualization library for electrophysiology data, with special focus on efficient rendering of large time series datasets.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange.svg)

## Overview

dspant_viz is designed to render large electrophysiology datasets efficiently using multiple backends (matplotlib, plotly). It provides specialized visualizations for time series and spike data with optimization strategies for handling datasets with millions of data points.

![Example Visualization](https://via.placeholder.com/800x400?text=Electrophysiology+Visualization+Example)

### Key Features

- **Multi-backend architecture**: Create visualizations that work with both Matplotlib and Plotly
- **Efficient large dataset handling**: Optimized for time series with millions of data points
- **Dask integration**: Lazy loading and efficient processing of large datasets
- **Plotly-resampler support**: Dynamic resampling of large datasets for interactive visualization
- **Consistent styling**: Theming systems for publication-quality figures
- **Interactive widgets**: Explore your data with IPython/Jupyter widgets

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. Install dspant_viz with:

```bash
# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | bash

# Install dspant_viz with uv
uv pip install dspant_viz

# With plotly-resampler for large dataset visualization
uv pip install "dspant_viz[resampler]"

# With all optional dependencies
uv pip install "dspant_viz[full]"
```

### Using pip

```bash
# Basic installation
pip install dspant_viz

# With plotly-resampler for large dataset visualization
pip install "dspant_viz[resampler]"
```

## Quick Start

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

# Render with plotly
fig_plotly = ts_plot.plot(backend="plotly")
```

### Spike Data Visualization

```python
import numpy as np
from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.spike.raster import RasterPlot

# Create spike data
spike_data = SpikeData(
    spikes={
        0: np.array([0.1, 0.5, 1.0]),
        1: np.array([0.2, 0.6, 1.2, 1.5]),
    }
)

# Create event times
event_times = np.array([0.0, 1.0, 2.0])

# Create raster plot
raster_plot = RasterPlot(
    data=spike_data,
    event_times=event_times,
    pre_time=0.5,
    post_time=0.5,
)

# Plot with different backends
fig_mpl, ax = raster_plot.plot(backend="mpl")
fig_plotly = raster_plot.plot(backend="plotly")
```

## Documentation

- [API Guide](./documents/API_GUIDE.md): Detailed documentation of components and usage
- [Examples](examples/): Example notebooks showing common use cases
- [API Reference](./documents/API_REFERENCE.md): Complete API reference documentation

## Performance Tips

For large datasets:
- Use Dask arrays with appropriate chunk sizes (10,000-30,000 samples)
- Enable dynamic resampling with Plotly (`use_resampler=True`)
- Use the `downsample` parameter with Matplotlib

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use dspant_viz in your research, please cite:

```
@software{dspant_viz,
  author = {The dspant_viz Contributors},
  title = {dspant_viz: Multi-backend visualization library for electrophysiology data},
  url = {https://github.com/yourusername/dspant_viz},
  version = {0.1.0},
  year = {2025},
}
```