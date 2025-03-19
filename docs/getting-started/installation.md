# Installation

This guide covers how to install dspant and its dependencies for different use cases.

## Prerequisites

dspant requires Python 3.8 or higher and depends on several scientific computing libraries:

- NumPy (>=1.20.0)
- SciPy (>=1.7.0)
- Dask (>=2022.2.0)
- Pandas (>=1.3.0)
- Polars (>=0.15.0)
- PyArrow (>=7.0.0)
- Matplotlib (>=3.4.0)
- Numba (>=0.54.0)

## Current Installation

For current installation 

```bash
git clone #repo
cd # to location
# if uv available
uv sync
# else after creating a venv
pip install -e 
```

### Checking Your Installation

To verify that dspant has been installed correctly:

```python
import dspant
print(dspant.__version__)

# Test loading some basic modules
from dspant.nodes import StreamNode
from dspant.engine import create_processing_node
from dspant.neuroproc.detection import create_negative_peak_detector

print("Installation successful!")
```

## Next Steps

Once you have dspant installed, proceed to the [Quickstart Guide](quickstart.md) to learn how to use the library.