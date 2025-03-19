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

## Standard Installation

For most users, the simplest way to install dspant is via pip:

```bash
pip install dspant
```

This will install dspant and all its required dependencies.

## Installation with Optional Features

dspant offers several optional features that can be installed depending on your needs:

```bash
# Install with TDT support
pip install dspant[tdt]

# Install with advanced visualization tools
pip install dspant[viz]

# Install with all optional dependencies
pip install dspant[all]
```

## Development Installation

If you want to contribute to dspant or use the latest development version, you can install directly from the repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/dspant.git
cd dspant

# Install in development mode
pip install -e .

# Install with development dependencies (testing, documentation, etc.)
pip install -e .[dev]
```

## Installing in a Virtual Environment

It's recommended to install dspant in a virtual environment to avoid conflicts with other packages:

```bash
# Create a virtual environment
python -m venv dspant-env

# Activate the environment
# On Windows:
dspant-env\Scripts\activate
# On macOS/Linux:
source dspant-env/bin/activate

# Install dspant
pip install dspant
```

## GPU Support

For accelerated processing on NVIDIA GPUs:

```bash
# Install with CUDA support
pip install dspant[cuda]
```

Note: GPU acceleration requires an NVIDIA GPU with CUDA support and appropriate CUDA drivers installed.

## Troubleshooting

### Common Installation Issues

#### Missing Compiler

Some dependencies may require a C compiler. If you encounter build errors:

- **Windows**: Install Visual C++ Build Tools
- **macOS**: Install Xcode Command Line Tools (`xcode-select --install`)
- **Linux**: Install GCC (`sudo apt-get install build-essential` or equivalent)

#### Memory Errors During Installation

If you encounter memory errors while installing, try:

```bash
pip install --no-cache-dir dspant
```

#### ImportError After Installation

If you encounter import errors after installation, ensure that you have all required dependencies:

```bash
pip install --upgrade -r requirements.txt
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