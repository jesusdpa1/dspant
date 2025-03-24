"""
DSPANT Neural Processing Extension

This package provides neural signal processing capabilities
that extend the core DSPANT package.
"""

__version__ = "0.1.0"

try:
    from dspant_neuroproc._rs import *

    __has_rust_extensions__ = True
except ImportError:
    __has_rust_extensions__ = False


def main():
    """Entry point when module is executed directly"""
    import dspant

    print(f"DSPANT Neural Processing Extension v{__version__}")
    print(f"Using DSPANT core v{dspant.__version__}")
    print(f"Rust extensions available: {__has_rust_extensions__}")
