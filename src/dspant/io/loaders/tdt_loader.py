"""
Loader for TDT data files.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import tdt

from dspant.core.internals import public_api


@public_api
def load_tdt_block(block_path: Union[str, Path], **kwargs) -> tdt.StructType:
    """
    Load a TDT block into memory.

    Args:
        block_path: Path to the TDT block
        **kwargs: Additional keyword arguments to pass to tdt.read_block

    Returns:
        Loaded TDT data structure
    """
    # Convert to Path if it's a string
    if isinstance(block_path, str):
        block_path = Path(block_path)

    # Ensure path exists
    if not block_path.exists():
        raise FileNotFoundError(f"TDT block path does not exist: {block_path}")

    # Load the block
    return tdt.read_block(str(block_path), **kwargs)


@public_api
def get_available_streams(block: tdt.StructType) -> Dict[str, Dict[str, Any]]:
    """
    Get information about available streams in a TDT block.

    Args:
        block: Loaded TDT block

    Returns:
        Dictionary with stream names and properties
    """
    if not hasattr(block, "streams") or not block.streams:
        return {}

    return {
        name: {
            "fs": stream.fs,
            "shape": stream.data.shape,
            "dtype": str(stream.data.dtype),
        }
        for name, stream in block.streams.items()
    }


@public_api
def get_available_epocs(block: tdt.StructType) -> Dict[str, Dict[str, Any]]:
    """
    Get information about available epocs in a TDT block.

    Args:
        block: Loaded TDT block

    Returns:
        Dictionary with epoc names and properties
    """
    if not hasattr(block, "epocs") or not block.epocs:
        return {}

    return {
        name: {
            "shape": epoc.data.shape
            if hasattr(epoc, "data") and epoc.data is not None
            else None,
            "onset_count": len(epoc.onset)
            if hasattr(epoc, "onset") and epoc.onset is not None
            else 0,
        }
        for name, epoc in block.epocs.items()
    }


@public_api
def list_tdt_tanks(directory: Union[str, Path]) -> List[Path]:
    """
    List TDT tank directories in the specified directory.

    Args:
        directory: Directory to search for TDT tanks

    Returns:
        List of paths to TDT tank directories
    """
    directory = Path(directory) if isinstance(directory, str) else directory

    # Look for directories that might be TDT tanks (contain block subdirectories)
    potential_tanks = [d for d in directory.iterdir() if d.is_dir()]

    # Filter to only include directories that have TDT block subdirectories
    tanks = []
    for tank in potential_tanks:
        # Check for block directories (typically start with a digit)
        blocks = [
            d
            for d in tank.iterdir()
            if d.is_dir()
            and (
                d.name.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9"))
                or d.name.lower().startswith("block")
            )
        ]
        if blocks:
            tanks.append(tank)

    return tanks
