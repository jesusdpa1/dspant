# src/dspant/io/exporters/tdt2zarr.py

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import zarr
import zarrs
from numcodecs import Blosc  # Import Blosc from numcodecs

# Configure zarrs as the codec pipeline
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def convert_tdt_stream_to_zarr(
    tdt_struct,
    save_path: Union[str, Path],
    time_segment: Optional[Tuple[float, float]] = None,
    chunk_size: int = 10000,
    compression: Literal["blosc", "zstd", None] = "blosc",
    compression_level: int = 5,
) -> Tuple[Dict[str, Any], Path]:
    """
    Convert a TDT stream to Zarr format.

    Args:
        tdt_struct: TDT stream structure
        save_path: Path where to save the data
        time_segment: Optional tuple of (start_time, end_time) in seconds
                     to extract only a portion of the data
        chunk_size: Number of samples per chunk
        compression: Compression algorithm to use
        compression_level: Compression level

    Returns:
        Tuple of (metadata_dict, zarr_path)
    """
    # Validate TDT structure
    if not hasattr(tdt_struct, "type_str") or tdt_struct.type_str != "streams":
        raise ValueError("Provided data is not a valid TDT stream.")

    # Convert save_path to Path
    save_path = Path(save_path)

    # Extract data
    data = tdt_struct.data
    fs = tdt_struct.fs

    # Handle single-channel data (1D array)
    if data.ndim == 1:
        # Reshape to 2D (1, samples)
        data = data.reshape(1, -1)
        original_shape = data.shape
    else:
        original_shape = data.shape

    # Apply time segment filtering if specified
    segment_info = None
    if time_segment is not None:
        start_time, end_time = time_segment
        # Convert time in seconds to sample indices
        start_sample = max(0, int(start_time * fs))
        end_sample = min(data.shape[1], int(end_time * fs))

        # Update data to only include the requested segment
        data = data[:, start_sample:end_sample]

        # Add time segment info to metadata
        segment_info = {
            "original_samples": original_shape[1],
            "segment_start_time": start_time,
            "segment_end_time": end_time,
            "segment_start_sample": start_sample,
            "segment_end_sample": end_sample,
            "segment_duration": end_time - start_time,
        }

    # Transpose data from (channels, samples) to (samples, channels)
    data = data.T

    # Prepare metadata
    metadata = prepare_metadata(
        tdt_struct,
        data.shape,  # Now (samples, channels)
        segment_info,
        compression,
        compression_level,
        chunk_size,
    )

    # Make sure save_path exists
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    # Create save name
    save_name = tdt_struct.name

    # Add time segment info to filename if provided
    if time_segment is not None:
        start_str = f"{time_segment[0]:.1f}".replace(".", "_")
        end_str = f"{time_segment[1]:.1f}".replace(".", "_")
        save_name = f"{save_name}_t{start_str}-{end_str}"

    # Create the zarr array path
    zarr_path = save_path / f"data_{save_name}.zarr"

    # Create chunks that are optimized for time series access patterns
    # Larger chunks in time dimension, complete channels in each chunk
    chunk_size = min(chunk_size, data.shape[0])  # Don't exceed data size
    chunks = (chunk_size, data.shape[1])

    # Set up codecs for Zarr v3
    codecs = []

    # Add BytesCodec for array->bytes conversion (required for BloscCodec)
    codecs.append(zarr.codecs.BytesCodec())

    # Configure compression codec
    if compression == "blosc":
        codecs.append(
            zarr.codecs.BloscCodec(
                cname="lz4",
                clevel=compression_level,
                shuffle="shuffle",  # Use string enum value instead of integer
            )
        )
    elif compression == "zstd":
        codecs.append(
            zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=compression_level,
                shuffle="shuffle",  # Use string enum value instead of integer
            )
        )

    # Create zarr array with appropriate v3 parameters
    zarr_array = zarr.create(
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        store=str(zarr_path),
        codecs=codecs,
        zarr_format=3,  # Explicitly use Zarr v3 format
    )

    # Write data
    zarr_array[:] = data

    # Write metadata as JSON attribute
    zarr_array.attrs["metadata"] = json.dumps(metadata)

    # Also save metadata as separate file for easier access
    metadata_path = save_path / f"metadata_{save_name}.json"
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    print(f"✅ Data saved to {zarr_path}")
    print(f"✅ Metadata saved to {metadata_path}")

    return metadata, zarr_path


def prepare_metadata(
    tdt_struct,
    data_shape: Tuple[int, int],
    segment_info: Optional[Dict[str, Any]] = None,
    compression: str = "blosc",
    compression_level: int = 5,
    chunk_size: int = 10000,
) -> Dict[str, Any]:
    """
    Prepare metadata from TDT stream structure.

    Args:
        tdt_struct: TDT stream structure
        data_shape: Shape of the data array (samples, channels)
        segment_info: Optional time segment information
        compression: Compression algorithm used
        compression_level: Compression level used
        chunk_size: Chunk size used

    Returns:
        Dictionary containing organized metadata
    """
    # Get channel information
    channel_info = tdt_struct.channel
    if not isinstance(channel_info, (list, np.ndarray)) or len(channel_info) == 1:
        # Convert to list with single element if scalar
        if not isinstance(channel_info, (list, np.ndarray)):
            channel_list = [channel_info]
        else:
            channel_list = list(channel_info)
    else:
        channel_list = list(channel_info)

    # Number of channels is now the second dimension
    n_channels = data_shape[1]
    n_samples = data_shape[0]

    # Create base metadata (now reflecting samples, channels order)
    base_metadata = {
        "name": tdt_struct.name,
        "fs": float(tdt_struct.fs),
        "number_of_samples": n_samples,
        "data_shape": data_shape,  # (samples, channels)
        "channel_numbers": n_channels,
        "channel_names": [str(ch) for ch in range(n_channels)],
        "channel_types": [str(tdt_struct.data.dtype) for _ in range(n_channels)],
        "axis_order": "samples_first",  # Explicitly document the axis order
    }

    # Create other metadata
    other_metadata = {
        "code": int(tdt_struct.code),
        "size": int(tdt_struct.size),
        "type": int(tdt_struct.type),
        "type_str": tdt_struct.type_str,
        "ucf": str(tdt_struct.ucf),
        "dform": int(tdt_struct.dform),
        "start_time": float(tdt_struct.start_time),
        "channel": [str(ch) for ch in channel_list],
        "storage_format": "zarr",
        "zarr_version": "3.0",  # Explicitly document the zarr version
    }

    # Add zarr-specific configuration
    codec_info = {}
    if compression == "blosc":
        codec_info = {
            "cname": "lz4",
            "clevel": compression_level,
            "shuffle": "shuffle",  # Use string value in metadata too
        }
    elif compression == "zstd":
        codec_info = {
            "cname": "zstd",
            "clevel": compression_level,
            "shuffle": "shuffle",  # Use string value in metadata too
        }

    other_metadata["zarr_config"] = {
        "chunk_size": chunk_size,
        "chunks": (chunk_size, n_channels),  # Document actual chunk shape
        "compression": compression,
        "compression_level": compression_level,
        "codec_info": codec_info,
    }

    # Add segment information if available
    if segment_info:
        other_metadata["segment_info"] = segment_info

    # Combine metadata
    metadata = {
        "source": type(tdt_struct).__name__,
        "base": base_metadata,
        "other": other_metadata,
    }

    return metadata
