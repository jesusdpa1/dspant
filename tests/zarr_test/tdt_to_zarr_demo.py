#!/usr/bin/env python3
# tdt_to_zarr_minimal.py
# %%
import os
from pathlib import Path

import dask.array as da
import tdt
import zarr
import zarrs
from dotenv import load_dotenv

# Configure zarrs as the codec pipeline
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

from dspant.io.exporters.tdt2zarr import convert_tdt_stream_to_zarr

# Load environment variables (for DATA_DIR)
load_dotenv()
# %%
# Data loading configuration
data_dir = Path(os.getenv("DATA_DIR"))
tdt_block_path = data_dir.joinpath(
    r"topoMapping/25-03-26_4902-1_testSubject_topoMapping/00_baseline"
)

# Create output directory
output_path = tdt_block_path.parent.joinpath("drv/zarr_test")
output_path.mkdir(parents=True, exist_ok=True)
# %%
# 1. Load TDT data
print(f"Loading TDT block from {tdt_block_path}...")
tdt_block = tdt.read_block(str(tdt_block_path), t1=0, t2=100)
# %%
# Select a stream to convert
stream_name = "RawG"
stream_obj = tdt_block.streams[stream_name]

# 2. Convert and save TDT stream to Zarr
print(f"Converting stream '{stream_name}' to Zarr format...")
metadata, zarr_path = convert_tdt_stream_to_zarr(
    stream_obj,
    save_path=output_path,
    chunk_size=10000,
    compression="zstd",
    compression_level=5,
)

# 3. Load the saved Zarr data
print(f"Loading Zarr data from {zarr_path}...")
# %%
# Method 1: Load using zarrs directly
zarrs_array = zarr.open(str(zarr_path))
print(f"Loaded with zarrs: shape={zarrs_array.shape}, dtype={zarrs_array.dtype}")

# Method 2: Load as a Dask array for processing
dask_array = da.from_zarr(str(zarr_path))
print(f"Loaded with dask: shape={dask_array.shape}, dtype={dask_array.dtype}")

print("Successfully saved and loaded TDT data as Zarr.")

# %%
import matplotlib.pyplot as plt 
#%%
plt.plot(dask_array)
# %%
