# %%
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import tdt
import zarr

# %%
home = Path.home()
tank_path = home.joinpath(r"data/emgContusion/25-02-05_9877-1_testSubject_emgContusion")
block_path = tank_path.joinpath(r"00_baseline-contusion")
test = tdt.read_block(str(block_path))

# %%
stream = "RawG"
name = test.streams[stream].name
code = test.streams[stream].code
size = test.streams[stream].size
type_ = test.streams[stream].type
type_str = test.streams[stream].type_str
ucf = test.streams[stream].ucf
fs = test.streams[stream].fs
dform = test.streams[stream].dform
start_time = test.streams[stream].start_time
data = test.streams[stream].data
channel = test.streams[stream].channel


# Define the metadata and data
metadata = {
    "name": name,
    "code": code,
    "size": size,
    "type": type_,
    "type_str": type_str,
    "ucf": ucf,
    "fs": fs,
    "dform": dform,
    "start_time": start_time,
    "channel": channel,
}
# %%
# Create a Zarr array
zarray = zarr.open("my_array.zarr", mode="w", shape=data.shape, dtype=data.dtype)
zarray[:] = data

# Add metadata
for key, value in metadata.items():
    zarray.attrs[key] = value

# %%

# Open the Zarr array
zarray = zarr.open("my_array.zarr", mode="r")
# %%
# Create a lazy Dask array from the Zarr array
darray = da.from_zarr("my_array.zarr", storage_options={"mmap": True}, chunks="auto")

# Print the Dask array (no computation is performed at this point)
darray
# %%
plt.plot(darray[1, :].compute())
# %%
# %%
