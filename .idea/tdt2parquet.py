# %%
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tdt

# %%
home = Path.home()
tank_path = home.joinpath(r"data/emgContusion/25-02-05_9877-1_testSubject_emgContusion")
block_path = tank_path.joinpath(r"00_baseline-contusion")
test = tdt.read_block(str(block_path))

# %%
test

# %%

# %%
stream = "RawG"
# %%
# Define the data

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

# Create a PyArrow table for the data
data_table = pa.Table.from_arrays(
    data,
    names=[str(i) for i in range(len(data))],  # give names to each columns
)

# Create a dictionary to store the metadata as key-value pairs
metadata = {
    "name": str(name),
    "code": str(code),
    "size": str(size),
    "type": str(type_),
    "type_str": str(type_str),
    "ucf": str(ucf),
    "fs": str(fs),
    "dform": str(dform),
    "start_time": str(start_time),
    "channel": str([str(x) for x in channel]),
}
# %%


# %%

my_schema = pa.schema(
    data_table.schema,
    metadata=metadata,
)
t2 = data_table.schema.add_metadata(metadata)

pq.write_table(data_table, f"{stream}.parquet", compression="snappy")


# %%
import dask.array as da
import matplotlib.pyplot as plt
#%%
# Read the Parquet table lazily
stream = "RawG"
df = da.from(f"{stream}.parquet")
# %%
plt.plot(df["0"][0:10000].compute())

# %%
