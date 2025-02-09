# %%
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tdt

# %%
# Set up the paths
home = Path(r"D:\SynapseBK")
tank_path = home.joinpath(r"emgContusion/25-02-05_9877-1_testSubject_emgContusion")
block_path = tank_path.joinpath(r"00_baseline-contusion")
# %%
# Read the block
test = tdt.read_block(str(block_path))

# Select the stream you're interested in
stream = "RawG"

# Extract relevant metadata from the stream
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

# Create a PyArrow Table for the data
data_table = pa.Table.from_arrays(
    data,
    names=[str(i) for i in range(len(data))],  # Give names to each column
)
# %%
# Create a dictionary to store the metadata as key-value pairs
metadata_raw = {
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

metadata_parquet = {
    key: str([str(x) for x in value]) if isinstance(value, list) else str(value)
    for key, value in metadata_raw.items()
}


# %%
# Create the folder
folder_name = f"{name}.ant"
folder_path = Path(folder_name)
folder_path.mkdir(parents=True, exist_ok=True)

# Create the metadata JSON file path
metadata_path = folder_path / f"metadata_{name}.json"
parquet_path = folder_path / f"data_{name}.parquet"

# Save the metadata to the JSON file
with open(metadata_path, "w") as metadata_file:
    json.dump(metadata_parquet, metadata_file, indent=4)


# Add metadata to the schema
data_table = data_table.replace_schema_metadata(metadata_parquet)
# Write the table with the updated schema to a Parquet file
pq.write_table(data_table, parquet_path, compression="snappy")


# %%
import dask
import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import butter, filtfilt


# %%
# Function to create a Butterworth filter
def butter_filter(lowcut, highcut, fs, order=4, btype="bandpass"):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype=btype, analog=False)


# Function to apply the filter lazily on a chunk
def apply_filter(chunk, lowcut, highcut, fs, order=4, btype="bandpass"):
    b, a = butter_filter(lowcut, highcut, fs, order, btype)
    return filtfilt(b, a, chunk, axis=0)


# For notch filter: specify the frequency to notch (e.g., 60 Hz) and adjust the bandpass parameters.
def apply_notch_filter(chunk, notch_freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    low = (notch_freq - 1) / nyquist
    high = (notch_freq + 1) / nyquist
    b, a = butter(4, [low, high], btype="bandstop")
    return filtfilt(b, a, chunk, axis=0)


# %%
# Stream name and file path
stream = "RawG"
file_path = f"{stream}.parquet"

# Memory-map the Parquet file
with pa.memory_map(file_path, "r") as mmap:
    # Read the table without loading into RAM
    table = pq.read_table(mmap)
    metadata = table.schema.metadata
    dask_array = da.from_array(
        table.to_pandas().values, chunks="auto"
    )  # Automatically chunked


# %%
# Set filter parameters
lowcut = 1.0  # Lower cutoff for bandpass
highcut = 30.0  # Upper cutoff for bandpass
fs = 1000  # Sampling frequency (Hz)

# Apply bandpass filter lazily (using Dask)
filtered_array = dask_array.map_blocks(
    apply_filter,
    lowcut=lowcut,
    highcut=highcut,
    fs=fs,
    order=4,
    btype="bandpass",
    dtype=dask_array.dtype,
)

# Apply notch filter lazily (using Dask)
notch_filtered_array = filtered_array.map_blocks(
    apply_notch_filter, notch_freq=60, fs=fs, quality_factor=30, dtype=dask_array.dtype
)

# %%
# You can now compute the results when necessary (e.g., for visualization)
# Here we compute only the filtered data
# filtered_data = filtered_array.compute()
notch_filtered_data = notch_filtered_array.compute()

# %%
# Plot the results for visualization (now that it's computed)
import matplotlib.pyplot as plt

plt.plot(
    notch_filtered_data[:1000000, 0]
)  # Plot the first 1000 samples of the notch-filtered signal
plt.title("Notch Filtered Signal (60Hz)")
plt.show()


# %%
