# %%
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tdt

# %% Set up paths
home = Path(r"E:/jpenalozaa/")
tank_path = home / "emgContusion/25-02-12_9882-1_testSubject_emgContusion"
block_path = tank_path / "16-49-56_stim"

# %% Read the block
test = tdt.read_block(str(block_path))
# %%
# Select the stream you're interested in
stream = "RawG"

# Extract relevant metadata from the stream
source = "tdt"

base_metadata = {
    "name": test.streams[stream].name,
    "fs": float(
        test.streams[stream].fs
    ),  # Ensure float for proper numerical representation
    "number_of_samples": test.streams[stream].data.shape[1],
    "data_shape": test.streams[stream].data.shape,
    "channel_numbers": len(test.streams[stream].channel),
    "channel_names": [str(ch) for ch in np.arange(test.streams[stream].data.shape[0])],
    "channel_types": [
        str(test.streams[stream].data[i, :].dtype)
        for i in np.arange(test.streams[stream].data.shape[0])
    ],
}

other_metadata = {
    "code": int(test.streams[stream].code),  # Convert np.uint32 to int
    "size": int(test.streams[stream].size),
    "type": int(test.streams[stream].type),
    "type_str": test.streams[stream].type_str,
    "ucf": str(test.streams[stream].ucf),  # Keep bool as string for consistency
    "dform": int(test.streams[stream].dform),
    "start_time": float(test.streams[stream].start_time),  # Convert np.float64 to float
    "channel": [
        str(ch) for ch in test.streams[stream].channel
    ],  # Convert to list of strings
}
# Combine into final JSON structure
metadata_json = {
    "source": source,
    "base": base_metadata,
    "other": other_metadata,
}

# %% Create the folder
folder_name = f"{test.streams[stream].name}.ant"
folder_path = Path(folder_name)
folder_path.mkdir(parents=True, exist_ok=True)

# Define file paths
metadata_path = folder_path / f"{stream}_metadata.json"
parquet_path = folder_path / f"{stream}_data.parquet"

# Save metadata to JSON file
with open(metadata_path, "w") as metadata_file:
    json.dump(metadata_json, metadata_file, indent=4)

# %% Prepare Parquet data
data = test.streams[stream].data
column_names = [str(i) for i in range(len(data))]  # Column names as string indices

# Convert metadata for embedding in Parquet
metadata_parquet = {
    key: json.dumps(value) if isinstance(value, (list, dict)) else str(value)
    for key, value in {**base_metadata, **other_metadata}.items()
}

# Create PyArrow Table with metadata
data_table = pa.Table.from_arrays(
    data,
    names=column_names,
    metadata={key.encode(): value.encode() for key, value in metadata_parquet.items()},
)

# Write Parquet file
pq.write_table(data_table, parquet_path, compression="snappy")

print(f"✅ Metadata saved to {metadata_path}")
print(f"✅ Data saved to {parquet_path}")


# %%
