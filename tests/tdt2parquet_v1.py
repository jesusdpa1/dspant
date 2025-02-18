# Select the stream you're interested in
tdt_type = "streams"
stream = "RawG"

# Extract relevant metadata from the stream
source = "tdt"

base_metadata = {
    "name": test[tdt_type][stream].name,
    "fs": float(
        test[tdt_type][stream].fs
    ),  # Ensure float for proper numerical representation
    "number_of_samples": test[tdt_type][stream].data.shape[1],
    "data_shape": test[tdt_type][stream].data.shape,
    "channel_numbers": len(test[tdt_type][stream].channel),
    "channel_names": [
        str(ch) for ch in np.arange(test[tdt_type][stream].data.shape[0])
    ],
    "channel_types": [
        str(test[tdt_type][stream].data[i, :].dtype)
        for i in np.arange(test[tdt_type][stream].data.shape[0])
    ],
}

other_metadata = {
    "code": int(test[tdt_type][stream].code),  # Convert np.uint32 to int
    "size": int(test[tdt_type][stream].size),
    "type": int(test[tdt_type][stream].type),
    "type_str": test[tdt_type][stream].type_str,
    "ucf": str(test[tdt_type][stream].ucf),  # Keep bool as string for consistency
    "dform": int(test[tdt_type][stream].dform),
    "start_time": float(
        test[tdt_type][stream].start_time
    ),  # Convert np.float64 to float
    "channel": [
        str(ch) for ch in test[tdt_type][stream].channel
    ],  # Convert to list of strings
}
# Combine into final JSON structure
metadata_json = {
    "source": source,
    "base": base_metadata,
    "other": other_metadata,
}

# %% Create the folder
folder_name = f"../data/{test[tdt_type][stream].name}.ant"
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
