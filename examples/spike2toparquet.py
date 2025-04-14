"""
simple guide to export spike2 files to parquet

requires python 3.9 sonpy and neo. numpy version 1.26 since 2.0 has an error when importing
"""

# %%
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from neo.io import CedIO

# %%
# Define paths
home_path = Path(
    r"E:\jpenalozaa\papers\2025_mp_emg diaphragm acquisition and processing\Spike File 4.11.2025"
)
recording_path = home_path / "taylor_recording.smrx"
output_path = home_path / "taylor_recording.ant"
output_path.mkdir(exist_ok=True)

# %%
# Load data
reader = CedIO(filename=recording_path)
block = reader.read()
# %%
segment = block[0].segments[0]
analog_signal = segment.analogsignals[0]

# %%
# Extract raw data
signal_array = analog_signal.magnitude.astype(np.float32)
fs = float(analog_signal.sampling_rate.rescale("Hz").magnitude)
n_samples = signal_array.shape[0]
n_ch = 1 if signal_array.ndim == 1 else signal_array.shape[1]
shape = [n_ch, n_samples]

# Ensure 2D for uniform handling
if signal_array.ndim == 1:
    signal_array = signal_array.reshape(1, -1)

channel_list = list(range(n_ch))
channel_names = [str(ch) for ch in channel_list]
channel_types = [str(signal_array[i, :].dtype) for i in range(n_ch)]
print(signal_array)
# %%
# Build metadata
base_metadata: Dict = {
    "name": "noisy_recording",
    "fs": fs,
    "number_of_samples": n_samples,
    "data_shape": shape,
    "channel_numbers": n_ch,
    "channel_names": channel_names,
    "channel_types": channel_types,
}

other_metadata: Dict = {
    "code": 0,
    "size": int(signal_array.size),
    "type": 0,
    "type_str": "float32",
    "ucf": "µV",
    "dform": 0,
    "start_time": float(segment.t_start.rescale("s").magnitude),
    "channel": channel_names,
    "save_path": str(output_path.relative_to(output_path.anchor)),
}

metadata_dict: Dict = {
    "source": "AnalogSignal",
    "base": base_metadata,
    "other": other_metadata,
}

# Metadata for parquet must be encoded as bytes
metadata_parquet = {
    "source": "AnalogSignal",
    "name": base_metadata["name"],
    "fs": str(base_metadata["fs"]),
    "channel_numbers": str(n_ch),
    "number_of_samples": str(n_samples),
}

# %%
# Convert to PyArrow Table with per-channel columns
pa_arrays = [pa.array(signal_array[:, i]) for i in range(n_ch)]
data_table = pa.Table.from_arrays(
    pa_arrays,
    names=channel_names,
    metadata={k.encode(): v.encode() for k, v in metadata_parquet.items()},
)

# Save Parquet file
save_name = output_path.stem
data_path = output_path / f"data_{save_name}.parquet"
pq.write_table(data_table, data_path, compression="snappy")
print(f"✅ Data saved to {data_path}")

# Save JSON metadata
with open(output_path / f"metadata_{save_name}.json", "w") as f:
    json.dump(metadata_dict, f, indent=4)

print(f"✅ Metadata saved to {output_path / f'metadata_{save_name}.json'}")

# %%
