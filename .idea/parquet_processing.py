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
file_path = f"{stream}.ant/data_{stream}.parquet"

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
lowcut = 15  # Lower cutoff for bandpass
highcut = 2000  # Upper cutoff for bandpass
fs = int(float(metadata[b"fs"]))  # Sampling frequency (Hz)

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
    apply_notch_filter, notch_freq=60, fs=fs, quality_factor=20, dtype=dask_array.dtype
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
    notch_filtered_data[:10000, 0]
)  # Plot the first 1000 samples of the notch-filtered signal
plt.title("Notch Filtered Signal (60Hz)")
plt.show()


# %%
