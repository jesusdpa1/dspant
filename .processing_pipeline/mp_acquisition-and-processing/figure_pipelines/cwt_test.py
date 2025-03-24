"""
Example script for integrating CWT processor with EMG analysis
"""

# %%

import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.filters import ButterFilter, FilterProcessor
from dspant.processor.spectral.ssqueeze_base import create_cwt_processor

# Set up styles
os.environ["SSQ_PARALLEL"] = "1"
sns.set_theme(style="darkgrid")
# %%
# ---- STEP 1: Load and preprocess data (using your existing code) ----

# Replace with your data path
home_path = Path(r"E:\jpenalozaa")
base_path = home_path.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)
emg_stream_path = base_path.joinpath(r"RawG.ant")
# %%
# Load EMG data
stream_emg = StreamNode(str(emg_stream_path))
stream_emg.load_metadata()
stream_emg.load_data()
print("EMG data loaded successfully")
stream_emg.summarize()

# Get sampling rate from the stream node
fs = stream_emg.fs

# Create filters for preprocessing
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)

# Create processing node with filters
processor = create_processing_node(stream_emg)

# Create processors for filtering
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Add filters to the processing node
processor.add_processor([notch_processor, bandpass_processor], group="filters")

# %%

tf_filtered = processor.process().persist()

# %%
samples_per_chunk = 5_000
rechunked_array = tf_filtered.rechunk({-2: samples_per_chunk, 1: -1})
# Create processor
processor = create_cwt_processor(wavelet="morlet", squeezing=True)
# %%
# Process with dask array
result = processor.process(tf_filtered[:100000, 0], fs=fs)
# %%
# Check result shape and type
assert isinstance(result, da.Array)
result_computed = result.compute()

# %%
import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy.visuals import imshow

# Extract data - remove singleton dimension if it exists
if result_computed.ndim == 3:
    tf_data = result_computed[:, :, 0]
else:
    tf_data = result_computed

# Create time axis
time_points = np.linspace(0, 100000 / fs, 100000)  # Assuming 10000 time points

# Create frequency axis
# If you ran with preserve_transform=True and have the actual frequencies:
# freqs = result_dict['ssq_freqs']  # For synchrosqueezed transform
# Or use an estimated frequency range:
freqs = np.linspace(0, fs / 2, tf_data.shape[0])[::-1]

# Visualization with SSQueezePy's imshow
plt.figure(figsize=(12, 6))
imshow(
    np.abs(tf_data),
    xticks=time_points,
    yticks=freqs,
    xlabel="Time [sec]",
    ylabel="Frequency [Hz]",
    title="CWT Time-Frequency Representation",
    cmap="magma",
)
plt.show()


# If you want to use the standard visualization method from the example:
def viz(x, Tx):
    plt.figure(figsize=(12, 6))
    plt.imshow(
        np.abs(Tx), aspect="auto", vmin=0, vmax=np.abs(Tx).max() * 0.2, cmap="turbo"
    )
    plt.colorbar(label="Magnitude")
    plt.title("Synchrosqueezed CWT")
    plt.xlabel("Time")
    plt.ylabel("Frequency Bin")
    plt.show()


# Visualize the synchrosqueezed transform
viz(None, tf_data)

# # For a more accurate visualization with proper axes:
# plt.figure(figsize=(12, 6))
# # plt.pcolormesh(time_points, freqs, np.abs(tf_data), shading="gouraud", cmap="turbo")
# # plt.colorbar(label="Magnitude")
# plt.xlabel("Time [sec]")
# plt.ylabel("Frequency [Hz]")
# plt.title("Synchrosqueezed CWT Time-Frequency Representation")
# plt.yscale("log")  # Use log scale for better frequency visualization
# plt.ylim(freqs[1], freqs[-1])  # Exclude frequency 0 for log scale
# plt.tight_layout()
# plt.show()

# %%
