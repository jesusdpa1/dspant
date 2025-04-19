"""
/stft_test.py
Example script for stft test
Author: jpenalozaa
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
from tqdm import tqdm

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processors.filters import ButterFilter, FilterProcessor
from dspant.processors.spatial import create_cmr_processor, create_whitening_processor
from dspant.processors.spectral import SpectrogramProcessor

# Set up styles
sns.set_theme(style="darkgrid")
# %%
# ---- STEP 1: Load and preprocess data (using your exact code) ----

home_path = Path(r"E:\jpenalozaa")
base_path = home_path.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
)

# home_path = Path.home()
# base_path = home_path.joinpath(
#     r"data/topoMapping/25-02-26_9881-2_testSubject_topoMapping/drv/drv_00_baseline"
# )
emg_stream_path = base_path.joinpath(r"RawG.ant")
# %%
# Load EMG data
stream_emg = StreamNode(str(emg_stream_path))
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()
# %%
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)
# %%
# Create processing node with filters
processor = create_processing_node(stream_emg)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Add processors to the processing node
processor.add_processor([notch_processor, bandpass_processor], group="filters")


spectrogram_processor = SpectrogramProcessor(
    n_fft=int(1024 * 2),  # Adjust based on your data characteristics
    hop_length=int(1024 / 4),  # Typically half of n_fft
    center=True,
)
# mfcc_processor = MFCCProcessor(
#     n_mfcc=40, melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 128, "power": 2.0}
# )
processor.add_processor([spectrogram_processor], "TF")
# %%
tf_filtered = processor.process(["filters"]).persist()
tf_stft = processor.process(["filters", "TF"]).compute()

# %%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the figure size and style
fig = plt.figure(figsize=(12, 10))  # Taller figure to accommodate both plots
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Create a 2x1 grid (signal on top, spectrogram on bottom) with space for colorbar
gs = GridSpec(2, 1, height_ratios=[1, 2])  # Spectrogram gets more vertical space

# Set which channel to plot
channel_to_plot = 0  # First channel

# Get dimensions from the STFT output
n_frequencies, n_times, n_channels = tf_stft.shape

# Create frequency scale for y-axis (based on your n_fft=1024)
fs = stream_emg.fs  # Get sampling rate
frequency_scale = np.linspace(0, fs / 2, n_frequencies)

# Calculate total duration and create appropriate time scale
total_duration = tf_filtered.shape[0] / fs
time_scale = np.linspace(0, total_duration, n_times)

# Set maximum time to display (in seconds)
max_time = 2.0  # Adjust as needed
max_time = min(max_time, total_duration)  # Ensure we don't exceed data length

# Calculate how many frames to display and samples for the raw signal
max_frames = int(max_time / total_duration * n_times)
max_samples = int(max_time * fs)

# Create time array for the raw signal
time_array = np.arange(max_samples) / fs

# Extract the magnitude from complex STFT values
if np.iscomplexobj(tf_stft):
    stft_magnitude = np.abs(tf_stft[:, :, channel_to_plot])
else:
    # If already magnitude, just use the values directly
    stft_magnitude = tf_stft[:, :, channel_to_plot]

# Plot the filtered signal on top
ax1 = plt.subplot(gs[0])
ax1.plot(
    time_array,
    tf_filtered[:max_samples, channel_to_plot],
    color=palette[0],
    linewidth=1.5,
)
ax1.set_xlim(0, max_time)
ax1.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.set_ylabel("Amplitude [μV]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.tick_params(labelsize=TICK_SIZE)
ax1.set_title("Filtered EMG Signal", fontsize=SUBTITLE_SIZE, weight="bold")

# Plot the STFT as a spectrogram on bottom
ax2 = plt.subplot(gs[1], sharex=ax1)  # Share x-axis with the top plot
im = ax2.pcolormesh(
    time_scale[:max_frames],
    frequency_scale,
    stft_magnitude[:, :max_frames],
    cmap="viridis",
    norm=LogNorm(
        vmin=np.max(stft_magnitude[:, :max_frames]) / 1000,
        vmax=np.max(stft_magnitude[:, :max_frames]),
    ),
)

# Set axis labels and title
ax2.set_xlabel("Time [s]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_ylabel("Frequency [Hz]", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_title("STFT Spectrogram", fontsize=SUBTITLE_SIZE, weight="bold")
ax2.tick_params(labelsize=TICK_SIZE)

# Limit y-axis to show frequencies up to 500 Hz (adjust as needed)
ax2.set_ylim(0, 2000)

# Add colorbar without affecting the main axes layout
from mpl_toolkits.axes_grid1 import make_axes_locatable

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="-1%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label("Magnitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
cbar.ax.tick_params(labelsize=TICK_SIZE)

# Add overall title
plt.suptitle("EMG Signal Analysis", fontsize=TITLE_SIZE, weight="bold", y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Make room for the suptitle
plt.show()
# %%
