"""
Functions to extract meps - Updated test script
Author: Jesus Penaloza
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.basic import TKEOProcessor
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
    plot_filter_response,
)
from dspant.processor.spatial import create_cmr_processor, create_whitening_processor
from dspant.processor.spectral import LFCCProcessor, MFCCProcessor, SpectrogramProcessor

sns.set_theme(style="darkgrid")
# %%

base_path = (
    r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"
)
# r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"

emg_stream_path = base_path + r"/RawG.ant"
hd_stream_path = base_path + r"/HDEG.ant"
# %%
# Load EMG data
stream_emg = StreamNode(emg_stream_path)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_emg.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
plt.figure(figsize=(15, 10))
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
bandpass_filter.plot_frequency_response(show_phase=True, cutoff_lines=True)
# %%
notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)
notch_filter.plot_frequency_response(title="60 Hz Notch Filter", cutoff_lines=True)
# %%
lowpass_filter = ButterFilter("lowpass", 100, order=4, fs=fs)
lowpass_filter.plot_frequency_response(show_phase=True, cutoff_lines=True)
# %%

highpass_filter = ButterFilter("highpass", 1, order=4, fs=fs)
highpass_filter.plot_frequency_response(show_phase=True, cutoff_lines=True)

# %%
# Create processing node with filters
processor_hd = create_processing_node(stream_emg)
# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
# %%
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
# %%
# Add processors to the processing node
processor_hd.add_processor([notch_processor, bandpass_processor], group="filters")

# View summary of the processing node
processor_hd.summarize()

# %%
# Apply filters and plot results
filter_data = processor_hd.process(["filters"]).persist()
# %%
# Plot filtered data
plt.figure(figsize=(15, 6))
time_axis = np.arange(40000) / fs  # Create time axis in seconds

plt.subplot(211)
plt.plot(time_axis, stream_emg.data[0:40000, 0], "k", alpha=0.7, label="Raw")
plt.title("Raw Data")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(time_axis, filter_data[0:40000, 0], "b", label="Filtered")
plt.title("Filtered Data (Notch + Bandpass)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Create and apply multiple different filter combinations

# Create a filter combination with whitening
processor_whitened = create_processing_node(stream_emg, name="Whitened")

# Create a whitening processor
whitening_processor = create_whitening_processor(apply_mean=True)

# Add processors
processor_whitened.add_processor(
    [notch_processor, bandpass_processor, whitening_processor], group="filters"
)

# Process data
whitened_data = processor_whitened.process(["filters"])

# Plot comparison
plt.figure(figsize=(15, 9))

plt.subplot(311)
plt.plot(time_axis, stream_emg.data[0:20000, 0], "k", alpha=0.7)
plt.title("Raw Data")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(312)
plt.plot(time_axis, filter_data[0:20000, 0], "b")
plt.title("Filtered Data (Notch + Bandpass)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(313)
plt.plot(time_axis, whitened_data[0:20000, 0], "g")
plt.title("Filtered + Whitened Data")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Plot spectrogram of the data
# Create a spectrogram processor
spec_processor = SpectrogramProcessor(n_fft=512, hop_length=128)

# Add it to a new processing node
processor_spec = create_processing_node(stream_emg, name="Spectrogram")
processor_spec.add_processor(
    [notch_processor, bandpass_processor, spec_processor], group="spectral"
)

# Process data
spec_data = processor_spec.process(["spectral"])

# Plot spectrogram
plt.figure(figsize=(12, 8))
plt.imshow(
    10
    * np.log10(spec_data[:, :1000, 0] + 1e-10),  # First 1000 time frames, first channel
    aspect="auto",
    origin="lower",
    cmap="viridis",
    extent=[0, 1000 * 128 / fs, 0, fs / 2],
)
plt.colorbar(label="Power (dB)")
plt.title("Spectrogram of HD-EMG Data")
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")
plt.show()
