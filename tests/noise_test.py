"""
Functions to extract MEPs with envelope detection - Complete test script
Author: Jesus Penaloza (Updated with envelope detection)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.basic import (
    TKEOProcessor,
    create_hilbert_envelope,
    create_rectify_smooth_envelope,
    create_rms_envelope,
    create_tkeo_envelope,
)
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)
from dspant.processor.quality_metrics import create_noise_estimation_processor
from dspant.processor.spatial import create_cmr_processor, create_whitening_processor
from dspant.processor.spectral import LFCCProcessor, MFCCProcessor, SpectrogramProcessor

sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"
#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

emg_stream_path = base_path + r"/RawG.ant"
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
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)

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
# %%
# View summary of the processing node
processor_hd.summarize()

# %%
# Apply filters and plot results
filter_data = processor_hd.process(group=["filters"]).persist()
# %%
channel_id = 0
# Plot filtered data
plt.figure(figsize=(15, 6))
time_axis = np.arange(40000) / fs  # Create time axis in seconds

plt.subplot(211)
plt.plot(time_axis, stream_emg.data[0:40000, channel_id], "k", alpha=0.7, label="Raw")
plt.title("Raw Data")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.plot(time_axis, filter_data[0:40000, channel_id], "b", label="Filtered")
plt.title("Filtered Data (Notch + Bandpass)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("filtered_data.png", dpi=300, bbox_inches="tight")
plt.show()

# %%


# Raw data noise estimation
noise_proc = create_noise_estimation_processor(
    method="mad",
    relative_start=0.1,
    relative_stop=0.9,
    random_seed=42,
    max_samples=5000,
    sample_percentage=0.1,
)
# %%
raw_noise_levels = noise_proc.process(stream_emg.data).compute()
print("Raw Data Noise Levels:", raw_noise_levels)
# %%
filtered_noise_levels = noise_proc.process(filter_data).compute()
print("Noise Levels:", filtered_noise_levels)

# %%
