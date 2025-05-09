"""
Script to showcase spike density plots
Author: Jesus Penaloza
"""

# %%

import os
import time
from pathlib import Path

import dask.array as da
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode
from dspant.processors.extractors.template_extractor import (
    extract_template_distributions,
)
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_neuroproc.processors.detection.peak_detector import (
    create_negative_peak_detector,
)
from dspant_neuroproc.visualization.spike_density_plots import (
    plot_combined_raster_density,
    plot_spike_density,
)

sns.set_theme(style="darkgrid")
load_dotenv()
# %%

data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(
    r"papers\2025_mp_emg diaphragm acquisition and processing\drv_15-40-17_stim\MonA.ant"
)

#     r"../data/24-12-16_5503-1_testSubject_emgContusion/drv_01_baseline-contusion"

# %%
stream_stim = StreamNode(str(emg_path))
stream_stim.load_metadata()
stream_stim.load_data()
# Print stream_emg summary
stream_stim.summarize()

fs = stream_stim.fs  # Get sampling rate from the stream node
# %%
stim_plots = plot_multi_channel_data(stream_stim.data, fs=fs, time_window=[200, 800])

# %%
pd_processor = create_negative_peak_detector(100, threshold_mode="absolute")
start = int(500 * fs)
end = int(800 * fs)
a = pd_processor.process(stream_stim.data[start:end, :1], fs=fs)
a = pd_processor.to_dataframe(a, fs)
# %%
a

# %%

we_processor = WaveformExtractor(stream_stim.data[start:end, :1], fs)
ab = we_processor.extract_waveforms(a["index"], time_unit="samples")
# %%

ac = extract_template_distributions(ab[0], normalization="minmax")
# %%
plt.plot(ac["template_mean"])

# %%
data_to_plot = stream_stim.data[start:end, :1]
plt.plot(data_to_plot[0:30, 0])
# %%
