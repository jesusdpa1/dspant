"""
script to test dspant_viz with real data
Author: Jesus Penaloza (Updated with envelope detection and onset detection)
"""

# %%
import os
import time
from pathlib import Path

import dask.array as da
import dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import (
    PSTHAnalyzer,
    plot_psth_with_raster,
)
from dspant_viz.core.themes_manager import (
    apply_matplotlib_theme,
    apply_plotly_theme,
    theme_manager,
)
from dspant_viz.widgets.waveform_inspector import WaveformInspector

# Apply the theme before creating the plot
apply_matplotlib_theme()  # For Matplotlib backend
apply_plotly_theme()  # For Plotly backend

# Import components
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot
from dspant_viz.visualization.stream.ts_raster import TimeSeriesRasterPlot

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()


# %%

data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(
    r"topoMapping/25-03-22_4896-2_testSubject_topoMapping/drv/drv_17-02-16_meps/RawG.ant"
)

# %%
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()

fs = stream_emg.fs  # Get sampling rate from the stream node
# %%
# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)
# Create processing node with filters
processor_emg = create_processing_node(stream_emg)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
filtered_emg = processor_emg.process(group=["filters"]).persist()
# %%
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data[0:1000000, :1]).persist()

st_tkeo_processor = create_single_threshold_detector(
    threshold_method="absolute",
    threshold_value=0.045,
    refractory_period=0.01,
    min_contraction_duration=0.01,
)
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)
# %%
emg_onsets = tkeo_epochs["onset_idx"].to_numpy() / fs  # in seconds needs to be
# %%

# Create a TimeSeriesPlot with Plotly backend and plotly-resampler
large_plot = TimeSeriesPlot(
    data=filtered_emg,
    sampling_rate=fs,
    time_window=[0, 40],
    title="Large Neural Recording with Dynamic Resampling and Events",
    color_mode="colormap",
    colormap="Viridis",
    normalize=True,
    grid=True,
    downsample=False,
)

fig_large = large_plot.plot(
    backend="plotly",
    use_resampler=True,
    max_n_samples=int(40 * fs),  # Show 10k points at a time
)
# %%
# # Add complex events to the large dataset plot
# large_event_annotator = EventAnnotator(
#     complex_events,
#     time_mode="seconds",
#     highlight_color="red",
#     marker_style="span",
#     alpha=0.3,
#     label_events=True,
# )
# large_event_annotator.plot(backend="plotly", ax=fig_large)

# Show the plot
fig_large.show()
# %%
