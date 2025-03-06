"""
Simple EMG data visualization with rerun-sdk
"""
# %%

import logging

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rerun as rr
import rerun.blueprint as rrb
import seaborn as sns

from dspant.emgproc.activity import (
    EMGOnsetDetector,
    create_absolute_threshold_detector,
)
from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.basic import (
    create_tkeo_envelope,
)
from dspant.processor.filters import (
    ButterFilter,
    FilterProcessor,
)

# %%
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
# %%
fig_bp = bandpass_filter.plot_frequency_response(
    show_phase=True, cutoff_lines=True, freq_scale="log", y_min=-80
)
# plt.show()  # This displays and clears the current figure
# plt.savefig("bandpass_filter.png", dpi=300, bbox_inches='tight')
# %%
notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)
fig_notch = notch_filter.plot_frequency_response(
    title="60 Hz Notch Filter", cutoff_lines=True, freq_scale="log", y_min=-80
)
plt.show()  # This displays and clears the current figure
# plt.savefig("notch_filter.png", dpi=300, bbox_inches='tight')
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
# Set up logging
# Set up logging
# Set up logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rerun
rr.init("EMG Analysis", spawn=True)

# Load the data from your script
logger.info("Loading filtered EMG data...")

# We'll reuse the variables from your script
# stream_emg and filter_data should already be loaded

# Get the sampling rate
fs = stream_emg.fs
logger.info(f"Sampling rate: {fs} Hz")

# Get data dimensions
n_channels = filter_data.shape[1]
logger.info(f"Number of channels: {n_channels}")

# Select a window to visualize (3 seconds)
window_seconds = 3
window_samples = int(window_seconds * fs)

# Time axis in seconds
time = np.arange(window_samples) / fs

# Setup styling for each channel - limit to max 4 channels for better visualization
channels_to_show = min(n_channels, 4)

for ch in range(channels_to_show):
    # Set up styling for raw data
    rr.log(
        f"emg/channel_{ch}/raw",
        rr.SeriesLine(color=[100, 100, 100, 255], name=f"Raw EMG - Ch {ch}", width=1),
        static=True,
    )

    # Set up styling for filtered data
    rr.log(
        f"emg/channel_{ch}/filtered",
        rr.SeriesLine(color=[0, 0, 255, 255], name=f"Filtered EMG - Ch {ch}", width=2),
        static=True,
    )

    # Set up styling for difference
    rr.log(
        f"emg/channel_{ch}/difference",
        rr.SeriesLine(color=[255, 0, 0, 255], name=f"Removed Noise - Ch {ch}", width=1),
        static=True,
    )

# Log the data for each channel efficiently
for ch in range(channels_to_show):
    # Get raw and filtered data
    raw_data = stream_emg.data[:window_samples, ch].compute()
    filtered_data = filter_data[:window_samples, ch].compute()
    diff_data = raw_data - filtered_data

    # Log each data point with its timestamp
    for i in range(window_samples):
        t = time[i]
        rr.set_time_seconds("time", t)

        # Log scalar data points
        rr.log(f"emg/channel_{ch}/raw", rr.Scalar(raw_data[i]))
        rr.log(f"emg/channel_{ch}/filtered", rr.Scalar(filtered_data[i]))
        rr.log(f"emg/channel_{ch}/difference", rr.Scalar(diff_data[i]))

# Create custom time series views for each channel
views = []
for ch in range(channels_to_show):
    # Create a TimeSeriesView for each channel
    views.append(
        rrb.TimeSeriesView(
            origin=f"/emg/channel_{ch}",
            name=f"Channel {ch}",
            # Show all three series in this view
            contents=[f"$origin/raw", f"$origin/filtered", f"$origin/difference"],
            # Set custom Y axis range if needed
            # axis_y=rrb.ScalarAxis(zoom_lock=False),
            # Configure legend
            plot_legend=rrb.PlotLegend(visible=True),
        )
    )

# Create a blueprint to organize the views
blueprint = rrb.Blueprint(
    *views,
    # Optional: Configure panels
    blueprint_panel=rrb.BlueprintPanel(state="collapsed"),
    selection_panel=rrb.SelectionPanel(state="collapsed"),
    time_panel=rrb.TimePanel(state="expanded"),
)

# Send the blueprint to Rerun
rr.send_blueprint(blueprint)

logger.info("Visualization complete!")

# %%
