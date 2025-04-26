"""
Functions to extract onset detection - Updated with dspant_viz Visualization
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

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode

# Import our Rust-accelerated version for comparison
from dspant.processors.basic.energy_rs import (
    create_tkeo_envelope_rs,
)
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.fir_filters import create_moving_average
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.double_threshold import (
    create_double_threshold_detector,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.visualization.events.events_test import EventAnnotator

# dspant_viz imports
from dspant_viz.visualization.stream.time_series import TimeSeriesPlot
from dspant_viz.visualization.stream.ts_area import TimeSeriesAreaPlot

sns.set_theme(style="darkgrid")
dotenv.load_dotenv()

# %%
base_path = Path(os.getenv("DATA_DIR"))
data_path = base_path.joinpath(
    r"papers\2025_mp_emg diaphragm acquisition and processing\Sample Ventilator Trace"
)

emg_right_path = data_path.joinpath(r"emg_r.ant")
emg_left_path = data_path.joinpath(r"emg_l.ant")
insp_path = data_path.joinpath(r"insp.ant")

# %%
# Load EMG data
stream_emg_r = StreamNode(str(emg_right_path))
stream_emg_r.load_metadata()
stream_emg_r.load_data()
stream_emg_r.summarize()

stream_emg_l = StreamNode(str(emg_left_path))
stream_emg_l.load_metadata()
stream_emg_l.load_data()
stream_emg_l.summarize()

stream_insp = StreamNode(str(insp_path))
stream_insp.load_metadata()
stream_insp.load_data()
stream_insp.summarize()

# %%
# Create and visualize filters before applying them
fs = stream_emg_l.fs  # Get sampling rate from the stream node

# Create filters with improved visualization
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)
lowpass_filter = create_lowpass_filter(50, fs)

# %%
# Create processing node with filters
processor_emg_r = create_processing_node(stream_emg_r)
processor_emg_l = create_processing_node(stream_emg_l)
processor_insp = create_processing_node(stream_insp)

# %%
# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)
lowpass_processor = FilterProcessor(
    filter_func=lowpass_filter.get_filter_function(), overlap_samples=40
)

# %%
# Add processors to the processing node
processor_emg_r.add_processor([notch_processor, bandpass_processor], group="filters")
processor_emg_l.add_processor([notch_processor, bandpass_processor], group="filters")
processor_insp.add_processor([notch_processor, lowpass_processor], group="filters")

# Apply filters and plot results
filtered_emg_r = processor_emg_r.process(group=["filters"]).persist()
filtered_emg_l = processor_emg_l.process(group=["filters"]).persist()
filtered_insp = processor_insp.process(group=["filters"]).persist()

# %%
# Combine filtered data for visualization
filtered_data = da.concatenate([filtered_emg_r, filtered_emg_l, filtered_insp], axis=1)

# %%
# Visualize filtered data with dspant_viz
# Apply themes for better visualization
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")

# Create TimeSeriesPlot
filtered_ts_plot = TimeSeriesPlot(
    data=filtered_data,
    sampling_rate=fs,
    title="Filtered EMG and Inspiration Signals",
    time_window=[100, 110],  # Same as original plot
    normalize=True,
    color_mode="colormap",
    colormap="viridis",
)

# Plot with Matplotlib
plt.figure(figsize=(15, 8))
fig_mpl, ax_mpl = filtered_ts_plot.plot(backend="mpl")
plt.show()

# %%
# TKEO Processing
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=4)
tkeo_data = tkeo_processor.process(filtered_emg_l, fs=fs).persist()

# Normalize TKEO data
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()
zscore_insp = zscore_processor.process(filtered_insp).persist()

# %%
# Activity Detection
st_tkeo_processor = create_double_threshold_detector(
    primary_threshold=0.035,
    secondary_threshold=0.045,
    refractory_period=0.02,
    min_contraction_duration=0.01,
)

st_insp_processor = create_double_threshold_detector(
    primary_threshold=0.1,
    secondary_threshold=0.2,
    secondary_points_required=100,
    refractory_period=0.02,
    min_contraction_duration=0.1,
)

# %%
# Detect epochs
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)

insp_epochs = st_insp_processor.process(zscore_insp, fs=fs).compute()
insp_epochs = st_insp_processor.to_dataframe(insp_epochs)

# %%
# Visualize TKEO with Activity Annotations
# Prepare event annotations
tkeo_events = {
    "start": tkeo_epochs["onset_idx"].to_numpy() / fs,
    "end": tkeo_epochs["offset_idx"].to_numpy() / fs,
    "label": [f"Event {i + 1}" for i in range(len(tkeo_epochs))],
}

# Create TimeSeriesAreaPlot for TKEO envelope
tkeo_area_plot = TimeSeriesAreaPlot(
    data=zscore_tkeo,
    sampling_rate=fs,
    title="TKEO Envelope with Activity Epochs",
    time_window=[100, 110],
    fill_to="zero",
    fill_alpha=0.3,
    color_mode="single",
    color="royalblue",
)

# Create event annotator
tkeo_event_annotator = EventAnnotator(
    events=tkeo_events,
    highlight_color="red",
    marker_style="span",
    alpha=0.2,
    label_events=True,
)

# Plot with Matplotlib
plt.figure(figsize=(15, 6))
fig_area_mpl, ax_area_mpl = tkeo_area_plot.plot(backend="mpl")

# Add event annotations
tkeo_event_annotator.plot(backend="mpl", ax=ax_area_mpl)
plt.show()

# %%
# Calculate instantaneous frequency
# Calculate the onset times
onset_times = tkeo_epochs["onset_idx"].to_numpy() / fs

# Calculate the time differences between onsets
time_diffs = np.diff(onset_times)

# Calculate the instantaneous frequency (in Hz)
instantaneous_freq = 1 / time_diffs

# Create an array of time points that correspond to the original signal
time_array = np.arange(len(filtered_emg_l)) / fs

# Optional: Visualize instantaneous frequency
plt.figure(figsize=(12, 4))
plt.plot(onset_times[1:], instantaneous_freq, marker="o")
plt.title("Instantaneous Frequency of EMG Activations")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.grid(True)
plt.show()

# %%
"""
Advanced Candlestick EMG Plot with Logarithmic Frequency Scale
"""
"""
Advanced Candlestick EMG Plot with Logarithmic Frequency Scale
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_log_candlestick_emg(
    onset_times,
    filtered_emg,
    fs,
    start_time=0,
    end_time=None,
    freq_aggregation_window=1.0,  # Now in seconds
    figsize=(15, 8),
):
    """
    Create an advanced plot with log-scaled candlestick frequency on top and EMG signal on bottom.

    Parameters
    ----------
    onset_times : np.ndarray
        Array of onset times
    filtered_emg : np.ndarray
        EMG signal data
    fs : float
        Sampling frequency
    start_time : float, optional
        Start time for visualization (default: 0)
    end_time : float, optional
        End time for visualization (default: None, full data length)
    freq_aggregation_window : float, optional
        Window size for frequency aggregation in seconds (default: 1.0)
    figsize : tuple, optional
        Figure size (default: (15, 8))
    """
    # Set color palette
    palette = sns.color_palette("colorblind")

    # Compute time array
    time_array = np.arange(len(filtered_emg)) / fs

    # Set end time if not provided
    if end_time is None:
        end_time = time_array[-1]

    # Filter data based on time window
    time_mask = (time_array >= start_time) & (time_array <= end_time)
    filtered_time = time_array[time_mask]
    filtered_emg_data = filtered_emg[time_mask]

    # Filter onset times within the time window
    filtered_onsets = onset_times[
        (onset_times >= start_time) & (onset_times <= end_time)
    ]

    # Compute candlestick data
    def aggregate_frequency_to_candlestick(onset_times, aggregation_window=1.0):
        """
        Convert instantaneous frequency to candlestick-like data.
        """
        # Calculate instantaneous frequency
        time_diffs = np.diff(onset_times)
        inst_freq = 1 / time_diffs
        inst_freq_times = onset_times[1:]

        # Create DataFrame
        df = pd.DataFrame({"time": inst_freq_times, "frequency": inst_freq})

        # Group by time windows
        grouped = df.groupby(
            pd.cut(
                df["time"],
                bins=np.arange(
                    start_time, end_time + aggregation_window, aggregation_window
                ),
            )
        )

        # Compute candlestick-like statistics
        candlestick_data = grouped.agg(
            {"time": "mean", "frequency": ["min", "max", "first", "last"]}
        ).reset_index()

        # Flatten column names
        candlestick_data.columns = ["window", "time", "low", "high", "open", "close"]

        return candlestick_data

    # Compute candlestick data
    candlestick_data = aggregate_frequency_to_candlestick(
        filtered_onsets, aggregation_window=freq_aggregation_window
    )

    # Create figure with GridSpec for more control
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Top subplot for candlesticks with log scale
    ax1 = plt.subplot(gs[0])
    plt.title(
        f"Instantaneous Frequency (Aggregated every {freq_aggregation_window}s)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot candlesticks with color variation
    for _, row in candlestick_data.iterrows():
        # Choose color based on whether price closed higher or lower
        color = palette[2] if row["close"] >= row["open"] else palette[3]

        # Ensure non-zero values for log scale
        low = max(row["low"], 1e-10)
        high = max(row["high"], 1e-10)
        open_val = max(row["open"], 1e-10)
        close_val = max(row["close"], 1e-10)

        # Plot the body of the candlestick
        plt.vlines(row["time"], low, high, color="black", linewidth=1.5)
        plt.bar(
            row["time"],
            close_val - open_val,
            bottom=min(open_val, close_val),
            width=freq_aggregation_window * 0.8,
            color=color,
            alpha=0.7,
        )

    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.yscale("log")  # Set logarithmic scale
    plt.grid(True, alpha=0.3, which="both")
    plt.xlim(start_time, end_time)

    # Bottom subplot for EMG signal
    ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.plot(filtered_time, filtered_emg_data, color=palette[0], linewidth=1.5)
    plt.title("EMG Signal", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Remove x-axis ticks from top subplot
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()


# Example usage function
def example_usage():
    """
    Example demonstration of the plotting function
    """
    # Simulate data (replace with your actual data loading)
    duration = 1200  # 20 minutes
    fs = 1000  # 1 kHz sampling rate

    # Generate sample data
    time_array = np.arange(duration * fs) / fs
    filtered_emg = np.sin(2 * np.pi * 10 * time_array) + np.random.normal(
        0, 0.1, duration * fs
    )

    # Simulate onset times
    onset_times = time_array[np.random.choice(len(time_array), 100, replace=False)]
    onset_times.sort()

    # Plot with different time windows and aggregation
    plot_log_candlestick_emg(
        onset_times,
        filtered_emg,
        fs,
        start_time=300,  # 5 minutes in
        end_time=900,  # 15 minutes in
        freq_aggregation_window=1.0,  # 1-second aggregation
    )


# Uncomment to run example
# example_usage()
"""
Advanced Candlestick EMG Plot with Logarithmic Frequency Scale
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_log_candlestick_emg(
    onset_times,
    filtered_emg,
    fs,
    start_time=0,
    end_time=None,
    freq_aggregation_window=5.0,  # Now in seconds
    figsize=(15, 8),
):
    """
    Create an advanced plot with log-scaled candlestick frequency on top and EMG signal on bottom.

    Parameters
    ----------
    onset_times : np.ndarray
        Array of onset times
    filtered_emg : np.ndarray
        EMG signal data
    fs : float
        Sampling frequency
    start_time : float, optional
        Start time for visualization (default: 0)
    end_time : float, optional
        End time for visualization (default: None, full data length)
    freq_aggregation_window : float, optional
        Window size for frequency aggregation in seconds (default: 5.0)
    figsize : tuple, optional
        Figure size (default: (15, 8))
    """
    # Set color palette
    palette = sns.color_palette("colorblind")

    # Compute time array
    time_array = np.arange(len(filtered_emg)) / fs

    # Set end time if not provided
    if end_time is None:
        end_time = time_array[-1]

    # Filter data based on time window
    time_mask = (time_array >= start_time) & (time_array <= end_time)
    filtered_time = time_array[time_mask]
    filtered_emg_data = filtered_emg[time_mask]

    # Filter onset times within the time window
    filtered_onsets = onset_times[
        (onset_times >= start_time) & (onset_times <= end_time)
    ]

    # Compute candlestick data
    def aggregate_frequency_to_candlestick(onset_times, aggregation_window=5.0):
        """
        Convert instantaneous frequency to candlestick-like data.
        """
        # Calculate instantaneous frequency
        time_diffs = np.diff(onset_times)
        inst_freq = 1 / time_diffs
        inst_freq_times = onset_times[1:]

        # Create DataFrame
        df = pd.DataFrame({"time": inst_freq_times, "frequency": inst_freq})

        # Group by time windows
        grouped = df.groupby(
            pd.cut(
                df["time"],
                bins=np.arange(
                    start_time, end_time + aggregation_window, aggregation_window
                ),
            ),
            observed=False,
        )

        # Compute candlestick-like statistics
        candlestick_data = grouped.agg(
            {"time": "mean", "frequency": ["min", "max", "first", "last"]}
        ).reset_index()

        # Flatten column names
        candlestick_data.columns = ["window", "time", "low", "high", "open", "close"]

        # Remove rows with no data
        candlestick_data = candlestick_data.dropna(subset=["time"])

        return candlestick_data

    # Compute candlestick data
    candlestick_data = aggregate_frequency_to_candlestick(
        filtered_onsets, aggregation_window=freq_aggregation_window
    )

    # Create figure with GridSpec for more control
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Top subplot for candlesticks with log scale
    ax1 = plt.subplot(gs[0])
    plt.title(
        f"Instantaneous Frequency (Aggregated every {freq_aggregation_window}s)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot candlesticks with color variation
    for _, row in candlestick_data.iterrows():
        # Choose color based on whether price closed higher or lower
        color = palette[2] if row["close"] >= row["open"] else palette[3]

        # Ensure non-zero values for log scale
        low = max(row["low"], 1e-10)
        high = max(row["high"], 1e-10)
        open_val = max(row["open"], 1e-10)
        close_val = max(row["close"], 1e-10)

        # Plot the body of the candlestick
        plt.vlines(row["time"], low, high, color="black", linewidth=1.5)
        plt.bar(
            row["time"],
            close_val - open_val,
            bottom=min(open_val, close_val),
            width=freq_aggregation_window * 0.8,
            color=color,
            alpha=0.7,
        )

    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.yscale("log")  # Set logarithmic scale
    plt.grid(True, alpha=0.3, which="both")
    plt.xlim(start_time, end_time)

    # Bottom subplot for EMG signal
    ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.plot(filtered_time, filtered_emg_data, color=palette[0], linewidth=1.5)
    plt.title("EMG Signal", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Remove x-axis ticks from top subplot
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()


# Example usage function
def example_usage():
    """
    Example demonstration of the plotting function
    """
    # Simulate data (replace with your actual data loading)
    duration = 1200  # 20 minutes
    fs = 1000  # 1 kHz sampling rate

    # Generate sample data
    time_array = np.arange(duration * fs) / fs
    filtered_emg = np.sin(2 * np.pi * 10 * time_array) + np.random.normal(
        0, 0.1, duration * fs
    )

    # Simulate onset times
    onset_times = time_array[np.random.choice(len(time_array), 100, replace=False)]
    onset_times.sort()

    # Plot with different time windows and aggregation
    plot_log_candlestick_emg(
        onset_times,
        filtered_emg,
        fs,
        start_time=300,  # 5 minutes in
        end_time=900,  # 15 minutes in
        freq_aggregation_window=5.0,  # 5-second aggregation
    )


# Uncomment to run example
example_usage()
"""
Advanced Candlestick EMG Plot with Logarithmic Frequency Scale
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_log_candlestick_emg(
    onset_times,
    filtered_emg,
    fs,
    start_time=0,
    end_time=None,
    freq_aggregation_window=1.0,  # Now in seconds
    figsize=(15, 8),
):
    """
    Create an advanced plot with log-scaled candlestick frequency on top and EMG signal on bottom.

    Parameters
    ----------
    onset_times : np.ndarray
        Array of onset times
    filtered_emg : np.ndarray
        EMG signal data
    fs : float
        Sampling frequency
    start_time : float, optional
        Start time for visualization (default: 0)
    end_time : float, optional
        End time for visualization (default: None, full data length)
    freq_aggregation_window : float, optional
        Window size for frequency aggregation in seconds (default: 1.0)
    figsize : tuple, optional
        Figure size (default: (15, 8))
    """
    # Set color palette
    palette = sns.color_palette("colorblind")

    # Compute time array
    time_array = np.arange(len(filtered_emg)) / fs

    # Set end time if not provided
    if end_time is None:
        end_time = time_array[-1]

    # Filter data based on time window
    time_mask = (time_array >= start_time) & (time_array <= end_time)
    filtered_time = time_array[time_mask]
    filtered_emg_data = filtered_emg[time_mask]

    # Filter onset times within the time window
    filtered_onsets = onset_times[
        (onset_times >= start_time) & (onset_times <= end_time)
    ]

    # Compute candlestick data
    def aggregate_frequency_to_candlestick(onset_times, aggregation_window=1.0):
        """
        Convert instantaneous frequency to candlestick-like data.
        """
        # Calculate instantaneous frequency
        time_diffs = np.diff(onset_times)
        inst_freq = 1 / time_diffs
        inst_freq_times = onset_times[1:]

        # Create DataFrame
        df = pd.DataFrame({"time": inst_freq_times, "frequency": inst_freq})

        # Group by time windows
        grouped = df.groupby(
            pd.cut(
                df["time"],
                bins=np.arange(
                    start_time, end_time + aggregation_window, aggregation_window
                ),
            )
        )

        # Compute candlestick-like statistics
        candlestick_data = grouped.agg(
            {"time": "mean", "frequency": ["min", "max", "first", "last"]}
        ).reset_index()

        # Flatten column names
        candlestick_data.columns = ["window", "time", "low", "high", "open", "close"]

        return candlestick_data

    # Compute candlestick data
    candlestick_data = aggregate_frequency_to_candlestick(
        filtered_onsets, aggregation_window=freq_aggregation_window
    )

    # Create figure with GridSpec for more control
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # Top subplot for candlesticks with log scale
    ax1 = plt.subplot(gs[0])
    plt.title(
        f"Instantaneous Frequency (Aggregated every {freq_aggregation_window}s)",
        fontsize=14,
        fontweight="bold",
    )

    # Plot candlesticks with color variation
    for _, row in candlestick_data.iterrows():
        # Choose color based on whether price closed higher or lower
        color = palette[2] if row["close"] >= row["open"] else palette[3]

        # Ensure non-zero values for log scale
        low = max(row["low"], 1e-10)
        high = max(row["high"], 1e-10)
        open_val = max(row["open"], 1e-10)
        close_val = max(row["close"], 1e-10)

        # Plot the body of the candlestick
        plt.vlines(row["time"], low, high, color="black", linewidth=1.5)
        plt.bar(
            row["time"],
            close_val - open_val,
            bottom=min(open_val, close_val),
            width=freq_aggregation_window * 0.8,
            color=color,
            alpha=0.7,
        )

    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.yscale("log")  # Set logarithmic scale
    plt.grid(True, alpha=0.3, which="both")
    plt.xlim(start_time, end_time)

    # Bottom subplot for EMG signal
    ax2 = plt.subplot(gs[1], sharex=ax1)
    plt.plot(filtered_time, filtered_emg_data, color=palette[0], linewidth=1.5)
    plt.title("EMG Signal", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Remove x-axis ticks from top subplot
    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.tight_layout()
    plt.show()


# Example usage function
def example_usage():
    """
    Example demonstration of the plotting function
    """
    # Simulate data (replace with your actual data loading)
    duration = 1200  # 20 minutes
    fs = 1000  # 1 kHz sampling rate

    # Generate sample data
    time_array = np.arange(duration * fs) / fs
    filtered_emg = np.sin(2 * np.pi * 10 * time_array) + np.random.normal(
        0, 0.1, duration * fs
    )

    # Simulate onset times
    onset_times = time_array[np.random.choice(len(time_array), 100, replace=False)]
    onset_times.sort()

    # Plot with different time windows and aggregation
    plot_log_candlestick_emg(
        onset_times,
        filtered_emg,
        fs,
        start_time=300,  # 5 minutes in
        end_time=900,  # 15 minutes in
        freq_aggregation_window=1.0,  # 1-second aggregation
    )


# Uncomment to run example
# example_usage()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the style
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Generate sample stock data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="B")
opens = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
closes = opens + np.random.normal(0, 2, len(dates))
highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 1, len(dates)))
lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 1, len(dates)))

# Create figure with careful sizing
fig, ax = plt.subplots(figsize=(16, 8))

# Color selection from palette
up_color = palette[2]  # Green-like color for up candles
down_color = palette[3]  # Red-like color for down candles


# Custom candlestick drawing
def draw_candlestick(ax, x, open_price, close_price, high, low, width=0.6, alpha=0.8):
    # Determine candle color
    if close_price >= open_price:
        color = up_color
        body_low = open_price
        body_high = close_price
    else:
        color = down_color
        body_low = close_price
        body_high = open_price

    # Draw the candle body (thick rectangle)
    body_width = width
    ax.add_patch(
        plt.Rectangle(
            (x - body_width / 2, body_low),
            body_width,
            body_high - body_low,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
    )

    # Draw the wick (thin line)
    ax.plot([x, x], [low, high], color="black", linewidth=1)


# Plot candlesticks
for i, (date, open_price, close_price, high, low) in enumerate(
    zip(dates, opens, closes, highs, lows)
):
    draw_candlestick(ax, mdates.date2num(date), open_price, close_price, high, low)

# Set x-axis to use dates
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Set y-axis limits with some padding
ax.set_ylim(min(lows) - 5, max(highs) + 5)

# Customize the plot with aesthetics from the original code
ax.set_title("Stock Price Movement", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Date", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Price", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

# Adjust tick sizes
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

# Add grid with slight transparency
ax.grid(True, linestyle="--", alpha=0.6)

# Tight layout with some padding
plt.tight_layout()

# Show the plot
plt.show()

# %%
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the style
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Generate onset frequency data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="B")


# Create onset frequency data with variation
def generate_onset_frequencies(num_periods, min_freq=1, max_freq=5):
    # Linear interpolation between min and max frequencies
    base_frequencies = np.linspace(min_freq, max_freq, num_periods)

    # Add random variation
    variations = np.random.normal(0, 0.5, num_periods)

    # Combine base frequencies with variations
    frequencies = base_frequencies + variations

    # Clip to ensure within min-max range
    frequencies = np.clip(frequencies, min_freq, max_freq)

    return frequencies


# Generate onset frequency data
opens = generate_onset_frequencies(len(dates))
closes = opens + np.random.normal(0, 0.5, len(dates))
highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.3, len(dates)))
lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.3, len(dates)))

# Calculate moving average
window_size = 5  # 5-day moving average
moving_avg = pd.Series(opens).rolling(window=window_size, center=True).mean()

# Create figure with careful sizing
fig, ax = plt.subplots(figsize=(16, 8))

# Color selection from palette
up_color = palette[2]  # Green-like color for up candles
down_color = palette[3]  # Red-like color for down candles
moving_avg_color = palette[1]  # Third color for moving average


# Custom candlestick drawing
def draw_candlestick(ax, x, open_price, close_price, high, low, width=0.6, alpha=0.8):
    # Determine candle color
    if close_price >= open_price:
        color = up_color
        body_low = open_price
        body_high = close_price
    else:
        color = down_color
        body_low = close_price
        body_high = open_price

    # Draw the candle body (thick rectangle)
    body_width = width
    ax.add_patch(
        plt.Rectangle(
            (x - body_width / 2, body_low),
            body_width,
            body_high - body_low,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
    )

    # Draw the wick (thin line)
    ax.plot([x, x], [low, high], color="black", linewidth=1)


# Plot candlesticks
for i, (date, open_price, close_price, high, low) in enumerate(
    zip(dates, opens, closes, highs, lows)
):
    draw_candlestick(ax, mdates.date2num(date), open_price, close_price, high, low)

# Plot moving average
ax.plot(
    mdates.date2num(dates),
    moving_avg,
    color=moving_avg_color,
    linewidth=3,
    label=f"{window_size}-day Moving Average",
)

# Set x-axis to use dates
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# Set y-axis limits with some padding
ax.set_ylim(min(lows) - 0.5, max(highs) + 0.5)

# Customize the plot with aesthetics from the original code
ax.set_title("Onset Frequency Variation", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Date", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Onset Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

# Adjust tick sizes
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

# Add grid with slight transparency
ax.grid(True, linestyle="--", alpha=0.6)

# Add legend
ax.legend(fontsize=TICK_SIZE)

# Tight layout with some padding
plt.tight_layout()

# Print summary statistics
print("Onset Frequency Data Summary:")
print(f"Average onset frequency: {np.mean(opens):.2f} Hz")
print(f"Minimum onset frequency: {np.min(lows):.2f} Hz")
print(f"Maximum onset frequency: {np.max(highs):.2f} Hz")
print(f"Moving Average Window Size: {window_size} days")

# Show the plot
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the style
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Simulation parameters
total_duration = 3600  # 1 hour in seconds
time_window = 60  # 60-second windows

# Generate time ranges
time_ranges = list(range(0, total_duration + time_window, time_window))
time_ranges = [
    (time_ranges[i], time_ranges[i + 1]) for i in range(len(time_ranges) - 1)
]


# Create onset frequency data with variation
def generate_onset_frequencies(time_ranges, min_freq=1, max_freq=5):
    frequencies = []
    for start, end in time_ranges:
        # Linear interpolation between min and max frequencies
        base_freq = np.interp(start, [0, total_duration], [min_freq, max_freq])

        # Add random variation
        variation = np.random.normal(0, 0.5)

        # Combine base frequency with variation
        freq = np.clip(base_freq + variation, min_freq, max_freq)

        frequencies.append(
            {
                "start_time": start,
                "end_time": end,
                "open": freq,
                "close": freq + np.random.normal(0, 0.5),
                "high": freq + np.abs(np.random.normal(0, 0.3)),
                "low": freq - np.abs(np.random.normal(0, 0.3)),
            }
        )

    return pd.DataFrame(frequencies)


# Generate onset frequency data
onset_data = generate_onset_frequencies(time_ranges)

# Calculate moving average
window_size = 5  # 5-window moving average
moving_avg = onset_data["open"].rolling(window=window_size, center=True).mean()

# Create figure with careful sizing
fig, ax = plt.subplots(figsize=(16, 8))

# Color selection from palette
up_color = palette[2]  # Green-like color for up candles
down_color = palette[3]  # Red-like color for down candles
moving_avg_color = palette[1]  # Third color for moving average


# Custom candlestick drawing
def draw_candlestick(ax, x, open_price, close_price, high, low, width=30, alpha=0.8):
    # Determine candle color
    if close_price >= open_price:
        color = up_color
        body_low = open_price
        body_high = close_price
    else:
        color = down_color
        body_low = close_price
        body_high = open_price

    # Draw the candle body (thick rectangle)
    body_width = width
    ax.add_patch(
        plt.Rectangle(
            (x - body_width / 2, body_low),
            body_width,
            body_high - body_low,
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=alpha,
        )
    )

    # Draw the wick (thin line)
    ax.plot([x, x], [low, high], color="black", linewidth=1)


# Plot candlesticks
for i, row in onset_data.iterrows():
    draw_candlestick(
        ax, row["start_time"], row["open"], row["close"], row["high"], row["low"]
    )

# Plot moving average
ax.plot(
    onset_data["start_time"],
    moving_avg,
    color=moving_avg_color,
    linewidth=3,
    label=f"{window_size}-window Moving Average",
)

# Customize the plot
ax.set_title("Onset Frequency Variation", fontsize=TITLE_SIZE, fontweight="bold")
ax.set_xlabel("Time (seconds)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
ax.set_ylabel("Onset Frequency (Hz)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

# Set x-axis limits
ax.set_xlim(0, total_duration)

# Adjust tick sizes
ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)

# Add grid with slight transparency
ax.grid(True, linestyle="--", alpha=0.6)

# Add legend
ax.legend(fontsize=TICK_SIZE)

# Tight layout with some padding
plt.tight_layout()

# Print summary statistics
print("Onset Frequency Data Summary:")
print(f"Total duration: {total_duration} seconds")
print(f"Window size: {time_window} seconds")
print(f"Average onset frequency: {onset_data['open'].mean():.2f} Hz")
print(f"Minimum onset frequency: {onset_data['low'].min():.2f} Hz")
print(f"Maximum onset frequency: {onset_data['high'].max():.2f} Hz")
print(f"Moving Average Window Size: {window_size} windows")

# Show the plot
plt.show()

# %%
