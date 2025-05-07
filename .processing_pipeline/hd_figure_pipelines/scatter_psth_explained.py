"""
Create scatterplot of spike counts between two neurons with regression line
Using EMG onsets as trial markers
Author: Jesus Penaloza
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from dotenv import load_dotenv

from dspant.engine import create_processing_node
from dspant.io.loaders.phy_kilosort_loarder import load_kilosort
from dspant.nodes import StreamNode
from dspant.processors.basic.energy_rs import create_tkeo_envelope_rs
from dspant.processors.basic.normalization import create_normalizer
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import (
    create_bandpass_filter,
    create_notch_filter,
)
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)

# %%
# Set up plotting style
sns.set_theme(style="ticks")
plt.rcParams.update({"font.size": 12})
load_dotenv()

# Load data
data_dir = Path(os.getenv("DATA_DIR"))
emg_path = data_dir.joinpath(r"test_/00_baseline/drv_00_baseline/RawG.ant")
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)
# %%
# Load EMG and spike data
print("Loading data...")
stream_emg = StreamNode(str(emg_path))
stream_emg.load_metadata()
stream_emg.load_data()
sorter_data = load_kilosort(sorter_path)
fs = stream_emg.fs
print(f"Data loaded. Sampling rate: {fs} Hz")

# Process EMG data
print("Processing EMG data...")
processor_emg = create_processing_node(stream_emg)

# Create filters
bandpass_filter = create_bandpass_filter(10, 4000, fs=fs, order=5)
notch_filter = create_notch_filter(60, q=60, fs=fs)

# Create processors
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Filter EMG data
processor_emg.add_processor([notch_processor, bandpass_processor], group="filters")
filtered_emg = processor_emg.process(group=["filters"]).persist()

# Apply TKEO
tkeo_processor = create_tkeo_envelope_rs(method="modified", cutoff_freq=15)
tkeo_data = tkeo_processor.process(filtered_emg, fs=fs).persist()

# Normalize TKEO
zscore_processor = create_normalizer("minmax")
zscore_tkeo = zscore_processor.process(tkeo_data).persist()

# Detect EMG onsets
print("Detecting EMG contractions...")
st_tkeo_processor = create_single_threshold_detector(
    threshold_method="absolute",
    threshold_value=0.1,
    refractory_period=0.1,  # 100ms refractory period to avoid duplicates
    min_contraction_duration=0.025,  # 25ms minimum duration
)
tkeo_epochs = st_tkeo_processor.process(zscore_tkeo, fs=fs).compute()
tkeo_epochs = st_tkeo_processor.to_dataframe(tkeo_epochs)

# Convert onsets to seconds
emg_onsets = tkeo_epochs["onset_idx"].to_numpy() / fs
print(f"Detected {len(emg_onsets)} EMG contractions")

# Define analysis window
WINDOW_DURATION = 1.0  # seconds per trial, centered on EMG onset
PRE_ONSET = 0.2  # seconds before onset
POST_ONSET = 0.8  # seconds after onset (total window = 1.0s)

# Get units to analyze
unit_ids = sorter_data.unit_ids
print(f"Available units: {len(unit_ids)}")

# Define two units to compare
unit_01_idx = 0  # Using first unit
unit_02_idx = 1  # Using second unit

# You can change these indices based on which units you want to analyze
unit_01 = unit_ids[unit_01_idx]
unit_02 = unit_ids[unit_02_idx]

print(f"Analyzing spike counts for Unit {unit_01} vs Unit {unit_02}")


# Function to count spikes within a time window
def count_spikes_in_window(sorter_data, unit_id, start_time, end_time):
    """Count spikes for a given unit within a time window"""
    # Get spike times for this unit in seconds
    unit_spike_times = (
        sorter_data.get_unit_spike_train(unit_id) / sorter_data.sampling_frequency
    )

    # Count spikes within the window
    mask = (unit_spike_times >= start_time) & (unit_spike_times < end_time)
    return np.sum(mask)


# Calculate spike counts for each EMG onset event
spike_counts_01 = []
spike_counts_02 = []

print("Counting spikes around each EMG onset...")
for i, onset_time in enumerate(emg_onsets):
    # Define the analysis window
    start_time = onset_time - PRE_ONSET
    end_time = onset_time + POST_ONSET

    # Check if window is within recording time
    if start_time < 0 or end_time > (sorter_data.spike_times.max() / fs):
        continue

    # Count spikes for each unit
    count_01 = count_spikes_in_window(sorter_data, unit_01, start_time, end_time)
    count_02 = count_spikes_in_window(sorter_data, unit_02, start_time, end_time)

    spike_counts_01.append(count_01)
    spike_counts_02.append(count_02)

    # Progress indicator
    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(emg_onsets)} events")

# Convert to numpy arrays
spike_counts_01 = np.array(spike_counts_01)
spike_counts_02 = np.array(spike_counts_02)

print(f"Analysis complete. Valid events: {len(spike_counts_01)}")
print(
    f"Unit {unit_01} spike count range: {np.min(spike_counts_01)}-{np.max(spike_counts_01)}"
)
print(
    f"Unit {unit_02} spike count range: {np.min(spike_counts_02)}-{np.max(spike_counts_02)}"
)

# Calculate correlation
correlation, p_value = stats.pearsonr(spike_counts_01, spike_counts_02)
print(f"Correlation: r = {correlation:.3f}, p = {p_value:.4f}")

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    spike_counts_01, spike_counts_02
)
print(f"Linear regression: y = {slope:.3f}x + {intercept:.3f}")

# Create the scatterplot
plt.figure(figsize=(7, 7))

# Plot scatter points
plt.scatter(spike_counts_01, spike_counts_02, alpha=0.7, s=40, color="black")

# Add regression line
x_range = np.linspace(min(spike_counts_01), max(spike_counts_01), 100)
plt.plot(x_range, slope * x_range + intercept, "r-", linewidth=2)

# Add a diagonal line (optional - to show y=x reference)
# plt.plot([min(spike_counts_01), max(spike_counts_02)],
#         [min(spike_counts_01), max(spike_counts_02)],
#         'k--', alpha=0.3, linewidth=1)

# Set axis limits with some padding
x_min = max(0, np.min(spike_counts_01) - 2)
x_max = np.max(spike_counts_01) + 2
y_min = max(0, np.min(spike_counts_02) - 2)
y_max = np.max(spike_counts_02) + 2

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Add a sample striped circle for demonstration like in the example image
# This could represent a specific trial or condition
circle_x = np.mean(spike_counts_01)
circle_y = np.mean(spike_counts_02)
circle = plt.Circle(
    (circle_x, circle_y), radius=2, fc="none", ec="black", hatch="///", alpha=1.0
)
plt.gca().add_patch(circle)

# Add axis labels and title
plt.xlabel(f"Spike count Unit {unit_01}")
plt.ylabel(f"Spike count Unit {unit_02}")
plt.title(f"Spike count correlation\nr = {correlation:.3f}, p = {p_value:.4f}")

# Add r-value annotation
plt.text(
    0.05,
    0.95,
    f"y = {slope:.2f}x + {intercept:.2f}",
    transform=plt.gca().transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
)

# Clean up plot
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Show the plot
plt.show()

# Optional: Save figure
# plt.savefig(f"spike_count_correlation_units_{unit_01}_{unit_02}.png", dpi=300)

# Create interactive unit selection
print("Creating interactive unit selection interface...")
try:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    # Get number of units
    n_units = len(unit_ids)

    # Create unit selection widgets
    unit_1_widget = widgets.Dropdown(
        options=[(f"Unit {unit_ids[i]}", i) for i in range(n_units)],
        value=unit_01_idx,
        description="Unit 1:",
    )

    unit_2_widget = widgets.Dropdown(
        options=[(f"Unit {unit_ids[i]}", i) for i in range(n_units)],
        value=unit_02_idx,
        description="Unit 2:",
    )

    window_widget = widgets.FloatSlider(
        value=WINDOW_DURATION,
        min=0.2,
        max=2.0,
        step=0.1,
        description="Window (s):",
        continuous_update=False,
    )

    # Create button to update plot
    update_button = widgets.Button(
        description="Update Plot",
        button_style="primary",
    )

    # Layout widgets horizontally
    controls = widgets.HBox(
        [unit_1_widget, unit_2_widget, window_widget, update_button]
    )

    # Create output area for plot
    output = widgets.Output()

    # Define update function
    def update_plot(b):
        with output:
            clear_output(wait=True)
            u1_idx = unit_1_widget.value
            u2_idx = unit_2_widget.value
            window_duration = window_widget.value

            if u1_idx == u2_idx:
                print("Please select different units")
                return

            unit_01 = unit_ids[u1_idx]
            unit_02 = unit_ids[u2_idx]

            # Calculate PRE and POST onset times
            pre_onset = window_duration * 0.2  # 20% before onset
            post_onset = window_duration * 0.8  # 80% after onset

            # Recalculate spike counts
            spike_counts_01 = []
            spike_counts_02 = []

            for onset_time in emg_onsets:
                start_time = onset_time - pre_onset
                end_time = onset_time + post_onset

                # Check if window is within recording time
                if start_time < 0 or end_time > (sorter_data.spike_times.max() / fs):
                    continue

                count_01 = count_spikes_in_window(
                    sorter_data, unit_01, start_time, end_time
                )
                count_02 = count_spikes_in_window(
                    sorter_data, unit_02, start_time, end_time
                )

                spike_counts_01.append(count_01)
                spike_counts_02.append(count_02)

            spike_counts_01 = np.array(spike_counts_01)
            spike_counts_02 = np.array(spike_counts_02)

            # Calculate correlation and regression
            correlation, p_value = stats.pearsonr(spike_counts_01, spike_counts_02)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                spike_counts_01, spike_counts_02
            )

            # Create plot
            plt.figure(figsize=(7, 7))

            # Plot scatter points
            plt.scatter(
                spike_counts_01, spike_counts_02, alpha=0.7, s=40, color="black"
            )

            # Add regression line
            x_range = np.linspace(min(spike_counts_01), max(spike_counts_01), 100)
            plt.plot(x_range, slope * x_range + intercept, "r-", linewidth=2)

            # Set axis limits with some padding
            x_min = max(0, np.min(spike_counts_01) - 2)
            x_max = np.max(spike_counts_01) + 2
            y_min = max(0, np.min(spike_counts_02) - 2)
            y_max = np.max(spike_counts_02) + 2

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Add a sample striped circle
            circle_x = np.mean(spike_counts_01)
            circle_y = np.mean(spike_counts_02)
            circle = plt.Circle(
                (circle_x, circle_y),
                radius=2,
                fc="none",
                ec="black",
                hatch="///",
                alpha=1.0,
            )
            plt.gca().add_patch(circle)

            # Add axis labels and title
            plt.xlabel(f"Spike count Unit {unit_01}")
            plt.ylabel(f"Spike count Unit {unit_02}")
            plt.title(
                f"Spike count correlation\nr = {correlation:.3f}, p = {p_value:.4f}"
            )

            # Add r-value annotation
            plt.text(
                0.05,
                0.95,
                f"y = {slope:.2f}x + {intercept:.2f}",
                transform=plt.gca().transAxes,
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # Clean up plot
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plt.show()

    # Connect button to update function
    update_button.on_click(update_plot)

    # Display widgets and initial plot
    display(controls)
    display(output)

    # Show initial plot
    with output:
        update_plot(None)

except ImportError:
    print("ipywidgets not available. Using static plot only.")

print("Script execution complete!")

# %%
