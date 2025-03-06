"""
Functions to extract onset detection and segments from EMG data
Author: Jesus Penaloza (Updated with envelope detection, onset detection, and segment extraction)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
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

# Import the new segment extraction functionality
from dspant.processor.segments import (
    create_centered_extractor,
    create_fixed_window_extractor,
    create_onset_offset_extractor,
)

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
# %%
notch_filter = ButterFilter("bandstop", (58, 62), order=4, fs=fs)
fig_notch = notch_filter.plot_frequency_response(
    title="60 Hz Notch Filter", cutoff_lines=True, freq_scale="log", y_min=-80
)
plt.show()
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
# ======= ENVELOPE DETECTION TESTING =======
# Create and apply multiple envelope detection methods

# First, get the filtered data as our base
base_data = filter_data


# %%
# 4. Create TKEO envelope pipeline
tkeo_pipeline = create_tkeo_envelope(
    method="modified", rectify=True, smooth=True, cutoff_freq=20, fs=fs
)

# Add to processing node
processor_tkeo = create_processing_node(stream_emg, name="TKEO")
# First add basic filtering
processor_tkeo.add_processor([notch_processor, bandpass_processor], group="preprocess")
# Then add envelope processors
for proc in tkeo_pipeline.get_group_processors("envelope"):
    processor_tkeo.add_processor(proc, group="envelope")

# Process data
tkeo_data = processor_tkeo.process(group=["preprocess", "envelope"]).persist()

# %%
# Define channel to analyze
channel_id = 1  # Change as needed for your data

# Create onset detectors with absolute thresholds
abs_detector = EMGOnsetDetector(
    threshold_method="absolute",  # Use absolute threshold
    threshold_value=tkeo_data.mean().compute(),  # Adjust this value based on your data scale
    min_duration=0.1,  # Minimum 100ms duration for valid activation
)
channel_data = tkeo_data[:, :].persist()
# Apply onset detection to envelope
# First to TKEO envelope (often gives good results for onset detection)
tkeo_onsets_abs = abs_detector.process(data=channel_data, fs=fs)

# Convert to Polars DataFrame for easier analysis
tkeo_abs_df = abs_detector.to_dataframe(tkeo_onsets_abs.compute())
# %%
# Filter for our channel of interest
tkeo_abs_df_ch = tkeo_abs_df.filter(pl.col("channel") == channel_id)
print(tkeo_abs_df_ch)
# Print activation statistics
print("\nAbsolute threshold activations stats:")
print(f"Total activations: {len(tkeo_abs_df_ch)}")
if len(tkeo_abs_df_ch) > 0:
    print(f"Mean duration: {tkeo_abs_df_ch['duration'].mean():.3f} s")
    print(f"Mean amplitude: {tkeo_abs_df_ch['amplitude'].mean():.3f}")

# %%
# ======= SEGMENT EXTRACTION =======
# Now, let's extract segments from the data using our new module

# 1. Extract segments using onset-offset pairs
# Get onsets and offsets for a specific channel from our detection results
onsets = tkeo_abs_df_ch.select("onset_idx").to_numpy().flatten()
offsets = tkeo_abs_df_ch.select("offset_idx").to_numpy().flatten()

# Sort onsets and offsets if needed
idx_sort = np.argsort(onsets)
onsets = onsets[idx_sort]
offsets = offsets[idx_sort]

print(f"Number of onsets: {len(onsets)}")
print(f"Example onsets: {onsets[:5]}")
print(f"Example offsets: {offsets[:5]}")

# %%
# Create an onset-offset extractor with padding to make all segments equal length
onset_offset_extractor = create_onset_offset_extractor(
    overlap_allowed=True,
    pad_mode="constant",
    pad_value=0.0,
    equalize_lengths=True,  # This ensures all segments have the same length
)
channel_data = tkeo_data[:, :].persist()
# Extract segments from filtered data using onset-offset pairs
segments_oo = onset_offset_extractor.process(
    data=channel_data,  # Use the filtered data
    onsets=onsets,  # Our detected onset indices
    offsets=offsets,  # Our detected offset indices
).compute()  # Compute to get concrete results

print(f"Extracted {segments_oo.shape[0]} segments using onset-offset method")
print(
    f"Segment shape: {segments_oo.shape}"
)  # Should be (n_segments, max_length, n_channels)

# %%
# ======= VISUALIZATION =======
# Let's visualize some of the extracted segments

# Choose a specific segment to visualize (select one with clear EMG activity)
segment_idx = 100 if len(onsets) > 0 else None

if segment_idx is not None:
    # Create a figure to visualize all three types of segments
    plt.figure(figsize=(15, 10))

    # 1. Plot onset-offset segment
    plt.subplot(1, 1, 1)
    segment_time = np.arange(segments_oo[segment_idx, :, channel_id].shape[0]) / fs
    plt.plot(segment_time, segments_oo[segment_idx, :, channel_id])
    plt.title(f"Onset-Offset Segment {segment_idx} (Channel {channel_id})")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # Mark the onset and offset points on the onset-offset segment
    # For onset-offset, the onset is at the beginning (0)
    plt.axvline(x=0, color="g", linestyle="--", label="Onset")
    segment_duration = (offsets[segment_idx] - onsets[segment_idx]) / fs
    plt.axvline(x=segment_duration, color="r", linestyle="--", label="Offset")
    plt.legend()

    plt.tight_layout()
    # plt.savefig("emg_segments_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
# %%
# Example of processing extracted segments
# Calculate features for each segment

if len(onsets) > 0:
    # Create a dataframe to store segment features
    segment_features = []

    # Calculate features for each onset-offset segment
    for i in range(segments_oo.shape[0]):
        segment = segments_oo[i, :, channel_id]

        # Remove padding (zeros at the end) if equalize_lengths was used
        if np.any(segment):  # Only if segment contains non-zero values
            # Find the last non-zero value
            non_zero_indices = np.where(segment != 0)[0]
            if len(non_zero_indices) > 0:
                last_non_zero = non_zero_indices[-1]
                segment = segment[: last_non_zero + 1]

        # Calculate features
        max_amplitude = np.max(np.abs(segment))
        mean_amplitude = np.mean(np.abs(segment))
        segment_energy = np.sum(segment**2)
        segment_rms = np.sqrt(np.mean(segment**2))
        segment_duration = (offsets[i] - onsets[i]) / fs

        # Store in a dictionary
        features = {
            "segment_id": i,
            "onset_idx": onsets[i],
            "offset_idx": offsets[i],
            "onset_time": onsets[i] / fs,
            "offset_time": offsets[i] / fs,
            "duration": segment_duration,
            "max_amplitude": max_amplitude,
            "mean_amplitude": mean_amplitude,
            "energy": segment_energy,
            "rms": segment_rms,
        }

        segment_features.append(features)

    # Convert to Polars DataFrame
    segment_features_df = pl.from_dicts(segment_features)

    # Display the features
    print("\nSegment Features:")
    print(segment_features_df.head(5))

    # Plot feature distributions
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.hist(segment_features_df["duration"], bins=20, alpha=0.7)
    plt.title("Segment Duration Distribution")
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")

    plt.subplot(2, 2, 2)
    plt.hist(segment_features_df["max_amplitude"], bins=20, alpha=0.7)
    plt.title("Maximum Amplitude Distribution")
    plt.xlabel("Max Amplitude")
    plt.ylabel("Count")

    plt.subplot(2, 2, 3)
    plt.hist(segment_features_df["rms"], bins=20, alpha=0.7)
    plt.title("RMS Distribution")
    plt.xlabel("RMS")
    plt.ylabel("Count")

    plt.subplot(2, 2, 4)
    plt.hist(segment_features_df["energy"], bins=20, alpha=0.7)
    plt.title("Energy Distribution")
    plt.xlabel("Energy")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig("emg_segment_features.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
channel_id = 1
# Calculate average segment and plot with confidence intervals
if len(onsets) > 0:
    # Calculate average segment (across all segments)
    avg_segment = np.mean(segments_oo[:, :, channel_id], axis=0)

    # Calculate confidence intervals (using standard deviation)
    std_segment = np.std(segments_oo[:, :, channel_id], axis=0)

    # Create percentile lines for multiple confidence levels
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    percentile_lines = np.percentile(segments_oo[:, :, channel_id], percentiles, axis=0)

    # Plot average waveform with confidence intervals in a style similar to the reference image
    plt.figure(figsize=(8, 5))

    # Plot all percentile lines (light gray)
    for i, p in enumerate(percentiles):
        if p != 50:  # Skip the median for now (will plot separately)
            plt.plot(
                np.arange(avg_segment.shape[0]) / fs,
                percentile_lines[i],
                color="lightgray",
                linewidth=1.0,
            )

    # Plot the average/median as a thick teal line
    plt.plot(
        np.arange(avg_segment.shape[0]) / fs,
        percentile_lines[3],  # This is the 50th percentile (median)
        color="teal",
        linewidth=3.0,
        label="Median",
    )

    # Add a more subtle grid
    plt.grid(True, alpha=0.3)

    # Remove top and right spines
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    # Set the y-axis limits a bit wider than the data range
    data_min = np.min(percentile_lines[0])
    data_max = np.max(percentile_lines[-1])
    y_range = data_max - data_min
    plt.ylim(data_min - 0.1 * y_range, data_max + 0.1 * y_range)

    # Add labels and title with year-like label similar to reference image
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("EMG Segment Waveforms")

    # Add a year-like label on the top right
    plt.text(
        0.95,
        0.95,
        f"n={segments_oo.shape[0]}",
        transform=plt.gca().transAxes,
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="top",
    )

    plt.tight_layout()
    plt.savefig("emg_average_waveform.png", dpi=300, bbox_inches="tight")
    plt.show()
# %%
