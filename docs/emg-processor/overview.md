# EMG Processing Overview

The EMG processing module provides specialized tools for analyzing and processing electromyography (EMG) signals. This module is designed to handle the unique characteristics of EMG data, including noise filtering, feature extraction, and analysis of muscle activation patterns.

## Key Features

- **Signal preprocessing**: Specialized filters for EMG denoising and artifact removal
- **Envelope extraction**: Methods for extracting muscle activation envelopes
- **Feature calculation**: Time and frequency domain feature extraction
- **Burst detection**: Algorithms for detecting muscle activation bursts
- **Fatigue analysis**: Tools for monitoring and analyzing muscle fatigue
- **Visualization**: Specialized plots for EMG signal visualization

## Module Structure

The EMG processing module is organized into several submodules:

- **Preprocessing**: Signal filtering and conditioning
- **Features**: Feature extraction in time and frequency domains
- **Detection**: Muscle activation detection algorithms
- **Analysis**: Advanced analysis tools for muscle activation patterns
- **Visualization**: Specialized plotting functions for EMG data

## Signal Preprocessing

EMG signals often require specialized preprocessing to remove artifacts and noise:

```python
from dspant.processor.filters import create_bandpass_filter
from dspant.processor.emg import create_emg_envelope_extractor

# Create EMG-specific filters
bandpass = create_bandpass_filter(
    low_hz=20,    # High-pass to remove motion artifacts
    high_hz=500,  # Low-pass to remove high-frequency noise
    order=4
)

# Create envelope extractor
envelope = create_emg_envelope_extractor(
    rectify=True,           # Full-wave rectification
    smoothing_ms=100,       # 100ms smoothing window
    method="rms"            # Root Mean Square method
)
```

## Feature Extraction

The module provides tools for extracting common EMG features:

```python
from dspant.processor.emg.features import (
    extract_amplitude_features,
    extract_frequency_features,
    extract_complexity_features
)

# Extract time-domain features
time_features = extract_amplitude_features(
    emg_data,
    features=["rms", "iemg", "mav", "wl", "zc", "ssc"],
    window_ms=250,
    overlap_ms=125
)

# Extract frequency-domain features
freq_features = extract_frequency_features(
    emg_data,
    features=["mnf", "mdf", "pkf", "mnp"],
    fs=1000,  # 1kHz sampling rate
    window_ms=250
)
```

## Muscle Activity Detection

Detect muscle activation bursts in EMG signals:

```python
from dspant.processor.emg.detection import create_muscle_activity_detector

# Create activity detector
detector = create_muscle_activity_detector(
    threshold_method="dynamic_rms",  # Dynamic RMS-based threshold
    threshold_factor=3.0,           # 3x RMS
    min_duration_ms=50,             # Minimum 50ms activation
    merge_distance_ms=25            # Merge activations within 25ms
)

# Apply detector
activations = detector.process(envelope_data, fs=1000)
```

## Fatigue Analysis

Tools for monitoring muscle fatigue during sustained contractions:

```python
from dspant.processor.emg.analysis import compute_fatigue_metrics

# Compute fatigue metrics
fatigue_metrics = compute_fatigue_metrics(
    emg_data,
    fs=1000,
    window_s=1.0,  # 1-second windows
    features=["mnf", "mdf", "rms_slope"]
)
```

## EMG-Specific Processing Pipeline

A complete EMG processing pipeline combining multiple steps:

```python
from dspant.nodes import StreamNode
from dspant.engine import create_processing_node
from dspant.processor.emg import create_emg_processing_pipeline

# Load EMG data
emg_node = StreamNode("path/to/emg_data.ant").load_data()

# Create processing node
proc_node = create_processing_node(emg_node)

# Create and add a complete EMG processing pipeline
emg_pipeline = create_emg_processing_pipeline(
    bandpass_range=(20, 500),     # Bandpass filter range
    notch_freq=50,                # Power line frequency
    envelope_smoothing_ms=100,    # Envelope smoothing window
    envelope_method="rms"         # RMS envelope method
)
proc_node.add_processor(emg_pipeline, group="emg_processing")

# Process data
processed_emg = proc_node.process()
```

## Visualization Functions

Specialized plotting functions for EMG data:

```python
from dspant.processor.emg.visualization import (
    plot_emg_with_envelope,
    plot_muscle_activations,
    plot_frequency_spectrum,
    plot_fatigue_progression
)

# Plot EMG signal with envelope
fig1 = plot_emg_with_envelope(
    raw_emg=emg_node.data,
    envelope=processed_emg,
    fs=emg_node.fs,
    time_window=(10, 15)  # 5-second window
)

# Plot detected muscle activations
fig2 = plot_muscle_activations(
    emg_data=emg_node.data,
    activations=activations,
    fs=emg_node.fs
)

# Plot frequency content changes over time
fig3 = plot_frequency_spectrum(
    emg_data=emg_node.data,
    fs=emg_node.fs,
    window_s=0.5,
    overlap=0.25
)
```

## Integration with Other Modules

The EMG processing module integrates with other dspant components:

- Works with the **StreamNode** system for efficient data handling
- Leverages the **Processing Pipeline** for flexible workflow construction
- Can be combined with **General Signal Processing** tools for custom analyses
- Outputs compatible with **Visualization** tools for publication-ready figures

