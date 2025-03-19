# General Signal Processing Overview

The general processing module provides a comprehensive set of tools for digital signal processing across various signal types. These processors implement common signal processing operations and can be used as building blocks for more specialized processing pipelines.

## Key Features

- **Filtering**: Various filter types including FIR, IIR, adaptive, and non-linear filters
- **Transforms**: Fourier, wavelet, and Hilbert transforms for frequency and time-frequency analysis
- **Feature extraction**: Amplitude, frequency, and statistical feature computation
- **Signal conditioning**: Normalization, baseline correction, and artifact removal
- **Dimensionality reduction**: PCA, ICA, and other techniques for reducing data complexity
- **Segmentation**: Windowing, epoch extraction, and continuous signal segmentation
- **Resampling**: Up-sampling, down-sampling, and interpolation methods

## Module Structure

The general processing module is organized into several submodules:

- **Filters**: Signal filtering components
- **Transforms**: Signal transformation operations
- **Features**: Feature extraction processors
- **Conditioning**: Signal conditioning processors
- **Dimensionality**: Dimensionality reduction techniques
- **Segmentation**: Signal segmentation tools
- **Resampling**: Sample rate conversion methods

## Filtering Operations

Filter processors provide various frequency-selective operations:

```python
from dspant.processor.filters import (
    create_bandpass_filter,
    create_lowpass_filter,
    create_highpass_filter,
    create_bandstop_filter,
    create_notch_filter
)

# Create a bandpass filter (300-3000 Hz)
bandpass = create_bandpass_filter(
    low_hz=300,
    high_hz=3000,
    order=4,
    filter_type="butterworth"
)

# Create a lowpass filter (cutoff at 100 Hz)
lowpass = create_lowpass_filter(
    cutoff_hz=100,
    order=6,
    filter_type="bessel"
)

# Create a 50 Hz notch filter (for power line noise)
notch = create_notch_filter(
    center_hz=50,
    q_factor=30
)
```

## Transform Operations

Transform processors convert signals between domains:

```python
from dspant.processor.transforms import (
    create_fft_processor,
    create_wavelet_processor,
    create_hilbert_processor
)

# FFT processor for spectral analysis
fft_proc = create_fft_processor(
    window="hann",
    nfft=1024,
    return_type="magnitude"
)

# Wavelet transform for time-frequency analysis
wavelet_proc = create_wavelet_processor(
    wavelet="morlet",
    scales=np.arange(1, 128),
    output="power"
)

# Hilbert transform for envelope and phase analysis
hilbert_proc = create_hilbert_processor(
    output="envelope"  # Options: envelope, phase, complex
)
```

## Feature Extraction

Feature extraction processors compute signal characteristics:

```python
from dspant.processor.features import (
    create_amplitude_feature_extractor,
    create_frequency_feature_extractor,
    create_statistical_feature_extractor
)

# Extract amplitude features
amp_features = create_amplitude_feature_extractor(
    features=["rms", "peak", "peak_to_peak", "mean_abs"],
    window_ms=500,
    step_ms=250
)

# Extract frequency features
freq_features = create_frequency_feature_extractor(
    features=["mean_freq", "median_freq", "band_power"],
    window_ms=1000,
    bands=[(0.5, 4), (4, 8), (8, 13), (13, 30)]  # Standard EEG bands
)

# Extract statistical features
stat_features = create_statistical_feature_extractor(
    features=["mean", "std", "skewness", "kurtosis", "percentile"],
    percentiles=[5, 50, 95]
)
```

## Signal Conditioning

Signal conditioning processors prepare signals for analysis:

```python
from dspant.processor.conditioning import (
    create_normalizer,
    create_baseline_corrector,
    create_artifact_remover
)

# Normalize signal to z-scores
normalizer = create_normalizer(
    method="zscore",  # Options: zscore, minmax, robust
    axis=0
)

# Correct baseline using polynomial fitting
baseline_corrector = create_baseline_corrector(
    method="polynomial",
    polynomial_order=3,
    baseline_region=(0, 1000)  # First 1000 samples
)

# Remove artifacts using threshold-based approach
artifact_remover = create_artifact_remover(
    method="threshold",
    threshold=5.0,  # 5 standard deviations
    window_ms=200
)
```

## Dimensionality Reduction

Dimensionality reduction processors simplify complex signals:

```python
from dspant.processor.dimensionality import (
    create_pca_processor,
    create_ica_processor
)

# PCA dimensionality reduction
pca_proc = create_pca_processor(
    n_components=10,
    whiten=True
)

# ICA for source separation
ica_proc = create_ica_processor(
    n_components=5,
    algorithm="extended-infomax"
)
```

## Segmentation

Segmentation processors divide continuous signals into chunks:

```python
from dspant.processor.segmentation import (
    create_windowing_processor,
    create_event_segmenter
)

# Window the signal into overlapping segments
windowing = create_windowing_processor(
    window_ms=1000,
    step_ms=500,
    window_type="hamming"
)

# Extract segments around events
event_segmenter = create_event_segmenter(
    pre_event_ms=200,
    post_event_ms=800,
    baseline_ms=100
)
```

## Resampling

Resampling processors change the sampling rate:

```python
from dspant.processor.resampling import (
    create_downsampler,
    create_upsampler,
    create_resampler
)

# Downsample by a factor
downsampler = create_downsampler(
    factor=4,
    filter_type="fir",
    filter_order=64
)

# Upsample by a factor
upsampler = create_upsampler(
    factor=2,
    filter_type="fir",
    filter_order=32
)

# Resample to specific sampling rate
resampler = create_resampler(
    output_fs=1000,  # Target sampling rate
    method="sinc"    # Options: sinc, linear, cubic
)
```

## Building Processing Pipelines

General processors can be combined into processing pipelines:

```python
from dspant.nodes import StreamNode
from dspant.engine import create_processing_node

# Load data
stream_node = StreamNode("path/to/data.ant").load_data()
proc_node = create_processing_node(stream_node)

# Create a sequence of processors
proc_node.add_processor(notch, group="preprocessing")
proc_node.add_processor(bandpass, group="preprocessing")
proc_node.add_processor(normalizer, group="preprocessing")
proc_node.add_processor(amp_features, group="feature_extraction")

# Process data through the pipeline
results = proc_node.process()
```

## Processor Customization

All processors can be customized beyond the factory functions:

```python
from dspant.processor.filters import ButterworthFilter
from dspant.engine.base import BaseProcessor

# Customize a filter directly
custom_filter = ButterworthFilter(
    cutoff_hz=(300, 3000),
    filter_type="bandpass",
    order=4,
    zero_phase=True,
    padtype="constant"
)

# Create a custom processor by subclassing BaseProcessor
class CustomProcessor(BaseProcessor):
    def __init__(self, parameter1, parameter2):
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self._overlap_samples = 100
        
    def process(self, data, fs=None, **kwargs):
        # Implement custom processing here
        return processed_data
        
    @property
    def overlap_samples(self):
        return self._overlap_samples
        
    @property
    def summary(self):
        return {
            "type": "CustomProcessor",
            "parameter1": self.parameter1,
            "parameter2": self.parameter2,
            "overlap": self._overlap_samples
        }
```

## Optimization Techniques

General processors include performance optimization features:

```python
# Create a processor with Numba acceleration
from dspant.processor.features import create_statistical_feature_extractor

# Numba-accelerated feature extractor
fast_features = create_statistical_feature_extractor(
    features=["mean", "std", "percentile"],
    percentiles=[5, 50, 95],
    use_numba=True  # Enable Numba acceleration
)

# Process with optimized chunk handling
results = proc_node.process(
    optimize_chunks=True,       # Automatically optimize chunk sizes
    persist_intermediates=True, # Store intermediate results
    num_workers=4               # Use 4 workers for parallel processing
)
```

## Common Signal Processing Workflows

The general processing module supports several common workflows:

1. **Signal cleaning**: Filter out noise and artifacts
2. **Feature engineering**: Extract informative features for machine learning
3. **Time-frequency analysis**: Analyze signal properties across time and frequency
4. **Event detection**: Identify important events in continuous signals
5. **Data reduction**: Compress and simplify complex signals

## Integration with Specialized Modules

General processors are building blocks for specialized modules:

- **Neural processing**: Specialized processors built on general processing concepts
- **EMG processing**: Muscle activity analysis using general signal processing foundations
- **EEG processing**: Brain activity analysis with signal processing fundamentals
