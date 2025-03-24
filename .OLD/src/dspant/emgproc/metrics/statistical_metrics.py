"""
Statistical metrics for EMG signal analysis.

This module provides a processor to compute various statistical metrics from EMG signals,
specifically designed to work with extracted contractile segments from waveform_extractor.py.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit
from scipy import stats

from ...engine.base import BaseProcessor


@jit(nopython=True, cache=True)
def _compute_time_domain_features_numba(
    signal: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Compute time domain features from EMG signal with Numba acceleration.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector

    Returns
    -------
    mean : float
        Mean absolute value (MAV)
    rms : float
        Root mean square (RMS)
    var : float
        Variance
    ssi : float
        Simple square integral (SSI)
    iemg : float
        Integrated EMG
    wl : float
        Waveform length
    zc : float
        Zero crossing rate
    ssc : float
        Slope sign change rate
    """
    n_samples = len(signal)

    # Skip if segment is too short
    if n_samples <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Initialize metrics
    mean = 0.0
    rms = 0.0
    var = 0.0
    ssi = 0.0
    iemg = 0.0
    wl = 0.0
    zc = 0.0
    ssc = 0.0

    # Mean absolute value (MAV)
    for i in range(n_samples):
        mean += abs(signal[i])
    mean /= n_samples

    # Root mean square (RMS)
    for i in range(n_samples):
        rms += signal[i] * signal[i]
    rms = np.sqrt(rms / n_samples)

    # Variance
    signal_mean = np.mean(signal)
    for i in range(n_samples):
        var += (signal[i] - signal_mean) ** 2
    var /= n_samples

    # Simple square integral (SSI)
    for i in range(n_samples):
        ssi += signal[i] * signal[i]

    # Integrated EMG (IEMG)
    for i in range(n_samples):
        iemg += abs(signal[i])

    # Waveform length (WL)
    for i in range(1, n_samples):
        wl += abs(signal[i] - signal[i - 1])

    # Zero crossing rate (ZC)
    threshold = 0.0001  # Small threshold to avoid noise
    for i in range(1, n_samples):
        if (
            (signal[i] > 0 and signal[i - 1] < 0)
            or (signal[i] < 0 and signal[i - 1] > 0)
        ) and abs(signal[i] - signal[i - 1]) >= threshold:
            zc += 1
    zc /= n_samples

    # Slope sign change rate (SSC)
    for i in range(1, n_samples - 1):
        if (
            (signal[i] > signal[i - 1] and signal[i] > signal[i + 1])
            or (signal[i] < signal[i - 1] and signal[i] < signal[i + 1])
        ) and (
            abs(signal[i] - signal[i - 1]) >= threshold
            or abs(signal[i] - signal[i + 1]) >= threshold
        ):
            ssc += 1
    ssc /= n_samples

    return mean, rms, var, ssi, iemg, wl, zc, ssc


@jit(nopython=True, cache=True)
def _compute_histogram_features_numba(
    signal: np.ndarray, n_bins: int = 20
) -> Tuple[float, float, float, float]:
    """
    Compute histogram-based features with Numba acceleration.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector
    n_bins : int
        Number of histogram bins

    Returns
    -------
    skewness : float
        Skewness of amplitude distribution
    kurtosis : float
        Kurtosis of amplitude distribution
    hist_entropy : float
        Entropy of the amplitude histogram
    median : float
        Median value
    """
    # Skip if segment is too short
    if len(signal) <= 1:
        return 0.0, 0.0, 0.0, 0.0

    # Compute histogram
    hist_range = (np.min(signal), np.max(signal))
    if hist_range[0] == hist_range[1]:  # Handle constant signal
        hist_counts = np.zeros(n_bins, dtype=np.float32)
        hist_counts[0] = len(signal)
    else:
        bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
        hist_counts = np.zeros(n_bins, dtype=np.float32)

        # Manual histogram computation
        bin_width = (hist_range[1] - hist_range[0]) / n_bins
        for i in range(len(signal)):
            bin_idx = int((signal[i] - hist_range[0]) / bin_width)
            if bin_idx == n_bins:  # Handle edge case for max value
                bin_idx = n_bins - 1
            hist_counts[bin_idx] += 1

    # Normalize histogram to get probability distribution
    hist_probs = hist_counts / len(signal)

    # Skewness (use NumPy's implementation formula)
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:  # Handle zero standard deviation
        skewness = 0.0
    else:
        skewness = 0.0
        for i in range(len(signal)):
            skewness += ((signal[i] - mean) / std) ** 3
        skewness = skewness / len(signal)

    # Kurtosis (use NumPy's implementation formula)
    if std == 0:  # Handle zero standard deviation
        kurtosis = 0.0
    else:
        kurtosis = 0.0
        for i in range(len(signal)):
            kurtosis += ((signal[i] - mean) / std) ** 4
        kurtosis = kurtosis / len(signal) - 3.0  # Excess kurtosis

    # Entropy
    hist_entropy = 0.0
    for p in hist_probs:
        if p > 0:  # Avoid log(0)
            hist_entropy -= p * np.log2(p)

    # Median
    median = np.median(signal)

    return skewness, kurtosis, hist_entropy, median


def compute_frequency_features(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Compute frequency domain features from EMG signal.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector
    fs : float
        Sampling frequency in Hz

    Returns
    -------
    features : Dict[str, float]
        Dictionary of frequency domain features
    """
    # Skip if segment is too short
    if len(signal) <= 2:
        return {
            "mean_freq": 0.0,
            "median_freq": 0.0,
            "peak_freq": 0.0,
            "total_power": 0.0,
            "spectral_entropy": 0.0,
        }

    # Compute power spectral density using appropriate segment length
    nperseg = min(len(signal), 256)

    # Use Welch's method if enough samples, otherwise use periodogram
    if len(signal) >= nperseg:
        f, psd = signal.welch(signal, fs, nperseg=nperseg)
    else:
        f, psd = signal.periodogram(signal, fs)

    # Skip if no valid PSD
    if len(psd) == 0 or np.sum(psd) == 0:
        return {
            "mean_freq": 0.0,
            "median_freq": 0.0,
            "peak_freq": 0.0,
            "total_power": 0.0,
            "spectral_entropy": 0.0,
        }

    # Total power
    total_power = np.sum(psd)

    # Mean frequency
    mean_freq = np.sum(f * psd) / total_power if total_power > 0 else 0.0

    # Median frequency (frequency that divides the power spectrum in half)
    if total_power > 0:
        cumulative_power = np.cumsum(psd)
        median_idx = np.where(cumulative_power >= total_power / 2)[0]
        if len(median_idx) > 0:
            median_freq = f[median_idx[0]]
        else:
            median_freq = 0.0
    else:
        median_freq = 0.0

    # Peak frequency
    peak_idx = np.argmax(psd)
    peak_freq = f[peak_idx] if peak_idx < len(f) else 0.0

    # Spectral entropy
    norm_psd = psd / total_power if total_power > 0 else np.zeros_like(psd)
    spectral_entropy = 0.0
    for p in norm_psd:
        if p > 0:  # Avoid log(0)
            spectral_entropy -= p * np.log2(p)

    return {
        "mean_freq": mean_freq,
        "median_freq": median_freq,
        "peak_freq": peak_freq,
        "total_power": total_power,
        "spectral_entropy": spectral_entropy,
    }


class StatisticalMetricsProcessor(BaseProcessor):
    """
    Processor for computing statistical metrics from EMG segments.

    This processor calculates various statistical features from EMG segments,
    including time-domain features, frequency-domain features, and histogram-based metrics.
    It is designed to work with 3D Dask arrays output from SegmentExtractionProcessor.
    """

    def __init__(
        self,
        metrics: List[str] = ["time", "frequency", "histogram"],
        n_histogram_bins: int = 20,
        use_numba: bool = True,
    ):
        """
        Initialize the EMG statistical metrics processor.

        Parameters
        ----------
        metrics : List[str]
            List of metric categories to compute
        n_histogram_bins : int
            Number of bins for histogram computation
        use_numba : bool
            Whether to use Numba acceleration for metrics computation
        """
        self.metrics = metrics
        self.n_histogram_bins = n_histogram_bins
        self.use_numba = use_numba
        self._overlap_samples = 0

    def process_segments(
        self, segments: da.Array, fs: Optional[float] = None, **kwargs
    ) -> Dict:
        """
        Compute statistical metrics for pre-extracted EMG segments.

        Parameters
        ----------
        segments : da.Array
            3D Dask array of segments (segments × samples × channels)
        fs : float, optional
            Sampling frequency (Hz), required for frequency-domain metrics
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        metrics : dict
            Dictionary of computed metrics for each segment and channel
        """
        if "frequency" in self.metrics and fs is None:
            raise ValueError(
                "Sampling frequency (fs) must be provided for frequency metrics"
            )

        # Get segment dimensions
        n_segments, n_samples, n_channels = segments.shape

        # Define chunking strategy for better parallelism
        chunk_size = min(100, n_segments)  # Process up to 100 segments at a time

        # Function to process a chunk of segments
        def process_chunk(segments_chunk):
            # Convert to numpy for processing
            segments_np = np.asarray(segments_chunk)
            chunk_segments, chunk_samples, chunk_channels = segments_np.shape

            # Initialize results
            chunk_results = {}

            for ch in range(chunk_channels):
                channel_results = []

                # Process each segment
                for seg_idx in range(chunk_segments):
                    segment_data = segments_np[seg_idx, :, ch]

                    # Skip NaN or zero segments
                    if np.isnan(segment_data).any() or np.all(segment_data == 0):
                        channel_results.append({})
                        continue

                    # Compute metrics for this segment
                    segment_metrics = {}

                    # Time domain features
                    if "time" in self.metrics:
                        if self.use_numba:
                            mav, rms, var, ssi, iemg, wl, zc, ssc = (
                                _compute_time_domain_features_numba(segment_data)
                            )

                            segment_metrics.update(
                                {
                                    "mean_abs_value": float(mav),
                                    "rms": float(rms),
                                    "variance": float(var),
                                    "simple_square_integral": float(ssi),
                                    "integrated_emg": float(iemg),
                                    "waveform_length": float(wl),
                                    "zero_crossing_rate": float(zc),
                                    "slope_sign_change_rate": float(ssc),
                                }
                            )
                        else:
                            # Fallback to non-Numba implementation
                            segment_metrics.update(
                                {
                                    "mean_abs_value": float(
                                        np.mean(np.abs(segment_data))
                                    ),
                                    "rms": float(np.sqrt(np.mean(segment_data**2))),
                                    "variance": float(np.var(segment_data)),
                                    "simple_square_integral": float(
                                        np.sum(segment_data**2)
                                    ),
                                    "integrated_emg": float(
                                        np.sum(np.abs(segment_data))
                                    ),
                                    "waveform_length": float(
                                        np.sum(np.abs(np.diff(segment_data)))
                                    ),
                                    "zero_crossing_rate": float(
                                        np.sum(np.diff(np.signbit(segment_data)) != 0)
                                        / len(segment_data)
                                    ),
                                    "slope_sign_change_rate": float(
                                        np.sum(
                                            np.diff(np.signbit(np.diff(segment_data)))
                                            != 0
                                        )
                                        / len(segment_data)
                                    ),
                                }
                            )

                    # Frequency domain features
                    if "frequency" in self.metrics and fs is not None:
                        freq_features = compute_frequency_features(segment_data, fs)
                        segment_metrics.update(freq_features)

                    # Histogram features
                    if "histogram" in self.metrics:
                        if self.use_numba:
                            skewness, kurtosis, hist_entropy, median = (
                                _compute_histogram_features_numba(
                                    segment_data, self.n_histogram_bins
                                )
                            )

                            segment_metrics.update(
                                {
                                    "skewness": float(skewness),
                                    "kurtosis": float(kurtosis),
                                    "histogram_entropy": float(hist_entropy),
                                    "median": float(median),
                                }
                            )
                        else:
                            # Fallback to non-Numba implementation
                            segment_metrics.update(
                                {
                                    "skewness": float(stats.skew(segment_data)),
                                    "kurtosis": float(stats.kurtosis(segment_data)),
                                    "median": float(np.median(segment_data)),
                                }
                            )

                            # Calculate histogram entropy manually
                            hist, _ = np.histogram(
                                segment_data, bins=self.n_histogram_bins
                            )
                            hist_probs = (
                                hist / np.sum(hist)
                                if np.sum(hist) > 0
                                else np.zeros_like(hist)
                            )
                            hist_entropy = -np.sum(
                                hist_probs * np.log2(hist_probs + 1e-10)
                            )
                            segment_metrics["histogram_entropy"] = float(hist_entropy)

                    channel_results.append(segment_metrics)

                # Add channel results to chunk results
                chunk_results[f"channel_{ch}"] = channel_results

            return chunk_results

        # Apply processing function to chunks
        results_blocks = []

        for i in range(0, n_segments, chunk_size):
            end_idx = min(i + chunk_size, n_segments)
            segment_chunk = segments[i:end_idx]

            # Process chunk and collect results
            chunk_result = da.delayed(process_chunk)(segment_chunk)
            results_blocks.append(chunk_result)

        # Compute all results
        computed_results = da.compute(*results_blocks)

        # Combine results from all chunks
        final_results = {}

        for ch in range(n_channels):
            channel_key = f"channel_{ch}"
            channel_results = []

            # Collect segment results for this channel from all chunks
            for chunk_result in computed_results:
                if channel_key in chunk_result:
                    channel_results.extend(chunk_result[channel_key])

            # Add combined channel results to final results
            final_results[channel_key] = channel_results

            # Compute summary statistics if we have valid segments
            valid_segments = [seg for seg in channel_results if seg]
            if valid_segments:
                # Get all available metrics
                all_metrics = set()
                for segment in valid_segments:
                    all_metrics.update(segment.keys())

                # Calculate statistics for each metric
                summary = {}
                for metric in all_metrics:
                    values = [
                        segment[metric]
                        for segment in valid_segments
                        if metric in segment
                    ]
                    if values:
                        summary[metric] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "median": float(np.median(values)),
                        }

                final_results[f"{channel_key}_summary"] = summary

        return final_results

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> Dict:
        """
        Process input data to compute statistical metrics.

        This is the main entry point that implements the BaseProcessor interface.

        Parameters
        ----------
        data : da.Array
            Input data array
        fs : float, optional
            Sampling frequency (Hz)
        **kwargs : dict
            Additional keyword arguments
            - segments: If data is a raw signal, this contains extracted segments

        Returns
        -------
        metrics : dict
            Dictionary of computed statistical metrics
        """
        # Check if segments are provided directly in kwargs
        segments = kwargs.get("segments", None)

        if segments is not None:
            # Process provided segments
            return self.process_segments(segments, fs, **kwargs)

        # If data is already in segment format (3D array), process directly
        if data.ndim == 3:
            return self.process_segments(data, fs, **kwargs)

        # If we get here, data is not in the expected format
        raise ValueError(
            "Input data must be a 3D array of segments (segments × samples × channels) "
            "or segments must be provided in kwargs"
        )

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap processing."""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration."""
        base_summary = super().summary
        base_summary.update(
            {
                "metrics": self.metrics,
                "n_histogram_bins": self.n_histogram_bins,
                "use_numba": self.use_numba,
            }
        )
        return base_summary


# Factory functions for creating processors with common configurations


def create_basic_statistical_processor() -> StatisticalMetricsProcessor:
    """
    Create a basic EMG statistical metrics processor.

    Returns
    -------
    processor : StatisticalMetricsProcessor
        Configured processor for basic statistical metrics
    """
    return StatisticalMetricsProcessor(
        metrics=["time"],
        use_numba=True,
    )


def create_comprehensive_statistical_processor() -> StatisticalMetricsProcessor:
    """
    Create a comprehensive EMG statistical metrics processor.

    Returns
    -------
    processor : StatisticalMetricsProcessor
        Configured processor for comprehensive statistical metrics
    """
    return StatisticalMetricsProcessor(
        metrics=["time", "frequency", "histogram"],
        n_histogram_bins=20,
        use_numba=True,
    )


# Helper function for direct analysis


def analyze_emg_segments(
    segments: Union[np.ndarray, da.Array],
    fs: Optional[float] = None,
    metrics: List[str] = ["time", "frequency", "histogram"],
) -> Dict:
    """
    Analyze EMG segments and compute statistical metrics.

    Parameters
    ----------
    segments : np.ndarray or da.Array
        3D array of segments (segments × samples × channels)
    fs : float, optional
        Sampling frequency in Hz (required for frequency metrics)
    metrics : List[str]
        List of metric categories to compute

    Returns
    -------
    metrics : dict
        Dictionary of computed statistical metrics
    """
    # Create processor
    processor = StatisticalMetricsProcessor(
        metrics=metrics,
        use_numba=True,
    )

    # Convert numpy to dask if needed
    if isinstance(segments, np.ndarray):
        # Create reasonable chunks - up to 100 segments per chunk
        chunk_size = min(100, segments.shape[0])
        segments_da = da.from_array(segments, chunks=(chunk_size, -1, -1))
    else:
        segments_da = segments

    # Process segments
    return processor.process(segments_da, fs=fs)
