"""
Spectral indices for EMG fatigue analysis.

This module provides implementations of common spectral indices used to
analyze muscle fatigue from EMG data in the time-frequency domain, including:
- Median frequency
- Mean frequency
- Spectral moments
- Frequency bands ratio
- Spectral entropy

These indices can be applied to time-frequency data produced from STFT, spectrogram,
or other time-frequency transformations.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
from numba import jit, prange

from ...engine.base import BaseProcessor


@jit(nopython=True, parallel=True, cache=True)
def _compute_median_frequency(
    spectrogram: np.ndarray, frequencies: np.ndarray, axis: int = 0
) -> np.ndarray:
    """
    Compute median frequency for each time frame.

    The median frequency is the frequency that divides the power spectrum
    into two regions with equal power.

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    frequencies : np.ndarray
        Frequency values corresponding to the spectrogram bins
    axis : int, default=0
        Axis along which to compute the median frequency (typically frequency axis)

    Returns
    -------
    median_freq : np.ndarray
        Median frequency for each time frame and channel
    """
    # Determine dimensions
    if spectrogram.ndim == 2:
        n_freqs, n_times = spectrogram.shape
        n_channels = 1
        spec = spectrogram.reshape(n_freqs, n_times, 1)  # Add singleton dimension
    else:
        n_freqs, n_times, n_channels = spectrogram.shape
        spec = spectrogram

    # Initialize output array
    median_freq = np.zeros((n_times, n_channels), dtype=np.float32)

    # Process each channel in parallel
    for c in prange(n_channels):
        for t in range(n_times):
            # Get power spectrum for current time frame
            power_spectrum = spec[:, t, c]

            # Skip if all values are zeros or NaNs
            if np.all(power_spectrum <= 0) or np.all(np.isnan(power_spectrum)):
                median_freq[t, c] = np.nan
                continue

            # Calculate cumulative sum of power
            total_power = np.sum(power_spectrum)
            cumsum = np.cumsum(power_spectrum) / total_power

            # Find the bin where cumulative power exceeds 0.5
            median_idx = np.searchsorted(cumsum, 0.5)

            # Handle edge case
            if median_idx >= len(frequencies):
                median_idx = len(frequencies) - 1

            # Linear interpolation for more accurate estimate
            if median_idx > 0 and median_idx < len(frequencies):
                prev_cumsum = cumsum[median_idx - 1]
                curr_cumsum = cumsum[median_idx]
                weight = (
                    (0.5 - prev_cumsum) / (curr_cumsum - prev_cumsum)
                    if curr_cumsum > prev_cumsum
                    else 0
                )
                median_freq[t, c] = (
                    frequencies[median_idx - 1] * (1 - weight)
                    + frequencies[median_idx] * weight
                )
            else:
                median_freq[t, c] = frequencies[median_idx]

    # Remove singleton dimension if input was 2D
    if spectrogram.ndim == 2:
        median_freq = median_freq.reshape(n_times)

    return median_freq


@jit(nopython=True, parallel=True, cache=True)
def _compute_mean_frequency(
    spectrogram: np.ndarray, frequencies: np.ndarray, axis: int = 0
) -> np.ndarray:
    """
    Compute mean frequency for each time frame.

    The mean frequency is the weighted average of frequencies with
    power spectrum values as weights.

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    frequencies : np.ndarray
        Frequency values corresponding to the spectrogram bins
    axis : int, default=0
        Axis along which to compute the mean frequency (typically frequency axis)

    Returns
    -------
    mean_freq : np.ndarray
        Mean frequency for each time frame and channel
    """
    # Determine dimensions
    if spectrogram.ndim == 2:
        n_freqs, n_times = spectrogram.shape
        n_channels = 1
        spec = spectrogram.reshape(n_freqs, n_times, 1)  # Add singleton dimension
    else:
        n_freqs, n_times, n_channels = spectrogram.shape
        spec = spectrogram

    # Initialize output array
    mean_freq = np.zeros((n_times, n_channels), dtype=np.float32)

    # Process each channel in parallel
    for c in prange(n_channels):
        for t in range(n_times):
            # Get power spectrum for current time frame
            power_spectrum = spec[:, t, c]

            # Skip if all values are zeros or NaNs
            if np.all(power_spectrum <= 0) or np.all(np.isnan(power_spectrum)):
                mean_freq[t, c] = np.nan
                continue

            # Calculate weighted average
            weighted_sum = np.sum(frequencies * power_spectrum)
            total_power = np.sum(power_spectrum)

            if total_power > 0:
                mean_freq[t, c] = weighted_sum / total_power
            else:
                mean_freq[t, c] = np.nan

    # Remove singleton dimension if input was 2D
    if spectrogram.ndim == 2:
        mean_freq = mean_freq.reshape(n_times)

    return mean_freq


@jit(nopython=True, parallel=True, cache=True)
def _compute_spectral_moment(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
    moment: int,
    central: bool = False,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute spectral moment for each time frame.

    The spectral moment is the weighted average of frequency raised to a power (moment)
    with power spectrum values as weights.

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    frequencies : np.ndarray
        Frequency values corresponding to the spectrogram bins
    moment : int
        The order of the moment to calculate
    central : bool, default=False
        If True, calculate central moment around the mean frequency
    axis : int, default=0
        Axis along which to compute the moment (typically frequency axis)

    Returns
    -------
    spectral_moment : np.ndarray
        Spectral moment for each time frame and channel
    """
    # Determine dimensions
    if spectrogram.ndim == 2:
        n_freqs, n_times = spectrogram.shape
        n_channels = 1
        spec = spectrogram.reshape(n_freqs, n_times, 1)  # Add singleton dimension
    else:
        n_freqs, n_times, n_channels = spectrogram.shape
        spec = spectrogram

    # Initialize output array
    spectral_moment = np.zeros((n_times, n_channels), dtype=np.float32)

    # Process each channel in parallel
    for c in prange(n_channels):
        for t in range(n_times):
            # Get power spectrum for current time frame
            power_spectrum = spec[:, t, c]

            # Skip if all values are zeros or NaNs
            if np.all(power_spectrum <= 0) or np.all(np.isnan(power_spectrum)):
                spectral_moment[t, c] = np.nan
                continue

            # Calculate total power
            total_power = np.sum(power_spectrum)

            if central and moment > 0:
                # Calculate mean frequency for central moment
                mean_freq = np.sum(frequencies * power_spectrum) / total_power
                # Calculate central moment
                spectral_moment[t, c] = (
                    np.sum(((frequencies - mean_freq) ** moment) * power_spectrum)
                    / total_power
                )
            else:
                # Calculate regular moment
                spectral_moment[t, c] = (
                    np.sum((frequencies**moment) * power_spectrum) / total_power
                )

    # Remove singleton dimension if input was 2D
    if spectrogram.ndim == 2:
        spectral_moment = spectral_moment.reshape(n_times)

    return spectral_moment


@jit(nopython=True, parallel=True, cache=True)
def _compute_frequency_bands_ratio(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
    low_band: Tuple[float, float],
    high_band: Tuple[float, float],
    axis: int = 0,
) -> np.ndarray:
    """
    Compute ratio between power in two frequency bands for each time frame.

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    frequencies : np.ndarray
        Frequency values corresponding to the spectrogram bins
    low_band : tuple
        Lower frequency band as (min_freq, max_freq)
    high_band : tuple
        Higher frequency band as (min_freq, max_freq)
    axis : int, default=0
        Axis along which to compute the ratio (typically frequency axis)

    Returns
    -------
    ratio : np.ndarray
        Frequency bands ratio for each time frame and channel
    """
    # Determine dimensions
    if spectrogram.ndim == 2:
        n_freqs, n_times = spectrogram.shape
        n_channels = 1
        spec = spectrogram.reshape(n_freqs, n_times, 1)  # Add singleton dimension
    else:
        n_freqs, n_times, n_channels = spectrogram.shape
        spec = spectrogram

    # Initialize output array
    ratio = np.zeros((n_times, n_channels), dtype=np.float32)

    # Find frequency indices for bands
    low_band_indices = np.where(
        (frequencies >= low_band[0]) & (frequencies <= low_band[1])
    )[0]
    high_band_indices = np.where(
        (frequencies >= high_band[0]) & (frequencies <= high_band[1])
    )[0]

    # Process each channel in parallel
    for c in prange(n_channels):
        for t in range(n_times):
            # Get power spectrum for current time frame
            power_spectrum = spec[:, t, c]

            # Calculate power in each band
            low_band_power = np.sum(power_spectrum[low_band_indices])
            high_band_power = np.sum(power_spectrum[high_band_indices])

            # Compute ratio
            if high_band_power > 0:
                ratio[t, c] = low_band_power / high_band_power
            else:
                ratio[t, c] = np.nan

    # Remove singleton dimension if input was 2D
    if spectrogram.ndim == 2:
        ratio = ratio.reshape(n_times)

    return ratio


@jit(nopython=True, parallel=True, cache=True)
def _compute_spectral_entropy(
    spectrogram: np.ndarray, normalize: bool = True, axis: int = 0
) -> np.ndarray:
    """
    Compute spectral entropy for each time frame.

    The spectral entropy quantifies the uniformity of the power spectrum.
    Lower values indicate more concentrated power distribution (less disorder).

    Parameters
    ----------
    spectrogram : np.ndarray
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    normalize : bool, default=True
        Whether to normalize entropy by log(n_freqs)
    axis : int, default=0
        Axis along which to compute the entropy (typically frequency axis)

    Returns
    -------
    entropy : np.ndarray
        Spectral entropy for each time frame and channel
    """
    # Determine dimensions
    if spectrogram.ndim == 2:
        n_freqs, n_times = spectrogram.shape
        n_channels = 1
        spec = spectrogram.reshape(n_freqs, n_times, 1)  # Add singleton dimension
    else:
        n_freqs, n_times, n_channels = spectrogram.shape
        spec = spectrogram

    # Initialize output array
    entropy = np.zeros((n_times, n_channels), dtype=np.float32)

    # Normalization factor
    norm_factor = np.log(n_freqs) if normalize else 1.0

    # Process each channel in parallel
    for c in prange(n_channels):
        for t in range(n_times):
            # Get power spectrum for current time frame
            power_spectrum = spec[:, t, c]

            # Skip if all values are zeros or NaNs
            if np.all(power_spectrum <= 0) or np.all(np.isnan(power_spectrum)):
                entropy[t, c] = np.nan
                continue

            # Normalize power spectrum to get probability distribution
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                p = power_spectrum / total_power

                # Calculate entropy
                # Use only non-zero values to avoid log(0)
                mask = p > 0
                if np.any(mask):
                    entropy[t, c] = -np.sum(p[mask] * np.log(p[mask])) / norm_factor
                else:
                    entropy[t, c] = 0
            else:
                entropy[t, c] = np.nan

    # Remove singleton dimension if input was 2D
    if spectrogram.ndim == 2:
        entropy = entropy.reshape(n_times)

    return entropy


class SpectralIndicesProcessor(BaseProcessor):
    """
    Processor for computing spectral indices from time-frequency data.

    This processor calculates various spectral indices commonly used in
    EMG fatigue analysis from spectrogram or time-frequency data.
    """

    def __init__(
        self,
        indices: List[str] = None,
        frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        band_ratios: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Initialize the spectral indices processor.

        Parameters
        ----------
        indices : List[str], optional
            List of indices to compute. Supported values are:
            - 'median_frequency': Median frequency
            - 'mean_frequency': Mean frequency
            - 'spectral_moment1': First spectral moment (mean frequency)
            - 'spectral_moment2': Second spectral moment
            - 'spectral_moment3': Third spectral moment
            - 'spectral_entropy': Spectral entropy
            If None, computes median and mean frequency.
        frequency_bands : Dict[str, Tuple[float, float]], optional
            Dictionary of frequency bands to calculate power for.
            Keys are band names, values are (min_freq, max_freq) tuples.
            Default: {'low': (20, 45), 'high': (45, 95), 'total': (20, 250)}
        band_ratios : List[Tuple[str, str]], optional
            List of band ratio pairs to compute as (numerator, denominator).
            Default: [('low', 'high')]
        """
        self._overlap_samples = 0  # No overlap needed for this processor

        # Default indices if none specified
        self.indices = indices or ["median_frequency", "mean_frequency"]

        # Default frequency bands
        self.frequency_bands = frequency_bands or {
            "low": (20, 45),
            "high": (45, 95),
            "total": (20, 250),
        }

        # Default band ratios
        self.band_ratios = band_ratios or [("low", "high")]

        # Validate band ratios
        for num, denom in self.band_ratios:
            if num not in self.frequency_bands or denom not in self.frequency_bands:
                raise ValueError(
                    f"Band ratio ({num}, {denom}) references undefined frequency bands. "
                    f"Available bands: {list(self.frequency_bands.keys())}"
                )

    def process(
        self, data: da.Array, fs: Optional[float] = None, **kwargs
    ) -> Dict[str, da.Array]:
        """
        Compute spectral indices from time-frequency data.

        Parameters
        ----------
        data : da.Array
            Time-frequency data (spectrogram) with shape:
            (freq_bins, time_frames) or (freq_bins, time_frames, channels)
        fs : float, optional
            Sampling frequency. Required if frequency_bins is not provided in kwargs.
        **kwargs : dict
            Additional parameters:
            - frequency_bins : np.ndarray, optional
              Frequency values corresponding to the spectrogram bins.
              If not provided, calculated from n_fft and fs.
            - n_fft : int, optional
              FFT size used to create the spectrogram (required if frequency_bins not provided)
            - compute_now : bool, default=False
              Whether to compute indices immediately or return dask arrays

        Returns
        -------
        indices : Dict[str, da.Array]
            Dictionary of computed spectral indices, each with shape (time_frames, channels)
        """
        # Get frequency bins
        frequency_bins = kwargs.get("frequency_bins", None)
        n_fft = kwargs.get("n_fft", None)

        # Calculate frequency bins if not provided
        if frequency_bins is None:
            if fs is None:
                raise ValueError(
                    "Either frequency_bins or sampling frequency (fs) must be provided"
                )

            if n_fft is None:
                # Infer n_fft from data shape if possible
                n_fft = (data.shape[0] - 1) * 2

            frequency_bins = np.linspace(0, fs / 2, data.shape[0])
        else:
            # Ensure frequency_bins is a numpy array
            frequency_bins = np.asarray(frequency_bins)

        # Compute now or return dask arrays
        compute_now = kwargs.get("compute_now", False)

        # Initialize results dictionary
        results = {}

        # Compute requested indices
        for index_name in self.indices:
            if index_name == "median_frequency":
                result = self._compute_index(
                    data, _compute_median_frequency, frequency_bins, compute_now
                )
                results["median_frequency"] = result

            elif index_name == "mean_frequency":
                result = self._compute_index(
                    data, _compute_mean_frequency, frequency_bins, compute_now
                )
                results["mean_frequency"] = result

            elif index_name.startswith("spectral_moment"):
                # Extract moment order from name
                try:
                    moment = int(index_name.replace("spectral_moment", ""))
                    central = kwargs.get("central_moments", False)
                    result = self._compute_index(
                        data,
                        _compute_spectral_moment,
                        frequency_bins,
                        compute_now,
                        moment=moment,
                        central=central,
                    )
                    results[index_name] = result
                except ValueError:
                    # Skip invalid moment specifications
                    continue

            elif index_name == "spectral_entropy":
                normalize = kwargs.get("normalize_entropy", True)
                result = self._compute_index(
                    data,
                    _compute_spectral_entropy,
                    frequency_bins,
                    compute_now,
                    normalize=normalize,
                )
                results["spectral_entropy"] = result

        # Compute frequency band powers
        for band_name, (min_freq, max_freq) in self.frequency_bands.items():
            # Create a mask for this frequency band
            band_mask = (frequency_bins >= min_freq) & (frequency_bins <= max_freq)

            if not np.any(band_mask):
                continue

            # Calculate band power by summing within the band
            def compute_band_power(spec):
                return np.sum(spec[band_mask, :], axis=0)

            result = data.map_blocks(
                compute_band_power,
                drop_axis=0,  # Remove the frequency dimension
                dtype=np.float32,
            )

            if compute_now:
                result = result.compute()

            results[f"{band_name}_band_power"] = result

        # Compute frequency band ratios
        for num_band, denom_band in self.band_ratios:
            num_key = f"{num_band}_band_power"
            denom_key = f"{denom_band}_band_power"

            if num_key in results and denom_key in results:
                ratio = results[num_key] / results[denom_key]
                results[f"{num_band}_to_{denom_band}_ratio"] = ratio

        return results

    def _compute_index(
        self,
        data: da.Array,
        func,
        frequency_bins: np.ndarray,
        compute_now: bool,
        **kwargs,
    ) -> da.Array:
        """
        Compute a spectral index using dask.

        Parameters
        ----------
        data : da.Array
            Spectrogram data
        func : callable
            Function to compute the index
        frequency_bins : np.ndarray
            Frequency values
        compute_now : bool
            Whether to compute immediately
        **kwargs : dict
            Additional parameters for the function

        Returns
        -------
        result : da.Array
            Computed index
        """

        # Define a function to process each block
        def process_block(x):
            return func(x, frequency_bins, **kwargs)

        # Apply function to dask array
        result = data.map_blocks(
            process_block,
            drop_axis=0,  # Remove the frequency dimension
            dtype=np.float32,
        )

        if compute_now:
            result = result.compute()

        return result

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "indices": self.indices,
                "frequency_bands": self.frequency_bands,
                "band_ratios": self.band_ratios,
            }
        )
        return base_summary


class FatigueIndicesProcessor(BaseProcessor):
    """
    Processor for computing fatigue indices from spectral data.

    This processor calculates fatigue-specific indices and trends from
    spectral indices like median frequency, mean frequency, and band ratios.
    """

    def __init__(
        self,
        window_size: int = 1,
        regression_method: Literal["linear", "robust"] = "linear",
    ):
        """
        Initialize the fatigue indices processor.

        Parameters
        ----------
        window_size : int, default=1
            Window size (in time points) for smoothing indices before trend calculation
        regression_method : str, default="linear"
            Method for calculating trends: "linear" or "robust"
        """
        self._overlap_samples = 0  # No overlap needed for this processor
        self.window_size = window_size
        self.regression_method = regression_method

    def process(
        self, spectral_indices: Dict[str, da.Array], **kwargs
    ) -> Dict[str, da.Array]:
        """
        Compute fatigue indices from spectral indices.

        Parameters
        ----------
        spectral_indices : Dict[str, da.Array]
            Dictionary of spectral indices from SpectralIndicesProcessor
        **kwargs : dict
            Additional parameters:
            - compute_now : bool, default=False
              Whether to compute indices immediately or return dask arrays

        Returns
        -------
        fatigue_indices : Dict[str, da.Array]
            Dictionary of computed fatigue indices
        """
        # Get compute flag
        compute_now = kwargs.get("compute_now", False)

        # Initialize results dictionary
        results = {}

        # Process each spectral index to derive fatigue indices
        for index_name, index_values in spectral_indices.items():
            # Compute trend and rate of change for select indices
            if index_name in [
                "median_frequency",
                "mean_frequency",
            ] or index_name.endswith("_ratio"):
                # Calculate slope (trend)
                slope = self._compute_trend(index_values, compute_now)
                results[f"{index_name}_slope"] = slope

                # Calculate normalized slope (percent change per minute)
                if compute_now:
                    # Initial value (average of first few samples)
                    init_val = np.mean(
                        index_values[: min(10, index_values.shape[0])].compute()
                    )
                    if init_val != 0:
                        norm_slope = slope / init_val * 100  # percent
                        results[f"{index_name}_percent_change"] = norm_slope

                # Calculate fatigue index (for median frequency)
                if index_name == "median_frequency":
                    if compute_now:
                        mdf = index_values.compute()
                        # Initial and final values
                        init_mdf = np.mean(mdf[: min(10, mdf.shape[0])])
                        final_mdf = np.mean(mdf[max(0, mdf.shape[0] - 10) :])

                        if init_mdf != 0:
                            fatigue_index = (
                                (init_mdf - final_mdf) / init_mdf * 100
                            )  # percent
                            results["fatigue_index"] = fatigue_index

        return results

    def _compute_trend(
        self, data: da.Array, compute_now: bool
    ) -> Union[da.Array, float]:
        """
        Compute trend (slope) from time series data.

        Parameters
        ----------
        data : da.Array
            Time series data
        compute_now : bool
            Whether to compute immediately

        Returns
        -------
        slope : da.Array or float
            Computed slope
        """
        if compute_now:
            # Convert to numpy for computation
            data_np = data.compute()

            # X values (time points)
            x = np.arange(data_np.shape[0])

            # Remove NaNs
            mask = ~np.isnan(data_np)
            if np.sum(mask) < 2:
                return np.nan

            x_valid = x[mask]
            y_valid = data_np[mask]

            if self.regression_method == "robust":
                # Robust regression using Theil-Sen estimator
                # (median of slopes between all point pairs)
                slopes = []
                for i in range(len(x_valid)):
                    for j in range(i + 1, len(x_valid)):
                        if x_valid[j] != x_valid[i]:  # Avoid division by zero
                            slopes.append(
                                (y_valid[j] - y_valid[i]) / (x_valid[j] - x_valid[i])
                            )

                if slopes:
                    return np.median(slopes)
                else:
                    return np.nan
            else:
                # Linear regression
                # Compute slope using least squares
                A = np.vstack([x_valid, np.ones(len(x_valid))]).T
                slope, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                return slope
        else:
            # Return a dask array for later computation
            # This is a placeholder as trend computation typically requires
            # the whole dataset
            return data.mean()  # Placeholder

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of processor configuration"""
        base_summary = super().summary
        base_summary.update(
            {
                "window_size": self.window_size,
                "regression_method": self.regression_method,
            }
        )
        return base_summary


# Factory functions for easy creation


def create_spectral_indices_processor(
    indices: List[str] = None,
    include_entropy: bool = False,
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> SpectralIndicesProcessor:
    """
    Create a spectral indices processor with standard parameters.

    Parameters
    ----------
    indices : List[str], optional
        List of indices to compute. If None, uses ['median_frequency', 'mean_frequency']
    include_entropy : bool, default=False
        Whether to include spectral entropy calculation
    frequency_bands : dict, optional
        Custom frequency bands definition

    Returns
    -------
    processor : SpectralIndicesProcessor
        Configured SpectralIndicesProcessor
    """
    # Default indices
    default_indices = ["median_frequency", "mean_frequency"]

    # Add entropy if requested
    if include_entropy:
        default_indices.append("spectral_entropy")

    # Use provided indices or defaults
    indices = indices or default_indices

    return SpectralIndicesProcessor(indices=indices, frequency_bands=frequency_bands)


def create_fatigue_analysis_processor(
    window_size: int = 1,
    use_robust_regression: bool = False,
) -> FatigueIndicesProcessor:
    """
    Create a fatigue indices processor with standard parameters.

    Parameters
    ----------
    window_size : int, default=1
        Window size for smoothing indices
    use_robust_regression : bool, default=False
        Whether to use robust regression method

    Returns
    -------
    processor : FatigueIndicesProcessor
        Configured FatigueIndicesProcessor
    """
    return FatigueIndicesProcessor(
        window_size=window_size,
        regression_method="robust" if use_robust_regression else "linear",
    )


def analyze_fatigue_from_spectrogram(
    spectrogram: da.Array,
    fs: float,
    frequency_bins: Optional[np.ndarray] = None,
    compute_now: bool = True,
) -> Dict[str, Any]:
    """
    Analyze fatigue from a spectrogram in one step.

    This function combines spectral indices calculation and fatigue
    analysis to produce a comprehensive set of fatigue metrics.

    Parameters
    ----------
    spectrogram : da.Array
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    fs : float
        Sampling frequency in Hz
    frequency_bins : np.ndarray, optional
        Frequency values corresponding to the spectrogram bins
    compute_now : bool, default=True
        Whether to compute indices immediately

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing all computed spectral and fatigue indices
    """
    # Create processors
    spectral_processor = create_spectral_indices_processor(
        include_entropy=True,
    )
    fatigue_processor = create_fatigue_analysis_processor(
        use_robust_regression=True,
    )

    # Calculate frequency bins if not provided
    if frequency_bins is None:
        n_fft = (spectrogram.shape[0] - 1) * 2
        frequency_bins = np.linspace(0, fs / 2, spectrogram.shape[0])

    # Compute spectral indices
    spectral_indices = spectral_processor.process(
        spectrogram,
        frequency_bins=frequency_bins,
        compute_now=compute_now,
    )

    # Compute fatigue indices
    fatigue_indices = fatigue_processor.process(
        spectral_indices,
        compute_now=compute_now,
    )

    # Combine results
    results = {}
    results.update(spectral_indices)
    results.update(fatigue_indices)

    return results


def extract_fatigue_metrics_from_segments(
    spectrogram: da.Array,
    segment_indices: List[Tuple[int, int]],
    fs: float,
    frequency_bins: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """
    Extract fatigue metrics from multiple segments of a spectrogram.

    This function analyzes fatigue metrics for each segment separately
    and returns the results organized by segment.

    Parameters
    ----------
    spectrogram : da.Array
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    segment_indices : List[Tuple[int, int]]
        List of (start, end) indices for each segment to analyze
    fs : float
        Sampling frequency in Hz
    frequency_bins : np.ndarray, optional
        Frequency values corresponding to the spectrogram bins

    Returns
    -------
    metrics : Dict[str, List[float]]
        Dictionary mapping metric names to lists of values (one per segment)
    """
    # Calculate frequency bins if not provided
    if frequency_bins is None:
        n_fft = (spectrogram.shape[0] - 1) * 2
        frequency_bins = np.linspace(0, fs / 2, spectrogram.shape[0])

    # Initialize results
    metrics = {
        "median_frequency_initial": [],
        "median_frequency_final": [],
        "median_frequency_slope": [],
        "fatigue_index": [],
        "low_high_ratio_change": [],
    }

    # Create processors
    spectral_processor = create_spectral_indices_processor(
        include_entropy=True,
    )
    fatigue_processor = create_fatigue_analysis_processor(
        use_robust_regression=True,
    )

    # Process each segment
    for start_idx, end_idx in segment_indices:
        # Extract segment
        segment = spectrogram[:, start_idx:end_idx]

        # Compute spectral indices
        spectral_indices = spectral_processor.process(
            segment,
            frequency_bins=frequency_bins,
            compute_now=True,
        )

        # Compute fatigue indices
        fatigue_indices = fatigue_processor.process(
            spectral_indices,
            compute_now=True,
        )

        # Extract key metrics
        if "median_frequency" in spectral_indices:
            mdf = spectral_indices["median_frequency"]
            if isinstance(mdf, da.Array):
                mdf = mdf.compute()

            # Initial and final values (average first/last 10%)
            seg_length = mdf.shape[0]
            initial_samples = max(1, int(seg_length * 0.1))
            final_samples = max(1, int(seg_length * 0.1))

            initial = np.nanmean(mdf[:initial_samples])
            final = np.nanmean(mdf[-final_samples:])

            metrics["median_frequency_initial"].append(float(initial))
            metrics["median_frequency_final"].append(float(final))

        # Add slopes and fatigue index
        if "median_frequency_slope" in fatigue_indices:
            slope = fatigue_indices["median_frequency_slope"]
            if hasattr(slope, "compute"):  # Check if it's a dask array
                slope = slope.compute()
            metrics["median_frequency_slope"].append(float(slope))

        if "fatigue_index" in fatigue_indices:
            fatigue_idx = fatigue_indices["fatigue_index"]
            metrics["fatigue_index"].append(float(fatigue_idx))

        # Low/high ratio change
        if "low_to_high_ratio" in spectral_indices:
            ratio = spectral_indices["low_to_high_ratio"]
            if isinstance(ratio, da.Array):
                ratio = ratio.compute()

            initial_ratio = np.nanmean(ratio[:initial_samples])
            final_ratio = np.nanmean(ratio[-final_samples:])

            if initial_ratio > 0:
                ratio_change = (final_ratio - initial_ratio) / initial_ratio * 100
            else:
                ratio_change = np.nan

            metrics["low_high_ratio_change"].append(float(ratio_change))

    return metrics


def compare_fatigue_between_segments(
    spectrogram: da.Array,
    segment_sets: Dict[str, List[Tuple[int, int]]],
    fs: float,
    frequency_bins: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare fatigue metrics between different sets of segments.

    This function is useful for comparing fatigue development between
    different conditions, exercises, or time periods.

    Parameters
    ----------
    spectrogram : da.Array
        Spectrogram data with shape (freq_bins, time_frames) or (freq_bins, time_frames, channels)
    segment_sets : Dict[str, List[Tuple[int, int]]]
        Dictionary mapping set names to lists of segment indices
    fs : float
        Sampling frequency in Hz
    frequency_bins : np.ndarray, optional
        Frequency values corresponding to the spectrogram bins

    Returns
    -------
    comparison : Dict[str, Dict[str, Dict[str, float]]]
        Nested dictionary with comparison results:
        - Top level: set names
        - Second level: metrics
        - Third level: statistics (mean, std, etc.)
    """
    # Initialize results
    comparison = {}

    # Process each set of segments
    for set_name, segments in segment_sets.items():
        # Extract metrics for this set
        metrics = extract_fatigue_metrics_from_segments(
            spectrogram, segments, fs, frequency_bins
        )

        # Compute statistics for each metric
        set_stats = {}
        for metric_name, values in metrics.items():
            # Filter out NaN values
            valid_values = [v for v in values if not np.isnan(v)]

            if valid_values:
                set_stats[metric_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "median": float(np.median(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "n_samples": len(valid_values),
                }
            else:
                set_stats[metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "median": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n_samples": 0,
                }

        comparison[set_name] = set_stats

    return comparison
