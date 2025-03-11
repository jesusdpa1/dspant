"""
Time domain indices for EMG signal analysis.

This module provides processors for computing common time domain indices from EMG signals,
specifically designed to work with extracted contractile segments from waveform_extractor.py.
It complements the spectral_indices module by focusing on time domain fatigue indicators.
"""

from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
from numba import jit

from ...engine.base import BaseProcessor


@jit(nopython=True, cache=True)
def _compute_time_domain_indices_numba(
    signal: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Compute time domain indices from EMG signal with Numba acceleration.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector (one segment)

    Returns
    -------
    mav : float
        Mean Absolute Value (average of absolute signal values)
    rms : float
        Root Mean Square (square root of the mean of the squared signal)
    iemg : float
        Integrated EMG (sum of absolute values of the signal)
    mmav : float
        Modified Mean Absolute Value (weighted average of absolute values)
    wl : float
        Waveform Length (cumulative length of the waveform)
    ssc : float
        Slope Sign Changes (number of times the slope changes sign)
    zc : float
        Zero Crossings (number of times the signal crosses zero)
    aac : float
        Average Amplitude Change (average rate of change of the signal)
    """
    n_samples = len(signal)

    # Skip if segment is too short
    if n_samples <= 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Initialize metrics
    mav = 0.0
    rms = 0.0
    iemg = 0.0
    mmav = 0.0
    wl = 0.0
    ssc = 0.0
    zc = 0.0
    aac = 0.0

    # Small threshold to avoid counting noise
    threshold = 0.0001 * np.max(np.abs(signal))

    # Mean Absolute Value (MAV)
    for i in range(n_samples):
        mav += abs(signal[i])
    mav /= n_samples

    # Root Mean Square (RMS)
    for i in range(n_samples):
        rms += signal[i] * signal[i]
    rms = np.sqrt(rms / n_samples)

    # Integrated EMG (IEMG)
    for i in range(n_samples):
        iemg += abs(signal[i])

    # Modified Mean Absolute Value (MMAV)
    weight_sum = 0.0
    for i in range(n_samples):
        weight = 1.0
        if i < 0.25 * n_samples or i > 0.75 * n_samples:
            weight = 0.5
        mmav += weight * abs(signal[i])
        weight_sum += weight
    mmav /= weight_sum

    # Waveform Length (WL)
    for i in range(1, n_samples):
        wl += abs(signal[i] - signal[i - 1])

    # Slope Sign Changes (SSC)
    for i in range(1, n_samples - 1):
        if (
            (signal[i] > signal[i - 1] and signal[i] > signal[i + 1])
            or (signal[i] < signal[i - 1] and signal[i] < signal[i + 1])
        ) and (
            abs(signal[i] - signal[i - 1]) >= threshold
            or abs(signal[i] - signal[i + 1]) >= threshold
        ):
            ssc += 1
    ssc /= (n_samples - 2) if n_samples > 2 else 1  # Normalize

    # Zero Crossings (ZC)
    for i in range(1, n_samples):
        if (
            (signal[i] > 0 and signal[i - 1] < 0)
            or (signal[i] < 0 and signal[i - 1] > 0)
        ) and abs(signal[i] - signal[i - 1]) >= threshold:
            zc += 1
    zc /= (n_samples - 1) if n_samples > 1 else 1  # Normalize

    # Average Amplitude Change (AAC)
    for i in range(1, n_samples):
        aac += abs(signal[i] - signal[i - 1])
    aac /= (n_samples - 1) if n_samples > 1 else 1

    return mav, rms, iemg, mmav, wl, ssc, zc, aac


@jit(nopython=True, cache=True)
def _compute_wilson_amplitude_numba(
    signal: np.ndarray, threshold_factor: float = 0.1
) -> float:
    """
    Compute Wilson Amplitude (WAMP) from EMG signal with Numba acceleration.

    WAMP counts the number of times that the difference between
    consecutive samples exceeds a threshold.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector
    threshold_factor : float
        Threshold as a factor of the signal's maximum amplitude

    Returns
    -------
    wamp : float
        Wilson Amplitude value (normalized by signal length)
    """
    n_samples = len(signal)

    # Skip if segment is too short
    if n_samples <= 1:
        return 0.0

    # Set threshold as a percentage of the maximum amplitude
    threshold = threshold_factor * np.max(np.abs(signal))

    wamp_count = 0
    for i in range(1, n_samples):
        if abs(signal[i] - signal[i - 1]) > threshold:
            wamp_count += 1

    # Normalize by signal length
    return wamp_count / (n_samples - 1) if n_samples > 1 else 0


@jit(nopython=True, cache=True)
def _compute_myopulse_percentage_rate_numba(
    signal: np.ndarray, threshold_factor: float = 0.5
) -> float:
    """
    Compute Myopulse Percentage Rate (MYOP) with Numba acceleration.

    MYOP is the percentage of time that the absolute value of the
    EMG signal exceeds a threshold.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector
    threshold_factor : float
        Threshold as a factor of the signal's mean absolute value

    Returns
    -------
    myop : float
        Myopulse Percentage Rate (0.0 to 1.0)
    """
    n_samples = len(signal)

    # Skip if segment is too short
    if n_samples <= 1:
        return 0.0

    # Set threshold as a percentage of mean absolute value
    mav = np.mean(np.abs(signal))
    threshold = threshold_factor * mav

    count_above_threshold = 0
    for i in range(n_samples):
        if abs(signal[i]) > threshold:
            count_above_threshold += 1

    return count_above_threshold / n_samples


@jit(nopython=True, cache=True)
def _compute_fatigue_indices_numba(
    signal: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Compute fatigue-specific time domain indices.

    These indices are particularly sensitive to muscle fatigue.

    Parameters
    ----------
    signal : np.ndarray
        Input signal vector

    Returns
    -------
    mdf_arv_ratio : float
        Ratio of median frequency to average rectified value
    rms_mav_ratio : float
        Ratio of RMS to MAV (increases with fatigue)
    peak_arv_ratio : float
        Ratio of peak amplitude to average amplitude
    mnf_mdf_ratio : float
        Ratio of mean frequency to median frequency (approx. using ZC)
    cv_amplitude : float
        Coefficient of variation of EMG amplitude (std/mean)
    """
    n_samples = len(signal)

    # Skip if segment is too short
    if n_samples <= 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Mean Absolute Value (MAV)
    mav = np.mean(np.abs(signal))

    # Root Mean Square (RMS)
    rms = np.sqrt(np.mean(np.power(signal, 2)))

    # RMS to MAV ratio (increases with fatigue)
    rms_mav_ratio = rms / mav if mav > 0 else 0.0

    # Peak to Average Ratio (PAR)
    peak = np.max(np.abs(signal))
    peak_arv_ratio = peak / mav if mav > 0 else 0.0

    # Coefficient of variation of amplitude
    std = np.std(np.abs(signal))
    cv_amplitude = std / mav if mav > 0 else 0.0

    # Approximation of MNF/MDF ratio using zero crossings
    # This is a simplified approach that doesn't require FFT
    # Based on the relationship between zero crossings and spectral content

    # Count zero crossings
    zc_count = 0
    for i in range(1, n_samples):
        if (signal[i] * signal[i - 1]) < 0:
            zc_count += 1

    # Normalize to get a value between 0 and 1
    zc_rate = zc_count / (n_samples - 1)

    # Use zero crossing rate as a proxy for frequency
    # Higher zc_rate ≈ higher mean frequency
    # As fatigue increases, MNF decreases more than MDF,
    # so the MNF/MDF ratio decreases

    # We'll use the inverse relationship to get a value that increases with fatigue
    mnf_mdf_ratio = 1.0 - zc_rate

    # MDF to ARV ratio (decreases with fatigue)
    # Using ZC as a proxy for MDF
    mdf_arv_ratio = zc_rate / mav if mav > 0 else 0.0

    return mdf_arv_ratio, rms_mav_ratio, peak_arv_ratio, mnf_mdf_ratio, cv_amplitude


@jit(nopython=True, cache=True)
def _compute_temporal_trend_indices_numba(
    signals: np.ndarray, window_size: int = 10
) -> Tuple[float, float, float, float]:
    """
    Compute trend indices from a sequence of EMG signals.

    These indices track changes in time domain features over multiple
    signal segments to quantify fatigue development.

    Parameters
    ----------
    signals : np.ndarray
        3D array of signals (segments × samples × channels)
    window_size : int
        Window size for smoothing

    Returns
    -------
    mav_slope : float
        Slope of MAV over time (normalized)
    rms_slope : float
        Slope of RMS over time (normalized)
    rms_mav_slope : float
        Slope of RMS/MAV ratio over time
    par_slope : float
        Slope of Peak-to-Average Ratio over time
    """
    n_segments, n_samples, n_channels = signals.shape

    if n_segments < 2 or n_channels < 1:
        return 0.0, 0.0, 0.0, 0.0

    # Initialize arrays to store values for each segment
    mav_values = np.zeros(n_segments)
    rms_values = np.zeros(n_segments)
    rms_mav_values = np.zeros(n_segments)
    par_values = np.zeros(n_segments)

    # Process each segment (using first channel only)
    for i in range(n_segments):
        signal = signals[i, :, 0]

        # Calculate MAV
        mav = np.mean(np.abs(signal))
        mav_values[i] = mav

        # Calculate RMS
        rms = np.sqrt(np.mean(np.power(signal, 2)))
        rms_values[i] = rms

        # Calculate RMS/MAV ratio
        rms_mav = rms / mav if mav > 0 else 0
        rms_mav_values[i] = rms_mav

        # Calculate Peak-to-Average Ratio
        peak = np.max(np.abs(signal))
        par = peak / mav if mav > 0 else 0
        par_values[i] = par

    # Apply moving average if window_size > 1
    if window_size > 1 and n_segments >= window_size:
        # Moving average for smoothing
        mav_smooth = np.zeros(n_segments - window_size + 1)
        rms_smooth = np.zeros(n_segments - window_size + 1)
        rms_mav_smooth = np.zeros(n_segments - window_size + 1)
        par_smooth = np.zeros(n_segments - window_size + 1)

        for i in range(n_segments - window_size + 1):
            mav_smooth[i] = np.mean(mav_values[i : i + window_size])
            rms_smooth[i] = np.mean(rms_values[i : i + window_size])
            rms_mav_smooth[i] = np.mean(rms_mav_values[i : i + window_size])
            par_smooth[i] = np.mean(par_values[i : i + window_size])
    else:
        # No smoothing
        mav_smooth = mav_values
        rms_smooth = rms_values
        rms_mav_smooth = rms_mav_values
        par_smooth = par_values

    # Calculate slopes using linear regression
    n = len(mav_smooth)

    # Time points (normalized to 0-1 range)
    x = np.arange(n) / max(1, n - 1)

    # Calculate linear regression for each metric
    slopes = np.zeros(4)

    for idx, values in enumerate([mav_smooth, rms_smooth, rms_mav_smooth, par_smooth]):
        # Skip if all values are the same (to avoid division by zero)
        if np.all(values == values[0]):
            slopes[idx] = 0.0
            continue

        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(values)

        # Calculate numerator and denominator for slope
        numerator = 0.0
        denominator = 0.0

        for i in range(n):
            x_diff = x[i] - x_mean
            y_diff = values[i] - y_mean
            numerator += x_diff * y_diff
            denominator += x_diff * x_diff

        # Calculate slope
        slope = numerator / denominator if denominator != 0 else 0.0

        # Normalize slope by mean value for percentage change
        norm_slope = slope / max(abs(y_mean), 1e-10) if y_mean != 0 else 0.0

        slopes[idx] = norm_slope

    return slopes[0], slopes[1], slopes[2], slopes[3]


class TimeDomainIndicesProcessor(BaseProcessor):
    """
    Processor for computing time domain indices from EMG segments.

    This processor calculates various time domain features from EMG segments,
    including standard indices (MAV, RMS, WL) and fatigue-specific indices.
    It is designed to work with the 3D dask arrays output from SegmentExtractionProcessor.
    """

    def __init__(
        self,
        indices: List[str] = ["basic", "fatigue"],
        wilson_threshold_factor: float = 0.1,
        myopulse_threshold_factor: float = 0.5,
        use_numba: bool = True,
        window_size: int = 1,
    ):
        """
        Initialize the EMG time domain indices processor.

        Parameters
        ----------
        indices : List[str]
            List of index categories to compute:
            - "basic": MAV, RMS, IEMG, WL, ZC, SSC
            - "fatigue": Fatigue-specific indices
            - "advanced": MMAV, WAMP, MYOP, etc.
        wilson_threshold_factor : float
            Threshold factor for WAMP calculation (as a factor of max amplitude)
        myopulse_threshold_factor : float
            Threshold factor for MYOP calculation (as a factor of MAV)
        use_numba : bool
            Whether to use Numba acceleration for computation
        window_size : int
            Window size for smoothing when computing trends
        """
        self.indices = indices
        self.wilson_threshold_factor = wilson_threshold_factor
        self.myopulse_threshold_factor = myopulse_threshold_factor
        self.use_numba = use_numba
        self.window_size = window_size
        self._overlap_samples = 0

    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> Dict:
        """
        Process function required by BaseProcessor interface.

        This is a wrapper around process_segments that ensures consistent API.

        Parameters
        ----------
        data : da.Array
            Input dask array - should be a 3D array of segments
        fs : float, optional
            Sampling frequency in Hz
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        Dict
            Dictionary containing computed indices
        """
        # Check if input is 3D (segments × samples × channels)
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D array (segments × samples × channels), got {data.ndim}D array"
            )

        return self.process_segments(data, fs, **kwargs)

    def process_segments(
        self, segments: da.Array, fs: Optional[float] = None, **kwargs
    ) -> Dict:
        """
        Compute time domain indices for pre-extracted EMG segments.

        Parameters
        ----------
        segments : da.Array
            3D Dask array of segments (segments × samples × channels)
        fs : float, optional
            Sampling frequency (Hz), not directly used for basic time domain indices
            but kept for API consistency
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        indices : dict
            Dictionary of computed indices for each segment and channel
        """
        # Get segment dimensions
        n_segments, n_samples, n_channels = segments.shape

        # Define chunking strategy for better parallelism
        chunk_size = min(100, n_segments)  # Process up to 100 segments at a time

        # Compute now or return dask arrays
        compute_now = kwargs.get("compute_now", True)

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

                    # Initialize segment indices
                    segment_indices = {}

                    # Compute basic time domain indices
                    if "basic" in self.indices:
                        if self.use_numba:
                            mav, rms, iemg, mmav, wl, ssc, zc, aac = (
                                _compute_time_domain_indices_numba(segment_data)
                            )

                            segment_indices.update(
                                {
                                    "mav": float(mav),  # Mean Absolute Value
                                    "rms": float(rms),  # Root Mean Square
                                    "iemg": float(iemg),  # Integrated EMG
                                    "wl": float(wl),  # Waveform Length
                                    "ssc": float(ssc),  # Slope Sign Changes
                                    "zc": float(zc),  # Zero Crossings
                                }
                            )
                        else:
                            # Fallback to non-Numba implementation
                            segment_indices.update(
                                {
                                    "mav": float(np.mean(np.abs(segment_data))),
                                    "rms": float(np.sqrt(np.mean(segment_data**2))),
                                    "iemg": float(np.sum(np.abs(segment_data))),
                                    "wl": float(np.sum(np.abs(np.diff(segment_data)))),
                                    "ssc": float(
                                        np.sum(
                                            np.diff(np.signbit(np.diff(segment_data)))
                                            != 0
                                        )
                                        / max(1, len(segment_data) - 2)
                                    ),
                                    "zc": float(
                                        np.sum(np.diff(np.signbit(segment_data)) != 0)
                                        / max(1, len(segment_data) - 1)
                                    ),
                                }
                            )

                    # Compute fatigue-specific indices
                    if "fatigue" in self.indices:
                        if self.use_numba:
                            (
                                mdf_arv_ratio,
                                rms_mav_ratio,
                                peak_arv_ratio,
                                mnf_mdf_ratio,
                                cv_amplitude,
                            ) = _compute_fatigue_indices_numba(segment_data)

                            segment_indices.update(
                                {
                                    "rms_mav_ratio": float(
                                        rms_mav_ratio
                                    ),  # RMS/MAV ratio (↑ with fatigue)
                                    "peak_arv_ratio": float(
                                        peak_arv_ratio
                                    ),  # Peak/ARV ratio
                                    "mnf_mdf_ratio": float(
                                        mnf_mdf_ratio
                                    ),  # Approx MNF/MDF ratio
                                    "cv_amplitude": float(
                                        cv_amplitude
                                    ),  # CV of amplitude
                                    "mdf_arv_ratio": float(
                                        mdf_arv_ratio
                                    ),  # MDF/ARV ratio (↓ with fatigue)
                                }
                            )
                        else:
                            # Fallback to non-Numba implementation
                            mav = np.mean(np.abs(segment_data))
                            rms = np.sqrt(np.mean(segment_data**2))
                            peak = np.max(np.abs(segment_data))
                            std = np.std(np.abs(segment_data))

                            # Calculate zero crossings for frequency estimation
                            zc_count = np.sum(np.diff(np.signbit(segment_data)) != 0)
                            zc_rate = zc_count / max(1, len(segment_data) - 1)

                            segment_indices.update(
                                {
                                    "rms_mav_ratio": float(rms / mav if mav > 0 else 0),
                                    "peak_arv_ratio": float(
                                        peak / mav if mav > 0 else 0
                                    ),
                                    "mnf_mdf_ratio": float(1.0 - zc_rate),
                                    "cv_amplitude": float(std / mav if mav > 0 else 0),
                                    "mdf_arv_ratio": float(
                                        zc_rate / mav if mav > 0 else 0
                                    ),
                                }
                            )

                    # Compute advanced time domain indices
                    if "advanced" in self.indices:
                        if self.use_numba:
                            # Add Modified Mean Absolute Value from basic computation
                            if "basic" in self.indices:
                                segment_indices["mmav"] = float(mmav)
                                segment_indices["aac"] = float(aac)
                            else:
                                _, _, _, mmav, _, _, _, aac = (
                                    _compute_time_domain_indices_numba(segment_data)
                                )
                                segment_indices["mmav"] = float(mmav)
                                segment_indices["aac"] = float(aac)

                            # Compute Wilson Amplitude
                            wamp = _compute_wilson_amplitude_numba(
                                segment_data, self.wilson_threshold_factor
                            )
                            segment_indices["wamp"] = float(wamp)

                            # Compute Myopulse Percentage Rate
                            myop = _compute_myopulse_percentage_rate_numba(
                                segment_data, self.myopulse_threshold_factor
                            )
                            segment_indices["myop"] = float(myop)

                        else:
                            # Fallback to non-Numba implementation
                            # Modified Mean Absolute Value
                            weights = np.ones(len(segment_data))
                            weights[: (len(segment_data) // 4)] = 0.5
                            weights[-(len(segment_data) // 4) :] = 0.5
                            mmav = np.average(np.abs(segment_data), weights=weights)
                            segment_indices["mmav"] = float(mmav)

                            # Average Amplitude Change
                            aac = np.mean(np.abs(np.diff(segment_data)))
                            segment_indices["aac"] = float(aac)

                            # Wilson Amplitude
                            threshold = self.wilson_threshold_factor * np.max(
                                np.abs(segment_data)
                            )
                            wamp = np.sum(
                                np.abs(np.diff(segment_data)) > threshold
                            ) / max(1, len(segment_data) - 1)
                            segment_indices["wamp"] = float(wamp)

                            # Myopulse Percentage Rate
                            threshold = self.myopulse_threshold_factor * np.mean(
                                np.abs(segment_data)
                            )
                            myop = np.sum(np.abs(segment_data) > threshold) / len(
                                segment_data
                            )
                            segment_indices["myop"] = float(myop)

                    channel_results.append(segment_indices)

                # Add channel results to chunk results
                chunk_results[f"channel_{ch}"] = channel_results

            # Compute trend indices if we have multiple segments
            if chunk_segments >= 2 and "fatigue" in self.indices:
                for ch in range(chunk_channels):
                    channel_trend_indices = {}

                    # Use numba implementation if available
                    if self.use_numba:
                        mav_slope, rms_slope, rms_mav_slope, par_slope = (
                            _compute_temporal_trend_indices_numba(
                                segments_np[:, :, ch : ch + 1],
                                window_size=self.window_size,
                            )
                        )

                        channel_trend_indices = {
                            "mav_slope": float(mav_slope),
                            "rms_slope": float(rms_slope),
                            "rms_mav_slope": float(rms_mav_slope),
                            "par_slope": float(par_slope),
                        }
                    else:
                        # Fallback to non-numba implementation
                        # Extract MAV, RMS, etc. from all segments
                        mav_values = []
                        rms_values = []
                        rms_mav_values = []
                        par_values = []

                        for seg_idx in range(chunk_segments):
                            results = chunk_results[f"channel_{ch}"][seg_idx]
                            if "mav" in results and "rms" in results:
                                mav = results["mav"]
                                rms = results["rms"]
                                mav_values.append(mav)
                                rms_values.append(rms)
                                rms_mav_values.append(rms / mav if mav > 0 else 0)

                                if "peak_arv_ratio" in results:
                                    par_values.append(results["peak_arv_ratio"])

                        # Calculate trends if we have enough data
                        if len(mav_values) >= 2:
                            # Apply moving average smoothing if window_size > 1
                            if (
                                self.window_size > 1
                                and len(mav_values) >= self.window_size
                            ):
                                # Moving average for smoothing
                                mav_smooth = []
                                rms_smooth = []
                                rms_mav_smooth = []
                                par_smooth = []

                                for i in range(len(mav_values) - self.window_size + 1):
                                    mav_smooth.append(
                                        np.mean(mav_values[i : i + self.window_size])
                                    )
                                    rms_smooth.append(
                                        np.mean(rms_values[i : i + self.window_size])
                                    )
                                    rms_mav_smooth.append(
                                        np.mean(
                                            rms_mav_values[i : i + self.window_size]
                                        )
                                    )
                                    if par_values:
                                        par_smooth.append(
                                            np.mean(
                                                par_values[i : i + self.window_size]
                                            )
                                        )
                            else:
                                # No smoothing
                                mav_smooth = mav_values
                                rms_smooth = rms_values
                                rms_mav_smooth = rms_mav_values
                                par_smooth = par_values

                            # Calculate slopes using linear regression
                            x = np.arange(len(mav_smooth)) / max(
                                1, len(mav_smooth) - 1
                            )  # Normalize to 0-1

                            # Function to calculate normalized slope
                            def calc_slope(y_values):
                                if len(y_values) < 2 or all(
                                    v == y_values[0] for v in y_values
                                ):
                                    return 0.0

                                # Calculate linear regression
                                p = np.polyfit(x, y_values, 1)
                                slope = p[0]

                                # Normalize by mean value
                                mean_val = np.mean(y_values)
                                if mean_val != 0:
                                    return slope / abs(mean_val)
                                return slope

                            channel_trend_indices = {
                                "mav_slope": float(calc_slope(mav_smooth)),
                                "rms_slope": float(calc_slope(rms_smooth)),
                                "rms_mav_slope": float(calc_slope(rms_mav_smooth)),
                            }

                            if par_smooth:
                                channel_trend_indices["par_slope"] = float(
                                    calc_slope(par_smooth)
                                )

                    # Add trend indices to results
                    chunk_results[f"channel_{ch}_trends"] = channel_trend_indices

            return chunk_results

        # Apply processing function to chunks
        if compute_now:
            # Compute directly
            results_blocks = []
            for i in range(0, n_segments, chunk_size):
                end_idx = min(i + chunk_size, n_segments)
                segment_chunk = segments[i:end_idx].compute()
                results_blocks.append(process_chunk(segment_chunk))
        else:
            # Create dask delayed objects
            import dask

            results_blocks = []
            for i in range(0, n_segments, chunk_size):
                end_idx = min(i + chunk_size, n_segments)
                segment_chunk = segments[i:end_idx]
                chunk_result = dask.delayed(process_chunk)(segment_chunk)
                results_blocks.append(chunk_result)

            # Return without computing
            return {"delayed_results": results_blocks}

        # Collect and merge results
        final_results = {}

        # Merge segment results for each channel
        for ch in range(n_channels):
            channel_key = f"channel_{ch}"
            trends_key = f"channel_{ch}_trends"

            # Collect all segment results for this channel
            all_segment_results = []
            for block in results_blocks:
                if channel_key in block:
                    all_segment_results.extend(block[channel_key])

            # Store merged segment results
            final_results[channel_key] = all_segment_results

            # Collect and merge trend results for this channel
            trend_results = {}
            for block in results_blocks:
                if trends_key in block:
                    for trend_name, trend_value in block[trends_key].items():
                        if trend_name not in trend_results:
                            trend_results[trend_name] = []
                        trend_results[trend_name].append(trend_value)

            # Average trend values from different blocks
            if trend_results:
                merged_trends = {}
                for trend_name, trend_values in trend_results.items():
                    merged_trends[trend_name] = float(np.mean(trend_values))
                final_results[trends_key] = merged_trends

        return final_results

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
                "indices": self.indices,
                "use_numba": self.use_numba,
                "window_size": self.window_size,
                "wilson_threshold_factor": self.wilson_threshold_factor,
                "myopulse_threshold_factor": self.myopulse_threshold_factor,
            }
        )
        return base_summary


def create_time_domain_analyzer(
    include_basic: bool = True,
    include_fatigue: bool = True,
    include_advanced: bool = False,
    window_size: int = 10,
    use_numba: bool = True,
) -> TimeDomainIndicesProcessor:
    """
    Create a time domain indices processor with common configuration.

    Parameters
    ----------
    include_basic : bool
        Whether to include basic time domain indices (MAV, RMS, WL, etc.)
    include_fatigue : bool
        Whether to include fatigue-specific indices
    include_advanced : bool
        Whether to include advanced time domain indices (MMAV, WAMP, MYOP)
    window_size : int
        Window size for smoothing when computing trends
    use_numba : bool
        Whether to use Numba acceleration

    Returns
    -------
    TimeDomainIndicesProcessor
        Configured time domain indices processor
    """
    indices = []
    if include_basic:
        indices.append("basic")
    if include_fatigue:
        indices.append("fatigue")
    if include_advanced:
        indices.append("advanced")

    return TimeDomainIndicesProcessor(
        indices=indices,
        window_size=window_size,
        use_numba=use_numba,
    )


def create_fatigue_analyzer(
    window_size: int = 15,
    use_numba: bool = True,
) -> TimeDomainIndicesProcessor:
    """
    Create a time domain indices processor optimized for fatigue analysis.

    This configuration focuses on indices that are sensitive to muscle fatigue
    and uses a larger window size for trend smoothing.

    Parameters
    ----------
    window_size : int
        Window size for smoothing when computing trends
    use_numba : bool
        Whether to use Numba acceleration

    Returns
    -------
    TimeDomainIndicesProcessor
        Configured time domain indices processor for fatigue analysis
    """
    return TimeDomainIndicesProcessor(
        indices=["basic", "fatigue", "advanced"],
        window_size=window_size,
        use_numba=use_numba,
        wilson_threshold_factor=0.15,
        myopulse_threshold_factor=0.6,
    )


def create_fast_analyzer(
    use_numba: bool = True,
) -> TimeDomainIndicesProcessor:
    """
    Create a lightweight time domain indices processor for quick analysis.

    This configuration computes only the most essential indices with
    minimal processing overhead.

    Parameters
    ----------
    use_numba : bool
        Whether to use Numba acceleration

    Returns
    -------
    TimeDomainIndicesProcessor
        Configured time domain indices processor for quick analysis
    """
    return TimeDomainIndicesProcessor(
        indices=["basic"],
        window_size=1,  # No smoothing
        use_numba=use_numba,
    )
