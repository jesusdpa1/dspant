"""
Spike-Triggered Average Waveform analysis for neural data.

This module provides functionality for calculating average waveforms aligned to spike times,
useful for characterizing action potential shapes, examining relationships between
spikes and local field potentials (LFPs), and studying neural response properties.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from dspant.core.internals import public_api
from dspant.nodes.sorter import SorterNode
from dspant.nodes.stream import StreamNode

from .base import BaseSpikeTransform


@public_api
class SpikeTriggeredWaveformAnalyzer(BaseSpikeTransform):
    """
    Spike-Triggered Average Waveform analyzer.

    This class computes the average waveform pattern around spike events,
    allowing characterization of spike shapes and their relationship to the
    underlying neural signal.
    """

    def __init__(
        self,
        pre_time_ms: float = 2.0,
        post_time_ms: float = 4.0,
        compute_std: bool = True,
        compute_median: bool = False,
        align_method: str = "peak",
    ):
        """
        Initialize the Spike-Triggered Waveform analyzer.

        Parameters
        ----------
        pre_time_ms : float
            Time before spike to include in window (milliseconds)
        post_time_ms : float
            Time after spike to include in window (milliseconds)
        compute_std : bool
            Whether to compute standard deviation across waveforms
        compute_median : bool
            Whether to compute median waveform (more robust to outliers)
        align_method : str
            Method for precise alignment within window:
            - "peak": Align to maximum amplitude (for positive spikes)
            - "trough": Align to minimum amplitude (for negative spikes)
            - "none": Use exact spike time without realignment
        """
        self.pre_time_ms = pre_time_ms
        self.post_time_ms = post_time_ms
        self.compute_std = compute_std
        self.compute_median = compute_median
        self.align_method = align_method

    @staticmethod
    @jit(nopython=True, cache=True)
    def _extract_waveform_windows(
        signal: np.ndarray,
        spike_times_samples: np.ndarray,
        pre_samples: int,
        post_samples: int,
    ) -> np.ndarray:
        """
        Extract waveform windows around spike times with Numba acceleration.

        Parameters
        ----------
        signal : np.ndarray
            Neural signal array (samples × channels)
        spike_times_samples : np.ndarray
            Spike times in samples
        pre_samples : int
            Number of samples before spike to include
        post_samples : int
            Number of samples after spike to include

        Returns
        -------
        windows : np.ndarray
            Extracted waveform windows (spikes × samples × channels)
        """
        n_spikes = len(spike_times_samples)
        n_samples = pre_samples + post_samples
        n_channels = signal.shape[1] if signal.ndim > 1 else 1

        # Reshape 1D signal to 2D for consistent handling
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        # Total signal length
        signal_length = signal.shape[0]

        # Output array: spikes × samples × channels
        windows = np.zeros((n_spikes, n_samples, n_channels), dtype=np.float32)

        # Extract window for each spike
        for i, spike_time in enumerate(spike_times_samples):
            # Calculate window boundaries
            start = spike_time - pre_samples
            end = spike_time + post_samples

            # Skip if window extends outside signal
            if start < 0 or end > signal_length:
                continue

            # Extract window for all channels
            for c in range(n_channels):
                windows[i, :, c] = signal[start:end, c]

        return windows

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _realign_windows(
        windows: np.ndarray,
        pre_samples: int,
        post_samples: int,
        align_method: str,
        max_shift: int = 5,
    ) -> np.ndarray:
        """
        Realign waveform windows for more precise spike alignment.

        Parameters
        ----------
        windows : np.ndarray
            Waveform windows (spikes × samples × channels)
        pre_samples : int
            Number of samples before spike
        post_samples : int
            Number of samples after spike
        align_method : str
            Alignment method ('peak', 'trough', 'none')
        max_shift : int
            Maximum allowed shift in samples

        Returns
        -------
        aligned_windows : np.ndarray
            Realigned waveform windows
        """
        n_spikes, n_samples, n_channels = windows.shape

        # If no alignment requested, return original windows
        if align_method == "none":
            return windows

        # Create output array
        aligned_windows = np.zeros_like(windows)

        # Calculate search window (limited to +/- max_shift samples around spike time)
        search_start = max(0, pre_samples - max_shift)
        search_end = min(n_samples, pre_samples + max_shift + 1)

        # Process each spike and channel
        for i in prange(n_spikes):
            for c in range(n_channels):
                # Get current window for this spike and channel
                window = windows[i, :, c]

                # Find alignment point based on method
                if align_method == "peak":
                    # Find maximum amplitude within search window
                    search_segment = window[search_start:search_end]
                    peak_idx = np.argmax(search_segment) + search_start
                elif align_method == "trough":
                    # Find minimum amplitude within search window
                    search_segment = window[search_start:search_end]
                    peak_idx = np.argmin(search_segment) + search_start
                else:
                    # Default to center if unknown method
                    peak_idx = pre_samples

                # Calculate shift amount
                shift = pre_samples - peak_idx

                # Apply shift by copying appropriate segments
                if shift == 0:
                    # No shift needed
                    aligned_windows[i, :, c] = window
                elif shift > 0:
                    # Shift right (need earlier samples)
                    if shift <= pre_samples:
                        # Can shift within window
                        aligned_windows[i, shift:, c] = window[:-shift]
                    else:
                        # Too much shift, keep original
                        aligned_windows[i, :, c] = window
                else:
                    # Shift left (need later samples)
                    if -shift < post_samples:
                        # Can shift within window
                        aligned_windows[i, :shift, c] = window[-shift:]
                    else:
                        # Too much shift, keep original
                        aligned_windows[i, :, c] = window

        return aligned_windows

    def transform(
        self,
        sorter: SorterNode,
        signal: Union[np.ndarray, StreamNode],
        unit_ids: Optional[List[int]] = None,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        filter_outliers: bool = False,
        max_deviation: float = 3.0,
        **kwargs,
    ) -> Dict:
        """
        Compute spike-triggered average waveforms for the specified units.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        signal : np.ndarray or StreamNode
            Neural signal data (samples × channels) or StreamNode
        unit_ids : list of int or None
            Units to include. If None, use all units.
        start_time_s : float
            Start time for analysis in seconds
        end_time_s : float or None
            End time for analysis in seconds. If None, use end of recording
        filter_outliers : bool
            Whether to filter outlier waveforms before averaging
        max_deviation : float
            Maximum allowed deviation (in std) for outlier rejection
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        result : dict
            Dictionary with spike-triggered waveform results:
            - 'time_bins': Time bins relative to spike (seconds)
            - 'staw_mean': Mean waveform (shape: time × channels × units)
            - 'unit_ids': List of unit IDs included in analysis
            - 'spike_counts': Number of spikes used per unit
            - If compute_std is True:
              - 'staw_std': Standard deviation of waveforms
            - If compute_median is True:
              - 'staw_median': Median waveform
            - If filter_outliers is True:
              - 'outlier_counts': Number of outliers removed per unit
        """
        # Validate the sorter
        self._validate_sorter(sorter)
        sampling_rate = sorter.sampling_frequency

        # Load signal data
        if isinstance(signal, StreamNode):
            # Load data from StreamNode
            signal_data = signal.load_data().compute()
            signal_fs = signal.fs

            # Check sampling rates match
            if abs(signal_fs - sampling_rate) > 1e-6:
                raise ValueError(
                    f"Signal sampling rate ({signal_fs} Hz) does not match "
                    f"sorter sampling rate ({sampling_rate} Hz)"
                )
        else:
            # Use provided numpy array
            signal_data = signal

        # Get time range in samples
        start_frame, end_frame = self._get_time_range_samples(
            sorter, start_time_s, end_time_s
        )

        # Get filtered unit IDs
        used_unit_ids = self._get_filtered_unit_ids(sorter, unit_ids)

        # Calculate window parameters
        pre_samples = int((self.pre_time_ms * sampling_rate) / 1000)
        post_samples = int((self.post_time_ms * sampling_rate) / 1000)
        window_samples = pre_samples + post_samples

        # Generate time bins relative to spike (centered at 0)
        time_bins = np.linspace(
            -self.pre_time_ms / 1000, self.post_time_ms / 1000, window_samples
        )

        # Ensure signal_data is 2D (samples × channels)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)

        n_channels = signal_data.shape[1]
        n_units = len(used_unit_ids)

        # Initialize outputs
        staw_mean = np.zeros((window_samples, n_channels, n_units), dtype=np.float32)
        spike_counts = np.zeros(n_units, dtype=np.int32)
        outlier_counts = np.zeros(n_units, dtype=np.int32) if filter_outliers else None

        if self.compute_std:
            staw_std = np.zeros_like(staw_mean)

        if self.compute_median:
            staw_median = np.zeros_like(staw_mean)

        # Process each unit
        for i, unit_id in enumerate(used_unit_ids):
            # Get spike times for this unit in the specified time range
            spike_train = sorter.get_unit_spike_train(
                unit_id, start_frame=start_frame, end_frame=end_frame
            )

            # Skip if no spikes
            if len(spike_train) == 0:
                continue

            # Count spikes for this unit
            spike_counts[i] = len(spike_train)

            # Extract waveform windows around each spike
            windows = self._extract_waveform_windows(
                signal_data, spike_train, pre_samples, post_samples
            )

            # Apply alignment if requested
            if self.align_method != "none":
                windows = self._realign_windows(
                    windows, pre_samples, post_samples, self.align_method
                )

            # Filter outliers if requested
            if filter_outliers:
                # For each channel
                for c in range(n_channels):
                    # Calculate mean and std across spikes
                    win_mean = np.mean(windows[:, :, c], axis=0)
                    win_std = np.std(windows[:, :, c], axis=0)

                    # Calculate deviation metric for each spike
                    deviation = np.mean(
                        np.abs(windows[:, :, c] - win_mean) / (win_std + 1e-6), axis=1
                    )

                    # Identify outliers
                    outlier_mask = deviation > max_deviation
                    outlier_counts[i] += np.sum(outlier_mask)

                    # Create mask for non-outliers
                    valid_mask = ~outlier_mask

                    # Skip if all are outliers
                    if np.sum(valid_mask) == 0:
                        continue

                    # Use only valid waveforms
                    windows = windows[valid_mask, :, :]

            # Compute mean waveform
            staw_mean[:, :, i] = np.mean(windows, axis=0)

            # Compute std if requested
            if self.compute_std:
                staw_std[:, :, i] = np.std(windows, axis=0)

            # Compute median if requested
            if self.compute_median:
                staw_median[:, :, i] = np.median(windows, axis=0)

        # Prepare result dictionary
        result = {
            "time_bins": time_bins,
            "staw_mean": staw_mean,
            "unit_ids": used_unit_ids,
            "spike_counts": spike_counts,
        }

        # Add optional outputs
        if self.compute_std:
            result["staw_std"] = staw_std

        if self.compute_median:
            result["staw_median"] = staw_median

        if filter_outliers:
            result["outlier_counts"] = outlier_counts

        return result
