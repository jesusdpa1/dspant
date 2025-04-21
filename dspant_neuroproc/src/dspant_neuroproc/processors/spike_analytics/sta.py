"""
Spike-Triggered Average (STA) implementation for neural data analysis.

This module provides functionality for calculating STA from spike data and
a continuous signal, useful for receptive field mapping and stimulus-response
analysis.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from dspant.core.internals import public_api
from dspant.nodes.sorter import SorterNode
from dspant.nodes.stream import StreamNode

from .base import BaseSpikeTransform


@public_api
class SpikeTriggeredAnalyzer(BaseSpikeTransform):
    """
    Spike-Triggered Average (STA) and covariance analyzer.

    This class computes the average stimulus preceding spikes, allowing identification
    of stimulus features that trigger neural responses.
    """

    def __init__(
        self,
        window_size_ms: float = 100.0,
        baseline_window_ms: Optional[Tuple[float, float]] = None,
        compute_stc: bool = False,
    ):
        """
        Initialize the STA analyzer.

        Parameters
        ----------
        window_size_ms : float
            Size of the analysis window in milliseconds
        baseline_window_ms : tuple of (float, float) or None
            Window for baseline calculation in ms, relative to spike time
            Example: (-150, -100) for baseline period 150-100ms before spikes
            If None, no baseline subtraction is performed
        compute_stc : bool
            Whether to compute Spike-Triggered Covariance (STC) in addition to STA
        """
        self.window_size_ms = window_size_ms
        self.baseline_window_ms = baseline_window_ms
        self.compute_stc = compute_stc

    @staticmethod
    @jit(nopython=True, cache=True)
    def _extract_windows(
        signal: np.ndarray,
        spike_times_samples: np.ndarray,
        pre_samples: int,
        post_samples: int,
    ) -> np.ndarray:
        """
        Extract signal windows around spike times with Numba acceleration.

        Parameters
        ----------
        signal : np.ndarray
            Continuous signal array (samples × channels)
        spike_times_samples : np.ndarray
            Spike times in samples
        pre_samples : int
            Number of samples before spike to include
        post_samples : int
            Number of samples after spike to include

        Returns
        -------
        windows : np.ndarray
            Extracted windows (spikes × samples × channels)
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

    def transform(
        self,
        sorter: SorterNode,
        signal: Union[np.ndarray, StreamNode],
        unit_ids: Optional[List[int]] = None,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        Compute STA for the specified units.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        signal : np.ndarray or StreamNode
            Continuous signal data (samples × channels) or StreamNode
        unit_ids : list of int or None
            Units to include. If None, use all units.
        start_time_s : float
            Start time for analysis in seconds
        end_time_s : float or None
            End time for analysis in seconds. If None, use end of recording
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        result : dict
            Dictionary with STA results:
            - 'time_bins': Time bins relative to spike (seconds)
            - 'sta': Spike-triggered average (shape: time × channels × units)
            - 'unit_ids': List of unit IDs included in analysis
            - 'spike_counts': Number of spikes used per unit
            - If compute_stc is True:
              - 'stc': Spike-triggered covariance (time × channels × channels × units)
            - If baseline correction was applied:
              - 'sta_baseline_corrected': Baseline-corrected STA
              - 'baseline_window': Baseline window used (seconds)
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
        pre_samples = int((self.window_size_ms * sampling_rate) / 1000)
        post_samples = int((self.window_size_ms * sampling_rate) / 1000)
        window_samples = pre_samples + post_samples

        # Generate time bins relative to spike (centered at 0)
        time_bins = np.linspace(
            -self.window_size_ms / 1000, self.window_size_ms / 1000, window_samples
        )

        # Ensure signal_data is 2D (samples × channels)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(-1, 1)

        n_channels = signal_data.shape[1]
        n_units = len(used_unit_ids)

        # Initialize outputs
        sta = np.zeros((window_samples, n_channels, n_units), dtype=np.float32)
        spike_counts = np.zeros(n_units, dtype=np.int32)

        if self.compute_stc:
            stc = np.zeros(
                (window_samples, n_channels, n_channels, n_units), dtype=np.float32
            )

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

            # Extract signal windows around each spike
            windows = self._extract_windows(
                signal_data, spike_train, pre_samples, post_samples
            )

            # Compute STA (mean across spikes)
            unit_sta = np.mean(windows, axis=0)
            sta[:, :, i] = unit_sta

            # Compute STC if requested
            if self.compute_stc:
                # For each time point and unit
                for t in range(window_samples):
                    # Get all spike-triggered stimuli at this time point
                    stim_slice = windows[:, t, :]

                    # Compute covariance
                    cov = np.cov(stim_slice, rowvar=False)
                    stc[t, :, :, i] = cov

        # Prepare baseline correction if requested
        if self.baseline_window_ms is not None:
            # Convert baseline window to indices
            baseline_start_ms = self.baseline_window_ms[0]
            baseline_end_ms = self.baseline_window_ms[1]

            # Convert to time indices
            ms_per_sample = 1000 / sampling_rate
            baseline_start_idx = int(
                (self.window_size_ms + baseline_start_ms) / ms_per_sample
            )
            baseline_end_idx = int(
                (self.window_size_ms + baseline_end_ms) / ms_per_sample
            )

            # Ensure indices are within bounds
            baseline_start_idx = max(0, min(baseline_start_idx, window_samples - 1))
            baseline_end_idx = max(0, min(baseline_end_idx, window_samples - 1))

            # Skip if baseline window is invalid
            if baseline_end_idx > baseline_start_idx:
                # Calculate baseline for each unit and channel
                sta_baseline = np.mean(
                    sta[baseline_start_idx:baseline_end_idx, :, :], axis=0
                )

                # Subtract baseline
                sta_corrected = np.zeros_like(sta)
                for i in range(n_units):
                    for c in range(n_channels):
                        sta_corrected[:, c, i] = sta[:, c, i] - sta_baseline[c, i]

        # Prepare result dictionary
        result = {
            "time_bins": time_bins,
            "sta": sta,
            "unit_ids": used_unit_ids,
            "spike_counts": spike_counts,
        }

        # Add STC if computed
        if self.compute_stc:
            result["stc"] = stc

        # Add baseline-corrected STA if computed
        if (
            self.baseline_window_ms is not None
            and baseline_end_idx > baseline_start_idx
        ):
            result["sta_baseline_corrected"] = sta_corrected
            result["baseline_window"] = (
                self.baseline_window_ms[0] / 1000,
                self.baseline_window_ms[1] / 1000,
            )

        return result
