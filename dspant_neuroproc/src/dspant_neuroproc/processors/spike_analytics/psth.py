"""
Peristimulus Time Histogram (PSTH) implementation for neural data analysis.

This module provides functionality for generating PSTH from spike data aligned
to stimulus or event triggers, useful for analyzing neural responses to stimuli.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from dspant.core.internals import public_api
from dspant.nodes.epoch import EpocNode
from dspant.nodes.sorter import SorterNode

from .base import BaseSpikeTransform
from .density_estimation import SpikeDensityEstimator


@public_api
class PSTHAnalyzer(BaseSpikeTransform):
    """
    Peristimulus Time Histogram (PSTH) generator for neural responses.

    This class computes spike histograms and densities aligned to stimulus events,
    allowing analysis of neural responses to stimuli or events.
    """

    def __init__(
        self,
        bin_size_ms: float = 10.0,
        window_size_ms: float = 1000.0,
        sigma_ms: Optional[float] = None,
        baseline_window_ms: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the PSTH analyzer.

        Parameters
        ----------
        bin_size_ms : float
            Size of time bins in milliseconds
        window_size_ms : float
            Total window size in milliseconds (typically centered on events)
        sigma_ms : float or None
            Standard deviation of Gaussian smoothing kernel in milliseconds
            If None, no smoothing is applied
        baseline_window_ms : tuple of (float, float) or None
            Window for baseline calculation in ms, relative to event onset
            Example: (-500, -100) for baseline period 500-100ms before events
            If None, no baseline normalization is performed
        """
        self.bin_size_ms = bin_size_ms
        self.window_size_ms = window_size_ms
        self.sigma_ms = sigma_ms
        self.baseline_window_ms = baseline_window_ms

        # Create density estimator if smoothing is requested
        if self.sigma_ms is not None:
            self.density_estimator = SpikeDensityEstimator(
                bin_size_ms=bin_size_ms, sigma_ms=sigma_ms
            )

    def _calculate_event_window_samples(
        self, sampling_rate: float
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate event window in samples.

        Parameters
        ----------
        sampling_rate : float
            Sampling rate in Hz

        Returns
        -------
        bin_size_samples : int
            Bin size in samples
        time_bins : np.ndarray
            Time bins relative to event onset in seconds
        """
        # Calculate bin size in samples
        bin_size_samples = int(self.bin_size_ms * sampling_rate / 1000)

        # Calculate window edges in samples
        half_window_ms = self.window_size_ms / 2
        pre_samples = int((half_window_ms * sampling_rate) / 1000)
        post_samples = int((half_window_ms * sampling_rate) / 1000)

        # Create bin edges
        n_bins = int(self.window_size_ms / self.bin_size_ms)
        bin_edges_samples = np.linspace(-pre_samples, post_samples, n_bins + 1)
        bin_centers_samples = (bin_edges_samples[:-1] + bin_edges_samples[1:]) / 2
        time_bins = bin_centers_samples / sampling_rate

        return bin_size_samples, time_bins

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _bin_spikes_by_events(
        spike_times: np.ndarray,
        event_times: np.ndarray,
        pre_samples: int,
        post_samples: int,
        bin_edges: np.ndarray,
    ) -> np.ndarray:
        """
        Bin spikes around event times with Numba acceleration.

        Parameters
        ----------
        spike_times : np.ndarray
            Spike times in samples
        event_times : np.ndarray
            Event/stimulus times in samples
        pre_samples : int
            Number of samples before events to include
        post_samples : int
            Number of samples after events to include
        bin_edges : np.ndarray
            Bin edges relative to event times in samples

        Returns
        -------
        binned_spikes : np.ndarray
            Binned spike counts (shape: n_bins × n_events)
        """
        n_events = len(event_times)
        n_bins = len(bin_edges) - 1

        # Output array: bins × events
        binned_spikes = np.zeros((n_bins, n_events), dtype=np.int32)

        # Process each event in parallel
        for e in prange(n_events):
            event_time = event_times[e]
            window_start = event_time - pre_samples
            window_end = event_time + post_samples

            # Find spikes within this event window
            for spike_time in spike_times:
                if window_start <= spike_time < window_end:
                    # Convert to time relative to event
                    rel_time = spike_time - event_time

                    # Find bin index
                    bin_idx = np.searchsorted(bin_edges, rel_time) - 1
                    if 0 <= bin_idx < n_bins:
                        binned_spikes[bin_idx, e] += 1

        return binned_spikes

    def transform(
        self,
        sorter: SorterNode,
        events: Union[np.ndarray, EpocNode],
        unit_ids: Optional[List[int]] = None,
        pre_time_ms: Optional[float] = None,
        post_time_ms: Optional[float] = None,
        **kwargs,
    ) -> Dict:
        """
        Compute PSTH for the specified units and events.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        events : np.ndarray or EpocNode
            Event/stimulus times in seconds (np.ndarray) or
            EpocNode containing event data
        unit_ids : list of int or None
            Units to include. If None, use all units.
        pre_time_ms : float or None
            Time before events to include (ms)
            If None, uses half of window_size_ms
        post_time_ms : float or None
            Time after events to include (ms)
            If None, uses half of window_size_ms
        **kwargs : dict
            Additional keyword arguments (unused)

        Returns
        -------
        result : dict
            Dictionary with PSTH results:
            - 'time_bins': Time bins relative to event onset (seconds)
            - 'psth_counts': Raw spike counts per bin (shape: bins × units)
            - 'psth_rates': Firing rates (Hz) per bin (shape: bins × units)
            - 'psth_sem': Standard error of mean (shape: bins × units)
            - 'unit_ids': List of unit IDs included in analysis
            - 'event_count': Number of events used in analysis
            - If smoothing was applied:
              - 'psth_smoothed': Smoothed firing rates (Hz)
            - If baseline normalization was applied:
              - 'psth_normalized': Baseline-normalized rates
              - 'baseline_window': Baseline window used (seconds)
        """
        # Validate the sorter
        self._validate_sorter(sorter)
        sampling_rate = sorter.sampling_frequency

        # Get filtered unit IDs
        used_unit_ids = self._get_filtered_unit_ids(sorter, unit_ids)

        # Set pre/post times if not specified
        if pre_time_ms is None:
            pre_time_ms = self.window_size_ms / 2
        if post_time_ms is None:
            post_time_ms = self.window_size_ms / 2

        # Convert event times
        if isinstance(events, EpocNode):
            # Extract event times from EpocNode
            if not hasattr(events, "data") or events.data is None:
                events.load_data()
            event_times_s = events.data["onset"].to_numpy()
        else:
            # Use provided numpy array
            event_times_s = np.asarray(events)

        # Convert event times to samples
        event_times_samples = (event_times_s * sampling_rate).astype(np.int64)

        # Calculate window parameters
        pre_samples = int((pre_time_ms * sampling_rate) / 1000)
        post_samples = int((post_time_ms * sampling_rate) / 1000)

        # Create bin edges relative to event onset
        n_bins = int((pre_time_ms + post_time_ms) / self.bin_size_ms)
        bin_size_samples = int(self.bin_size_ms * sampling_rate / 1000)
        bin_edges_samples = np.linspace(-pre_samples, post_samples, n_bins + 1)
        bin_centers_samples = (bin_edges_samples[:-1] + bin_edges_samples[1:]) / 2
        time_bins = bin_centers_samples / sampling_rate

        # Initialize output arrays
        n_units = len(used_unit_ids)
        psth_counts = np.zeros((n_bins, n_units), dtype=np.int32)

        # Process each unit
        for i, unit_id in enumerate(used_unit_ids):
            # Get spike times for this unit (in samples)
            spike_train = sorter.get_unit_spike_train(unit_id)

            # Bin spikes around events
            unit_psth = self._bin_spikes_by_events(
                spike_train,
                event_times_samples,
                pre_samples,
                post_samples,
                bin_edges_samples,
            )

            # Sum across events (unit_psth shape: bins × events)
            psth_counts[:, i] = np.sum(unit_psth, axis=1)

        # Convert to firing rates (Hz)
        n_events = len(event_times_s)
        bin_size_s = self.bin_size_ms / 1000
        psth_rates = psth_counts / (n_events * bin_size_s)

        # Calculate standard error of the mean
        psth_sem = np.zeros_like(psth_rates)
        for i, unit_id in enumerate(used_unit_ids):
            # Get spike times for this unit (in samples)
            spike_train = sorter.get_unit_spike_train(unit_id)

            # Bin spikes around events
            unit_psth = self._bin_spikes_by_events(
                spike_train,
                event_times_samples,
                pre_samples,
                post_samples,
                bin_edges_samples,
            )

            # Calculate firing rates for each event (Hz)
            event_rates = unit_psth / bin_size_s

            # Calculate SEM
            psth_sem[:, i] = np.std(event_rates, axis=1) / np.sqrt(n_events)

        # Prepare result dictionary
        result = {
            "time_bins": time_bins,
            "psth_counts": psth_counts,
            "psth_rates": psth_rates,
            "psth_sem": psth_sem,
            "unit_ids": used_unit_ids,
            "event_count": n_events,
        }

        # Apply smoothing if requested
        if self.sigma_ms is not None:
            # Transpose to match density estimator input shape (time × units)
            smoothed = self.density_estimator.smooth(psth_counts.T).T

            # Scale by number of events and bin size
            result["psth_smoothed"] = smoothed / (n_events * bin_size_s)

        # Apply baseline normalization if requested
        if self.baseline_window_ms is not None:
            # Convert baseline window to indices
            baseline_start_s = self.baseline_window_ms[0] / 1000
            baseline_end_s = self.baseline_window_ms[1] / 1000
            baseline_idx = np.where(
                (time_bins >= baseline_start_s) & (time_bins <= baseline_end_s)
            )[0]

            # Skip if no bins in baseline window
            if len(baseline_idx) > 0:
                # Calculate baseline for each unit
                baseline_data = (
                    result["psth_smoothed"] if "psth_smoothed" in result else psth_rates
                )

                # Mean baseline rate for each unit
                baseline_means = np.mean(baseline_data[baseline_idx, :], axis=0)

                # Normalize (data - baseline) / baseline
                normalized = np.zeros_like(baseline_data)
                for i in range(normalized.shape[1]):
                    if baseline_means[i] > 0:
                        normalized[:, i] = (
                            baseline_data[:, i] - baseline_means[i]
                        ) / baseline_means[i]

                result["psth_normalized"] = normalized
                result["baseline_window"] = (baseline_start_s, baseline_end_s)

        return result
