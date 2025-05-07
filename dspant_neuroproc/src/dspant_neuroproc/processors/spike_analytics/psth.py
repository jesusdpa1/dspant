"""
Peristimulus Time Histogram (PSTH) implementation for neural data analysis.

This module provides functionality for generating PSTH from spike data aligned
to stimulus or event triggers, useful for analyzing neural responses to stimuli.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from dspant.core.internals import public_api
from dspant.nodes.epoch import EpocNode
from dspant.nodes.sorter import SorterNode

from .base import BaseSpikeTransform
from .density_estimation import SpikeDensityEstimator

try:
    from dspant_neuroproc._rs import (
        compute_psth,
        compute_psth_all,
        compute_psth_parallel,
        compute_raster_data,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    import numba


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
        # Ensure parameters are the correct types for Rust
        self.bin_size_ms = np.float32(bin_size_ms)
        self.window_size_ms = np.float32(window_size_ms)
        self.sigma_ms = np.float32(sigma_ms) if sigma_ms is not None else None

        # For baseline window, we don't convert to float32 yet since it's a tuple
        if baseline_window_ms is not None:
            self.baseline_window_ms = (
                float(baseline_window_ms[0]),
                float(baseline_window_ms[1]),
            )
        else:
            self.baseline_window_ms = None

        # Create density estimator if smoothing is requested and Rust is not available
        if not _HAS_RUST and self.sigma_ms is not None:
            self.density_estimator = SpikeDensityEstimator(
                bin_size_ms=bin_size_ms, sigma_ms=cast(float, sigma_ms)
            )

    @staticmethod
    @numba.jit(nopython=True, parallel=True, cache=True)
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
        for e in numba.prange(n_events):
            event_time = event_times[e]
            window_start = event_time - pre_samples
            window_end = event_time + post_samples

            # Find spikes within this window
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
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
            Additional keyword arguments

        Returns
        -------
        result : dict
            Dictionary with PSTH results including raster data with multiple trials
        """
        # Validate the sorter
        self._validate_sorter(sorter)
        sampling_rate = np.float32(sorter.sampling_frequency)

        # Get filtered unit IDs with explicit int casting
        if unit_ids is None:
            used_unit_ids = [int(uid) for uid in sorter.unit_ids]
        else:
            used_unit_ids = [int(uid) for uid in unit_ids]

        # Set pre/post times if not specified
        if pre_time_ms is None:
            pre_time_ms = np.float32(self.window_size_ms / 2.0)
        else:
            pre_time_ms = np.float32(pre_time_ms)

        if post_time_ms is None:
            post_time_ms = np.float32(self.window_size_ms / 2.0)
        else:
            post_time_ms = np.float32(post_time_ms)

        # Convert event times
        if isinstance(events, EpocNode):
            # Extract event times from EpocNode
            if not hasattr(events, "data") or events.data is None:
                events.load_data()
            event_times_s = events.data["onset"].to_numpy().astype(np.float32)
        else:
            # Use provided numpy array with explicit float32 casting
            event_times_s = np.asarray(events, dtype=np.float32)

        # Convert pre/post times to seconds with explicit float32 casting
        pre_time_s = np.float32(pre_time_ms / 1000.0)
        post_time_s = np.float32(post_time_ms / 1000.0)
        bin_size_s = np.float32(self.bin_size_ms / 1000.0)

        # Calculate smoothing sigma in bins (if applicable)
        smoothing_sigma = None
        if self.sigma_ms is not None:
            smoothing_sigma = np.float32(self.sigma_ms / self.bin_size_ms)

        if _HAS_RUST:
            # Prepare spike trains with proper casting for Rust
            spike_trains = []
            for unit_id in used_unit_ids:
                # Get spike train with explicit int32 casting
                spike_train = sorter.get_unit_spike_train(unit_id).astype(np.int32)

                # Convert to seconds with explicit float32 casting
                spike_times = (spike_train / sampling_rate).astype(np.float32)

                spike_trains.append(spike_times)

            # Use Rust implementation for multi-unit PSTH
            rust_result = compute_psth_all(
                spike_trains,
                event_times_s,
                used_unit_ids,
                pre_time_s,
                post_time_s,
                bin_size_s,
                smoothing_sigma,
            )

            # Get time bins
            time_bins = rust_result["time_bins"]

            # Extract individual unit PSTH data
            psth_data = rust_result["psth_data"]

            # Restructure data for compatibility with existing code
            # Ensure proper shape with a non-zero number of units
            if len(psth_data) > 0:
                # Each unit data has psth_counts as a 2D array (bins x events)
                unit_data = psth_data[0]
                n_bins = unit_data["psth_counts"].shape[0]
                n_events = unit_data["psth_counts"].shape[1]
                n_units = len(psth_data)

                # Initialize arrays
                psth_counts = np.zeros((n_bins, n_events, n_units), dtype=np.float32)

                # Fill with data from each unit
                for i, unit_data in enumerate(psth_data):
                    psth_counts[:, :, i] = unit_data["psth_counts"]

                # Calculate rates
                psth_rates = psth_counts / (
                    float(event_times_s.size) * float(bin_size_s)
                )

                # Calculate standard error of mean (SEM)
                psth_sem = np.zeros_like(psth_rates)
                for i in range(n_units):
                    counts = psth_counts[:, :, i]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        valid_trials = np.sum(counts > 0, axis=1)
                        valid_trials[valid_trials == 0] = 1  # Avoid division by zero
                        std_vals = np.std(counts, axis=1)
                        psth_sem[:, :, i] = np.reshape(
                            std_vals / np.sqrt(valid_trials), (-1, 1)
                        )

                    # Replace NaN/Inf with zeros
                    psth_sem[:, :, i] = np.nan_to_num(psth_sem[:, :, i])

                # Process smoothed data if available
                psth_smoothed = None
                if "psth_smoothed" in unit_data:
                    psth_smoothed = np.zeros_like(psth_counts)
                    for i, unit_data in enumerate(psth_data):
                        if "psth_smoothed" in unit_data:
                            psth_smoothed[:, :, i] = unit_data["psth_smoothed"]

                    # Convert to rates
                    psth_smoothed = psth_smoothed / (
                        float(event_times_s.size) * float(bin_size_s)
                    )
            else:
                # Handle the case with no units
                n_bins = len(time_bins)
                n_events = len(event_times_s)
                psth_counts = np.zeros((n_bins, n_events, 0), dtype=np.float32)
                psth_rates = np.zeros_like(psth_counts)
                psth_sem = np.zeros_like(psth_counts)
                psth_smoothed = None

            # Compute raster data for each unit
            raster_data = []
            for i, unit_id in enumerate(used_unit_ids):
                spike_times = spike_trains[i]

                # Skip if no spikes
                if len(spike_times) == 0:
                    raster_data.append(
                        {
                            "unit_id": unit_id,
                            "trials": np.array([], dtype=np.int32),
                            "spike_times": np.array([], dtype=np.float32),
                        }
                    )
                    continue

                # Use Rust implementation for raster data
                unit_raster = compute_raster_data(
                    spike_times,
                    event_times_s,
                    pre_time_s,
                    post_time_s,
                )

                raster_data.append(
                    {
                        "unit_id": unit_id,
                        "trials": unit_raster["trials"],
                        "spike_times": unit_raster["spike_times"],
                    }
                )

            # Prepare result dictionary
            result = {
                "time_bins": time_bins,
                "psth_counts": psth_counts,
                "psth_rates": psth_rates,
                "psth_sem": psth_sem,
                "unit_ids": used_unit_ids,
                "event_count": len(event_times_s),
                "raster_data": raster_data,
            }

            # Add smoothed data if available
            if psth_smoothed is not None:
                result["psth_smoothed"] = psth_smoothed

            # Apply baseline normalization if requested
            if self.baseline_window_ms is not None:
                # Convert baseline window to indices with explicit float32 casting
                baseline_start_s = np.float32(self.baseline_window_ms[0] / 1000.0)
                baseline_end_s = np.float32(self.baseline_window_ms[1] / 1000.0)
                baseline_idx = np.where(
                    (time_bins >= baseline_start_s) & (time_bins <= baseline_end_s)
                )[0]

                # Skip if no bins in baseline window
                if len(baseline_idx) > 0:
                    # Calculate baseline for each unit
                    baseline_data = (
                        result["psth_smoothed"]
                        if "psth_smoothed" in result
                        else result["psth_rates"]
                    )

                    # Mean baseline rate for each unit
                    baseline_means = np.mean(baseline_data[baseline_idx, :, :], axis=0)

                    # Normalize (data - baseline) / baseline
                    normalized = np.zeros_like(baseline_data)
                    for i in range(normalized.shape[2]):  # For each unit
                        for e in range(normalized.shape[1]):  # For each event
                            if baseline_means[e, i] > 0:
                                normalized[:, e, i] = (
                                    baseline_data[:, e, i] - baseline_means[e, i]
                                ) / baseline_means[e, i]

                    result["psth_normalized"] = normalized
                    result["baseline_window"] = (
                        float(baseline_start_s),
                        float(baseline_end_s),
                    )

            return result
        else:
            # Python implementation with improved performance
            # Convert event times to samples
            event_times_samples = (event_times_s * sampling_rate).astype(np.int32)

            # Calculate window parameters
            pre_samples = int((pre_time_ms * sampling_rate) / 1000.0)
            post_samples = int((post_time_ms * sampling_rate) / 1000.0)

            # Create bin edges relative to event onset
            n_bins = int((pre_time_ms + post_time_ms) / self.bin_size_ms)
            bin_size_samples = int(self.bin_size_ms * sampling_rate / 1000.0)
            bin_edges_samples = np.linspace(
                -pre_samples, post_samples, n_bins + 1, dtype=np.float32
            )
            bin_centers_samples = (bin_edges_samples[:-1] + bin_edges_samples[1:]) / 2.0
            time_bins = bin_centers_samples / sampling_rate

            # Initialize output arrays
            n_units = len(used_unit_ids)
            n_events = len(event_times_s)

            # Create 3D array: bins × events × units
            psth_counts = np.zeros((n_bins, n_events, n_units), dtype=np.float32)

            # Storage for raster data
            raster_data = []

            # Process each unit
            for i, unit_id in enumerate(used_unit_ids):
                # Get spike times for this unit (in samples)
                spike_train = sorter.get_unit_spike_train(unit_id).astype(np.int32)

                # Storage for this unit's raster data
                unit_spike_times = []
                unit_trial_indices = []

                # Use optimized Numba function to bin spikes
                unit_counts = self._bin_spikes_by_events(
                    spike_train,
                    event_times_samples,
                    pre_samples,
                    post_samples,
                    bin_edges_samples,
                )

                # Store in the 3D array
                psth_counts[:, :, i] = unit_counts

                # Process raster data - need to iterate to extract spike times
                for trial_idx, event_time in enumerate(event_times_samples):
                    # Calculate window boundaries
                    window_start = event_time - pre_samples
                    window_end = event_time + post_samples

                    # Find spikes within this window
                    mask = (spike_train >= window_start) & (spike_train < window_end)
                    window_spikes = spike_train[mask]

                    if len(window_spikes) > 0:
                        # Convert to seconds relative to event onset
                        rel_times_sec = (window_spikes - event_time) / sampling_rate

                        # Store spikes for this trial
                        unit_spike_times.extend(rel_times_sec.tolist())
                        unit_trial_indices.extend([trial_idx] * len(rel_times_sec))

                # Store raster data for this unit
                raster_data.append(
                    {
                        "unit_id": unit_id,
                        "trials": np.array(unit_trial_indices, dtype=np.int32),
                        "spike_times": np.array(unit_spike_times, dtype=np.float32),
                    }
                )

            # Convert to firing rates (Hz)
            bin_size_s = float(self.bin_size_ms / 1000.0)
            psth_rates = psth_counts / (n_events * bin_size_s)

            # Calculate standard error of the mean across events/trials
            psth_sem = np.zeros_like(psth_rates)
            for i in range(n_units):
                counts = psth_counts[:, :, i]
                with np.errstate(divide="ignore", invalid="ignore"):
                    valid_trials = np.maximum(
                        np.sum(counts > 0, axis=1), 1
                    )  # Avoid division by zero
                    std_vals = np.std(counts, axis=1)
                    psth_sem[:, :, i] = np.reshape(
                        std_vals / np.sqrt(valid_trials), (-1, 1)
                    )

                # Replace NaN/Inf with zeros
                psth_sem[:, :, i] = np.nan_to_num(psth_sem[:, :, i])

            # Prepare result dictionary
            result = {
                "time_bins": time_bins,
                "psth_counts": psth_counts,
                "psth_rates": psth_rates,
                "psth_sem": psth_sem,
                "unit_ids": used_unit_ids,
                "event_count": n_events,
                "raster_data": raster_data,
            }

            # Apply smoothing if requested
            if self.sigma_ms is not None and hasattr(self, "density_estimator"):
                # Initialize smoothed array
                psth_smoothed = np.zeros_like(psth_counts)

                # Apply smoothing for each unit and trial
                for i in range(n_units):
                    for e in range(n_events):
                        # Extract this trial's spike counts
                        trial_counts = psth_counts[:, e, i]

                        # Apply smoothing using density estimator
                        smoothed = self.density_estimator.smooth(
                            trial_counts.reshape(-1, 1)
                        )

                        # Store result
                        psth_smoothed[:, e, i] = smoothed.flatten()

                # Scale by number of events and bin size to get rates
                result["psth_smoothed"] = psth_smoothed / (n_events * bin_size_s)

            # Apply baseline normalization if requested
            if self.baseline_window_ms is not None:
                # Convert baseline window to indices
                baseline_start_s = float(self.baseline_window_ms[0] / 1000.0)
                baseline_end_s = float(self.baseline_window_ms[1] / 1000.0)
                baseline_idx = np.where(
                    (time_bins >= baseline_start_s) & (time_bins <= baseline_end_s)
                )[0]

                # Skip if no bins in baseline window
                if len(baseline_idx) > 0:
                    # Calculate baseline for each unit
                    baseline_data = (
                        result["psth_smoothed"]
                        if "psth_smoothed" in result
                        else result["psth_rates"]
                    )

                    # Mean baseline rate for each unit
                    baseline_means = np.mean(baseline_data[baseline_idx, :, :], axis=0)

                    # Normalize (data - baseline) / baseline
                    normalized = np.zeros_like(baseline_data)
                    for i in range(normalized.shape[2]):
                        for e in range(normalized.shape[1]):
                            if baseline_means[e, i] > 0:
                                normalized[:, e, i] = (
                                    baseline_data[:, e, i] - baseline_means[e, i]
                                ) / baseline_means[e, i]

                    result["psth_normalized"] = normalized
                    result["baseline_window"] = (baseline_start_s, baseline_end_s)

            return result

    def compute_single_unit_psth(
        self,
        sorter: SorterNode,
        unit_id: int,
        events: Union[np.ndarray, EpocNode],
        pre_time_ms: Optional[float] = None,
        post_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute PSTH for a single unit (optimized version).

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        unit_id : int
            Unit ID to analyze
        events : np.ndarray or EpocNode
            Event/stimulus times in seconds
        pre_time_ms : float or None
            Time before events to include (ms)
        post_time_ms : float or None
            Time after events to include (ms)

        Returns
        -------
        Dict
            PSTH results for this unit
        """
        # Validate unit ID
        unit_id = int(unit_id)
        if unit_id not in sorter.unit_ids:
            raise ValueError(f"Unit ID {unit_id} not found in sorter")

        # Convert parameters
        sampling_rate = np.float32(sorter.sampling_frequency)

        # Set pre/post times if not specified
        if pre_time_ms is None:
            pre_time_ms = np.float32(self.window_size_ms / 2.0)
        else:
            pre_time_ms = np.float32(pre_time_ms)

        if post_time_ms is None:
            post_time_ms = np.float32(self.window_size_ms / 2.0)
        else:
            post_time_ms = np.float32(post_time_ms)

        # Convert to seconds
        pre_time_s = np.float32(pre_time_ms / 1000.0)
        post_time_s = np.float32(post_time_ms / 1000.0)
        bin_size_s = np.float32(self.bin_size_ms / 1000.0)

        # Get spike times
        spike_train = sorter.get_unit_spike_train(unit_id).astype(np.int32)
        spike_times_s = (spike_train / sampling_rate).astype(np.float32)

        # Convert event times
        if isinstance(events, EpocNode):
            if not hasattr(events, "data") or events.data is None:
                events.load_data()
            event_times_s = events.data["onset"].to_numpy().astype(np.float32)
        else:
            event_times_s = np.asarray(events, dtype=np.float32)

        # Calculate smoothing sigma in bins (if applicable)
        smoothing_sigma = None
        if self.sigma_ms is not None:
            smoothing_sigma = np.float32(self.sigma_ms / self.bin_size_ms)

        if _HAS_RUST:
            # Use Rust implementation with smoothing if requested
            if smoothing_sigma is not None:
                counts, time_bins, smoothed = compute_psth_parallel(
                    spike_times_s,
                    event_times_s,
                    pre_time_s,
                    post_time_s,
                    bin_size_s,
                    smoothing_sigma,
                )
            else:
                # Use basic version without smoothing
                counts, time_bins = compute_psth(
                    spike_times_s,
                    event_times_s,
                    pre_time_s,
                    post_time_s,
                    bin_size_s,
                )
                smoothed = None

            # Calculate raster data
            raster = compute_raster_data(
                spike_times_s,
                event_times_s,
                pre_time_s,
                post_time_s,
            )

            # Convert to rates
            n_events = len(event_times_s)
            rates = counts / (n_events * bin_size_s)

            # Calculate SEM
            with np.errstate(divide="ignore", invalid="ignore"):
                valid_trials = np.maximum(np.sum(counts > 0, axis=1), 1)
                std_vals = np.std(counts, axis=1)
                sem = np.reshape(std_vals / np.sqrt(valid_trials), (-1, 1))

            # Prepare result
            result = {
                "unit_id": unit_id,
                "time_bins": time_bins,
                "psth_counts": counts,
                "psth_rates": rates,
                "psth_sem": sem,
                "event_count": n_events,
                "raster_data": {
                    "unit_id": unit_id,
                    "trials": raster["trials"],
                    "spike_times": raster["spike_times"],
                },
            }

            # Add smoothed data if available
            if smoothed is not None:
                result["psth_smoothed"] = smoothed / (n_events * bin_size_s)

            return result
        else:
            # Use Python implementation for a single unit
            # Will be implemented if needed - for now, just use transform
            return self.transform(
                sorter,
                events,
                unit_ids=[unit_id],
                pre_time_ms=pre_time_ms,
                post_time_ms=post_time_ms,
            )
