"""
Peristimulus Time Histogram (PSTH) implementation for neural data analysis.

This module provides functionality for generating PSTH from spike data aligned
to stimulus or event triggers, useful for analyzing neural responses to stimuli.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numba
import numpy as np

from dspant.core.internals import public_api
from dspant.nodes.epoch import EpocNode
from dspant.nodes.sorter import SorterNode

from .base import BaseSpikeTransform

try:
    from dspant_neuroproc._rs import (
        bin_spikes_by_events,
        compute_psth,
        compute_psth_all,
        compute_psth_parallel,
        compute_raster_data,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


# Define Numba-accelerated functions outside the class for efficiency
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

                # Find bin index using binary search
                bin_idx = np.searchsorted(bin_edges, rel_time) - 1
                if 0 <= bin_idx < n_bins:
                    binned_spikes[bin_idx, e] += 1

    return binned_spikes


@numba.jit(nopython=True, cache=True)
def _gaussian_kernel(sigma: float, kernel_size: int) -> np.ndarray:
    """
    Create a Gaussian kernel for smoothing.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian in bins
    kernel_size : int
        Size of the kernel window

    Returns
    -------
    kernel : np.ndarray
        Normalized Gaussian kernel
    """
    # Ensure kernel size is odd for symmetric kernel
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create kernel
    kernel = np.zeros(kernel_size, dtype=np.float32)
    center = kernel_size // 2

    # Calculate Gaussian values
    for i in range(kernel_size):
        x = float(i - center)
        kernel[i] = np.exp(-(x * x) / (2 * sigma * sigma))

    # Normalize kernel
    return kernel / np.sum(kernel)


@numba.jit(nopython=True, parallel=True, cache=True)
def _apply_gaussian_smoothing(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to binned spike data.

    Parameters
    ----------
    data : np.ndarray
        Binned spike data (shape: n_bins × n_events)
    sigma : float
        Standard deviation of Gaussian kernel in bins

    Returns
    -------
    smoothed : np.ndarray
        Smoothed data
    """
    if sigma <= 0.0:
        return data.copy()

    # Create Gaussian kernel
    kernel_radius = int(np.ceil(sigma * 3.0))
    kernel_size = kernel_radius * 2 + 1
    kernel = np.zeros(kernel_size, dtype=np.float32)

    # Compute kernel values
    scale = -0.5 / (sigma * sigma)
    for i in range(kernel_size):
        x = float(i) - (kernel_size - 1) / 2.0
        kernel[i] = np.exp(x * x * scale)

    # Normalize kernel
    kernel = kernel / np.sum(kernel)

    # Get dimensions
    n_bins, n_events = data.shape
    smoothed = np.zeros_like(data)

    # Process each event in parallel
    for e in numba.prange(n_events):
        data_col = data[:, e]

        # Convolve with kernel
        for i in range(n_bins):
            weighted_sum = 0.0

            for k in range(kernel_size):
                # Index in data with mirrored boundaries
                idx = i + k - (kernel_size // 2)

                # Apply boundary conditions (mirror)
                if idx < 0:
                    idx = -idx
                elif idx >= n_bins:
                    idx = 2 * n_bins - idx - 2

                # Add weighted value
                if 0 <= idx < n_bins:
                    weighted_sum += data_col[idx] * kernel[k]

            smoothed[i, e] = weighted_sum

    return smoothed


@numba.jit(nopython=True, parallel=True, cache=True)
def _compute_raster_data(
    spike_times: np.ndarray,
    event_times: np.ndarray,
    pre_time: float,
    post_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute raster data for visualization.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike times in seconds
    event_times : np.ndarray
        Event times in seconds
    pre_time : float
        Time before events in seconds
    post_time : float
        Time after events in seconds

    Returns
    -------
    spike_times : np.ndarray
        Relative spike times to event onset
    trial_indices : np.ndarray
        Trial indices for each spike
    """
    # Estimate maximum number of spikes (worst case)
    max_spikes = len(spike_times) * len(event_times)
    rel_times = np.zeros(max_spikes, dtype=np.float32)
    trial_indices = np.zeros(max_spikes, dtype=np.int32)

    # Counter for actual spikes
    count = 0

    # Process each event
    for e_idx in numba.prange(len(event_times)):
        event_time = event_times[e_idx]
        window_start = event_time - pre_time
        window_end = event_time + post_time

        # Find spikes within window
        for spike_time in spike_times:
            if window_start <= spike_time < window_end:
                # Store relative time and trial index
                if count < max_spikes:
                    rel_times[count] = spike_time - event_time
                    trial_indices[count] = e_idx
                    count += 1

    # Return only used portion of arrays
    return rel_times[:count], trial_indices[:count]


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
        # Store parameters as float32 for compatibility with Rust
        self.bin_size_ms = np.float32(bin_size_ms)
        self.window_size_ms = np.float32(window_size_ms)
        self.sigma_ms = np.float32(sigma_ms) if sigma_ms is not None else None

        # Store baseline window
        if baseline_window_ms is not None:
            self.baseline_window_ms = (
                float(baseline_window_ms[0]),
                float(baseline_window_ms[1]),
            )
        else:
            self.baseline_window_ms = None

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

        # Get filtered unit IDs
        if unit_ids is None:
            used_unit_ids = [int(uid) for uid in sorter.unit_ids]
        else:
            used_unit_ids = [
                int(uid) for uid in unit_ids if int(uid) in sorter.unit_ids
            ]

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

        # Convert to samples and seconds with explicit casting
        pre_time_s = np.float32(pre_time_ms / 1000.0)
        post_time_s = np.float32(post_time_ms / 1000.0)
        bin_size_s = np.float32(self.bin_size_ms / 1000.0)

        # Calculate window parameters
        pre_samples = int(pre_time_s * sampling_rate)
        post_samples = int(post_time_s * sampling_rate)

        # Calculate smoothing sigma in bins (if applicable)
        smoothing_sigma = None
        if self.sigma_ms is not None:
            smoothing_sigma = np.float32(self.sigma_ms / self.bin_size_ms)

        if _HAS_RUST:
            # Use Rust implementation for better performance
            # Prepare spike trains for Rust
            spike_trains = []
            for unit_id in used_unit_ids:
                spike_train = sorter.get_unit_spike_train(unit_id).astype(np.int32)
                spike_times = (spike_train / sampling_rate).astype(np.float32)
                spike_trains.append(spike_times)

            # Use Rust implementation for PSTH computation
            result = compute_psth_all(
                spike_trains,
                event_times_s,
                used_unit_ids,
                pre_time_s,
                post_time_s,
                bin_size_s,
                smoothing_sigma,
            )

            # Extract result data
            time_bins = result["time_bins"]
            psth_data = result["psth_data"]
            n_events = result["event_count"]

            # Process data from Rust implementation
            if len(psth_data) > 0:
                # Extract dimensions from the first unit data
                unit_data = psth_data[0]
                n_bins = unit_data["psth_counts"].shape[0]
                n_events = unit_data["psth_counts"].shape[1]
                n_units = len(psth_data)

                # Initialize arrays
                psth_counts = np.zeros((n_bins, n_units), dtype=np.float32)

                # Fill data from each unit
                for i, unit_data in enumerate(psth_data):
                    # Average across events
                    psth_counts[:, i] = np.mean(unit_data["psth_counts"], axis=1)

                # Calculate rates and SEM
                bin_size_s = float(self.bin_size_ms / 1000.0)
                psth_rates = psth_counts / bin_size_s

                # Get smoothed data if available
                psth_smoothed = None
                if "psth_smoothed" in unit_data:
                    psth_smoothed = np.zeros_like(psth_counts)
                    for i, unit_data in enumerate(psth_data):
                        if "psth_smoothed" in unit_data:
                            psth_smoothed[:, i] = np.mean(
                                unit_data["psth_smoothed"], axis=1
                            )

                    # Convert to rates
                    psth_smoothed = psth_smoothed / bin_size_s
            else:
                # Handle case with no units
                n_bins = len(time_bins)
                psth_counts = np.zeros((n_bins, 0), dtype=np.float32)
                psth_rates = np.zeros_like(psth_counts)
                psth_smoothed = None

            # Compute raster data for each unit
            raster_data = []
            for i, unit_id in enumerate(used_unit_ids):
                # Get spike train
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

                # Compute raster data
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

        else:
            # Pure Python implementation
            # Create bin edges relative to event onset
            n_bins = int((pre_time_ms + post_time_ms) / self.bin_size_ms)
            bin_edges_samples = np.linspace(
                -pre_samples, post_samples, n_bins + 1, dtype=np.float32
            )
            time_bins = np.linspace(-pre_time_s, post_time_s, n_bins, dtype=np.float32)

            # Convert event times to samples
            event_times_samples = (event_times_s * sampling_rate).astype(np.int32)

            # Initialize output arrays
            n_units = len(used_unit_ids)
            n_events = len(event_times_s)
            psth_counts = np.zeros((n_bins, n_units), dtype=np.float32)

            # Storage for raster data
            raster_data = []

            # Process each unit
            for i, unit_id in enumerate(used_unit_ids):
                # Get spike times for this unit (in samples)
                spike_train = sorter.get_unit_spike_train(unit_id).astype(np.int32)

                # Use optimized function to bin spikes (call the standalone function)
                unit_counts = _bin_spikes_by_events(
                    spike_train,
                    event_times_samples,
                    pre_samples,
                    post_samples,
                    bin_edges_samples,
                )

                # Average across events
                psth_counts[:, i] = np.mean(unit_counts, axis=1)

                # Compute raster data for this unit
                spike_times_s = (spike_train / sampling_rate).astype(np.float32)
                rel_times, trial_indices = _compute_raster_data(
                    spike_times_s,
                    event_times_s,
                    pre_time_s,
                    post_time_s,
                )

                raster_data.append(
                    {
                        "unit_id": unit_id,
                        "trials": trial_indices,
                        "spike_times": rel_times,
                    }
                )

            # Convert to firing rates (Hz)
            bin_size_s = float(self.bin_size_ms / 1000.0)
            psth_rates = psth_counts / bin_size_s

            # Apply smoothing if requested
            psth_smoothed = None
            if smoothing_sigma is not None:
                psth_smoothed = np.zeros_like(psth_counts)

                # Process each unit
                for i in range(n_units):
                    # Average counts across events first for stable smoothing
                    unit_counts = psth_counts[:, i].reshape(-1, 1)

                    # Apply Gaussian smoothing (call the standalone function)
                    smoothed = _apply_gaussian_smoothing(unit_counts, smoothing_sigma)

                    # Store result
                    psth_smoothed[:, i] = smoothed.flatten()

                # Convert to rates
                psth_smoothed = psth_smoothed / bin_size_s

        # Prepare result dictionary
        result = {
            "time_bins": time_bins,
            "psth_counts": psth_counts,
            "psth_rates": psth_rates,
            "unit_ids": used_unit_ids,
            "event_count": len(event_times_s),
            "raster_data": raster_data,
        }

        # Add smoothed data if available
        if psth_smoothed is not None:
            result["psth_smoothed"] = psth_smoothed

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
                    psth_smoothed if psth_smoothed is not None else psth_rates
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
        # Just call transform with a single unit ID
        return self.transform(
            sorter,
            events,
            unit_ids=[int(unit_id)],
            pre_time_ms=pre_time_ms,
            post_time_ms=post_time_ms,
        )
