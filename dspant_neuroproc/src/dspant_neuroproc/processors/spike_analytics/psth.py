"""
Peristimulus Time Histogram (PSTH) implementation for neural data analysis.

This module provides functionality for generating PSTH from spike data aligned
to stimulus or event triggers, useful for analyzing neural responses to stimuli.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
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
            Additional keyword arguments

        Returns
        -------
        result : dict
            Dictionary with PSTH results including raster data with multiple trials
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
        n_events = len(event_times_s)
        psth_counts = np.zeros((n_bins, n_units), dtype=np.int32)

        # Storage for raster data
        raster_data = []

        # Process each unit
        for i, unit_id in enumerate(used_unit_ids):
            # Get spike times for this unit (in samples)
            spike_train = sorter.get_unit_spike_train(unit_id)

            # Storage for this unit's raster data
            unit_spike_times = []
            unit_trial_indices = []

            # Process each event/trial
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

                    # Bin spikes for PSTH calculation
                    for rel_time in window_spikes - event_time:
                        bin_idx = np.searchsorted(bin_edges_samples, rel_time) - 1
                        if 0 <= bin_idx < n_bins:
                            psth_counts[bin_idx, i] += 1

            # Store raster data for this unit
            raster_data.append(
                {
                    "unit_id": unit_id,
                    "trials": np.array(unit_trial_indices),
                    "spike_times": np.array(unit_spike_times),
                }
            )

        # The rest of the function remains the same as in the previous implementation
        # (PSTH rate calculation, SEM calculation, smoothing, baseline normalization)

        # Convert to firing rates (Hz)
        bin_size_s = self.bin_size_ms / 1000
        psth_rates = psth_counts / (n_events * bin_size_s)

        # Calculate standard error of the mean across events/trials
        psth_sem = np.zeros_like(psth_rates)
        for i, unit_id in enumerate(used_unit_ids):
            # Skip if no spikes for this unit
            if np.sum(psth_counts[:, i]) == 0:
                continue

            # Get spike times for this unit
            spike_train = sorter.get_unit_spike_train(unit_id)

            # Calculate rates for each event/trial
            event_rates = np.zeros((n_bins, n_events), dtype=np.float32)

            for e, event_time in enumerate(event_times_samples):
                window_start = event_time - pre_samples
                window_end = event_time + post_samples

                # Find spikes within this window
                window_mask = (spike_train >= window_start) & (spike_train < window_end)
                window_spikes = spike_train[window_mask]

                # Bin spikes for this event
                if len(window_spikes) > 0:
                    for rel_time in window_spikes - event_time:
                        bin_idx = np.searchsorted(bin_edges_samples, rel_time) - 1
                        if 0 <= bin_idx < n_bins:
                            event_rates[bin_idx, e] += 1

            # Convert counts to rates
            event_rates = event_rates / bin_size_s

            # Calculate SEM (avoiding division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                valid_trials = np.sum(event_rates > 0, axis=1)
                valid_trials[valid_trials == 0] = 1  # Avoid division by zero
                psth_sem[:, i] = np.std(event_rates, axis=1) / np.sqrt(valid_trials)

            # Handle any NaN values
            psth_sem[:, i] = np.nan_to_num(psth_sem[:, i])

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


def plot_psth_with_raster(
    psth_result: Dict,
    unit_index: int = 0,
    figsize: Tuple[float, float] = (10, 8),
    raster_color: str = "#2D3142",
    raster_alpha: float = 0.7,
    psth_color: str = "orange",
    show_sem: bool = True,
    sem_alpha: float = 0.3,
    show_smoothed: bool = True,
    marker_size: float = 4,
    marker_type: str = "|",
    title: Optional[str] = None,
    show_grid: bool = True,
    normalize_psth: bool = False,
    ylim_raster: Optional[Tuple[float, float]] = None,
    ylim_psth: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    raster_height_ratio: float = 2.0,
):
    """
    Create a combined raster plot and PSTH aligned to event onsets.

    Parameters
    ----------
    psth_result : dict
        Output from PSTHAnalyzer.transform()
    unit_index : int
        Index of the unit to display (if multiple units in result)
    figsize : tuple
        Figure size as (width, height)
    raster_color : str
        Color for raster plot markers
    raster_alpha : float
        Alpha transparency for raster markers
    psth_color : str
        Color for PSTH line
    show_sem : bool
        Whether to show standard error of mean
    sem_alpha : float
        Alpha transparency for SEM shading
    show_smoothed : bool
        Whether to show smoothed PSTH if available
    marker_size : float
        Size of raster markers
    marker_type : str
        Type of marker for raster plot
    title : str or None
        Title for the figure
    show_grid : bool
        Whether to show grid lines
    normalize_psth : bool
        Whether to show normalized PSTH (if available)
    ylim_raster : tuple or None
        Y-axis limits for raster plot
    ylim_psth : tuple or None
        Y-axis limits for PSTH plot
    xlim : tuple or None
        X-axis limits for both plots
    raster_height_ratio : float
        Ratio of raster height to PSTH height (default: 2.0)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing both plots
    axes : list of matplotlib.axes.Axes
        List containing [raster_ax, psth_ax]
    """
    # Check if raster data is available
    if "raster_data" not in psth_result:
        raise ValueError("PSTH result does not contain raster data.")

    # Get unit ID
    unit_ids = psth_result["unit_ids"]
    if unit_index >= len(unit_ids):
        raise ValueError(
            f"Unit index {unit_index} out of range (max: {len(unit_ids) - 1})"
        )

    unit_id = unit_ids[unit_index]

    # Create figure with two subplots (shared x-axis)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [raster_height_ratio, 1]},
    )
    raster_ax, psth_ax = axes

    # Get time bins
    time_bins = psth_result["time_bins"]

    # Get raster data for this unit
    raster_data = psth_result["raster_data"][unit_index]
    spike_times = raster_data["spike_times"]
    trial_indices = raster_data["trials"]

    # Check if we have events
    n_events = psth_result["event_count"]

    # Plot raster
    if len(spike_times) > 0:
        raster_ax.scatter(
            spike_times,
            trial_indices,
            marker=marker_type,
            s=marker_size,
            color=raster_color,
            alpha=raster_alpha,
            linewidths=marker_size / 4 if marker_type != "|" else 1,
        )

    # Set raster labels
    raster_ax.set_ylabel("Trial")
    if title:
        raster_ax.set_title(title)
    else:
        raster_ax.set_title(f"Unit {unit_id} - {n_events} trials")

    # Set y-limits for raster if provided
    if ylim_raster is not None:
        raster_ax.set_ylim(ylim_raster)
    else:
        # Set y-limits to show all trials, even if some have no spikes
        raster_ax.set_ylim(-0.5, n_events - 0.5)

    # Plot PSTH
    if normalize_psth and "psth_normalized" in psth_result:
        # Use normalized PSTH if requested and available
        psth_data = psth_result["psth_normalized"][:, unit_index]
        y_label = "Normalized firing rate"
    elif show_smoothed and "psth_smoothed" in psth_result:
        # Use smoothed PSTH if requested and available
        psth_data = psth_result["psth_smoothed"][:, unit_index]
        y_label = "Firing rate (Hz)"
    else:
        # Use binned PSTH
        psth_data = psth_result["psth_rates"][:, unit_index]
        y_label = "Firing rate (Hz)"

    # Plot PSTH line
    psth_ax.plot(time_bins, psth_data, color=psth_color, linewidth=2)

    # Plot SEM if requested
    if show_sem and not normalize_psth:  # SEM not applicable to normalized data
        sem = psth_result["psth_sem"][:, unit_index]

        # Make sure SEM values are valid
        valid_sem = np.isfinite(sem)
        if np.any(valid_sem):
            psth_ax.fill_between(
                time_bins[valid_sem],
                psth_data[valid_sem] - sem[valid_sem],
                psth_data[valid_sem] + sem[valid_sem],
                color=psth_color,
                alpha=sem_alpha,
            )

    # Add baseline window if available
    if "baseline_window" in psth_result:
        start, end = psth_result["baseline_window"]
        # Add baseline shading to both plots
        for ax in axes:
            ax.axvspan(start, end, color="gray", alpha=0.2)
            # Add text label if there's room
            if end - start > 0.05:
                ax.text(
                    (start + end) / 2,
                    ax.get_ylim()[1] * 0.9,
                    "baseline",
                    ha="center",
                    va="top",
                    fontsize=8,
                    alpha=0.7,
                )

    # Set PSTH labels
    psth_ax.set_xlabel("Time from event onset (s)")
    psth_ax.set_ylabel(y_label)

    # Set y-limits for PSTH if provided
    if ylim_psth is not None:
        psth_ax.set_ylim(ylim_psth)

    # Set x-limits if provided
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(xlim)

    # Add vertical line at event onset (time=0)
    for ax in axes:
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)
        if show_grid:
            ax.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()

    return fig, axes
