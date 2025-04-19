from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit, prange

from dspant.nodes.sorter import SorterNode


class SpikeDensityEstimator:
    """
    Converts discrete spike events into continuous firing rate estimates
    using binning and Gaussian smoothing.
    """

    def __init__(
        self,
        bin_size_ms: float = 10.0,
        sigma_ms: float = 20.0,
        kernel_window_ms: Optional[float] = None,
    ):
        """
        Initialize the spike density estimator.

        Parameters
        ----------
        bin_size_ms : float
            Size of time bins in milliseconds
        sigma_ms : float
            Standard deviation of Gaussian smoothing kernel in milliseconds
        kernel_window_ms : float or None
            Width of kernel window in milliseconds. If None, uses 5*sigma_ms.
        """
        self.bin_size_ms = bin_size_ms
        self.sigma_ms = sigma_ms
        self.kernel_window_ms = (
            kernel_window_ms if kernel_window_ms is not None else 5 * sigma_ms
        )

    def bin_spikes(
        self,
        sorter: SorterNode,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        unit_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Bin spike times into a histogram.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        start_time_s : float
            Start time for analysis in seconds
        end_time_s : float or None
            End time for analysis in seconds. If None, use the end of the recording.
        unit_ids : list of int or None
            Units to include. If None, use all units.

        Returns
        -------
        time_bins : np.ndarray
            Time bin centers in seconds
        binned_spikes : np.ndarray
            Spike counts per bin (shape: time_bins × units)
        used_unit_ids : list of int
            Unit IDs included in the analysis
        """
        if sorter.sampling_frequency is None:
            raise ValueError("Sorter node must have sampling frequency set")

        sampling_rate = sorter.sampling_frequency

        # Convert times to samples
        start_frame = int(start_time_s * sampling_rate)
        if end_time_s is None:
            # Use the last spike time as the end time
            end_frame = int(np.max(sorter.spike_times))
        else:
            end_frame = int(end_time_s * sampling_rate)

        # Calculate bin edges and centers
        bin_size_samples = int(self.bin_size_ms * sampling_rate / 1000)
        n_bins = (end_frame - start_frame) // bin_size_samples + 1
        bin_edges_samples = np.arange(
            start_frame, start_frame + (n_bins + 1) * bin_size_samples, bin_size_samples
        )
        bin_centers_samples = bin_edges_samples[:-1] + bin_size_samples // 2

        # Use requested units or all units
        if unit_ids is None:
            unit_ids = sorter.unit_ids
        used_unit_ids = [u for u in unit_ids if u in sorter.unit_ids]

        # Create output array
        binned_spikes = np.zeros(
            (len(bin_centers_samples), len(used_unit_ids)), dtype=np.int32
        )

        # Bin spikes for each unit
        for i, unit_id in enumerate(used_unit_ids):
            spike_train = sorter.get_unit_spike_train(
                unit_id, start_frame=start_frame, end_frame=end_frame
            )
            if len(spike_train) > 0:
                # Subtract start_frame to align with bins
                aligned_spikes = spike_train - start_frame
                bin_indices = np.digitize(aligned_spikes, bin_edges_samples) - 1
                # Count spikes in each bin
                for idx in bin_indices:
                    if 0 <= idx < len(bin_centers_samples):
                        binned_spikes[idx, i] += 1

        # Convert back to seconds for output
        time_bins = bin_centers_samples / sampling_rate

        return time_bins, binned_spikes, used_unit_ids

    @staticmethod
    @jit(nopython=True, cache=True)
    def _gaussian_kernel(sigma: float, window_size: int) -> np.ndarray:
        """
        Create a normalized Gaussian kernel.

        Parameters
        ----------
        sigma : float
            Standard deviation of Gaussian (in bins)
        window_size : int
            Width of kernel window (in bins)

        Returns
        -------
        kernel : np.ndarray
            Normalized Gaussian kernel
        """
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd size for symmetric kernel

        x = np.arange(window_size) - (window_size // 2)
        kernel = np.exp(-(x**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)  # Normalize

    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _apply_gaussian_smoothing(
        binned_data: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        """
        Apply Gaussian smoothing to binned spike data.

        Parameters
        ----------
        binned_data : np.ndarray
            Binned spike counts (shape: time_bins × units)
        kernel : np.ndarray
            Gaussian kernel for smoothing

        Returns
        -------
        smoothed_data : np.ndarray
            Smoothed spike density
        """
        n_bins, n_units = binned_data.shape
        half_window = len(kernel) // 2
        smoothed = np.zeros_like(binned_data, dtype=np.float32)

        # Apply smoothing for each unit in parallel
        for u in prange(n_units):
            for t in range(n_bins):
                # Calculate convolution at this time point
                conv_sum = 0.0
                for k in range(len(kernel)):
                    idx = t + k - half_window
                    if 0 <= idx < n_bins:
                        conv_sum += binned_data[idx, u] * kernel[k]
                smoothed[t, u] = conv_sum

        return smoothed

    def smooth(self, binned_spikes: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to binned spike data.

        Parameters
        ----------
        binned_spikes : np.ndarray
            Binned spike counts (shape: time_bins × units)

        Returns
        -------
        smoothed_rates : np.ndarray
            Smoothed spike density estimate
        """
        # Convert ms to bins
        sigma_bins = self.sigma_ms / self.bin_size_ms
        window_size_bins = int(self.kernel_window_ms / self.bin_size_ms)

        # Create kernel
        kernel = self._gaussian_kernel(sigma_bins, window_size_bins)

        # Apply smoothing
        smoothed = self._apply_gaussian_smoothing(binned_spikes, kernel)

        # Convert to firing rate (spikes/s)
        smoothed_rates = smoothed * (1000 / self.bin_size_ms)

        return smoothed_rates

    def estimate(
        self,
        sorter: SorterNode,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        unit_ids: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Estimate spike density from a sorter node.

        Parameters
        ----------
        sorter : SorterNode
            SorterNode containing spike data
        start_time_s : float
            Start time for analysis in seconds
        end_time_s : float or None
            End time for analysis in seconds. If None, use the end of the recording.
        unit_ids : list of int or None
            Units to include. If None, use all units.

        Returns
        -------
        time_bins : np.ndarray
            Time bin centers in seconds
        spike_density : np.ndarray
            Smoothed spike density in Hz (shape: time_bins × units)
        used_unit_ids : list of int
            Unit IDs included in the analysis
        """
        # First bin the spikes
        time_bins, binned_spikes, used_unit_ids = self.bin_spikes(
            sorter, start_time_s, end_time_s, unit_ids
        )

        # Then smooth the binned data
        smoothed_rates = self.smooth(binned_spikes)

        return time_bins, smoothed_rates, used_unit_ids
