from typing import Dict, List, Literal, Optional, Tuple, Union

import dask.array as da
import numpy as np
import polars as pl
from numba import jit

from dspant.core.internals import public_api

from .base_extractor import BaseExtractor


@public_api
class WaveformExtractor(BaseExtractor):
    """
    Advanced spike waveform extraction processor.

    Supports flexible extraction of spike waveforms from raw neural data,
    with options for alignment, filtering, and advanced preprocessing.
    """

    def __init__(self, data: Union[np.ndarray, da.Array], fs: float):
        """
        Initialize WaveformExtractor.

        Parameters:
        -----------
        data : array
            Raw neural data (samples × channels)
        fs : float
            Sampling frequency in Hz
        """
        # Ensure data is 2D
        self.data = self._check_data_shape(da.asarray(data))
        self.fs = fs

        # Validate data
        if self.data.ndim != 2:
            raise ValueError("Data must be 2-dimensional (samples × channels)")

    def extract_waveforms(
        self,
        spike_times: Union[np.ndarray, pl.Series, pl.DataFrame],
        pre_samples: int = 10,
        post_samples: int = 40,
        align_to_min: bool = True,
        align_window: int = 5,
        max_jitter: int = 3,
        time_unit: Literal["seconds", "samples"] = "seconds",
        channel_selection: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[da.Array, np.ndarray, Dict]:
        """
        Extract spike waveforms with advanced alignment options.

        Parameters:
        -----------
        spike_times : array-like
            Spike detection times
        pre_samples : int, optional
            Samples before spike peak (default: 10)
        post_samples : int, optional
            Samples after spike peak (default: 40)
        align_to_min : bool, optional
            Align to negative peak if True, positive peak if False (default: True)
        align_window : int, optional
            Window size for peak alignment (default: 5)
        max_jitter : int, optional
            Maximum allowed shift during alignment (default: 3)
        time_unit : {'seconds', 'samples'}, optional
            Unit of spike times (default: 'seconds')
        channel_selection : int or list, optional
            Specific channel(s) to extract waveforms from

        Returns:
        --------
        waveforms : da.Array
            Extracted spike waveforms
        valid_spike_times : np.ndarray
            Times of successfully extracted spikes
        metadata : Dict
            Extraction metadata
        """
        # Validate and convert spike times to samples
        spike_samples = self._convert_time_to_samples(
            self._validate_time_input(spike_times), self.fs, time_unit
        )

        # Channel selection
        if channel_selection is not None:
            if isinstance(channel_selection, int):
                channel_selection = [channel_selection]
            data = self.data[:, channel_selection]
        else:
            data = self.data

        # Prepare storage for waveforms
        waveform_chunks = []
        valid_indices = []
        waveform_polarities = []

        # Extract waveforms
        for i, spike_sample in enumerate(spike_samples):
            try:
                # Check spike time is within data bounds
                if (
                    spike_sample < pre_samples
                    or spike_sample >= data.shape[0] - post_samples
                ):
                    continue

                # Extract waveform chunk
                waveform_chunk = data[
                    spike_sample - pre_samples : spike_sample + post_samples + 1, :
                ]

                # Verify chunk size
                if waveform_chunk.shape[0] != pre_samples + post_samples + 1:
                    continue

                # Determine polarity
                is_negative = (
                    np.min(waveform_chunk[:, 0]) < 0
                    if align_to_min
                    else np.max(waveform_chunk[:, 0]) > 0
                )

                waveform_chunks.append(waveform_chunk)
                valid_indices.append(i)
                waveform_polarities.append(is_negative)

            except Exception as e:
                print(f"Error extracting waveform at sample {spike_sample}: {str(e)}")
                continue

        # Handle case of no valid waveforms
        if not waveform_chunks:
            return (
                da.empty(
                    (0, pre_samples + post_samples + 1, data.shape[1]), dtype=data.dtype
                ),
                np.array([]),
                {"n_spikes": 0},
            )

        # Stack waveforms
        waveforms = da.stack(waveform_chunks)

        # Filter spike times
        valid_spike_times = spike_samples[valid_indices]

        # Prepare metadata
        metadata = {
            "n_spikes": len(valid_spike_times),
            "pre_samples": pre_samples,
            "post_samples": post_samples,
            "aligned": align_to_min,
            "channels": (
                channel_selection
                if channel_selection is not None
                else list(range(data.shape[1]))
            ),
        }

        # Optional: Apply alignment
        if align_to_min:
            waveforms = self._align_waveforms(
                waveforms, pre_samples, align_window, max_jitter, waveform_polarities
            )

        return waveforms, valid_spike_times, metadata

    @staticmethod
    @jit(nopython=True, cache=True)
    def _align_waveform(
        waveform: np.ndarray,
        pre_samples: int,
        align_window: int,
        max_jitter: int,
        is_negative: bool,
    ) -> Tuple[np.ndarray, int]:
        """
        Align a single waveform to its peak.

        Numba-accelerated alignment function.
        """
        waveform_length = waveform.shape[0]
        n_channels = waveform.shape[1]

        # Search for peak within alignment window
        search_start = max(0, pre_samples - align_window)
        search_end = min(waveform_length, pre_samples + align_window + 1)

        # Search first channel for peak
        search_window = waveform[search_start:search_end, 0]

        # Find peak position
        if is_negative:
            peak_idx = np.argmin(search_window) + search_start
        else:
            peak_idx = np.argmax(search_window) + search_start

        # Calculate shift
        shift = pre_samples - peak_idx

        # Apply shift if within jitter limits
        if abs(shift) <= max_jitter:
            aligned = np.zeros_like(waveform)

            # Apply shift
            if shift > 0:
                aligned[shift:, :] = waveform[:-shift, :]
            elif shift < 0:
                aligned[:shift, :] = waveform[-shift:, :]
            else:
                aligned = waveform.copy()

            return aligned, shift

        # If shift is too large, return original waveform
        return waveform, 0

    def _align_waveforms(
        self,
        waveforms: da.Array,
        pre_samples: int,
        align_window: int,
        max_jitter: int,
        polarities: List[bool],
    ) -> da.Array:
        """
        Align multiple waveforms using Numba-accelerated method.
        """
        # Prepare list to store aligned waveforms
        aligned_chunks = []

        # Align each waveform
        for i, (waveform, is_negative) in enumerate(zip(waveforms, polarities)):
            # Compute the waveform
            waveform_np = waveform.compute()

            # Align
            aligned_waveform, _ = self._align_waveform(
                waveform_np, pre_samples, align_window, max_jitter, is_negative
            )

            aligned_chunks.append(
                da.from_array(aligned_waveform, chunks=waveform.chunks)
            )

        # Stack aligned waveforms
        return da.stack(aligned_chunks)
