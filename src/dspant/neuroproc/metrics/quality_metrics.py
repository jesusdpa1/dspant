"""
Quality metrics for neural spike sorting.

This module provides computation of standard quality metrics for evaluating
the results of spike sorting algorithms. These metrics help assess unit isolation
quality and identify well-isolated single units.

The metrics are computed on spike data that has already been extracted and sorted,
accepting spike times and unit assignments as input rather than performing signal processing.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numba import jit, prange
from scipy.ndimage import gaussian_filter1d

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False


def compute_num_spikes(
    spike_times: Dict[str, List[np.ndarray]], unit_ids: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Compute the number of spikes across segments for each unit.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times.
        Each entry in the list represents spike times (in samples) for one recording segment.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    num_spikes : Dict[str, int]
        The number of spikes, across all segments, for each unit ID.
    """
    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    num_spikes = {}
    for unit_id in unit_ids:
        if unit_id not in spike_times:
            continue
        n = sum(len(segment_spikes) for segment_spikes in spike_times[unit_id])
        num_spikes[unit_id] = n

    return num_spikes


def compute_firing_rates(
    spike_times: Dict[str, List[np.ndarray]],
    total_duration: float,
    unit_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute the firing rate across segments for each unit.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times.
        Each entry in the list represents spike times (in samples) for one recording segment.
    total_duration : float
        Total duration of the recording in seconds.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    firing_rates : Dict[str, float]
        The firing rate (in Hz), across all segments, for each unit ID.
    """
    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    if total_duration is None or total_duration <= 0:
        raise ValueError("Total duration must be provided and greater than 0")

    firing_rates = {}
    num_spikes = compute_num_spikes(spike_times, unit_ids)
    for unit_id in unit_ids:
        if unit_id in num_spikes:
            firing_rates[unit_id] = num_spikes[unit_id] / total_duration

    return firing_rates


def compute_presence_ratios(
    spike_times: Dict[str, List[np.ndarray]],
    sampling_frequency: float,
    total_duration: float,
    bin_duration_s: float = 60.0,
    mean_fr_ratio_thresh: float = 0.0,
    unit_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate the presence ratio, the fraction of time the unit is firing above a certain threshold.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    sampling_frequency : float
        Sampling frequency in Hz.
    total_duration : float
        Total duration of the recording in seconds.
    bin_duration_s : float, default: 60.0
        The duration of each bin in seconds.
    mean_fr_ratio_thresh : float, default: 0.0
        The unit is considered active in a bin if its firing rate during that bin
        is strictly above `mean_fr_ratio_thresh` times its mean firing rate.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    presence_ratios : Dict[str, float]
        The presence ratio for each unit ID.
    """
    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    # Convert mean_fr_ratio_thresh to float
    mean_fr_ratio_thresh = float(mean_fr_ratio_thresh)
    if mean_fr_ratio_thresh < 0:
        raise ValueError(
            f"Expected positive float for mean_fr_ratio_thresh. Provided value: {mean_fr_ratio_thresh}"
        )

    if mean_fr_ratio_thresh > 1:
        warnings.warn(
            "mean_fr_ratio_thresh parameter above 1 might lead to low presence ratios."
        )

    # Calculate bins in samples
    total_samples = int(total_duration * sampling_frequency)
    bin_duration_samples = int(bin_duration_s * sampling_frequency)
    num_bin_edges = total_samples // bin_duration_samples + 1
    bin_edges = np.arange(num_bin_edges) * bin_duration_samples

    presence_ratios = {}

    if total_samples < bin_duration_samples:
        warnings.warn(
            f"Bin duration of {bin_duration_s}s is larger than recording duration. Presence ratios are set to NaN."
        )
        for unit_id in unit_ids:
            presence_ratios[unit_id] = np.nan
        return presence_ratios

    for unit_id in unit_ids:
        if unit_id not in spike_times:
            presence_ratios[unit_id] = np.nan
            continue

        # Concatenate spike times across segments
        all_spikes = []
        segment_offset = 0
        for segment_spikes in spike_times[unit_id]:
            if len(segment_spikes) > 0:
                all_spikes.append(segment_spikes + segment_offset)
            segment_offset += (
                bin_duration_samples  # Simple approximation for segment boundaries
            )

        if all_spikes:
            all_spikes = (
                np.concatenate(all_spikes)
                if any(len(spk) > 0 for spk in all_spikes)
                else np.array([])
            )
        else:
            all_spikes = np.array([])

        unit_fr = len(all_spikes) / total_duration
        bin_n_spikes_thres = math.floor(unit_fr * bin_duration_s * mean_fr_ratio_thresh)

        presence_ratios[unit_id] = presence_ratio(
            all_spikes,
            total_samples,
            bin_edges=bin_edges,
            bin_n_spikes_thres=bin_n_spikes_thres,
        )

    return presence_ratios


def compute_isi_violations(
    spike_times: Dict[str, List[np.ndarray]],
    sampling_frequency: float,
    total_duration: float,
    isi_threshold_ms: float = 1.5,
    min_isi_ms: float = 0.0,
    unit_ids: Optional[List[str]] = None,
) -> namedtuple:
    """
    Calculate Inter-Spike Interval (ISI) violations.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    sampling_frequency : float
        Sampling frequency in Hz.
    total_duration : float
        Total duration of the recording in seconds.
    isi_threshold_ms : float, default: 1.5
        Threshold for classifying adjacent spikes as an ISI violation, in ms.
        This is the biophysical refractory period.
    min_isi_ms : float, default: 0
        Minimum possible inter-spike interval, in ms.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    isi_violations : namedtuple
        A named tuple containing:
        - isi_violations_ratio: Dict[str, float] - The relative firing rate of the hypothetical
          neurons that are generating the ISI violations.
        - isi_violations_count: Dict[str, int] - Number of ISI violations.
    """
    res = namedtuple("isi_violation", ["isi_violations_ratio", "isi_violations_count"])

    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    # Convert ms to seconds
    isi_threshold_s = isi_threshold_ms / 1000
    min_isi_s = min_isi_ms / 1000

    isi_violations_ratio = {}
    isi_violations_count = {}

    for unit_id in unit_ids:
        if unit_id not in spike_times:
            isi_violations_ratio[unit_id] = np.nan
            isi_violations_count[unit_id] = 0
            continue

        # Convert spike times to seconds
        spike_train_list = []
        for segment_spikes in spike_times[unit_id]:
            if len(segment_spikes) > 0:
                spike_train_list.append(segment_spikes / sampling_frequency)

        if not any(len(train) > 0 for train in spike_train_list):
            isi_violations_ratio[unit_id] = np.nan
            isi_violations_count[unit_id] = 0
            continue

        ratio, _, count = isi_violations(
            spike_train_list, total_duration, isi_threshold_s, min_isi_s
        )

        isi_violations_ratio[unit_id] = ratio
        isi_violations_count[unit_id] = count

    return res(isi_violations_ratio, isi_violations_count)


def compute_amplitude_cutoffs(
    spike_amplitudes: Dict[str, np.ndarray],
    num_histogram_bins: int = 100,
    histogram_smoothing_value: int = 3,
    amplitudes_bins_min_ratio: int = 5,
    unit_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    spike_amplitudes : Dict[str, np.ndarray]
        Dictionary mapping unit IDs to arrays of spike amplitudes.
    num_histogram_bins : int, default: 100
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set
        to NaN.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    amplitude_cutoffs : Dict[str, float]
        Estimated fraction of missing spikes, based on the amplitude distribution, for each unit ID.
    """
    if unit_ids is None:
        unit_ids = list(spike_amplitudes.keys())

    all_fraction_missing = {}

    for unit_id in unit_ids:
        if unit_id not in spike_amplitudes:
            all_fraction_missing[unit_id] = np.nan
            continue

        amplitudes = spike_amplitudes[unit_id]

        all_fraction_missing[unit_id] = amplitude_cutoff(
            amplitudes,
            num_histogram_bins,
            histogram_smoothing_value,
            amplitudes_bins_min_ratio,
        )

    return all_fraction_missing


def compute_snrs(
    templates: Dict[str, np.ndarray],
    noise_levels: Dict[str, float],
    peak_sign: str = "neg",
    unit_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute signal-to-noise ratio for each unit.

    Parameters
    ----------
    templates : Dict[str, np.ndarray]
        Dictionary mapping unit IDs to templates (average waveforms).
    noise_levels : Dict[str, float]
        Dictionary mapping unit IDs to noise standard deviation.
    peak_sign : str, default: "neg"
        The sign of the peak to use for SNR computation.
        Can be "neg", "pos", or "both".
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    snrs : Dict[str, float]
        Signal-to-noise ratio for each unit.
    """
    if unit_ids is None:
        unit_ids = list(templates.keys())

    snrs = {}

    for unit_id in unit_ids:
        if unit_id not in templates or unit_id not in noise_levels:
            snrs[unit_id] = np.nan
            continue

        template = templates[unit_id]
        noise = noise_levels[unit_id]

        if peak_sign == "neg":
            peak_amp = np.min(template)
        elif peak_sign == "pos":
            peak_amp = np.max(template)
        else:  # "both"
            peak_amp = np.max(np.abs(template))

        snrs[unit_id] = np.abs(peak_amp) / noise

    return snrs


# Helper functions


def presence_ratio(
    spike_train: np.ndarray,
    total_length: int,
    bin_edges: np.ndarray = None,
    num_bin_edges: int = None,
    bin_n_spikes_thres: int = 0,
) -> float:
    """
    Calculate the presence ratio for a single unit.

    Parameters
    ----------
    spike_train : np.ndarray
        Spike times for this unit, in samples.
    total_length : int
        Total length of the recording in samples.
    bin_edges : np.array, optional
        Pre-computed bin edges (mutually exclusive with num_bin_edges).
    num_bin_edges : int, optional
        The number of bins edges to use to compute the presence ratio
        (mutually exclusive with bin_edges).
    bin_n_spikes_thres : int, default: 0
        Minimum number of spikes within a bin to consider the unit active.

    Returns
    -------
    presence_ratio : float
        The presence ratio for one unit.
    """
    assert bin_edges is not None or num_bin_edges is not None, (
        "Use either bin_edges or num_bin_edges"
    )
    assert bin_n_spikes_thres >= 0

    if bin_edges is not None:
        bins = bin_edges
        num_bin_edges = len(bin_edges)
    else:
        bins = num_bin_edges

    h, _ = np.histogram(spike_train, bins=bins)

    return np.sum(h > bin_n_spikes_thres) / (num_bin_edges - 1)


def isi_violations(
    spike_trains: List[np.ndarray],
    total_duration_s: float,
    isi_threshold_s: float = 0.0015,
    min_isi_s: float = 0,
) -> Tuple[float, float, int]:
    """
    Calculate Inter-Spike Interval (ISI) violations.

    Parameters
    ----------
    spike_trains : list of np.ndarrays
        The spike times for each recording segment for one unit, in seconds.
    total_duration_s : float
        The total duration of the recording (in seconds).
    isi_threshold_s : float, default: 0.0015
        Threshold for classifying adjacent spikes as an ISI violation, in seconds.
        This is the biophysical refractory period.
    min_isi_s : float, default: 0
        Minimum possible inter-spike interval, in seconds.
        This is the artificial refractory period enforced
        by the data acquisition system or post-processing algorithms.

    Returns
    -------
    isi_violations_ratio : float
        The isi violation ratio.
    isi_violations_rate : float
        Rate of contaminating spikes as a fraction of overall rate.
    isi_violation_count : int
        Number of violations.
    """
    num_violations = 0
    num_spikes = 0

    isi_violations_ratio = np.float64(np.nan)
    isi_violations_rate = np.float64(np.nan)
    isi_violations_count = np.float64(np.nan)

    for spike_train in spike_trains:
        if len(spike_train) < 2:
            continue
        isis = np.diff(spike_train)
        num_spikes += len(spike_train)
        num_violations += np.sum(isis < isi_threshold_s)

    violation_time = 2 * num_spikes * (isi_threshold_s - min_isi_s)

    if num_spikes > 0:
        total_rate = num_spikes / total_duration_s
        violation_rate = num_violations / violation_time
        isi_violations_ratio = violation_rate / total_rate
        isi_violations_rate = num_violations / total_duration_s
        isi_violations_count = num_violations

    return isi_violations_ratio, isi_violations_rate, isi_violations_count


def amplitude_cutoff(
    amplitudes: np.ndarray,
    num_histogram_bins: int = 500,
    histogram_smoothing_value: int = 3,
    amplitudes_bins_min_ratio: int = 5,
) -> float:
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes.

    Parameters
    ----------
    amplitudes : ndarray_like
        The amplitudes (in uV) of the spikes for one unit.
    num_histogram_bins : int, default: 500
        The number of bins to use to compute the amplitude histogram.
    histogram_smoothing_value : int, default: 3
        Controls the smoothing applied to the amplitude histogram.
    amplitudes_bins_min_ratio : int, default: 5
        The minimum ratio between number of amplitudes for a unit and the number of bins.
        If the ratio is less than this threshold, the amplitude_cutoff for the unit is set
        to NaN.

    Returns
    -------
    fraction_missing : float
        Estimated fraction of missing spikes, based on the amplitude distribution.
    """
    if len(amplitudes) / num_histogram_bins < amplitudes_bins_min_ratio:
        return np.nan
    else:
        h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

        pdf = gaussian_filter1d(h, histogram_smoothing_value)
        support = b[:-1]
        bin_size = np.mean(np.diff(support))
        peak_index = np.argmax(pdf)

        pdf_above = np.abs(pdf[peak_index:] - pdf[0])

        if len(np.where(pdf_above == pdf_above.min())[0]) > 1:
            warnings.warn(
                "Amplitude PDF does not have a unique minimum! More spikes might be required for a correct "
                "amplitude_cutoff computation!"
            )

        G = np.argmin(pdf_above) + peak_index
        fraction_missing = np.sum(pdf[G:]) * bin_size
        fraction_missing = np.min([fraction_missing, 0.5])

        return fraction_missing


if HAVE_NUMBA:

    @numba.jit(nopython=True, nogil=True, cache=False)
    def _compute_nb_violations_numba(spike_train: np.ndarray, t_r: int) -> int:
        """
        Compute the number of refractory period violations.

        Parameters
        ----------
        spike_train : np.ndarray
            Spike times for one unit
        t_r : int
            Refractory period in samples

        Returns
        -------
        n_v : int
            Number of violations
        """
        n_v = 0
        N = len(spike_train)

        for i in range(N):
            for j in range(i + 1, N):
                diff = spike_train[j] - spike_train[i]

                if diff > t_r:
                    break

                n_v += 1

        return n_v


def compute_refrac_period_violations(
    spike_times: Dict[str, List[np.ndarray]],
    sampling_frequency: float,
    total_duration: float,
    refractory_period_ms: float = 1.0,
    censored_period_ms: float = 0.0,
    unit_ids: Optional[List[str]] = None,
) -> namedtuple:
    """
    Calculate the number of refractory period violations.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    sampling_frequency : float
        Sampling frequency in Hz.
    total_duration : float
        Total duration of the recording in seconds.
    refractory_period_ms : float, default: 1.0
        The period (in ms) where no 2 good spikes can occur.
    censored_period_ms : float, default: 0.0
        The period (in ms) where no 2 spikes can occur (because they are not detected, or
        because they were removed by another mean).
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    rp_violations : namedtuple
        A named tuple containing:
        - rp_contamination: Dict[str, float] - Estimated contamination based on refractory period violations.
        - rp_violations: Dict[str, int] - Number of refractory period violations.
    """
    if not HAVE_NUMBA:
        warnings.warn(
            "Numba is required for refrac_period_violations. Install with 'pip install numba'."
        )
        return None

    res = namedtuple("rp_violations", ["rp_contamination", "rp_violations"])

    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    t_c = int(round(censored_period_ms * sampling_frequency * 1e-3))
    t_r = int(round(refractory_period_ms * sampling_frequency * 1e-3))

    nb_violations = {}
    rp_contamination = {}

    # We need to compute total samples and perform spike count
    T = int(total_duration * sampling_frequency)
    n_spikes = compute_num_spikes(spike_times)

    for unit_id in unit_ids:
        if unit_id not in spike_times:
            nb_violations[unit_id] = 0
            rp_contamination[unit_id] = np.nan
            continue

        total_violations = 0

        # Process each segment
        for segment_spikes in spike_times[unit_id]:
            if len(segment_spikes) > 0:
                total_violations += _compute_nb_violations_numba(segment_spikes, t_r)

        nb_violations[unit_id] = n_v = total_violations
        N = n_spikes.get(unit_id, 0)

        if N == 0:
            rp_contamination[unit_id] = np.nan
        else:
            D = 1 - n_v * (T - 2 * N * t_c) / (N**2 * (t_r - t_c))
            rp_contamination[unit_id] = 1 - math.sqrt(D) if D >= 0 else 1.0

    return res(rp_contamination, nb_violations)


def compute_quality_metrics(
    spike_times: Dict[str, List[np.ndarray]],
    sampling_frequency: float,
    total_duration: float,
    spike_amplitudes: Optional[Dict[str, np.ndarray]] = None,
    templates: Optional[Dict[str, np.ndarray]] = None,
    noise_levels: Optional[Dict[str, float]] = None,
    unit_ids: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    **metric_params,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Compute a set of quality metrics for sorted units.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    sampling_frequency : float
        Sampling frequency in Hz.
    total_duration : float
        Total duration of the recording in seconds.
    spike_amplitudes : Dict[str, np.ndarray], optional
        Dictionary mapping unit IDs to arrays of spike amplitudes.
    templates : Dict[str, np.ndarray], optional
        Dictionary mapping unit IDs to templates (average waveforms).
    noise_levels : Dict[str, float], optional
        Dictionary mapping unit IDs to noise standard deviation.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.
    metrics : List[str], optional
        List of metrics to compute. If None, compute all applicable metrics.
        Available metrics: 'num_spikes', 'firing_rate', 'presence_ratio', 'isi_violation',
        'rp_violation', 'amplitude_cutoff', 'snr'.
    **metric_params : dict
        Additional parameters for specific metrics.

    Returns
    -------
    quality_metrics : Dict[str, Dict[str, Union[float, int]]]
        Dictionary mapping metric names to dictionaries of unit metric values.
    """
    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    # Determine which metrics to compute
    all_metrics = [
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "isi_violation",
        "rp_violation",
        "amplitude_cutoff",
        "snr",
    ]

    if metrics is None:
        # Determine which metrics we can compute based on available data
        metrics = ["num_spikes", "firing_rate", "presence_ratio", "isi_violation"]

        if HAVE_NUMBA:
            metrics.append("rp_violation")

        if spike_amplitudes is not None:
            metrics.append("amplitude_cutoff")

        if templates is not None and noise_levels is not None:
            metrics.append("snr")
    else:
        # Validate requested metrics
        unknown_metrics = [m for m in metrics if m not in all_metrics]
        if unknown_metrics:
            raise ValueError(
                f"Unknown metrics: {unknown_metrics}. Available metrics: {all_metrics}"
            )

        # Check if we have the required data for each metric
        if "amplitude_cutoff" in metrics and spike_amplitudes is None:
            warnings.warn(
                "'amplitude_cutoff' metric requires spike_amplitudes. Skipping."
            )
            metrics.remove("amplitude_cutoff")

        if "snr" in metrics and (templates is None or noise_levels is None):
            warnings.warn("'snr' metric requires templates and noise_levels. Skipping.")
            metrics.remove("snr")

        if "rp_violation" in metrics and not HAVE_NUMBA:
            warnings.warn("'rp_violation' metric requires numba. Skipping.")
            metrics.remove("rp_violation")

    # Initialize results
    results = {}

    # Compute requested metrics
    for metric in metrics:
        if metric == "num_spikes":
            results[metric] = compute_num_spikes(spike_times, unit_ids)

        elif metric == "firing_rate":
            results[metric] = compute_firing_rates(
                spike_times, total_duration, unit_ids
            )

        elif metric == "presence_ratio":
            bin_duration_s = metric_params.get("bin_duration_s", 60.0)
            mean_fr_ratio_thresh = metric_params.get("mean_fr_ratio_thresh", 0.0)

            results[metric] = compute_presence_ratios(
                spike_times,
                sampling_frequency,
                total_duration,
                bin_duration_s,
                mean_fr_ratio_thresh,
                unit_ids,
            )

        elif metric == "isi_violation":
            isi_threshold_ms = metric_params.get("isi_threshold_ms", 1.5)
            min_isi_ms = metric_params.get("min_isi_ms", 0.0)

            res = compute_isi_violations(
                spike_times,
                sampling_frequency,
                total_duration,
                isi_threshold_ms,
                min_isi_ms,
                unit_ids,
            )

            # Handle namedtuple result
            results["isi_violations_ratio"] = res.isi_violations_ratio
            results["isi_violations_count"] = res.isi_violations_count

        elif metric == "rp_violation":
            refractory_period_ms = metric_params.get("refractory_period_ms", 1.0)
            censored_period_ms = metric_params.get("censored_period_ms", 0.0)

            res = compute_refrac_period_violations(
                spike_times,
                sampling_frequency,
                total_duration,
                refractory_period_ms,
                censored_period_ms,
                unit_ids,
            )

            # Handle namedtuple result
            results["rp_contamination"] = res.rp_contamination
            results["rp_violations"] = res.rp_violations

        elif metric == "amplitude_cutoff":
            num_histogram_bins = metric_params.get("num_histogram_bins", 100)
            histogram_smoothing_value = metric_params.get(
                "histogram_smoothing_value", 3
            )
            amplitudes_bins_min_ratio = metric_params.get(
                "amplitudes_bins_min_ratio", 5
            )

            results[metric] = compute_amplitude_cutoffs(
                spike_amplitudes,
                num_histogram_bins,
                histogram_smoothing_value,
                amplitudes_bins_min_ratio,
                unit_ids,
            )

        elif metric == "snr":
            peak_sign = metric_params.get("peak_sign", "neg")

            results[metric] = compute_snrs(templates, noise_levels, peak_sign, unit_ids)

    return results


def compute_synchrony_metrics(
    spike_times: Dict[str, List[np.ndarray]],
    sampling_frequency: float,
    synchrony_sizes: List[int] = None,
    unit_ids: Optional[List[str]] = None,
) -> namedtuple:
    """
    Compute synchrony metrics, which represent the rate of occurrences of
    spikes at the exact same sample index, with synchrony sizes 2, 4 and 8.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    sampling_frequency : float
        Sampling frequency in Hz.
    synchrony_sizes : List[int], optional
        The synchrony sizes to compute. If None, uses [2, 4, 8].
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    synchrony_metrics : namedtuple
        A named tuple containing:
        - sync_spike_X: Dict[str, float] - The synchrony metric for synchrony size X.
          One field for each synchrony size.
    """
    if synchrony_sizes is None:
        synchrony_sizes = [2, 4, 8]

    # Create named tuple dynamically
    fields = [f"sync_spike_{size}" for size in synchrony_sizes]
    res = namedtuple("synchrony_metrics", fields)

    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    # Flatten all spike times into a sorted array with unit labels
    all_spikes = []
    all_units = []

    for unit_id in spike_times:
        for segment_spikes in spike_times[unit_id]:
            if len(segment_spikes) > 0:
                all_spikes.extend(segment_spikes)
                all_units.extend([unit_id] * len(segment_spikes))

    if not all_spikes:
        # No spikes found
        empty_dict = {unit_id: 0 for unit_id in unit_ids}
        return res(*[empty_dict for _ in synchrony_sizes])

    # Sort spikes by time
    sort_idx = np.argsort(all_spikes)
    all_spikes = np.array(all_spikes)[sort_idx]
    all_units = np.array(all_units)[sort_idx]

    # Count spikes per unit
    spike_counts = {unit_id: 0 for unit_id in unit_ids}
    for unit_id in all_units:
        if unit_id in spike_counts:
            spike_counts[unit_id] += 1

    # Find synchronous events
    unique_times, counts = np.unique(all_spikes, return_counts=True)

    # Count synchronous spikes for each unit and synchrony size
    synchrony_metrics_dict = {}
    for sync_idx, synchrony_size in enumerate(synchrony_sizes):
        sync_id_metrics_dict = {unit_id: 0 for unit_id in unit_ids}

        # Filter for times with at least the minimum synchrony size
        sync_times = unique_times[counts >= synchrony_size]

        for sync_time in sync_times:
            # Find all spikes at this time
            mask = all_spikes == sync_time
            units_with_sync = all_units[mask]

            # Count for each unit
            for unit_id in set(units_with_sync):
                if unit_id in sync_id_metrics_dict:
                    sync_id_metrics_dict[unit_id] += 1

        # Normalize by spike count
        for unit_id in unit_ids:
            if spike_counts[unit_id] > 0:
                sync_id_metrics_dict[unit_id] /= spike_counts[unit_id]

        synchrony_metrics_dict[f"sync_spike_{synchrony_size}"] = sync_id_metrics_dict

    return res(*[synchrony_metrics_dict[field] for field in fields])


def compute_drift_metrics(
    spike_times: Dict[str, List[np.ndarray]],
    spike_positions: Dict[str, np.ndarray],
    sampling_frequency: float,
    total_duration: float,
    interval_s: int = 60,
    min_spikes_per_interval: int = 100,
    direction: str = "y",
    min_fraction_valid_intervals: float = 0.5,
    min_num_bins: int = 2,
    unit_ids: Optional[List[str]] = None,
) -> namedtuple:
    """
    Compute drift metrics using estimated spike locations.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    spike_positions : Dict[str, np.ndarray]
        Dictionary mapping unit IDs to arrays of spike positions (x, y, z).
        Each array should have shape (n_spikes, 3) or include fields 'x', 'y', 'z'.
    sampling_frequency : float
        Sampling frequency in Hz.
    total_duration : float
        Total duration of the recording in seconds.
    interval_s : int, default: 60
        Interval length is seconds for computing spike depth.
    min_spikes_per_interval : int, default: 100
        Minimum number of spikes for computing depth in an interval.
    direction : str, default: "y"
        The direction along which drift metrics are estimated ("x", "y", or "z").
    min_fraction_valid_intervals : float, default: 0.5
        The fraction of valid (not NaN) position estimates to estimate drifts.
    min_num_bins : int, default: 2
        Minimum number of bins required to return a valid metric value.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    drift_metrics : namedtuple
        A named tuple containing:
        - drift_ptp: Dict[str, float] - The drift signal peak-to-peak.
        - drift_std: Dict[str, float] - The drift signal standard deviation.
        - drift_mad: Dict[str, float] - The drift signal median absolute deviation.
    """
    res = namedtuple("drift_metrics", ["drift_ptp", "drift_std", "drift_mad"])

    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    # Check if we have both spike times and positions
    missing_units = [
        u for u in unit_ids if u not in spike_times or u not in spike_positions
    ]
    if missing_units:
        warnings.warn(f"Missing spike times or positions for units: {missing_units}")
        for u in missing_units:
            if u in unit_ids:
                unit_ids.remove(u)

    if not unit_ids:
        empty_dict = {}
        return res(empty_dict, empty_dict, empty_dict)

    # Validate direction
    position_data = next(iter(spike_positions.values()))
    if hasattr(position_data, "dtype") and direction in position_data.dtype.names:
        # Structured array with named fields
        pass
    elif isinstance(position_data, np.ndarray) and position_data.ndim == 2:
        # Array with shape (n_spikes, 3)
        direction_map = {"x": 0, "y": 1, "z": 2}
        if direction not in direction_map:
            raise ValueError(
                f"Invalid direction '{direction}'. Must be 'x', 'y', or 'z'."
            )
        direction = direction_map[direction]
    else:
        raise ValueError(
            "Spike positions must be a structured array with fields 'x', 'y', 'z' or an array with shape (n_spikes, 3)"
        )

    interval_samples = int(interval_s * sampling_frequency)
    total_samples = int(total_duration * sampling_frequency)

    # Check if recording is long enough
    if total_samples < min_num_bins * interval_samples:
        warnings.warn(
            "The recording is too short given the specified 'interval_s' and "
            "'min_num_bins'. Drift metrics will be set to NaN"
        )
        empty_dict = {unit_id: np.nan for unit_id in unit_ids}
        return res(empty_dict, empty_dict, empty_dict)

    # Compute number of intervals
    num_intervals = total_samples // interval_samples
    bins = np.arange(num_intervals + 1) * interval_samples

    # Initialize results
    drift_ptps = {}
    drift_stds = {}
    drift_mads = {}

    # Compute median positions for each unit
    for unit_id in unit_ids:
        # Concatenate spike times across segments
        all_spikes = []
        segment_offset = 0
        for segment_spikes in spike_times[unit_id]:
            if len(segment_spikes) > 0:
                all_spikes.append(segment_spikes + segment_offset)
            segment_offset += (
                interval_samples  # Simple approximation for segment boundaries
            )

        if not all_spikes:
            drift_ptps[unit_id] = np.nan
            drift_stds[unit_id] = np.nan
            drift_mads[unit_id] = np.nan
            continue

        all_spikes = (
            np.concatenate(all_spikes)
            if any(len(spk) > 0 for spk in all_spikes)
            else np.array([])
        )

        if len(all_spikes) < min_spikes_per_interval:
            drift_ptps[unit_id] = np.nan
            drift_stds[unit_id] = np.nan
            drift_mads[unit_id] = np.nan
            continue

        # Get positions
        positions = spike_positions[unit_id]
        if isinstance(direction, int):
            # Array with shape (n_spikes, 3)
            pos_values = positions[:, direction]
        else:
            # Structured array with named fields
            pos_values = positions[direction]

        # Compute reference position (median across all spikes)
        reference_position = np.median(pos_values)

        # Compute median position in each interval
        median_positions = np.nan * np.ones(num_intervals)
        for interval in range(num_intervals):
            start_frame = bins[interval]
            end_frame = bins[interval + 1]

            # Find spikes in this interval
            mask = (all_spikes >= start_frame) & (all_spikes < end_frame)
            if np.sum(mask) >= min_spikes_per_interval:
                median_positions[interval] = np.median(pos_values[mask])

        # Compute position differences from reference
        position_diffs = median_positions - reference_position

        # Check if we have enough valid positions
        if np.sum(~np.isnan(position_diffs)) < min_fraction_valid_intervals * len(
            position_diffs
        ):
            drift_ptps[unit_id] = np.nan
            drift_stds[unit_id] = np.nan
            drift_mads[unit_id] = np.nan
            continue

        # Compute drift metrics
        drift_ptps[unit_id] = np.nanmax(position_diffs) - np.nanmin(position_diffs)
        drift_stds[unit_id] = np.nanstd(position_diffs)
        drift_mads[unit_id] = np.nanmedian(
            np.abs(position_diffs - np.nanmedian(position_diffs))
        )

    return res(drift_ptps, drift_stds, drift_mads)


def compute_firing_ranges(
    spike_times: Dict[str, List[np.ndarray]],
    sampling_frequency: float,
    total_duration: float,
    bin_size_s: float = 5,
    percentiles: Tuple[float, float] = (5, 95),
    unit_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate firing range, the range between the 5th and 95th percentiles of the firing rates
    distribution computed in non-overlapping time bins.

    Parameters
    ----------
    spike_times : Dict[str, List[np.ndarray]]
        Dictionary mapping unit IDs to lists of spike times (in samples).
        Each entry in the list represents spike times for one recording segment.
    sampling_frequency : float
        Sampling frequency in Hz.
    total_duration : float
        Total duration of the recording in seconds.
    bin_size_s : float, default: 5
        The size of the bin in seconds.
    percentiles : tuple, default: (5, 95)
        The percentiles to compute.
    unit_ids : List[str], optional
        List of unit IDs to compute for. If None, use all units.

    Returns
    -------
    firing_ranges : Dict[str, float]
        The firing range for each unit.
    """
    if unit_ids is None:
        unit_ids = list(spike_times.keys())

    bin_size_samples = int(bin_size_s * sampling_frequency)
    total_samples = int(total_duration * sampling_frequency)

    if total_samples < bin_size_samples:
        warnings.warn(
            f"Bin size of {bin_size_s}s is larger than recording duration. Firing ranges are set to NaN."
        )
        return {unit_id: np.nan for unit_id in unit_ids}

    firing_ranges = {}

    for unit_id in unit_ids:
        if unit_id not in spike_times:
            firing_ranges[unit_id] = np.nan
            continue

        # Collect firing rates across all segments
        firing_rates = []

        # Process each segment
        for segment_spikes in spike_times[unit_id]:
            if len(segment_spikes) == 0:
                continue

            # Create bins for this segment
            segment_length = max(segment_spikes) + 1 if len(segment_spikes) > 0 else 0
            edges = np.arange(0, segment_length + 1, bin_size_samples)
            if len(edges) <= 1:
                continue

            # Compute histogram and firing rates
            counts, _ = np.histogram(segment_spikes, bins=edges)
            rates = counts / bin_size_s
            firing_rates.extend(rates)

        if not firing_rates:
            firing_ranges[unit_id] = np.nan
            continue

        # Compute percentile range
        firing_rates = np.array(firing_rates)
        firing_ranges[unit_id] = np.percentile(
            firing_rates, percentiles[1]
        ) - np.percentile(firing_rates, percentiles[0])

    return firing_ranges
