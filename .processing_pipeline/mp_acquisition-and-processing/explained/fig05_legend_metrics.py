"""
/fig05_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 5 legend (Revision 1, BSPC-D-26-03892R1).
Runs fig05_explained_common_reference.py unchanged (runpy) and measures its arrays.

Notes: the CMR reference is the sample-wise median of ALL 32 HDEG channels; the figure shows
channels 0-3, each trace normalized to its own maximum (panels A and B separately), so
amplitudes are compared here in physical units.

Metrics (displayed channels 0-3, full processed window, first/last 1 s excluded):
  - Mean pairwise Pearson r between channels, before/after CMR, overall / bursts / expiration.
  - Per-channel burst SNR (Eq. 1, mean-square power) and expiration RMS, before/after.
  - CMR reference: fraction of its energy within ±15 ms of cardiac peaks.
"""

# %%
import os
import runpy
from itertools import combinations
from pathlib import Path

import mp_metrics as mpm
import numpy as np
from scipy import signal

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig05_explained_common_reference.py"))

FS = float(g["FS"])
CH = list(g["SELECTED_CHANNELS"])
trim = int(1.0 * FS)
before = np.asarray(g["filter_data_computed"])[trim:-trim, CH]
after = np.asarray(g["cmr_data_computed"])[trim:-trim, CH]
reference = np.asarray(g["cmr_reference"].compute()).ravel()[trim:-trim]
n_total_channels = int(np.asarray(g["filter_data_computed"]).shape[1])
N = before.shape[0]

# %%
# Burst mask: mean of per-channel normalized 50 ms RMS envelopes after CMR, p10 + 0.3 (p99 - p10), runs >= 0.1 s
w = int(0.05 * FS)
envs = []
for c in range(after.shape[1]):
    e = np.sqrt(np.convolve(after[:, c] ** 2, np.ones(w) / w, mode="same"))
    envs.append(e / np.percentile(e, 99))
env = np.mean(envs, axis=0)
p10, p99 = np.percentile(env, [10, 99])
active = env > p10 + 0.3 * (p99 - p10)
edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
burst = np.zeros(N, bool)
n_bursts = 0
for s0, s1 in zip(starts, stops):
    if (s1 - s0) / FS >= 0.1:
        burst[s0:s1] = True
        n_bursts += 1
guard = int(0.05 * FS)
expiration = ~np.convolve(burst, np.ones(2 * guard + 1), mode="same").astype(bool)


def mean_pairwise_r(x, mask=None):
    x = x if mask is None else x[mask]
    r = np.corrcoef(x.T)
    return float(np.mean([r[i, j] for i, j in combinations(range(x.shape[1]), 2)]))


def pairwise_matrix(x):
    return np.round(np.corrcoef(x.T), 3).tolist()


# %%
# Cardiac peaks on the reference (positive peaks, 8 MAD, 0.1 s refractory - as in Figure 4)
med = np.median(reference)
mad = np.median(np.abs(reference - med))
peaks, _ = signal.find_peaks(reference, height=med + 8 * mad, distance=int(0.1 * FS))
hw = int(0.015 * FS)
in_win = np.zeros(N, bool)
for p in peaks:
    in_win[max(0, p - hw) : p + hw] = True

per_channel = {}
for k, ch in enumerate(CH):
    b, a = before[:, k], after[:, k]
    per_channel[f"ch{ch}"] = {
        "burst_snr_db_before": float(10 * np.log10(np.mean(b[burst] ** 2) / np.mean(b[expiration] ** 2))),
        "burst_snr_db_after": float(10 * np.log10(np.mean(a[burst] ** 2) / np.mean(a[expiration] ** 2))),
        "expiration_rms_change_db": float(20 * np.log10(np.sqrt(np.mean(a[expiration] ** 2)) / np.sqrt(np.mean(b[expiration] ** 2)))),
        "burst_rms_change_db": float(20 * np.log10(np.sqrt(np.mean(a[burst] ** 2)) / np.sqrt(np.mean(b[burst] ** 2)))),
        # burst samples outside ±15 ms of cardiac peaks: change not attributable to ECG removal
        "burst_rms_change_db_excluding_ecg": float(
            20 * np.log10(np.sqrt(np.mean(a[burst & ~in_win] ** 2)) / np.sqrt(np.mean(b[burst & ~in_win] ** 2)))
        ),
    }

metrics = {
    "parameters": {
        "fs_hz": FS,
        "analysis_window_s": N / FS,
        "displayed_channels": CH,
        "reference_channels": n_total_channels,
        "reference": "sample-wise median of all channels",
        "bandpass_hz": [20, 2000],
        "bandpass_order": 5,
        "notch_hz": 60,
        "notch_q": 30,
        "figure_normalization": "each trace scaled to its own max, panels A and B separately",
        "figure_tick_labels": "MaxNLocator shows positions 0-3 bottom-up = channels 3,2,1,0",
    },
    "n_bursts": n_bursts,
    "mean_pairwise_r": {
        "before_all": mean_pairwise_r(before),
        "after_all": mean_pairwise_r(after),
        "before_expiration": mean_pairwise_r(before, expiration),
        "after_expiration": mean_pairwise_r(after, expiration),
        "before_bursts": mean_pairwise_r(before, burst),
        "after_bursts": mean_pairwise_r(after, burst),
    },
    "pairwise_r_matrix_before": pairwise_matrix(before),
    "pairwise_r_matrix_after": pairwise_matrix(after),
    "per_channel": per_channel,
    "reference": {
        "n_cardiac_peaks": int(len(peaks)),
        "heart_rate_bpm": float(60 * FS / np.median(np.diff(peaks))),
        "energy_fraction_within_15ms_of_peaks": float(np.sum(reference[in_win] ** 2) / np.sum(reference**2)),
        "time_fraction_within_15ms_of_peaks": float(in_win.mean()),
    },
}
mpm.save_metrics(metrics, mpm.figure_dir() / "fig05_legend_metrics.json")
for k in ("n_bursts", "mean_pairwise_r", "per_channel", "reference"):
    v = metrics[k]
    print(k, {a: (round(b, 3) if isinstance(b, float) else {c: round(d, 2) for c, d in b.items()} if isinstance(b, dict) else b) for a, b in v.items()} if isinstance(v, dict) else v)
