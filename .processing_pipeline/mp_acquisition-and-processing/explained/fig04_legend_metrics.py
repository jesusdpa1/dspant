"""
/fig04_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 4 legend (Revision 1, BSPC-D-26-03892R1).
Runs fig04_explained_template-subtraction.py unchanged (runpy) and measures its arrays.

Metrics:
  - Cardiac residual: energy in ±15 ms windows around each R-peak that falls in expiration,
    before vs. after template subtraction (dB).
  - Burst SNR (Eq. 1, mean-square power) before vs. after subtraction, one burst mask for both.
  - Template: number of beats averaged, beats inside bursts, least-squares scale factors.
"""

# %%
import os
import runpy
from pathlib import Path

import mp_metrics as mpm
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig04_explained_template-subtraction.py"))

FS = float(g["FS"])
before = np.asarray(g["emg_data"]).ravel()  # filtered EMG, channel 0, 0-10 s
after = np.asarray(g["cleaned_emg_alt"])[:, 0]
peaks = np.asarray(g["r_peak_indices"]).astype(int)
template = np.asarray(g["template"])
half_win = int(g["half_win"])
N = len(before)

# %%
# Burst mask from the cleaned signal (same rule as Figure 3): 50 ms RMS, p10 + 0.3 (p99 - p10), runs >= 0.1 s
RMS_WIN = int(0.05 * FS)
env = np.sqrt(np.convolve(after**2, np.ones(RMS_WIN) / RMS_WIN, mode="same"))
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

# %%
# Cardiac residual around R-peaks during expiration
valid = peaks[(peaks - half_win >= 0) & (peaks + half_win < N)]
exp_peaks = [p for p in valid if expiration[p - half_win : p + half_win].all()]
burst_peaks = [p for p in valid if burst[p]]


def window_energy(x, idx):
    return float(sum(np.sum(x[p - half_win : p + half_win] ** 2) for p in idx))


cardiac_residual_db = 10 * np.log10(window_energy(after, exp_peaks) / window_energy(before, exp_peaks))


def burst_snr_db(x):
    return float(10 * np.log10(np.mean(x[burst] ** 2) / np.mean(x[expiration] ** 2)))


# Least-squares scale factor of the template at each beat (as used by the subtractor, without lag search)
def scale_at(p):
    seg = before[p - half_win : p - half_win + len(template)]
    return float(np.dot(seg, template) / np.dot(template, template)) if len(seg) == len(template) else np.nan


metrics = {
    "parameters": {
        "fs_hz": FS,
        "window_s": N / FS,
        "bandpass_hz": [20, 2000],
        "bandpass_order": 5,
        "notch_hz": 60,
        "notch_q": 30,
        "r_peak_reference": "200 Hz lowpass (order 5) of the filtered EMG channel",
        "r_peak_threshold_mad": 8.0,
        "refractory_s": 0.1,
        "template_window_ms": 1000 * len(template) / FS,
        "subtraction": "per-beat cross-correlation alignment + least-squares scaling",
    },
    "n_r_peaks": int(len(peaks)),
    "heart_rate_bpm": float(60 * FS / np.median(np.diff(np.sort(peaks)))),
    "n_bursts": n_bursts,
    "n_peaks_in_expiration": len(exp_peaks),
    "n_peaks_in_bursts": len(burst_peaks),
    "cardiac_residual_db_expiration": float(cardiac_residual_db),
    "burst_snr_db_before": burst_snr_db(before),
    "burst_snr_db_after": burst_snr_db(after),
    "scale_median_expiration": float(np.nanmedian([scale_at(p) for p in exp_peaks])),
    "scale_median_bursts": float(np.nanmedian([scale_at(p) for p in burst_peaks])),
}
mpm.save_metrics(metrics, mpm.figure_dir() / "fig04_legend_metrics.json")
for k, v in metrics.items():
    if k != "parameters":
        print(k, round(v, 3) if isinstance(v, float) else v)
