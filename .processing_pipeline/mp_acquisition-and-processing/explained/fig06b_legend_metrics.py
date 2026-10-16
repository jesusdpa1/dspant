"""
/fig06b_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 6 and ECG/EMG separation figure legends
(Revision 1, BSPC-D-26-03892R1, Reviewer #3 C11-C12). Runs fig06b_explained_ecg_emg_separation.py
unchanged (runpy; transforms and ECG separation come from the parquet cache).

Metrics (30 s analysis segment of Figure 6, channel 0):
  - Per inspiratory burst: median and mean power frequency (Welch PSD) of the filtered signal
    with ECG and after ECG template subtraction; paired difference (ECG bias).
  - Stability: linear trend of burst median frequency over time (Hz/min). Anesthetized baseline,
    not a fatigue protocol.
  - Extracted ECG: median frequency, frequency below which 95% of its power lies, power below 500 Hz.
  - ECG-removed bursts: fraction of power in 100-1000 Hz (Figure 6 legend statement).
"""

# %%
import os
import runpy
from pathlib import Path

import mp_metrics as mpm
import numpy as np
from scipy import signal, stats

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig06b_explained_ecg_emg_separation.py"))

FS = float(g["FS"])
with_ecg = np.asarray(g["x"])
clean = np.asarray(g["emg_clean"])
ecg = np.asarray(g["ecg_component"])
N = len(clean)

# %%
# Burst mask from the ECG-removed signal (same rule as Figures 3-5): 50 ms RMS, p10 + 0.3 (p99 - p10), runs >= 0.1 s
w = int(0.05 * FS)
env = np.sqrt(np.convolve(clean**2, np.ones(w) / w, mode="same"))
p10, p99 = np.percentile(env, [10, 99])
active = env > p10 + 0.3 * (p99 - p10)
edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
bursts = [(a, b) for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]) if (b - a) / FS >= 0.1]
# exclude bursts touching the segment edges (filter / transform edge effects)
bursts = [(a, b) for a, b in bursts if a > int(0.5 * FS) and b < N - int(0.5 * FS)]

NPERSEG = 2048  # 84 ms at 24.4 kHz (11.9 Hz resolution)


def psd(x):
    return signal.welch(x, fs=FS, nperseg=min(NPERSEG, len(x)))


def median_mean_freq(x):
    f, p = psd(x)
    c = np.cumsum(p)
    return float(f[np.searchsorted(c, c[-1] / 2)]), float(np.sum(f * p) / np.sum(p))


rows = []
for a, b in bursts:
    fm_e, fmean_e = median_mean_freq(with_ecg[a:b])
    fm_c, fmean_c = median_mean_freq(clean[a:b])
    rows.append((a / FS, fm_e, fm_c, fmean_e, fmean_c))
rows = np.array(rows)
t_min = rows[:, 0] / 60
trend = stats.linregress(t_min, rows[:, 2])

# Extracted ECG over the whole segment
f, p = psd(ecg)
c = np.cumsum(p) / np.sum(p)
ecg_median = float(f[np.searchsorted(c, 0.5)])
ecg_f95 = float(f[np.searchsorted(c, 0.95)])
ecg_below_500 = float(np.sum(p[f < 500]) / np.sum(p))

# Fraction of ECG-removed burst power in 100-1000 Hz
band = []
for a, b in bursts:
    fb, pb = psd(clean[a:b])
    band.append(np.sum(pb[(fb >= 100) & (fb <= 1000)]) / np.sum(pb))


def ms(v):
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}


metrics = {
    "parameters": {
        "fs_hz": FS,
        "segment_s": N / FS,
        "welch_nperseg": NPERSEG,
        "burst_mask": "ECG-removed signal, 50 ms RMS, p10 + 0.3 (p99 - p10), >= 0.1 s, edge bursts excluded",
    },
    "n_bursts": len(bursts),
    "burst_duration_s": ms([(b - a) / FS for a, b in bursts]),
    "median_freq_hz_with_ecg": ms(rows[:, 1]),
    "median_freq_hz_ecg_removed": ms(rows[:, 2]),
    "median_freq_bias_hz": ms(rows[:, 1] - rows[:, 2]),
    "mean_freq_hz_with_ecg": ms(rows[:, 3]),
    "mean_freq_hz_ecg_removed": ms(rows[:, 4]),
    "mean_freq_bias_hz": ms(rows[:, 3] - rows[:, 4]),
    "median_freq_paired_wilcoxon_p": float(stats.wilcoxon(rows[:, 1], rows[:, 2]).pvalue),
    "median_freq_trend_hz_per_min": {"slope": float(trend.slope), "r": float(trend.rvalue), "p": float(trend.pvalue)},
    "ecg": {"median_freq_hz": ecg_median, "f95_hz": ecg_f95, "power_fraction_below_500hz": ecg_below_500},
    "burst_power_fraction_100_1000hz_ecg_removed": ms(band),
}
mpm.save_metrics(metrics, mpm.figure_dir() / "fig06b_legend_metrics.json")
for k, v in metrics.items():
    if k != "parameters":
        print(k, v)
