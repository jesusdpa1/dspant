"""
/fig03_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 3 legend (Revision 1, BSPC-D-26-03892R1).
Runs fig03_explained_noise_filters_ffc.py unchanged (runpy) and measures the arrays it plots,
so the numbers always correspond to the figure.

Metrics (no ground truth exists for in vivo data):
  - Burst SNR (Eq. 1): 10 log10(P_burst / P_expiration), P = mean square, using ONE burst mask
    for all methods (derived from the wavelet-packet output, 50 ms moving RMS).
  - Line-noise power: Welch PSD power in 60·k ± 1 Hz (k = 1..33, up to 2 kHz), relative to raw.
  - Runtime: median wall time over repeats, per second of single-channel data.
"""

# %%
import os
import runpy
import time
from pathlib import Path

import dask.array as da
import mp_metrics as mpm
import numpy as np
from scipy import signal

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig03_explained_noise_filters_ffc.py"))

FS = float(g["FS"])
N = int(g["DATA_END"] - g["DATA_START"])  # 10 s window shown in the figure
outputs = {
    "raw": np.asarray(g["raw_window"]),
    "bandpass_notch": np.asarray(g["filter_data"][:N, 0]),
    "visushrink": np.asarray(g["wavelet_filtered_window"][:N]),
    "ffc": np.asarray(g["ffc_data"][:N, 0]),
    "wavelet_packet": np.asarray(g["whp_data"][:N, 0]),
}

# %%
# Burst / expiration mask from the wavelet-packet output (cleanest trace), shared by all methods
RMS_WIN = int(0.05 * FS)
MIN_BURST_S = 0.1
GUARD_S = 0.05
ref = outputs["wavelet_packet"]
env = np.sqrt(np.convolve(ref**2, np.ones(RMS_WIN) / RMS_WIN, mode="same"))
# checked: the same 6 bursts are found with a 0.2 s window and a midpoint threshold
p10, p99 = np.percentile(env, [10, 99])
thr = p10 + 0.3 * (p99 - p10)
active = env > thr
# keep only supra-threshold runs longer than MIN_BURST_S (rejects isolated ECG spikes)
edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
burst = np.zeros(N, bool)
for s0, s1 in zip(starts, stops):
    if (s1 - s0) / FS >= MIN_BURST_S:
        burst[s0:s1] = True
guard = int(GUARD_S * FS)
expiration = ~np.convolve(burst, np.ones(2 * guard + 1), mode="same").astype(bool)
n_bursts = int(sum((s1 - s0) / FS >= MIN_BURST_S for s0, s1 in zip(starts, stops)))


def burst_snr_db(x):
    return float(10 * np.log10(np.mean(x[burst] ** 2) / np.mean(x[expiration] ** 2)))


# %%
# Line-noise power at 60 Hz harmonics (Welch, 0.5 Hz resolution)
def harmonic_power(x):
    f, p = signal.welch(x, fs=FS, nperseg=int(2 * FS))
    band = np.zeros_like(f, bool)
    for k in range(1, int(2000 // 60) + 1):
        band |= np.abs(f - 60 * k) <= 1.0
    return float(np.sum(p[band]))


raw_harm = harmonic_power(outputs["raw"])

# %%
# Runtime per second of data (single channel, 10 s window, median of repeats)
REPEATS = 5
raw_da = da.from_array(outputs["raw"][:, None], chunks=(N, 1))
bp_da = da.from_array(outputs["bandpass_notch"][:, None], chunks=(N, 1))
runners = {
    "bandpass_notch": lambda: np.asarray(
        g["bandpass_filter"].filter(g["notch_filter"].filter(raw_da, use_rust=True), use_rust=True)
    ),
    "visushrink": lambda: g["wavelet_filter"](outputs["raw"]),
    "ffc": lambda: g["ffc_filter"].process(bp_da, FS).compute(),
    "wavelet_packet": lambda: g["whp_filter"].process(bp_da, FS).compute(),
}
runtime = {}
for name, fn in runners.items():
    fn()  # warm-up (JIT / caches)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    runtime[name] = float(np.median(times) / (N / FS) * 1000)  # ms per s of data

# %%
metrics = {
    "parameters": {
        "fs_hz": FS,
        "window_s": N / FS,
        "bandpass_hz": [20, 2000],
        "bandpass_order": 5,
        "notch_hz": 60,
        "notch_q": 30,
        "wavelet": {"name": "db4", "levels": 6, "threshold": "VisuShrink sigma*sqrt(2 ln N), sigma = MAD(finest detail)/0.6745, soft", "input": "raw"},
        "ffc_hz": 60,
        "wp_fundamental_hz": 60,
        "burst_mask": {"source": "wavelet_packet output", "rms_window_s": 0.05, "threshold": "p10 + 0.3 (p99 - p10) of envelope", "min_burst_s": MIN_BURST_S, "guard_s": GUARD_S, "n_bursts": n_bursts},
        "runtime_repeats": REPEATS,
    },
    "burst_snr_db": {k: burst_snr_db(v) for k, v in outputs.items()},
    "harmonic_power_db_vs_raw": {k: float(10 * np.log10(harmonic_power(v) / raw_harm)) for k, v in outputs.items()},
    "runtime_ms_per_s": runtime,
}
mpm.save_metrics(metrics, mpm.figure_dir() / "fig03_legend_metrics.json")
for k in ("burst_snr_db", "harmonic_power_db_vs_raw", "runtime_ms_per_s"):
    print(k, {n: round(v, 2) for n, v in metrics[k].items()})
print("bursts in mask:", n_bursts, "| burst s:", burst.sum() / FS, "| expiration s:", expiration.sum() / FS)
