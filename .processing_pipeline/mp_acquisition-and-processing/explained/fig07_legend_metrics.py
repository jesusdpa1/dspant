"""
/fig07_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 7 legend (Revision 1, BSPC-D-26-03892R1, Reviewer #3 C9:
TKEO vs. RMS envelope robustness under low SNR). Runs fig07_explained_combined_envelopes.py unchanged (runpy)
and reuses its processors (abs + 50 ms causal moving average, 50 ms causal moving RMS, modified TKEO + 20 Hz
zero-phase Butterworth lowpass).

Analysis window: 30 s starting at the figure segment (5-35 s), channel 0, ECG not removed (as in the figure).
  - Lag: cross-correlation lag of each envelope vs. a zero-phase reference (centered 50 ms moving RMS).
  - ECG false peaks: expiration-period envelope peaks > 50% of the median burst peak.
  - Burst/expiration contrast: 20 log10 for amplitude envelopes (MA, RMS), 10 log10 for TKEO (energy).
  - Low-SNR robustness: band-limited (20-2000 Hz) Gaussian noise added at 20, 10, 5, 0 dB relative to burst RMS
    (3 seeds); onsets detected per envelope with an adaptive threshold p10 + 0.3 (p99 - p10) and a 20 ms hold,
    compared with the same envelope's noise-free onsets (match window ±100 ms).
"""

# %%
import os
import runpy
from pathlib import Path

import dask.array as da
import mp_metrics as mpm
import numpy as np
from scipy import signal


os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig07_explained_combined_envelopes.py"))

FS = float(g["FS"])
CH = int(g["CHANNEL_TO_PLOT"])
s0 = int(g["START"])
x = np.asarray(g["filter_data"][s0 : s0 + int(30 * FS), CH : CH + 1])  # (N, 1)
N = len(x)
WIN = int(g["window_size"])


def envelopes(sig):
    """The three envelopes of Figure 7 E-G, computed with the figure's own processors."""
    d = da.from_array(sig, chunks=(len(sig), 1))
    ma = g["moving_avg_processor"].process(g["abs_processor"].process(d)).compute()[:, 0]
    rms = g["moving_rms_processor"].process(d).compute()[:, 0]
    tk = g["lowpass_processor"].process(g["tkeo_processor"].process(d), fs=FS).compute()[:, 0]
    n = min(len(ma), len(rms), len(tk))
    return {"abs_ma": ma[:n], "rms": rms[:n], "tkeo_lp": tk[:n]}


UNITS_DB = {"abs_ma": 20, "rms": 20, "tkeo_lp": 10}  # amplitude vs energy envelopes

clean_env = envelopes(x)
ref = g["create_moving_rms"](window_size=WIN, center=True).process(da.from_array(x, chunks=(N, 1))).compute()[:, 0]
n = min(len(ref), *(len(v) for v in clean_env.values()))
ref = ref[:n]
clean_env = {k: v[:n] for k, v in clean_env.items()}


def mask_bursts(env, min_s=0.1):
    p10, p99 = np.percentile(env, [10, 99])
    active = env > p10 + 0.3 * (p99 - p10)
    e = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    return [(a, b) for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]) if (b - a) / FS >= min_s]


edge = int(0.5 * FS)
bursts = [(a, b) for a, b in mask_bursts(ref) if a > edge and b < n - edge]
burst = np.zeros(n, bool)
for a, b in bursts:
    burst[a:b] = True
guard = int(0.05 * FS)
expiration = ~np.convolve(burst, np.ones(2 * guard + 1), mode="same").astype(bool)
expiration[:edge] = expiration[-edge:] = False


def zscore(v):
    return (v - v.mean()) / v.std()


lag = {k: mpm.lag_ms(zscore(ref), zscore(v), FS, max_lag_ms=100) for k, v in clean_env.items()}

contrast, false_peaks = {}, {}
for k, v in clean_env.items():
    contrast[k] = float(UNITS_DB[k] * np.log10(v[burst].mean() / v[expiration].mean()))
    burst_peak = np.median([v[a:b].max() for a, b in bursts])
    pk, _ = signal.find_peaks(v, height=0.5 * burst_peak, distance=int(0.1 * FS))
    false_peaks[k] = int(np.sum(expiration[pk]))


# %%
# Low-SNR robustness
def onsets(env, hold_s=0.02):
    hold = int(hold_s * FS)
    return np.array([a for a, b in mask_bursts(env, min_s=hold_s) if b - a >= hold and edge < a < n - edge])


burst_rms = np.sqrt(np.mean(x[:n, 0][burst] ** 2))
sos_band = signal.butter(4, [20, 2000], btype="band", fs=FS, output="sos")
true_onsets = np.array([a for a, _ in bursts])  # from the zero-phase reference envelope (noise-free)


def score(env, k):
    """Onsets of one envelope, lag-corrected, matched to the true burst onsets (±100 ms)."""
    on = onsets(env) - int(round(lag[k] / 1000 * FS))
    errs, matched = [], set()
    for t in true_onsets:
        d = np.abs(on - t) if len(on) else np.array([np.inf])
        j = int(np.argmin(d))
        if d[j] <= 0.1 * FS and j not in matched:
            errs.append(d[j] / FS * 1000)
            matched.add(j)
    return errs, len(true_onsets) - len(errs), len(on) - len(matched)


def summarize(errs, missed, false, n_ref, minutes, contrast_vals):
    return {
        "median_abs_onset_error_ms": float(np.median(errs)) if errs else None,
        "p90_abs_onset_error_ms": float(np.percentile(errs, 90)) if errs else None,
        "missed_pct": 100 * missed / n_ref,
        "false_onsets_per_min": false / minutes,
        "contrast_db": float(np.mean(contrast_vals)),
    }


robust = {}
minutes = n / FS / 60
robust["clean"] = {}
for k, v in clean_env.items():
    e, m, f = score(v, k)
    robust["clean"][k] = summarize(e, m, f, len(true_onsets), minutes, [contrast[k]])
for snr_db in (20, 10, 5, 0):
    acc = {k: {"e": [], "m": 0, "f": 0, "c": []} for k in clean_env}
    for seed in range(3):
        rng = np.random.default_rng(seed)
        noise = signal.sosfiltfilt(sos_band, rng.standard_normal(len(x)))
        noise *= burst_rms / 10 ** (snr_db / 20) / noise.std()
        env_n = envelopes((x + noise[:, None]).astype(x.dtype))  # Rust envelopes expect float32
        for k, v in env_n.items():
            v = v[:n]
            e, m, f = score(v, k)
            acc[k]["e"] += e
            acc[k]["m"] += m
            acc[k]["f"] += f
            acc[k]["c"].append(UNITS_DB[k] * np.log10(v[burst].mean() / v[expiration].mean()))
    robust[f"{snr_db}dB"] = {k: summarize(a["e"], a["m"], a["f"], 3 * len(true_onsets), 3 * minutes, a["c"]) for k, a in acc.items()}

metrics = {
    "parameters": {
        "fs_hz": FS,
        "window_s": n / FS,
        "segment_start_s": s0 / FS,
        "envelope_window_ms": 1000 * WIN / FS,
        "ma_rms": "causal (center=False)",
        "tkeo": "modified, then 4th-order Butterworth 20 Hz lowpass, zero-phase",
        "reference": "centered 50 ms moving RMS",
        "noise": "Gaussian, 4th-order Butterworth 20-2000 Hz (zero-phase), SNR re burst RMS, 3 seeds",
        "onset_rule": "p10 + 0.3 (p99 - p10) of each envelope, >= 20 ms above threshold; lag-corrected; matched to true burst onsets (zero-phase reference) within ±100 ms",
    },
    "n_bursts": len(bursts),
    "lag_ms_vs_zero_phase_reference": lag,
    "burst_expiration_contrast_db": contrast,
    "ecg_false_peaks_in_expiration": false_peaks,
    "expiration_s": float(expiration.sum() / FS),
    "low_snr": robust,
}
mpm.save_metrics(metrics, mpm.figure_dir() / "fig07_legend_metrics.json")
for k, v in metrics.items():
    if k not in ("parameters", "low_snr"):
        print(k, v)
for snr, d in robust.items():
    print(snr, {k: {a: (round(b, 1) if isinstance(b, float) else b) for a, b in r.items()} for k, r in d.items()})
