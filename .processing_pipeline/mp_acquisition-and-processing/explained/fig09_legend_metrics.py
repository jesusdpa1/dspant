"""
/fig09_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 9 legend (Revision 1, BSPC-D-26-03892R1).
Runs fig09_explained_ventilatory_simplified.py unchanged (runpy) and measures the windows it plots
(Neural Trigger OFF: 375-395 s; ON: 0-20 s), using the un-normalized signals:
left EMGdia TKEO envelope (modified TKEO, 4 Hz lowpass) and inspiratory pressure (60 Hz notch, 50 Hz lowpass).

Events: onsets where the signal exceeds p10 + 0.3 (p99 - p10) of that window for >= 0.1 s.
  - Ventilator rate from pressure onsets; EMGdia burst rate from envelope onsets.
  - EMG-to-pressure delay: nearest pressure onset minus EMG onset; "coincident" if |delay| <= 100 ms.
  - Burst amplitude: envelope peak within each burst; variability as coefficient of variation (CV).
"""

# %%
import os
import runpy
from pathlib import Path

import mp_metrics as mpm
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig09_explained_ventilatory_simplified.py"))

FS = float(g["fs"])
env_all = np.asarray(g["tkeo_data"].compute()).ravel()
press_all = np.asarray(g["filtered_insp"].compute()).ravel()
WINDOWS = {"off": (g["OFF_START"], g["OFF_END"]), "on": (g["ON_START"], g["ON_END"])}
COINCIDENT_S = 0.1


def events(x, min_s=0.1):
    p10, p99 = np.percentile(x, [10, 99])
    active = x > p10 + 0.3 * (p99 - p10)
    e = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    return [(a, b) for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]) if (b - a) / FS >= min_s]


def cv(v):
    return float(np.std(v, ddof=1) / np.mean(v))


metrics = {
    "parameters": {
        "fs_hz": FS,
        "emg_side": "left (emg_l)",
        "emg_filter": "bandpass 20-2000 Hz (order 5) + notch 60 Hz (Q 30)",
        "envelope": "modified TKEO, 4 Hz lowpass (min-max normalized per window only for display)",
        "pressure_filter": "notch 60 Hz (Q 30) + 50 Hz lowpass",
        "windows_s": WINDOWS,
        "event_rule": "p10 + 0.3 (p99 - p10), >= 0.1 s",
        "coincident_window_s": COINCIDENT_S,
    }
}
for name, (t0, t1) in WINDOWS.items():
    i0, i1 = int(t0 * FS), int(t1 * FS)
    env, press = env_all[i0:i1], press_all[i0:i1]
    dur_min = (i1 - i0) / FS / 60
    emg_ev, pr_ev = events(env), events(press)
    emg_on = np.array([a for a, _ in emg_ev])
    pr_on = np.array([a for a, _ in pr_ev])
    peaks = np.array([env[a:b].max() for a, b in emg_ev])
    delays = np.array([(pr_on[np.argmin(np.abs(pr_on - o))] - o) / FS for o in emg_on]) if len(pr_on) else np.array([])
    coincident = np.abs(delays) <= COINCIDENT_S
    m = {
        "ventilator_rate_per_min": len(pr_on) / dur_min,
        "emg_burst_rate_per_min": len(emg_on) / dur_min,
        "n_emg_bursts": int(len(emg_on)),
        "n_pressure_breaths": int(len(pr_on)),
        "delay_s_mean": float(np.mean(delays)),
        "delay_s_sd": float(np.std(delays, ddof=1)),
        "pct_emg_bursts_coincident": float(100 * coincident.mean()),
        "burst_peak_cv": cv(peaks),
    }
    # EMG bursts followed by a pressure onset within 0-500 ms (triggered breaths)
    m["pct_emg_followed_by_breath_within_500ms"] = float(100 * np.mean((delays > 0) & (delays <= 0.5)))
    trig = delays[(delays > 0) & (delays <= 0.5)]
    m["delay_s_followed_mean"] = float(np.mean(trig))
    m["delay_s_followed_sd"] = float(np.std(trig, ddof=1))
    m["delays_s_all"] = [round(float(d), 3) for d in delays]
    if coincident.any() and (~coincident).any():
        m["peak_ratio_coincident_vs_not"] = float(np.median(peaks[coincident]) / np.median(peaks[~coincident]))
    metrics[name] = m

mpm.save_metrics(metrics, mpm.figure_dir() / "fig09_legend_metrics.json")
for k in ("off", "on"):
    print(k, {a: round(b, 3) if isinstance(b, float) else b for a, b in metrics[k].items()})
