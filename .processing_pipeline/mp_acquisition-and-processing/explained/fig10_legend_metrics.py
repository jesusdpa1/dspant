"""
/fig10_legend_metrics.py
Author: jpenalozaa
Description: Computes the values quoted in the Figure 10 legend (Revision 1, BSPC-D-26-03892R1).
Runs fig10_explained_stim.py unchanged (runpy) and measures the window it plots (400-500 s).

  - Stimulation monitor (MonA, channel 0): pulses detected as negative (cathodic) peaks below 50% of the
    window minimum, >= 2 ms apart; trains separated by gaps > 50 ms. Reports intra-train pulse frequency,
    pulses and duration per train, train rate, and stimulation start time.
  - Breathing before stimulation: EMGdia burst rate from the TKEO envelope (p10 + 0.3 (p99 - p10), >= 0.1 s)
    between the window start and the first stimulation train.
EMGdia during stimulation contains stimulus artifacts (shown for visualization) and is not quantified.
"""

# %%
import os
import runpy
from pathlib import Path

import mp_metrics as mpm
import numpy as np
from scipy import signal

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
g = runpy.run_path(str(HERE / "fig10_explained_stim.py"))

FS = float(g["FS"])
i0, i1 = int(g["START"]), int(g["END"])
stim = np.asarray(g["STREAM_STIM"].data[i0:i1, 0].compute()).ravel()
env = np.asarray(g["TKEO_DATA"][i0:i1, 1].compute()).ravel()

# %%
# Stimulation pulses and trains
pulses, _ = signal.find_peaks(-stim, height=0.5 * -stim.min(), distance=int(0.002 * FS))
gaps = np.diff(pulses) / FS
breaks = np.where(gaps > 0.05)[0]
train_starts = np.concatenate([[pulses[0]], pulses[breaks + 1]])
train_ends = np.concatenate([pulses[breaks], [pulses[-1]]])
pulses_per_train = np.diff(np.concatenate([[0], breaks + 1, [len(pulses)]]))
intra = gaps[gaps <= 0.05]
complete = slice(0, len(train_starts) - 1)  # last train may be truncated by the window end
train_dur = (train_ends - train_starts)[complete] / FS
train_period = np.diff(train_starts) / FS

# %%
# Breathing rate before stimulation onset
pre = env[: max(train_starts[0] - int(0.5 * FS), 1)]
p10, p99 = np.percentile(pre, [10, 99])
active = pre > p10 + 0.3 * (p99 - p10)
e = np.diff(np.concatenate([[0], active.astype(int), [0]]))
pre_bursts = [(a, b) for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]) if (b - a) / FS >= 0.1]
pre_min = len(pre) / FS / 60


def ms(v):
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}


metrics = {
    "parameters": {
        "fs_hz": FS,
        "window_s": [i0 / FS, i1 / FS],
        "emg_channel": 1,
        "emg_filter": "bandpass 20-2000 Hz (order 5) + notch 60 Hz (Q 30)",
        "envelope": "modified TKEO, 4 Hz lowpass",
        "pulse_rule": "negative peaks < 50% of window minimum, >= 2 ms apart; train gap > 50 ms",
    },
    "stim_start_s_in_window": float(train_starts[0] / FS),
    "n_trains": int(len(train_starts)),
    "intra_train_pulse_freq_hz": float(1 / np.median(intra)),
    "pulses_per_train": ms(pulses_per_train[complete]),
    "train_duration_s": ms(train_dur),
    "train_rate_per_min": float(60 / np.median(train_period)),
    "train_period_s": ms(train_period),
    "pre_stim_breathing_rate_per_min": float(len(pre_bursts) / pre_min),
    "pre_stim_burst_duration_s": ms([(b - a) / FS for a, b in pre_bursts]),
    "pre_stim_window_s": float(len(pre) / FS),
}
mpm.save_metrics(metrics, mpm.figure_dir() / "fig10_legend_metrics.json")
for k, v in metrics.items():
    if k != "parameters":
        print(k, v)
