"""
/fig01_fig02_legend_metrics.py
Author: jpenalozaa
Description: Computes the timing values quoted in the Figure 1 and Figure 2 legends
(Revision 1, BSPC-D-26-03892R1). The figures themselves are unchanged; this script
mirrors the filter parameters of fig01_explained_iir-vs-fir.py and
fig02_explained_combined_filters-phase-delay.py — keep them in sync.
"""

# %%
import mp_metrics as mpm
import numpy as np
from scipy import signal

# Figure 1 parameters (fig01_explained_iir-vs-fir.py)
FIG1_FS = 8000
FIG1_CUTOFF = 50
FIG1_ORDER = 4
FIG1_MA_LENGTH = 161

# Figure 2 panel I parameters (fig02_explained_combined_filters-phase-delay.py)
FIG2_PHASE_FS = 1000
FIG2_PHASE_CUTOFF = 50
FIG2_ORDER = 4

# %%
# Figure 1
butter = signal.butter(FIG1_ORDER, FIG1_CUTOFF / (FIG1_FS / 2), btype="low")
ma = (np.ones(FIG1_MA_LENGTH) / FIG1_MA_LENGTH, np.array([1.0]))

t = np.linspace(0, 1.0, FIG1_FS)  # same time base as the figure script
step = np.zeros_like(t)
step_idx = int(len(t) * 0.2)
step[step_idx:] = 1.0

fig1 = {
    "butterworth": {
        "phase_delay_ms_at_5hz": mpm.phase_delay_ms(butter, 5, FIG1_FS),
        "phase_delay_ms_at_cutoff": mpm.phase_delay_ms(butter, FIG1_CUTOFF, FIG1_FS),
        "group_delay_ms_at_5hz": mpm.group_delay_ms(butter, 5, FIG1_FS),
        "step": mpm.step_metrics(t, signal.lfilter(*butter, step), t[step_idx]),
    },
    "moving_average": {
        "phase_delay_ms_at_5hz": mpm.phase_delay_ms(ma, 5, FIG1_FS),
        "phase_delay_ms_constant": (FIG1_MA_LENGTH - 1) / 2 / FIG1_FS * 1000,
        "minus3db_hz": mpm.minus3db_frequency(ma, FIG1_FS),
        "first_null_hz": FIG1_FS / FIG1_MA_LENGTH,
        "step": mpm.step_metrics(t, signal.lfilter(*ma, step), t[step_idx]),
    },
}

# Figure 1 panel E/F fidelity, original signal (5 + 50 + 200 + 1000 Hz + unseeded noise).
# Reference = 5 Hz component (below both filters' cutoff). The figure uses unseeded noise,
# so fidelity is reported as mean ± SD over N_REALIZATIONS noise draws.
N_REALIZATIONS = 100
EDGE_TRIM = int(0.1 * FIG1_FS)
fig1_clean = np.sin(2 * np.pi * 5 * t)
fig1_mixed = (
    fig1_clean
    + 0.7 * np.sin(2 * np.pi * 50 * t)
    + 0.5 * np.sin(2 * np.pi * 200 * t)
    + 0.3 * np.sin(2 * np.pi * 1000 * t)
)
rng = np.random.default_rng(0)
fid = {"butterworth": [], "moving_average": []}
for _ in range(N_REALIZATIONS):
    noisy = fig1_mixed + 0.1 * rng.standard_normal(len(t))
    fid["butterworth"].append(mpm.fidelity(fig1_clean, noisy, signal.filtfilt(*butter, noisy), EDGE_TRIM))
    fid["moving_average"].append(mpm.fidelity(fig1_clean, noisy, signal.filtfilt(*ma, noisy), EDGE_TRIM))
for name, runs in fid.items():
    fig1[name]["fidelity_vs_5hz_mean"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    fig1[name]["fidelity_vs_5hz_sd"] = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
    filt = butter if name == "butterworth" else ma
    fig1[name]["gain_db_zero_phase"] = {f"{f}hz": mpm.attenuation_db(filt, f, FIG1_FS, passes=2) for f in (5, 50, 200, 1000)}
fig1["butterworth"]["minus3db_hz"] = mpm.minus3db_frequency(butter, FIG1_FS)
fig1["butterworth"]["rolloff_db_per_octave"] = mpm.rolloff_db_per_octave(butter, 4 * FIG1_CUTOFF, FIG1_FS)
dense = np.logspace(np.log10(FIG1_FS / FIG1_MA_LENGTH * 1.01), np.log10(FIG1_FS / 2 * 0.999), 20000)
fig1["moving_average"]["peak_sidelobe_db"] = float(mpm.magnitude_db(ma, dense, FIG1_FS).max())

# %%
# Figure 2 panels A-H (test signals of the figure script: fs 16 kHz, 0.5 s, filtfilt with padlen)
FIG2_FS = 16000
FIG2_NOTCH_Q = 30
FIG2_PADLEN = 4000
t2 = np.linspace(0, 0.5, int(FIG2_FS * 0.5))
trim2 = int(0.1 * FIG2_FS)
base = np.sin(2 * np.pi * 5 * t2)
hf500 = 0.5 * np.sin(2 * np.pi * 500 * t2)
nyq2 = FIG2_FS / 2
fig2_filters = {
    "lowpass": signal.butter(FIG2_ORDER, 100 / nyq2, btype="low"),
    "highpass": signal.butter(FIG2_ORDER, 10 / nyq2, btype="high"),
    "bandpass": signal.butter(FIG2_ORDER, [20 / nyq2, 2000 / nyq2], btype="band"),
    "notch": signal.iirnotch(60 / nyq2, FIG2_NOTCH_Q),
}
cases = {  # (filter, input, clean reference or None, frequencies to report)
    "lowpass": (base + hf500, base, (5, 500)),
    "highpass": (base + hf500, hf500, (5, 500)),
    "bandpass": (base + 0.5 * np.sin(2 * np.pi * 5000 * t2) + 0.7 * np.sin(2 * np.pi * 10 * t2), None, (5, 10, 5000)),
    "notch": (base + 0.8 * np.sin(2 * np.pi * 60 * t2), base, (5, 60)),
}
fig2 = {}
for name, (x_in, clean, freqs) in cases.items():
    filt = fig2_filters[name]
    y = signal.filtfilt(*filt, x_in, padlen=FIG2_PADLEN)
    entry = {"gain_db_zero_phase": {f"{f}hz": mpm.attenuation_db(filt, f, FIG2_FS, passes=2) for f in freqs}}
    if clean is not None:
        entry["fidelity"] = mpm.fidelity(clean, x_in, y, trim2)
    else:
        entry["output_rms_over_input_rms"] = float(np.sqrt(np.mean(y[trim2:-trim2] ** 2) / np.mean(x_in[trim2:-trim2] ** 2)))
    fig2[name] = entry
notch_f = np.linspace(1, 200, 200000)
notch_mag = mpm.magnitude_db(fig2_filters["notch"], notch_f, FIG2_FS)
below = notch_f[notch_mag <= -3.0103]
fig2["notch"]["minus3db_band_hz"] = [float(below.min()), float(below.max())]

# %%
# Figure 2, panel I (same test signal as the figure script)
phase_ba = signal.butter(FIG2_ORDER, FIG2_PHASE_CUTOFF / (FIG2_PHASE_FS / 2), btype="low")
tp = np.linspace(0, 1.0, FIG2_PHASE_FS, endpoint=False)
x = (
    np.sin(2 * np.pi * 5 * tp)
    + 0.5 * np.sin(2 * np.pi * 20 * tp)
    + 0.3 * np.sin(2 * np.pi * 40 * tp)
    + 0.2 * np.sin(2 * np.pi * 100 * tp)
)
for loc in [int(0.25 * FIG2_PHASE_FS), int(0.5 * FIG2_PHASE_FS), int(0.75 * FIG2_PHASE_FS)]:
    x[loc] += 1.5
y_single = signal.lfilter(*phase_ba, x)
y_zero = signal.filtfilt(*phase_ba, x)

fig2["panel_I"] = {
        "phase_delay_ms_at_5hz": mpm.phase_delay_ms(phase_ba, 5, FIG2_PHASE_FS),
        "phase_delay_ms_at_cutoff": mpm.phase_delay_ms(phase_ba, FIG2_PHASE_CUTOFF, FIG2_PHASE_FS),
        "measured_lag_ms_single_pass": mpm.lag_ms(y_zero, y_single, FIG2_PHASE_FS),
        "measured_lag_ms_zero_phase": mpm.lag_ms(x, y_zero, FIG2_PHASE_FS),
}

out = mpm.figure_dir()
mpm.save_metrics(fig1, out / "fig01_legend_metrics.json")
mpm.save_metrics(fig2, out / "fig02_legend_metrics.json")
for name, d in (("Figure 1", fig1), ("Figure 2", fig2)):
    print(name)
    for k, v in d.items():
        print(" ", k, v)
