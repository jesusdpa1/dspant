"""
/mp_metrics.py
Author: jpenalozaa
Description: Quantitative metrics shared by the manuscript figure scripts.
Every number printed on a figure or quoted in a legend is computed here and
written to a JSON file next to the figure, so legends never drift from code.
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def figure_dir() -> Path:
    """FIGURE_DIR from the environment, falling back to ./img next to the scripts."""
    env = os.getenv("FIGURE_DIR")
    path = Path(env) if env else Path(__file__).resolve().parent / "img"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(metrics: dict, path: Path) -> None:
    """Write metrics as pretty JSON (numpy scalars converted to Python floats)."""

    def _convert(obj):
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    path.write_text(json.dumps(_convert(metrics), indent=2))
    print(f"Metrics saved to: {path}")


# --------------------------------------------------------------------------
# Signal-fidelity metrics (require a known clean reference)
# --------------------------------------------------------------------------
def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """SNR of `estimate` against the known `clean` reference: 10 log10(P_clean / P_error)."""
    error = estimate - clean
    return 10 * np.log10(np.mean(clean**2) / np.mean(error**2))


def delta_snr_db(clean: np.ndarray, raw: np.ndarray, filtered: np.ndarray) -> float:
    """SNR gain of the filtered signal relative to the raw (noisy) input."""
    return snr_db(clean, filtered) - snr_db(clean, raw)


def rmse(clean: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - clean) ** 2)))


def pearson_r(clean: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.corrcoef(clean, estimate)[0, 1])


def fidelity(clean, raw, filtered, trim: int = 0) -> dict:
    """ΔSNR, RMSE and Pearson r, optionally trimming `trim` samples at each edge."""
    sl = slice(trim, len(clean) - trim if trim else None)
    c, x, y = clean[sl], raw[sl], filtered[sl]
    return {
        "snr_in_db": snr_db(c, x),
        "snr_out_db": snr_db(c, y),
        "delta_snr_db": delta_snr_db(c, x, y),
        "rmse": rmse(c, y),
        "pearson_r": pearson_r(c, y),
    }


# --------------------------------------------------------------------------
# Frequency-domain filter characteristics
# --------------------------------------------------------------------------
def magnitude_db(filt, freqs: np.ndarray, fs: float) -> np.ndarray:
    """Magnitude response in dB at `freqs` (Hz). `filt` is SOS array or (b, a)."""
    if isinstance(filt, np.ndarray) and filt.ndim == 2:
        _, h = signal.sosfreqz(filt, worN=freqs, fs=fs)
    else:
        b, a = filt
        _, h = signal.freqz(b, a, worN=freqs, fs=fs)
    return 20 * np.log10(np.maximum(np.abs(h), 1e-12))


def attenuation_db(filt, freq: float, fs: float, passes: int = 1) -> float:
    """Gain (dB, negative = attenuation) at one frequency; passes=2 for filtfilt."""
    return float(passes * magnitude_db(filt, np.array([freq]), fs)[0])


def minus3db_frequency(filt, fs: float, f_lo: float = 0.1, f_hi=None, kind="low") -> float:
    """First frequency where the single-pass response crosses -3 dB."""
    f_hi = f_hi or fs / 2 * 0.999
    freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), 20000)
    mag = magnitude_db(filt, freqs, fs)
    if kind == "low":
        idx = np.argmax(mag <= -3.0103)
    else:  # high-pass: last frequency still below -3 dB
        idx = np.where(mag <= -3.0103)[0][-1]
    return float(freqs[idx])


def rolloff_db_per_octave(filt, f_ref: float, fs: float) -> float:
    """Asymptotic roll-off measured between f_ref and 2·f_ref (dB/octave)."""
    m1, m2 = magnitude_db(filt, np.array([f_ref, 2 * f_ref]), fs)
    return float(m2 - m1)


def group_delay_ms(filt, freq: float, fs: float) -> float:
    """Group delay (ms) at `freq` for a single (causal) pass."""
    if isinstance(filt, np.ndarray) and filt.ndim == 2:
        total = 0.0
        for section in filt:
            _, gd = signal.group_delay((section[:3], section[3:]), w=[freq], fs=fs)
            total += gd[0]
    else:
        _, gd = signal.group_delay(filt, w=[freq], fs=fs)
        total = gd[0]
    return float(total / fs * 1000)


def phase_delay_ms(filt, freq: float, fs: float) -> float:
    """Phase delay (ms) at `freq` for a single pass: -unwrapped phase / (2π f)."""
    freqs = np.linspace(0, freq, 4096)[1:]
    if isinstance(filt, np.ndarray) and filt.ndim == 2:
        _, h = signal.sosfreqz(filt, worN=freqs, fs=fs)
    else:
        b, a = filt
        _, h = signal.freqz(b, a, worN=freqs, fs=fs)
    phase = np.unwrap(np.angle(h))[-1]
    return float(-phase / (2 * np.pi * freq) * 1000)


def group_delay_curve_ms(filt, freqs: np.ndarray, fs: float) -> np.ndarray:
    """Group delay curve (ms) for plotting."""
    return np.array([group_delay_ms(filt, f, fs) for f in freqs])


# --------------------------------------------------------------------------
# Time-domain characteristics
# --------------------------------------------------------------------------
def step_metrics(t: np.ndarray, step_out: np.ndarray, step_time: float, tol: float = 0.02) -> dict:
    """Rise time (10-90 %), overshoot (%) and settling time (±tol band), all relative to step onset."""
    final = step_out[-1]
    after = t >= step_time
    tt, yy = t[after] - step_time, step_out[after]
    t10 = tt[np.argmax(yy >= 0.1 * final)]
    t90 = tt[np.argmax(yy >= 0.9 * final)]
    outside = np.where(np.abs(yy - final) > tol * abs(final))[0]
    settling = tt[outside[-1] + 1] if len(outside) else 0.0
    return {
        "rise_time_ms": (t90 - t10) * 1000,
        "overshoot_pct": max(0.0, (yy.max() - final) / final * 100),
        "settling_time_ms": settling * 1000,
    }


def lag_ms(reference: np.ndarray, delayed: np.ndarray, fs: float, max_lag_ms: float = 100) -> float:
    """Lag (ms) that maximizes cross-correlation; positive = `delayed` lags `reference`."""
    max_lag = int(max_lag_ms / 1000 * fs)
    ref = reference - reference.mean()
    dly = delayed - delayed.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    xc = [np.dot(ref[max(0, -k): len(ref) - max(0, k)], dly[max(0, k): len(dly) - max(0, -k)]) for k in lags]
    return float(lags[int(np.argmax(xc))] / fs * 1000)
