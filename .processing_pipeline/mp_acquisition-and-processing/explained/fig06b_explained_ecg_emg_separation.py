"""
/fig06b_explained_ecg_emg_separation.py
Author: jpenalozaa
Description: Time-frequency separation of EMGdia and ECG (Revision 1, BSPC-D-26-03892R1, Reviewer #3 C11-C12).
Uses the same signal segment, transforms and color normalization as Figure 6
(fig06_explained_cwt_stft.py is run unchanged via runpy), and the ECG template subtraction
of Figure 4 with identical parameters. Left column: EMGdia after ECG subtraction;
right column: the extracted ECG component (filtered signal minus ECG-subtracted signal).
"""

# %%
import gc
import os
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
from mp_cache import cached
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter
from ssqueezepy import cwt, stft

from dspant.pattern.detection.peak import create_positive_peak_detector
from dspant.pattern.subtraction.correlation import create_correlation_subtractor
from dspant.processors.extractors.template_extractor import extract_template
from dspant.processors.extractors.waveform_extractor import WaveformExtractor
from dspant.processors.filters import FilterProcessor
from dspant.processors.filters.iir_filters import create_lowpass_filter

HERE = Path(__file__).resolve().parent

# %%
# Figure 6 (same recording, filters, 30 s analysis segment, wavelet, STFT settings, display window)
g6 = runpy.run_path(str(HERE / "fig06_explained_cwt_stft.py"))
plt.close("all")
FS = g6["FS"]
filter_data = g6["filter_data"]
START, END = g6["START"], g6["END"]
START_LONG, END_LONG = g6["START_LONG"], g6["END_LONG"]
x = g6["x"]  # filtered signal, channel 0, 30 s (input to the Figure 6 transforms)
wavelet = g6["wavelet"]
STFT_MASK, CWT_MASK = g6["STFT_MASK"], g6["CWT_MASK"]
STFT_FREQS_FILTERED, CWT_FREQS_FILTERED = g6["STFT_FREQS_FILTERED"], g6["CWT_FREQS_FILTERED"]
TIME_LONG = g6["TIME_LONG"]

# Figure 6 normalization bounds (5th / 99th percentile of the ECG-contaminated power)
STFT_BOUNDS = np.percentile(np.abs(g6["STFT_CHANNEL_LONG_FILTERED"]) ** 2, [5, 99])
CWT_BOUNDS = np.percentile(np.abs(g6["CWT_CHANNEL_LONG_FILTERED"]) ** 2, [5, 99])

# %%
# ECG template subtraction - identical parameters to Figure 4
CWT_MARGIN = int(2 * FS)
def _compute_separation():
    lowpass_filter = create_lowpass_filter(200, fs=FS, order=5)
    lowpass_processor = FilterProcessor(filter_func=lowpass_filter.get_filter_function(), overlap_samples=40)
    reference_data = lowpass_processor.process(filter_data, FS)[START:END, 0].persist()
    detector = create_positive_peak_detector(threshold=8.0, threshold_mode="mad", refractory_period=0.1)
    peaks = detector.detect(reference_data, fs=FS)["index"].compute()

    half_win = int((30 / 1000) * FS) // 2  # 30 ms template window
    waveforms = WaveformExtractor(filter_data[:, 0], FS).extract_waveforms(
        peaks, pre_samples=half_win, post_samples=half_win, time_unit="samples"
    )
    template = extract_template(waveforms[0], axis=0)
    clean = (
        create_correlation_subtractor()
        .process(data=filter_data[START:END, :1], template=template[:, :1], indices=peaks, fs=FS, mode=None)
        .compute()[:, 0]
    )
    ecg = x - clean

    # Same transforms as Figure 6 (computed on the 30 s segment, then cut to the displayed window)
    def stft_window(sig):
        return stft(sig)[::-1][:, START_LONG:END_LONG][STFT_MASK, :]

    def cwt_window(sig):
        # computed on the displayed window plus a 2 s margin (memory); Figure 6 scales keep the frequency axis identical
        coeffs, _ = cwt(sig[: END_LONG + CWT_MARGIN], wavelet, scales=g6["scales"])
        out = coeffs[:, START_LONG:END_LONG][CWT_MASK, :].copy()
        del coeffs
        gc.collect()
        return out

    return {
        "r_peaks": np.asarray(peaks),
        "emg_clean": clean,
        "ecg_component": ecg,
        "stft_emg": stft_window(clean),
        "stft_ecg": stft_window(ecg),
        "cwt_emg": cwt_window(clean),
        "cwt_ecg": cwt_window(ecg),
    }


SEPARATION_PARAMS = {
    **g6["TF_PARAMS"],
    "ecg_reference_lowpass": [200, 5],
    "r_peak": {"threshold_mad": 8.0, "refractory_s": 0.1},
    "template_ms": 30,
    "subtraction": "correlation",
    "display_window": [START_LONG, END_LONG],
    "cwt_segment": "display window + 2 s margin, Figure 6 scales",
}
sep = cached("fig06b_separation", SEPARATION_PARAMS, _compute_separation)
r_peak_indices = sep["r_peaks"]
emg_clean, ecg_component = sep["emg_clean"], sep["ecg_component"]
print(f"Detected {len(r_peak_indices)} R-peaks.")


def normalized_power(coeffs, bounds):
    return np.clip((np.abs(coeffs) ** 2 - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)


STFT_EMG = normalized_power(sep["stft_emg"], STFT_BOUNDS)
STFT_ECG = normalized_power(sep["stft_ecg"], STFT_BOUNDS)
CWT_EMG = normalized_power(sep["cwt_emg"], CWT_BOUNDS)
CWT_ECG = normalized_power(sep["cwt_ecg"], CWT_BOUNDS)

# %%
# Plot - for each transform: original (ECG-contaminated, as in Figure 6) followed by the
# separated components (ECG-removed EMGdia on top of the extracted ECG). Template subtraction
# itself is illustrated in Figure 4, so time series are not repeated here.
FONT_SIZE = 16
SUBTITLE_SIZE = int(FONT_SIZE * 0.8)
AXIS_LABEL_SIZE = int(FONT_SIZE * 0.7)
TICK_SIZE = int(FONT_SIZE * 0.65)
COLORMAP = g6["COLORMAP"]
CONTRAST_STFT, CONTRAST_CWT = g6["CONTRAST_STFT"], g6["CONTRAST_CWT"]
MIN_FREQ = g6["MIN_FREQ"]
MAX_FREQ_DISPLAY = 2000  # reduced from Figure 6 (4000 Hz); all EMGdia and ECG power of interest lies below
TIME_POOL = 41  # samples (1.7 ms) per displayed time bin, finer than one rendered pixel at 600 dpi


def pool_time(power, k=TIME_POOL):
    n = power.shape[1] // k * k
    return power[:, :n].reshape(power.shape[0], -1, k).mean(axis=2)


T_POOLED = TIME_LONG[: len(TIME_LONG) // TIME_POOL * TIME_POOL].reshape(-1, TIME_POOL).mean(axis=1)
STFT_ORIG = normalized_power(g6["STFT_CHANNEL_LONG_FILTERED"], STFT_BOUNDS)
CWT_ORIG = normalized_power(g6["CWT_CHANNEL_LONG_FILTERED"], CWT_BOUNDS)

mpu.set_publication_style()
FIG = plt.figure(figsize=(7, 6.5))
# two groups (STFT, CWT), each with three equally spaced rows: original | ECG removed | ECG
outer = GridSpec(2, 2, figure=FIG, width_ratios=[20, 1], hspace=0.22, wspace=0.02)

blocks = [
    ("STFT", STFT_FREQS_FILTERED, STFT_ORIG, STFT_EMG, STFT_ECG, CONTRAST_STFT),
    ("CWT", CWT_FREQS_FILTERED, CWT_ORIG, CWT_EMG, CWT_ECG, CONTRAST_CWT),
]
panels = []
for k, (name, freqs, p_orig, p_emg, p_ecg, gamma) in enumerate(blocks):
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=1)
    g_rows = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[k, 0], hspace=0.15)  # equal spacing within group
    specs = [
        (g_rows[0, 0], "EMGdia with ECG", p_orig, name),
        (g_rows[1, 0], "EMGdia (ECG removed)", p_emg, None),
        (g_rows[2, 0], "ECG", p_ecg, None),
    ]
    for spec, label, power, title in specs:
        ax = FIG.add_subplot(spec)
        mesh = ax.pcolormesh(T_POOLED, freqs, pool_time(power), cmap=COLORMAP, shading="auto", norm=norm, rasterized=True)
        mpu.format_axis(
            ax, title=title, ylabel="Hz", xlim=(0, 5), ylim=(MIN_FREQ, MAX_FREQ_DISPLAY),
            title_fontsize=SUBTITLE_SIZE, label_fontsize=AXIS_LABEL_SIZE, tick_fontsize=TICK_SIZE,
        )
        ax.set_yticks([500, 1000, 1500])
        fmt = ScalarFormatter(useMathText=True)  # scientific notation as in Figure 6
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
        ax.text(0.005, 0.94, label, transform=ax.transAxes, ha="left", va="top", color="white",
                fontsize=TICK_SIZE, fontweight="bold")
        panels.append(ax)
    cax = FIG.add_subplot(outer[k, 1])  # one colorbar per transform
    plt.colorbar(mesh, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cax.tick_params(labelsize=TICK_SIZE)

for ax in panels[:-1]:
    ax.tick_params(labelbottom=False)
panels[-1].set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE)
for ax in panels:
    ax.tick_params(axis="both", pad=-3, labelsize=TICK_SIZE)
    ax.yaxis.set_label_coords(-0.05, 0.5)
for ax in (panels[1], panels[2], panels[4], panels[5]):
    ax.yaxis.get_offset_text().set_visible(False)  # same frequency scale as the top panel of each block (x10^3)
for ax, label in zip(panels, "ABCDEF"):
    mpu.add_panel_label(ax, label, x_offset_factor=0.075, y_offset_factor=0.05, fontsize=SUBTITLE_SIZE)

# %%
FIGURE_TITLE = "fig06b_ecg_emg_time_frequency_separation"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")
mpu.save_figure(FIG, FIGURE_PATH, dpi=600)
plt.show()
print(f"Figure saved to: {FIGURE_PATH}")
