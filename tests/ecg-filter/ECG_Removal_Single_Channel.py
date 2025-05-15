#%% 
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
import scipy.signal
from scipy.signal import find_peaks, correlate, medfilt, convolve

# Fix Intel OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()
# Add src repo
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
# emgonset repo imports
from emgonset.processing.filters import create_bandpass_filter, create_notch_filter
from emgonset.utils.io import EMGData
from emgonset.visualization.general_plots import plot_multi_channel_data

# === File paths ===
data_dir = Path(os.getenv("DATA_DIR"))
data_path = data_dir.joinpath(
    r"X:\data\becca\drv_00_baseline_25-02-26_9881-2_testSubject_topoMapping"
)
reference_channel_path = data_path.joinpath("referenceChannel.ant")
emg_path = data_path.joinpath("RawG.ant")

# === Load data A ===
ref_ = EMGData(reference_channel_path)
emg_ = EMGData(emg_path)
fs = ref_.fs
ref_data = ref_.load_data()
emg_data = emg_.load_data()
# Use channel 0
ref_signal = ref_data[:, 0]
emg_signal = emg_data[:, 0] # 1st channel only 

# === Filtering ===
notch = create_notch_filter(60)
bp = create_bandpass_filter(5, 400)
notch.initialize(fs)
bp.initialize(fs)
ref_filt = bp(notch(ref_signal))
emg_filt = bp(notch(emg_signal))

#%% 
# Parameters
fs = ref_.fs

# ECG subtraction 
window_ms = 60 # Template width in milliseconds (+- 30 ms) 
window_samples = int((window_ms/ 1000) * fs) 
half_win = window_samples // 2

# Burst smoothing
smoothing_ms = 200
smoothing_samples = int((smoothing_ms/ 1000) * fs) 

# R-Peak detection parameters
peak_height = 0.0005
peak_distance = int(0.4 * fs) 


# %%
# View data
img_ref_filtered = plot_multi_channel_data(ref_filt, time_window=[0, 1], fs=fs)

img_emg_filtered = plot_multi_channel_data(emg_filt, time_window=[0, 1], fs=fs)


#%% 
ref_smoothed = medfilt(ref_filt, kernel_size = 31) 

# First pass high confidence peaks
r1_ = find_peaks(ref_smoothed, prominence = 0.0002, distance = int(0.3 * fs))[0]

# Second pass lower prominence, shorter minimum spacing
r2_ = find_peaks(ref_smoothed, prominence=0.0001, distance = int(0.15*fs))[0]

# Merge and deduplicate 
r_peaks = np.unique(np.concatenate([r1_, r2_]))

# Filter out peaks too close to signal edge 
r_peaks = r_peaks[(r_peaks > half_win) & (r_peaks < len(ref_filt)-half_win)]

#Print how many peaks were detected 
print(f"Detected { len(r_peaks)} R-peaks after combined passes." )

# %%
# Step 2 Template creation 
templates = np.array([emg_filt[p - half_win : p + half_win] for p in r_peaks])
template = np.mean(templates, axis=0)
# %%
# Step 3 Template subtraction 
emg_clean = emg_filt.copy()
for p in r_peaks: 
    segment = emg_clean[p-half_win : p + half_win]
    corr = correlate(segment, template, mode = 'valid')
    shift = np.argmax(corr) - (len(corr) // 2)
    start = p - half_win + shift 
    end = start + window_samples
    if start > 0 and end < len(emg_clean):
        segment = emg_clean[start:end]
        scale = np.dot(segment, template) / np.dot(template, template)
        emg_clean[start:end] -= scale * template
    

# %%
# Step 4 Burst detection 
rectified = np.abs(emg_clean) 
envelope = convolve(rectified, np.ones(smoothing_samples) / smoothing_samples, mode = 'same')
threshold = envelope.mean() + 1.5 * envelope.std()
burst_mask = envelope > threshold 
burst_signal = emg_clean * burst_mask

# %%
#Step 5 Plot full results
time = np.arange(len(emg_filt)) /fs
plt.figure(figsize = (15, 10))
           
plt.subplot (4, 1, 1) 
plt.plot(time, emg_filt, label = "Filtered EMG Channel 0")
plt.ylabel("Amplitude") 
plt.legend()

plt.subplot(4,1,2) 
plt.plot(time, ref_filt, label = "Reference Filtered")
plt.plot (time[r_peaks], ref_smoothed[r_peaks], "ko", label = "Detected R-Peaks")
plt.ylabel = ("Amplitude") 
plt.legend()

plt.subplot(4, 1, 3) 
plt.plot(time, emg_clean, label = "Attempted ECG-Subtracted EMG") 
plt.ylabel ("Amplitude") 
plt.legend()

plt.subplot(4, 1, 4) 
plt.plot(time, burst_signal, label = "Attempted Detected Bursts")
plt.xlabel = ("Time(s)")
plt.ylabel = ("Amplitude")
plt.legend ()

plt.tight_layout()
plt.suptitle("Burst Detection with ECG Template Subtraction", y =1.02)
plt.show()

         
# %%
#Step 6 plot first 3 seconds 
duration_sec = 3 
samples_to_plot = int (duration_sec * fs) 
time = np.arange(samples_to_plot) / fs 

r_peaks_trimmed = r_peaks[r_peaks < samples_to_plot]

plt.figure(figsize = (15, 10))

plt.figure(figsize=(15, 8))
plt.subplot(3, 1, 1)
plt.plot(time, emg_filt[:samples_to_plot], label="Filtered EMG Channel 0")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, ref_filt[:samples_to_plot], label="Reference Filtered")
plt.plot(time[r_peaks_trimmed], ref_filt[r_peaks_trimmed], 'ko', label="Detected R-Peaks")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, emg_clean[:samples_to_plot], label="ECG-Subtracted EMG")
plt.legend()



plt.tight_layout()
plt.suptitle("Zoomed View: First 3 Seconds", y=1.02)
plt.show()
         



