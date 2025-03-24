# %%
import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ssqueezepy import Wavelet, cwt, imshow, ssq_cwt
from ssqueezepy.experimental import scale_to_freq
from tqdm import tqdm

from dspant.engine import create_processing_node
from dspant.nodes import StreamNode
from dspant.processor.filters import ButterFilter, FilterProcessor

# Set up styles
os.environ["SSQ_PARALLEL"] = "1"
sns.set_theme(style="darkgrid")
# %%
# ---- STEP 1: Load and preprocess data (using your existing code) ----
# 2413.48640768 seconds
# Replace with your data path
home_path = Path(r"E:\jpenalozaa")
base_path = home_path.joinpath(r"ssq_cwt_data\drv\drv_01_baseline-hemisection")
emg_stream_path = base_path.joinpath(r"RawG.ant")
# %%
# Load EMG data
stream_emg = StreamNode(str(emg_stream_path))
stream_emg.load_metadata()
stream_emg.load_data()
print("EMG data loaded successfully")
stream_emg.summarize()

# Get sampling rate from the stream node
fs = stream_emg.fs

# Create filters for preprocessing
bandpass_filter = ButterFilter("bandpass", (20, 2000), order=4, fs=fs)
notch_filter = ButterFilter("bandstop", (59, 61), order=4, fs=fs)

# Create processing node with filters
processor = create_processing_node(stream_emg)

# Create processors for filtering
notch_processor = FilterProcessor(
    filter_func=notch_filter.get_filter_function(), overlap_samples=40
)
bandpass_processor = FilterProcessor(
    filter_func=bandpass_filter.get_filter_function(), overlap_samples=40
)

# Add filters to the processing node
processor.add_processor([notch_processor, bandpass_processor], group="filters")

# %%

tf_filtered = processor.process().persist()
# %%

start = int(fs * 1355)
end = int(fs * 1360)
sample_data = tf_filtered[start:end, 1]

wavelet = Wavelet()
t = np.linspace(start, end, end - start)


Twxo, Wxo, ssq_freq, scales = ssq_cwt(np.array(sample_data), fs=fs)


# %%
ikw = dict(
    # xticks=t,
    abs=1,
    xlabel="Time [sec]",
    ylabel="Frequency [Hz]",
    norm=(np.min(abs(Twxo)), np.max(abs(Twxo)) / 4),
)
plt.grid(False)
imshow(
    Twxo[50:270, :],
    **ikw,
    yticks=ssq_freq[50:270],
)
# %%

plt.imshow(
    np.abs(Twxo),
    aspect="auto",
    cmap="magma",
    vmin=np.float32(4.789342e-08) * 2,
    vmax=np.float32(5.7670695e-06) / 4,
)


# %%
def viz(x, Tx, Wx):
    plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo")
    plt.show()
    plt.imshow(np.abs(Tx), aspect="auto", vmin=np.abs(Twxo), vmax=0.2, cmap="turbo")
    plt.show()


# %%# Define signal ####################################
N = 2048
t = np.linspace(0, 10, N, endpoint=False)
xo = np.cos(2 * np.pi * 2 * (np.exp(t / 2.2) - 1))
xo += xo[::-1]  # add self reflected
x = xo + np.sqrt(2) * np.random.randn(N)  # add noise

plt.plot(xo)
plt.show()
plt.plot(x)
plt.show()

# %%# CWT + SSQ CWT ####################################
Twxo, Wxo, *_ = ssq_cwt(xo)
viz(xo, Twxo, Wxo)

Twx, Wx, *_ = ssq_cwt(x)
viz(x, Twx, Wx)
