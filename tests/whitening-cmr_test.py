"""
Functions to extract meps
author: Jesus Penaloza

"""

# %%

import dask
import matplotlib.pyplot as plt
import seaborn as sns

# Add at the top of your script, after imports
from dask.distributed import Client, LocalCluster

from dspant.core.nodes.data import EpocNode, StreamNode
from dspant.core.nodes.stream_processing import ProcessingNode
from dspant.preprocessing.common_reference import create_cmr_processor
from dspant.preprocessing.whiten import create_whitening_processor
from dspant.processing.filters import (
    FilterProcessor,
    create_bandpass_filter,
    create_lowpass_filter,
    create_notch_filter,
)
from dspant.processing.time_frequency import (
    LFCCProcessor,
    MFCCProcessor,
    SpectrogramProcessor,
)
from dspant.processing.transforms import TKEOProcessor

# %%
# Create a local cluster with multiple workers
# Adjust the number of workers based on your CPU cores


sns.set_theme(style="darkgrid")
# %%

base_path = r"E:\jpenalozaa\topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline"

emg_stream_path = base_path + r"\RawG.ant"
hd_stream_path = base_path + r"\HDEG.ant"
# %%
stream_emg = StreamNode(emg_stream_path)
stream_emg.load_metadata()
stream_emg.load_data()
# Print stream_emg summary
stream_emg.summarize()
# %%
stream_hd = StreamNode(hd_stream_path)
stream_hd.load_metadata()
stream_hd.load_data()
# Print stream summary
stream_hd.summarize()

# %%
processor_hd = ProcessingNode(stream_hd)
notch = FilterProcessor(filter_func=create_notch_filter(60), overlap_samples=1200)

# Add some processors
bandpass = FilterProcessor(
    create_bandpass_filter(lowcut=300, highcut=8000), overlap_samples=1200
)


cmr = create_cmr_processor()

processor_hd.add_processor([notch, bandpass], group="filters")
processor_hd.add_processor(cmr, group="preprocessing")


# %%
processor_hd.summarize()
# %%
a = processor_hd.process()

# %%

plt.plot(a[0:20000, 0])
# %%
whitening = create_whitening_processor()
processor_hd.add_processor(whitening, group="preprocessing")
w = processor_hd.process()
# %%
whiten_ = w[0:20000, :].compute()
# %%
plt.plot(whiten_[0:20000, 2])

# %%
plt.plot(whiten_[:, 1])
# %%
