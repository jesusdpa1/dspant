# %%
import os
import time
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dotenv import load_dotenv
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from rastermap import Rastermap
from scipy import stats

from dspant.processors.spatial.common_reference_rs import create_cmr_processor_rs
from dspant.processors.spatial.whiten_rs import create_whitening_processor_rs
from dspant.visualization.general_plots import plot_multi_channel_data
from dspant_emgproc.processors.activity_detection.single_threshold import (
    create_single_threshold_detector,
)
from dspant_neuroproc.processors.spike_analytics.psth import PSTHAnalyzer

sns.set_theme(style="darkgrid")
load_dotenv()
# %%
# Data loading configuration
BASE_DIR = Path(os.getenv("BASE_DIR"))
DATA_DIR = BASE_DIR.joinpath(
    r"hd_paper/small_contacts/25-03-26_4902-1_testSubject_topoMapping"
)

ZARR_PATH = DATA_DIR.joinpath(r"drv_zarr/drv_sliced_5min_00_baseline_emg.zarr")

SORTER_PATH = DATA_DIR.joinpath(
    r"drv_ks4/drv_sliced_5min_00_baseline/ks4_output_2025-06-17_13-43/sorter_output"
)
# %%
SPIKE_TIMES_PATH = SORTER_PATH.joinpath("spike_times.npy")
SPIKE_CLUSTERS_PATH = SORTER_PATH.joinpath("spike_clusters.npy")

# %%
# Load Kilosort spike data
spike_times = np.load(SPIKE_TIMES_PATH)[:10000000]
spike_clusters = np.load(SPIKE_CLUSTERS_PATH)[:10000000]
# cluster_info = np.load("cluster_info.npy")  # or cluster_groups.csv
# %%
from scipy.sparse import csr_array

st_bin = 100000

st = spike_times.squeeze()
clu = spike_clusters.squeeze()
if len(st) != len(clu):
    raise ValueError("spike times and clusters must have same length")
spks = csr_array(
    (np.ones(len(st), "uint8"), (clu, np.floor(st / st_bin * 1000).astype("int")))
)
spike_matrix = spks.todense().astype("float32")

# %%
import dask.array as da
import numpy as np
from scipy.sparse import csr_array

st_bin = 100000

# Convert to dask arrays for chunked processing
st = da.from_array(spike_times.squeeze(), chunks="auto")
clu = da.from_array(spike_clusters.squeeze(), chunks="auto")

if len(st) != len(clu):
    raise ValueError("spike times and clusters must have same length")


# Process in chunks
def create_sparse_chunk(st_chunk, clu_chunk):
    spks = csr_array(
        (
            np.ones(len(st_chunk), "uint8"),
            (clu_chunk, np.floor(st_chunk / st_bin * 1000).astype("int")),
        )
    )
    return spks.todense().astype("float32")


# Map function across chunks and stack results
spike_matrix = da.map_blocks(
    create_sparse_chunk,
    st,
    clu,
    dtype="float32",
    new_axis=[0],  # Adjust based on your output shape
)

# Compute when needed
result = spike_matrix.persist()

# %%
# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.imshow(result[:, 0:10000], aspect="auto", cmap="viridis")
plt.xlabel("Time bins")
plt.ylabel("Neurons (sorted)")
plt.title("Rastermap-sorted neural activity")
plt.show()
# %%

# %%


# %%
