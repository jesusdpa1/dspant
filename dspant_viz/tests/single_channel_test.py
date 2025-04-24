# %%
import numpy as np
import dask.array as da
from dspant_viz.core.base import BasePlotModel
from dspant_viz.widgets.tsx.single_channel_widget import SingleChannelPlotWidget, StreamlitSingleChannelPlot

# %%
# Create large dataset
time = np.linspace(0, 100, 1_000_000)
data = np.sin(time) + np.random.normal(0, 0.1, time.shape)
data_2d = data.reshape(-1, 1)  # Reshape to [samples, channels]

# Convert to Dask array for lazy loading
dask_data = da.from_array(data_2d, chunks=(10000, 1))

# %%
# Create plot model
plot_model = BasePlotModel(
    data=dask_data,
    channel=0,
    fs=10000.0
)
# %%
# Create and show widget
streamlit_plot = StreamlitSingleChannelPlot(plot_model)
streamlit_plot.show()
