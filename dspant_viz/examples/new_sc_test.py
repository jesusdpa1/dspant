# %%
import matplotlib.pyplot as plt

from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.spike.raster import RasterPlot

# %%
# Create mock spike data
data = SpikeData(
    spikes={
        "trial_1": [0.1, 0.5, 1.0],
        "trial_2": [0.2, 0.6, 1.2, 1.5],
        "trial_3": [0.15, 0.7, 1.1],
    },
    unit_id=42,
)

# Create RasterPlot instance
raster_plot = RasterPlot(
    data,
    marker_size=6,
    marker_color="teal",
    marker_alpha=0.8,
    marker_type="|",
    show_event_onset=True,
    show_grid=True,
)
# %%
# Plot with Matplotlib
fig, ax = raster_plot.plot(backend="mpl")
plt.show()

# Plot with Plotly
fig_plotly = raster_plot.plot(backend="plotly")
fig_plotly.show()

# %%
