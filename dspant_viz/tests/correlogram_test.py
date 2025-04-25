# %%
# Crosscorrelogram Analysis
# =========================
#
# This notebook demonstrates a complete workflow for visualizing
# crosscorrelograms using the dspant_viz package:
#
# 1. Generate synthetic unit spike data
# 2. Test individual Crosscorrelogram components
# 3. Test Matplotlib and Plotly backends
# 4. Test the interactive CrosscorrelogramInspector widget

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Import dspant_viz components
from dspant_viz.core.data_models import SpikeData
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.visualization.spike.correlogram import CorrelogramPlot
from dspant_viz.widgets.correlogram_inspector import CorrelogramInspector

# Apply themes
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")

# %%
# Set random seed for reproducibility
np.random.seed(42)


# Generate continuous spike train data for multiple units
def generate_spike_data(duration=30.0, n_units=5, rates=(3, 10)):
    """Generate continuous spike data for multiple units with varying firing rates"""
    spike_data = {}

    # Create units with different firing rates
    for unit_id in range(n_units):
        # Random firing rate between min and max rates
        rate = np.random.uniform(rates[0], rates[1])

        # Generate spikes using Poisson process
        n_spikes = np.random.poisson(rate * duration)
        spikes = np.sort(np.random.uniform(0, duration, n_spikes))

        # Store spikes for this unit
        spike_data[unit_id] = spikes

        print(f"Unit {unit_id}: {n_spikes} spikes, mean rate = {rate:.2f} Hz")

    return spike_data


# %%
# Generate spike data
duration = 30.0  # 30 seconds of recording
spike_data = generate_spike_data(duration=duration, n_units=5)

# Create SpikeData object (continuous spike times)
neural_data = SpikeData(spikes=spike_data)

# %%
# 1. Test Autocorrelogram Components
# ----------------------------------

# Test Autocorrelogram for Unit 0
autocorr_plot = CorrelogramPlot(
    data=neural_data,
    unit01=0,  # First unit
    bin_width=0.01,  # 10 ms bins
    window_size=0.5,  # ±500 ms window
    normalize=True,
)

# Plot with Matplotlib
plt.figure(figsize=(10, 5))
fig_mpl, ax_mpl = autocorr_plot.plot(backend="mpl")
plt.title("Autocorrelogram for Unit 0 (Matplotlib)")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = autocorr_plot.plot(backend="plotly")
fig_plotly.update_layout(
    title="Autocorrelogram for Unit 0 (Plotly)",
    height=400,
)
fig_plotly.show()

# %%
# 2. Test Crosscorrelogram Components
# -----------------------------------

# Test Crosscorrelogram between Unit 0 and Unit 1
crosscorr_plot = CorrelogramPlot(
    data=neural_data,
    unit01=0,  # Reference unit
    unit02=1,  # Comparison unit
    bin_width=0.01,  # 10 ms bins
    window_size=0.5,  # ±500 ms window
    normalize=True,
)

# Plot with Matplotlib
plt.figure(figsize=(10, 5))
fig_mpl, ax_mpl = crosscorr_plot.plot(backend="mpl")
plt.title("Crosscorrelogram between Units 0 and 1 (Matplotlib)")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = crosscorr_plot.plot(backend="plotly")
fig_plotly.update_layout(
    title="Crosscorrelogram between Units 0 and 1 (Plotly)",
    height=400,
)
fig_plotly.show()

# %%
# 3. Test Update Functionality
# ----------------------------

# Update autocorrelogram parameters
autocorr_plot.update(
    unit01=2,  # Change to Unit 2
    bin_width=0.05,  # Wider bins
    window_size=1.0,  # Larger window
)

# Plot updated Matplotlib version
plt.figure(figsize=(10, 5))
fig_mpl, ax_mpl = autocorr_plot.plot(backend="mpl")
plt.title("Updated Autocorrelogram for Unit 2 (Matplotlib)")
plt.tight_layout()
plt.show()

# %%
# 4. Test CrosscorrelogramInspector Widget
# ----------------------------------------

# Create inspector widget with Matplotlib backend
inspector_mpl = CorrelogramInspector(
    spike_data=neural_data,
    backend="mpl",
)

# Display the widget
print("Matplotlib Crosscorrelogram Inspector:")
inspector_mpl.display()

# Create inspector widget with Plotly backend
inspector_plotly = CorrelogramInspector(
    spike_data=neural_data,
    backend="plotly",
)

# Display the widget
print("Plotly Crosscorrelogram Inspector:")
inspector_plotly.display()

# %%
# Summary
# =======
# We've demonstrated a complete workflow for crosscorrelogram analysis:
#
# 1. Generated continuous spike data for multiple units
# 2. Tested autocorrelogram visualization
# 3. Tested crosscorrelogram visualization
# 4. Demonstrated parameter updates
# 5. Created interactive widgets for both Matplotlib and Plotly backends
#
# This showcases the flexibility of the crosscorrelogram visualization
# in dspant_viz for exploring neural spike train relationships.
# %%
