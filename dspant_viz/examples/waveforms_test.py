# %%
# Waveform Visualization Analysis
# ===============================
#
# This notebook demonstrates a complete workflow for visualizing
# neural waveforms using the dspant_viz package:
#
# 1. Generate synthetic waveform data
# 2. Test individual Waveform components
# 3. Test Matplotlib and Plotly backends
# 4. Test different visualization options

# Import necessary libraries
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Import dspant_viz components
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.visualization.spike.waveforms import WaveformPlot
from dspant_viz.widgets.waveform_inspector import WaveformInspector

# Apply themes
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")

# %%
# Set random seed for reproducibility
np.random.seed(42)


# Generate synthetic waveform data
def generate_waveform_data(
    n_neurons=3, samples_per_waveform=100, n_waveforms=50, sampling_rate=30000.0
):
    """
    Generate synthetic multi-unit waveform data using Dask array.

    Parameters
    ----------
    n_neurons : int, optional
        Number of neurons to simulate
    samples_per_waveform : int, optional
        Number of samples in each waveform
    n_waveforms : int, optional
        Number of waveforms per neuron
    sampling_rate : float, optional
        Sampling rate in Hz

    Returns
    -------
    da.Array
        Synthetic waveform data
    """
    # Create waveforms with some variation
    waveforms = np.zeros((n_neurons, samples_per_waveform, n_waveforms))

    for neuron in range(n_neurons):
        # Base waveform shape
        base_waveform = np.zeros(samples_per_waveform)
        peak_index = samples_per_waveform // 2

        # Create a spike-like waveform with variation
        base_waveform[peak_index - 10 : peak_index + 10] = np.linspace(0, 1, 20)
        base_waveform[peak_index : peak_index + 20] *= -1

        # Add variation for each waveform
        for w in range(n_waveforms):
            noise = np.random.normal(0, 0.1, samples_per_waveform)
            variation_scale = 1 + np.random.uniform(-0.2, 0.2)
            waveforms[neuron, :, w] = base_waveform * variation_scale + noise

    # Convert to Dask array
    return da.from_array(waveforms, chunks=(1, samples_per_waveform, n_waveforms))


# %%
# Generate waveform data
waveforms = generate_waveform_data()

# %%
# 1. Test Individual Waveform Visualization
# -----------------------------------------

# Test individual waveforms for Unit 0
individual_plot = WaveformPlot(
    waveforms=waveforms,
    unit_id=0,
    num_waveforms=10,
    sampling_rate=30000.0,
    template=False,
    normalization="zscore",
    color_mode="colormap",
    colormap="viridis",
    line_width=1.5,
    alpha=0.7,
)

# Plot with Matplotlib
plt.figure(figsize=(10, 5))
fig_mpl, ax_mpl = individual_plot.plot(backend="mpl")
plt.title("Individual Waveforms for Unit 0 (Matplotlib)")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = individual_plot.plot(backend="plotly")
fig_plotly.update_layout(
    title="Individual Waveforms for Unit 0 (Plotly)",
    height=400,
)
fig_plotly.show()

# %%
# 2. Test Template Waveform Visualization
# ---------------------------------------

# Test template waveform for Unit 1
template_plot = WaveformPlot(
    waveforms=waveforms,
    unit_id=1,
    sampling_rate=30000.0,
    template=True,
    normalization="minmax",
    color_mode="single",
    colormap="blue",
    line_width=2.0,
    alpha=0.8,
)

# Plot with Matplotlib
plt.figure(figsize=(10, 5))
fig_mpl, ax_mpl = template_plot.plot(backend="mpl")
plt.title("Template Waveform for Unit 1 (Matplotlib)")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = template_plot.plot(backend="plotly")
fig_plotly.update_layout(
    title="Template Waveform for Unit 1 (Plotly)",
    height=400,
)
fig_plotly.show()

# %%
# 3. Test Different Normalization Methods
# ---------------------------------------

# Test different normalization approaches
normalization_methods = [None, "zscore", "minmax"]

for method in normalization_methods:
    norm_plot = WaveformPlot(
        waveforms=waveforms,
        unit_id=2,
        num_waveforms=15,
        sampling_rate=30000.0,
        template=False,
        normalization=method,
        color_mode="colormap",
        colormap="colorblind",
        line_width=1.0,
        alpha=0.6,
    )

    # Plot with Matplotlib
    plt.figure(figsize=(10, 5))
    fig_mpl, ax_mpl = norm_plot.plot(backend="mpl")
    plt.title(f"Waveforms for Unit 2 - Normalization: {method or 'None'} (Matplotlib)")
    plt.tight_layout()
    plt.show()
# %%

# Create and display the widget
inspector = WaveformInspector(waveforms, backend="plotly")
inspector.display()
# %%
# Summary
# =======
# We've demonstrated a comprehensive workflow for waveform visualization:
#
# 1. Generated synthetic multi-unit waveform data
# 2. Visualized individual waveforms
# 3. Plotted template (mean) waveforms
# 4. Explored different normalization techniques
# 5. Tested both Matplotlib and Plotly backends
#
# This showcases the flexibility of the WaveformPlot
# in dspant_viz for exploring neural waveform characteristics.
# %%
