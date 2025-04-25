# %%
# Multi-Unit Neural Spike Visualization Testing
# ============================================
#
# This notebook demonstrates a complete workflow for visualizing multi-unit neural spike data
# using the dspant_viz package. We'll:
#
# 1. Generate synthetic multi-unit spike data
# 2. Test individual components (RasterPlot, PSTHPlot)
# 3. Test the composite visualization (RasterPSTHComposite)
# 4. Test the interactive widget (PSTHRasterInspector)

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Import dspant_viz components
from dspant_viz.core.data_models import SpikeData
from dspant_viz.core.themes_manager import apply_matplotlib_theme, apply_plotly_theme
from dspant_viz.visualization.composites.raster_psth import RasterPSTHComposite
from dspant_viz.visualization.spike.psth import PSTHPlot
from dspant_viz.visualization.spike.raster import RasterPlot
from dspant_viz.widgets.psth_raster_inspector import PSTHRasterInspector

# Use default themes
apply_matplotlib_theme()  # Uses 'seaborn-darkgrid' by default
apply_plotly_theme()  # Uses 'seaborn' by default

# Specify a specific theme
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")
# %%
# Set random seed for reproducibility
np.random.seed(42)


# Create a function to generate synthetic spike data for multiple units
def generate_synthetic_spikes(
    n_units=2,
    n_trials=20,
    time_window=(-1.0, 1.0),
    baseline_rates=[3, 5],
    response_rates=[20, 35],
    response_start=0.05,
    response_duration=0.2,
    jitter=0.01,
):
    """
    Generate synthetic multi-unit spike data.

    Parameters
    ----------
    n_units : int
        Number of units to generate
    n_trials : int
        Number of trials per unit
    time_window : tuple of (float, float)
        Time window for spike generation (pre_time, post_time)
    baseline_rates : list of float
        Baseline firing rates for each unit (Hz)
    response_rates : list of float
        Response firing rates for each unit (Hz)
    response_start : float
        Time after stimulus onset when response begins (s)
    response_duration : float
        Duration of the response (s)
    jitter : float
        Temporal jitter in the response timing (s)

    Returns
    -------
    dict
        Multi-unit spike data: {unit_id: {trial_id: [spike_times]}}
    """
    pre_time, post_time = time_window
    total_duration = post_time - pre_time

    multi_unit_spikes = {}

    for unit_idx in range(n_units):
        baseline_rate = baseline_rates[unit_idx % len(baseline_rates)]
        response_rate = response_rates[unit_idx % len(response_rates)]

        unit_spikes = {}
        for trial in range(n_trials):
            # Baseline activity (random spikes at baseline_rate)
            n_baseline_spikes = np.random.poisson(baseline_rate * total_duration)
            baseline_spike_times = np.random.uniform(
                pre_time, post_time, n_baseline_spikes
            )

            # Stimulus response (higher firing rate after stimulus onset)
            n_response_spikes = np.random.poisson(response_rate * response_duration)

            # Add jitter to response timing
            response_jitter = response_start + np.random.normal(
                0, jitter, n_response_spikes
            )
            response_spike_times = (
                np.random.uniform(0, response_duration, n_response_spikes)
                + response_jitter
            )

            # Combine and sort all spike times
            all_spikes = np.concatenate([baseline_spike_times, response_spike_times])
            all_spikes = all_spikes[
                (all_spikes >= pre_time) & (all_spikes <= post_time)
            ]
            all_spikes.sort()

            # Store spikes for this trial
            unit_spikes[f"Trial {trial + 1}"] = all_spikes.tolist()

        multi_unit_spikes[unit_idx] = unit_spikes

    return multi_unit_spikes


# %%
# Generate multi-unit spike data
multi_unit_spikes = generate_synthetic_spikes(
    n_units=3,
    n_trials=15,
    baseline_rates=[3, 5, 8],
    response_rates=[25, 40, 60],
    response_start=0.05,
    response_duration=0.2,
    jitter=0.01,
)

# Create SpikeData object
spike_data = SpikeData(spikes=multi_unit_spikes)

# Display a summary of the generated data
print("Generated spike data summary:")
for unit_id, trials in multi_unit_spikes.items():
    total_spikes = sum(len(spikes) for spikes in trials.values())
    print(f"Unit {unit_id}: {len(trials)} trials, {total_spikes} total spikes")

    # Show sample of first few trials
    for trial_id, spikes in list(trials.items())[:2]:
        print(f"  {trial_id}: {len(spikes)} spikes")
    print("  ...")

# %%
# 1. Test the RasterPlot component
# --------------------------------

# Initialize RasterPlot for Unit 0
raster_plot = RasterPlot(
    data=spike_data,
    marker_size=6,
    marker_color="blue",
    marker_alpha=0.8,
    marker_type="|",
    unit_id=0,
)

# Plot with matplotlib
plt.figure(figsize=(10, 5))
fig, ax = raster_plot.plot(backend="mpl")
plt.title("Unit 0 Raster Plot (Matplotlib)")
plt.tight_layout()
plt.show()

# Update to Unit 1 with different color
raster_plot.update(unit_id=1, marker_color="green")
plt.figure(figsize=(10, 5))
fig, ax = raster_plot.plot(backend="mpl")
plt.title("Unit 1 Raster Plot (Matplotlib)")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = raster_plot.plot(backend="plotly")
fig_plotly.update_layout(
    title="Unit 1 Raster Plot (Plotly)",
    height=400,
)
fig_plotly.show()

# %%
# 2. Test the PSTHPlot component
# -----------------------------

# Initialize PSTHPlot for Unit 0
psth_plot = PSTHPlot(
    data=spike_data,
    bin_width=0.05,
    time_window=(-1.0, 1.0),
    line_color="crimson",
    line_width=2,
    show_sem=True,
    sem_alpha=0.3,
    unit_id=0,
)

# Plot with matplotlib
plt.figure(figsize=(10, 5))
fig, ax = psth_plot.plot(backend="mpl")
plt.title("Unit 0 PSTH (Matplotlib)")
plt.tight_layout()
plt.show()

# Update to Unit 2 with different parameters
psth_plot.update(unit_id=2, line_color="purple", bin_width=0.1)
plt.figure(figsize=(10, 5))
fig, ax = psth_plot.plot(backend="mpl")
plt.title("Unit 2 PSTH with 100ms bins (Matplotlib)")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = psth_plot.plot(backend="plotly")
fig_plotly.update_layout(
    title="Unit 2 PSTH with 100ms bins (Plotly)",
    height=400,
)
fig_plotly.show()

# %%
# 3. Test the RasterPSTHComposite component
# ---------------------------------------

# Initialize RasterPSTHComposite for Unit 0
composite = RasterPSTHComposite(
    spike_data=spike_data,
    bin_width=0.05,
    time_window=(-1.0, 1.0),
    raster_color="navy",
    raster_alpha=0.8,
    psth_color="crimson",
    show_sem=True,
    sem_alpha=0.3,
    marker_size=6,
    marker_type="|",
    title="Unit 0 Neural Response",
    show_grid=True,
    raster_height_ratio=2.0,
    unit_id=0,
)

# Plot with matplotlib
fig, axes = composite.plot(backend="mpl")
plt.tight_layout()
plt.show()

# Update to Unit 1
composite.update(
    unit_id=1,
    title="Unit 1 Neural Response",
    raster_color="darkgreen",
    psth_color="orange",
)
fig, axes = composite.plot(backend="mpl")
plt.tight_layout()
plt.show()

# Plot with Plotly
fig_plotly = composite.plot(backend="plotly")
fig_plotly.update_layout(height=600)
fig_plotly.show()

# %%
# 4. Add a baseline window for comparison
# -------------------------------------

# Update parameters to show baseline period
composite.update(time_window=(-1.0, 1.0), title="Unit 1 Response with Baseline")

# Define baseline window
baseline_window = (-0.8, -0.3)  # -800 to -300 ms before stimulus

# Plot with matplotlib and add baseline shading
fig, axes = composite.plot(backend="mpl")

# Add baseline shading to both subplots
for ax in axes:
    ax.axvspan(baseline_window[0], baseline_window[1], color="gray", alpha=0.2)
    ax.text(
        (baseline_window[0] + baseline_window[1]) / 2,
        ax.get_ylim()[1] * 0.9,
        "baseline",
        ha="center",
        va="top",
        fontsize=9,
        color="gray",
    )

plt.tight_layout()
plt.show()

# %%
# 5. Test the PSTHRasterInspector widget
# -----------------------------------
# Create more units for an interesting demo
multi_unit_data = generate_synthetic_spikes(
    n_units=5,
    n_trials=20,
    baseline_rates=[3, 5, 8, 2, 10],
    response_rates=[25, 40, 60, 15, 80],
    response_start=0.05,
    response_duration=0.2,
    jitter=0.01,
)

# Create SpikeData object with multiple units
multi_unit_spike_data = SpikeData(spikes=multi_unit_data)

# Create inspector widget with Matplotlib backend
inspector_mpl = PSTHRasterInspector(
    spike_data=multi_unit_spike_data,
    bin_width=0.05,
    time_window=(-1.0, 1.0),
    backend="mpl",
    raster_color="navy",
    psth_color="crimson",
    show_sem=True,
    raster_height_ratio=2.5,
)

# Display the widget
print("Use the slider below to explore different units:")
inspector_mpl.display()

# %%
# Alternatively, create inspector with Plotly backend for interactive visualization
inspector_plotly = PSTHRasterInspector(
    spike_data=multi_unit_spike_data,
    bin_width=0.05,
    time_window=(-1.0, 1.0),
    backend="plotly",
    raster_color="darkblue",
    psth_color="firebrick",
    show_sem=True,
    raster_height_ratio=2.0,
)

# Display the widget
print("Interactive Plotly version (use slider to switch units):")
inspector_plotly.display()

# %%
# Summary
# =======
# We've demonstrated a complete workflow for multi-unit neural spike visualization:
#
# 1. Created a SpikeData model that stores and organizes multi-unit spike data
# 2. Tested individual visualization components (RasterPlot, PSTHPlot)
# 3. Tested combined visualization (RasterPSTHComposite) showing both raster and PSTH together
# 4. Created an interactive inspector widget for exploring multiple units
#
# This approach allows for efficient visualization and exploration of complex multi-unit
# neural data with support for both static (Matplotlib) and interactive (Plotly) backends.
