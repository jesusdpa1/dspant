# %%
# Event-Triggered Neural Response Analysis
# ============================================
#
# This notebook demonstrates a complete workflow for visualizing event-triggered
# neural responses using the dspant_viz package:
#
# 1. Generate synthetic unit spike data and event triggers
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

# Apply themes
apply_matplotlib_theme("ggplot")
apply_plotly_theme("seaborn")

# %%
# Set random seed for reproducibility
np.random.seed(42)


# Generate continuous spike train data for multiple units
def generate_spike_data(duration=30.0, n_units=3, rates=(3, 10)):
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


# Generate event times (e.g., stimulus onsets)
def generate_event_times(duration=30.0, n_events=10, min_spacing=1.0):
    """Generate event times with minimum spacing between events"""
    events = []
    last_event = 0

    while len(events) < n_events and last_event < duration:
        # Add minimum spacing plus random additional time
        next_event = last_event + min_spacing + np.random.exponential(2.0)

        if next_event < duration:
            events.append(next_event)
            last_event = next_event
        else:
            break

    return np.array(events)


# %%
# Generate data
duration = 30.0  # 30 seconds of recording
spike_data = generate_spike_data(duration=duration, n_units=3)
event_times = generate_event_times(duration=duration, n_events=15)

# Create SpikeData object (continuous spike times)
neural_data = SpikeData(spikes=spike_data)

print(f"\nGenerated {len(event_times)} events at times: {event_times}")

# %%
# 1. Test the RasterPlot component (trial-based visualization)
# ------------------------------------------------------------

# Parameters for event-triggered analysis
pre_time = 1.0  # 1 second before event
post_time = 1.5  # 1.5 seconds after event

# Initialize RasterPlot for Unit 0
raster_plot = RasterPlot(
    data=neural_data,
    event_times=event_times,  # Add events
    pre_time=pre_time,
    post_time=post_time,
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
    data=neural_data,
    event_times=event_times,  # Add events
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.05,  # 50ms bins
    line_color="crimson",
    line_width=2,
    show_sem=True,
    sem_alpha=0.3,
    unit_id=0,
    sigma=0.1,  # Add smoothing (100ms Gaussian kernel)
)

# Plot with matplotlib
plt.figure(figsize=(10, 5))
fig, ax = psth_plot.plot(backend="mpl")
plt.title("Unit 0 PSTH (Matplotlib)")
plt.tight_layout()
plt.show()

# Update to Unit 2 with different parameters
psth_plot.update(unit_id=2, line_color="purple", bin_width=0.1, sigma=0.15)
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
    spike_data=neural_data,
    event_times=event_times,  # Add events
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.05,
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
    sigma=0.1,  # Add smoothing
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
composite.update(unit_id=2, title="Unit 2 Response with Baseline")

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
# Generate more spike data for an interesting demo
more_units_data = generate_spike_data(duration=duration, n_units=5, rates=(2, 15))

# Create SpikeData object with multiple units
multi_unit_data = SpikeData(spikes=more_units_data)

# Create inspector widget with Matplotlib backend
inspector_mpl = PSTHRasterInspector(
    spike_data=multi_unit_data,
    event_times=event_times,  # Add events
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.05,
    backend="mpl",
    raster_color="navy",
    psth_color="crimson",
    show_sem=True,
    raster_height_ratio=2.5,
    sigma=0.1,  # Add smoothing
)

# Display the widget
print("Use the slider below to explore different units:")
inspector_mpl.display()

# %%
# Alternatively, create inspector with Plotly backend for interactive visualization
inspector_plotly = PSTHRasterInspector(
    spike_data=multi_unit_data,
    event_times=event_times,  # Add events
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.05,
    backend="plotly",
    raster_color="darkblue",
    psth_color="firebrick",
    show_sem=True,
    raster_height_ratio=2.0,
    sigma=0.1,  # Add smoothing
)

# Display the widget
print("Interactive Plotly version (use slider to switch units):")
inspector_plotly.display()

# %%
# 6. Compare different event subsets
# ---------------------------------

# Create first half vs. second half of events comparison
early_events = event_times[event_times < duration / 2]
late_events = event_times[event_times >= duration / 2]

print(f"Early events (n={len(early_events)}): {early_events}")
print(f"Late events (n={len(late_events)}): {late_events}")

# Create two composite plots with different event subsets
early_composite = RasterPSTHComposite(
    spike_data=neural_data,
    event_times=early_events,
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.05,
    title="Early Events Response (Unit 0)",
    unit_id=0,
    raster_color="darkblue",
    psth_color="darkred",
    sigma=0.1,
)

late_composite = RasterPSTHComposite(
    spike_data=neural_data,
    event_times=late_events,
    pre_time=pre_time,
    post_time=post_time,
    bin_width=0.05,
    title="Late Events Response (Unit 0)",
    unit_id=0,
    raster_color="darkgreen",
    psth_color="darkorange",
    sigma=0.1,
)

# Plot both for comparison
fig_early, _ = early_composite.plot(backend="mpl")
plt.tight_layout()
plt.show()

fig_late, _ = late_composite.plot(backend="mpl")
plt.tight_layout()
plt.show()

# %%
# Summary
# =======
# We've demonstrated a complete workflow for event-triggered neural response analysis:
#
# 1. Created continuous spike data and event triggers
# 2. Used event times to create trial-based visualizations from continuous data
# 3. Tested individual components (RasterPlot, PSTHPlot)
# 4. Combined visualizations with the RasterPSTHComposite
# 5. Created an interactive widget for exploring multiple units
# 6. Demonstrated comparing responses to different event subsets
#
# This architecture enables flexible analysis of neural responses to events
# while maintaining a clear separation between continuous and event-triggered
# visualizations.
