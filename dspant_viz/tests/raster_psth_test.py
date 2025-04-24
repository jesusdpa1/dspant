# %%
# RasterPSTHComposite Testing Notebook
# ===================================
#
# This notebook demonstrates the usage of the RasterPSTHComposite component
# with both Matplotlib and Plotly backends.

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Import dspant_viz components
from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.composites.raster_psth import RasterPSTHComposite

# %% [markdown]
# ## Generate Test Data
#
# First, we'll create some synthetic spike data to simulate neural responses to stimuli.

# %%
# Set random seed for reproducibility
np.random.seed(42)


# Create a function to generate synthetic spike data
def generate_test_spikes(
    n_trials=20,
    pre_time=-1.0,  # seconds before stimulus
    post_time=1.0,  # seconds after stimulus
    baseline_rate=5,  # spikes per second
    response_rate=20,  # spikes per second during response
    response_start=0.05,  # seconds after stimulus
    response_duration=0.2,  # seconds
    jitter=0.01,  # temporal jitter in seconds
):
    """Generate synthetic spike data for testing."""
    total_duration = post_time - pre_time
    trial_spikes = {}

    for trial in range(n_trials):
        # Baseline activity (random spikes at baseline_rate)
        n_baseline_spikes = np.random.poisson(baseline_rate * total_duration)
        baseline_spike_times = np.random.uniform(pre_time, post_time, n_baseline_spikes)

        # Stimulus response (higher firing rate after stimulus onset)
        n_response_spikes = np.random.poisson(response_rate * response_duration)

        # Add jitter to response timing
        response_jitter = response_start + np.random.normal(
            0, jitter, n_response_spikes
        )
        response_spike_times = (
            np.random.uniform(0, response_duration, n_response_spikes) + response_jitter
        )

        # Combine and sort all spike times
        all_spikes = np.concatenate([baseline_spike_times, response_spike_times])
        all_spikes = all_spikes[(all_spikes >= pre_time) & (all_spikes <= post_time)]
        all_spikes.sort()

        # Store spikes for this trial
        trial_spikes[f"Trial {trial + 1}"] = all_spikes.tolist()

    return trial_spikes


# Generate spike data
trial_spikes = generate_test_spikes(
    n_trials=30,
    pre_time=-1.0,
    post_time=1.0,
    baseline_rate=5,
    response_rate=40,
    response_start=0.05,
    response_duration=0.2,
)

# Create SpikeData object
spike_data = SpikeData(spikes=trial_spikes, unit_id=1)

# Display a summary of the generated data
print(f"Generated {len(trial_spikes)} trials with the following spike counts:")
for trial, spikes in list(trial_spikes.items())[:5]:
    print(f"{trial}: {len(spikes)} spikes")
print("...")

# %% [markdown]
# ## Create RasterPSTHComposite Visualization
#
# Now, let's create a RasterPSTHComposite visualization with the generated spike data.

# %%
# Create RasterPSTHComposite
composite = RasterPSTHComposite(
    spike_data=spike_data,
    bin_width=0.05,  # 50 ms bins
    time_window=(-0.8, 0.8),  # -800 to +800 ms
    raster_color="navy",
    raster_alpha=0.8,
    psth_color="crimson",
    show_sem=True,
    sem_alpha=0.3,
    marker_size=4,
    marker_type="|",
    title="Unit 1 Response to Stimulus",
    show_grid=True,
    raster_height_ratio=2.5,  # Make raster plot taller
)

# %% [markdown]
# ## Matplotlib Backend
#
# First, let's visualize using the Matplotlib backend.

# %%
# Plot with Matplotlib backend
fig, axes = composite.plot(backend="mpl")

# Customize further if needed
axes[0].set_ylabel("Trial #")
axes[1].set_xlabel("Time from stimulus onset (s)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Plotly Backend
#
# Now, let's visualize the same data using the Plotly backend for interactive exploration.

# %%
# Plot with Plotly backend
fig_plotly = composite.plot(backend="plotly")

# Update layout for better display in Jupyter
fig_plotly.update_layout(
    height=500,
    width=500,
    title_font_size=16,
)

# Display the interactive plot
fig_plotly.show()

# %% [markdown]
# ## Modifying Parameters
#
# One advantage of the composite design is that we can easily update parameters and regenerate the visualization.

# %%
# Update parameters
composite.update(
    time_window=(-0.5, 0.5),  # Zoom in to -500 to +500 ms
    raster_color="darkgreen",
    psth_color="orange",
    bin_width=0.02,  # Finer bins (20 ms)
    title="Unit 1 Response (Zoomed)",
)

# Plot with updated parameters
fig_updated, axes_updated = composite.plot(backend="mpl")
plt.show()

# %% [markdown]
# ## Adding Baseline Comparison
#
# Let's add a baseline period to the visualization to demonstrate normalization.

# %%
# Update parameters to show normalized PSTH with baseline
composite.update(
    time_window=(-1.0, 1.0),  # Show full range
    normalize_psth=True,  # Normalize PSTH
    title="Unit 1 Response (Normalized to Baseline)",
)

# Create a baseline period
baseline_window = (-0.8, -0.3)  # -800 to -300 ms before stimulus

# Add baseline shading to the visualization
fig_baseline, axes_baseline = composite.plot(backend="mpl")

# Add baseline shading
for ax in axes_baseline:
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

plt.show()

# %% [markdown]
# ## Conclusion
#
# The RasterPSTHComposite provides a cohesive way to visualize spike data with synchronized raster and PSTH plots. The component supports both Matplotlib and Plotly backends, giving users flexibility in their visualization workflow.
#
# Key features demonstrated:
#
# 1. Multi-backend support (Matplotlib and Plotly)
# 2. Easy parameter updates
# 3. Synchronized time axes between raster and PSTH
# 4. Support for SEM visualization
# 5. Flexible customization options
