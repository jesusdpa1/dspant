# %%
import matplotlib.pyplot as plt
import numpy as np

from dspant_neuroproc.processors.spike_analytics.correlogram import (
    SpikeCovarianceAnalyzer,
)


class DummySorter:
    def __init__(self, spike_trains, sampling_frequency=30000):
        self.spike_trains = spike_trains
        self.unit_ids = list(spike_trains.keys())
        self.sampling_frequency = sampling_frequency

    def get_unit_spike_train(self, unit_id):
        return self.spike_trains[unit_id]


# Parameters
n_units = 20
duration_sec = 10
sampling_rate = 30000
spikes_per_unit = 100

# Generate spike trains for 20 units
np.random.seed(42)
spike_trains = {
    unit_id: np.sort(
        np.random.randint(0, duration_sec * sampling_rate, spikes_per_unit)
    ).astype(np.int32)
    for unit_id in range(n_units)
}

sorter = DummySorter(spike_trains, sampling_frequency=sampling_rate)
analyzer = SpikeCovarianceAnalyzer(bin_size_ms=1.0, window_size_ms=100.0)

# Compute all autocorrelograms and crosscorrelograms
correlograms = {}
for i in range(n_units):
    for j in range(n_units):
        if i == j:
            result = analyzer.compute_autocorrelogram(sorter, unit_id=i)
        else:
            result = analyzer.compute_crosscorrelogram(sorter, unit1=i, unit2=j)
        correlograms[(i, j)] = result

# %%
# Plot 20x20 grid of correlograms
fig, axs = plt.subplots(n_units, n_units, figsize=(24, 24), sharex=True, sharey=True)
fig.suptitle(
    "Correlograms: Autocorrelograms (diag) and Crosscorrelograms (off-diag)",
    fontsize=16,
)

for i in range(n_units):
    for j in range(n_units):
        ax = axs[i, j]
        result = correlograms[(i, j)]
        if result is None:
            continue

        values = result.get("autocorrelogram", result.get("crosscorrelogram"))
        time_bins = result["time_bins"]
        width = time_bins[1] - time_bins[0]

        ax.bar(time_bins, values, width=width, color="black")
        ax.set_xticks([])
        ax.set_yticks([])

        if i == n_units - 1:
            ax.set_xlabel(f"U{j}", fontsize=6)
        if j == 0:
            ax.set_ylabel(f"U{i}", fontsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# %%
