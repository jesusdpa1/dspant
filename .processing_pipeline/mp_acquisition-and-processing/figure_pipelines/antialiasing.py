# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

# Set the colorblind-friendly palette
sns.set_palette("colorblind")
palette = sns.color_palette("colorblind")

# Add Montserrat font
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Montserrat"]

# Set the figure size and style
plt.figure(figsize=(12, 9))
sns.set_theme(style="darkgrid")

# Define font sizes with appropriate scaling
TITLE_SIZE = 18
SUBTITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
CAPTION_SIZE = 13

# Time axis with high resolution for the original signal
t = np.linspace(0, 1, 1000)

# Create the original signal (high frequency)
f_original = 15  # Hz
original_signal = np.sin(2 * np.pi * f_original * t)

# Create inadequate sampling points (below Nyquist rate)
sample_rate = 20  # Hz (less than 2*f_original)
t_samples = np.linspace(0, 1, sample_rate)
samples = np.sin(2 * np.pi * f_original * t_samples)

# Create the aliased signal by interpolating through the sample points
interpolation = interp1d(
    t_samples, samples, kind="cubic", bounds_error=False, fill_value="extrapolate"
)
aliased_signal = interpolation(t)

# Create a grid for multiple subplots
gs = GridSpec(2, 1, height_ratios=[1, 1])

# Plot the original high-frequency signal
ax1 = plt.subplot(gs[0])
ax1.plot(t, original_signal, color=palette[0], linewidth=2)
ax1.set_title(
    f"Original Signal (sampling frequency = {f_original} Hz)",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)
ax1.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax1.set_xlim(0, 1)
ax1.set_ylim(-1.2, 1.2)
ax1.tick_params(labelsize=TICK_SIZE)

# Combined plot showing both sampling points and aliased signal
ax2 = plt.subplot(gs[1], sharex=ax1)

# First plot the original signal as a faint background line
ax2.plot(
    t,
    original_signal,
    color=palette[0],
    alpha=0.3,
    linewidth=1,
    label="Original Signal (faint)",
)

# Then plot the aliased signal
ax2.plot(
    t, aliased_signal, color=palette[2], linewidth=2.5, label="Aliased Signal (5 Hz)"
)

# Add dotted vertical lines connecting sampling points to time axis
for i, (t_sample, sample) in enumerate(zip(t_samples, samples)):
    ax2.plot(
        [t_sample, t_sample],
        [0, sample],
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )

# Then plot the sampling points
ax2.plot(
    t_samples, samples, "o", color=palette[3], markersize=8, label="Sampling Points"
)

# Add legend
ax2.legend(loc="upper right", fontsize=TICK_SIZE)

# Set title and labels
ax2.set_title(
    f"Sampling Points and Resulting Aliased Signal",
    fontsize=SUBTITLE_SIZE,
    weight="bold",
)
ax2.set_xlabel("Time (s)", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_ylabel("Amplitude", fontsize=AXIS_LABEL_SIZE, weight="bold")
ax2.set_ylim(-1.2, 1.2)
ax2.tick_params(labelsize=TICK_SIZE)

# Calculate the apparent frequency for the caption
f_alias = sample_rate - f_original

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.subplots_adjust(hspace=0.4)

# Display the plot
plt.show()

# To save the figure
# plt.savefig('aliasing_visualization.png', dpi=300, bbox_inches='tight')
# %%
