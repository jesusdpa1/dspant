"""
EMG Time Series Visualization in Proper Perspective
Author: jpenalozaa
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mp_plotting_utils as mpu
import numpy as np

# Constants
NUM_CHANNELS = 5
NUM_SAMPLES = 2500
SAMPLING_RATE = 1000  # Hz
TITLE = "EMGtoHD"

# Create random time series data for testing
np.random.seed(42)  # For reproducibility
data = np.zeros((NUM_SAMPLES, NUM_CHANNELS))

# Generate synthetic data with some patterns
t = np.linspace(0, 2.5, NUM_SAMPLES)  # 2.5 seconds of data
for i in range(NUM_CHANNELS):
    # Base signal - sine wave with increasing frequency per channel
    base_freq = 2 + i  # Increasing frequency per channel
    signal = np.sin(2 * np.pi * base_freq * t)

    # Add some noise
    noise = 0.15 * np.random.randn(NUM_SAMPLES)

    # Add a burst in the middle
    burst = 0.5 * np.exp(-((t - (0.3 + 0.15 * i)) ** 2) / (2 * 0.05**2))

    # Combine and add to data matrix
    data[:, i] = signal + noise + burst

# Normalize each channel for better visualization
normalized_data = np.zeros_like(data, dtype=float)
for i in range(NUM_CHANNELS):
    channel_data = data[:, i]
    max_abs = np.max(np.abs(channel_data))
    if max_abs > 0:
        normalized_data[:, i] = channel_data / max_abs * 0.45  # Scale to avoid overlap
    else:
        normalized_data[:, i] = channel_data

# Create time array
time = np.arange(NUM_SAMPLES) / SAMPLING_RATE

# Apply the publication style
mpu.set_publication_style()

# Create figure with custom background color facecolor="#FBF1C8"
fig = plt.figure(
    figsize=(18, 7),
)
# Add the title at the top right
fig.text(1.0, 0.98, TITLE, ha="right", va="top", fontsize=12, fontstyle="italic")

# Create main plot area with the background color
ax = fig.add_subplot(
    111,
)
ax.set_title(
    "EMG Multichannel Recordings in Perspective",
    fontsize=mpu.FONT_SIZES["title"],
    color="darkblue",
)

# Parameters for perspective
vertical_spacing = 1.0  # Vertical spacing between channels
skew_factor = 0.5  # Amount of horizontal skew for perspective

# Calculate the bounds for the perspective box
max_time = time[-1]
max_vert = vertical_spacing * NUM_CHANNELS  # Add an extra position for space
max_horiz_shift = skew_factor * max_vert

# Create vertical positions array for each channel - shifted up by one position
# Start at position 1 instead of 0 to leave space for the x-axis
y_positions = np.arange(1, NUM_CHANNELS + 1) * vertical_spacing

# Draw the perspective box outlines (the dashed lines defining the 3D space)
# Left diagonal
ax.plot([0, max_horiz_shift], [0, max_vert], "k--", linewidth=1, alpha=0.7)
# Right diagonal
ax.plot(
    [max_time, max_time + max_horiz_shift], [0, max_vert], "k--", linewidth=1, alpha=0.7
)
# Top horizontal
ax.plot(
    [max_horiz_shift, max_time + max_horiz_shift],
    [max_vert, max_vert],
    "k--",
    linewidth=1,
    alpha=0.7,
)
# Bottom horizontal (perspective-aligned x-axis)
ax.plot([0, max_time], [0, 0], "k-", linewidth=1)

# Plot each channel with shifted positions
channel_colors = [
    "#2D3142",
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
]  # Dark primary, blue, orange, green, red

for i in range(NUM_CHANNELS):
    # Calculate vertical position (starting from position 1, not 0)
    vert_pos = y_positions[i]

    # Calculate horizontal shift for perspective
    horiz_shift = (i + 1) * skew_factor  # Adjust skew to match vertical position

    # Shift time values for perspective
    shifted_time = time + horiz_shift

    # Shift data vertically
    shifted_data = normalized_data[:, i] + vert_pos

    # Plot time series with appropriate color
    ax.plot(shifted_time, shifted_data, color=channel_colors[i], linewidth=1.5)

    # Add channel label
    ax.text(
        shifted_time[0] - 0.05,
        vert_pos,
        f"Ch {i + 1}",
        ha="right",
        va="center",
        fontsize=mpu.FONT_SIZES["annotation"],
    )

    # Add horizontal reference line (very light)
    ax.axhline(
        y=vert_pos,
        xmin=horiz_shift / max(ax.get_xlim()),
        xmax=1,
        color="lightgray",
        linestyle="-",
        linewidth=0.5,
    )

# Configure axes
ax.set_xlim(-0.3, max_time + max_horiz_shift + 0.3)
ax.set_ylim(-0.6, max_vert + 0.6)

# Hide default axes
ax.yaxis.set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.xaxis.set_visible(False)

# Add custom x-axis ticks and labels that follow the perspective
num_ticks = 6  # Number of tick marks you want
for i in range(num_ticks):
    tick_time = i * (max_time / (num_ticks - 1))

    # Draw tick mark
    ax.plot([tick_time, tick_time], [0, -0.1], "k-", linewidth=1)

    # Add tick label
    ax.text(
        tick_time,
        -0.2,
        f"{tick_time:.1f}",
        ha="center",
        va="top",
        fontsize=mpu.FONT_SIZES["tick_label"],
    )

    # Add perspective dash lines for each tick that extend up through the perspective box
    # Calculate the corresponding points at the top of the perspective
    tick_time_top = tick_time + max_horiz_shift

    # Draw the dashed line in perspective
    ax.plot([tick_time, tick_time_top], [0, max_vert], "k--", linewidth=0.7, alpha=0.3)

# Add time axis label
ax.text(
    max_time / 2,
    -0.5,
    "Time (s)",
    ha="center",
    va="top",
    fontsize=mpu.FONT_SIZES["axis_label"],
    fontweight="bold",
)

# Finalize the figure
mpu.finalize_figure(
    fig, title="EMG Signal Analysis", title_fontsize=mpu.FONT_SIZES["title"]
)

# Show the plot
plt.show()
# %%
