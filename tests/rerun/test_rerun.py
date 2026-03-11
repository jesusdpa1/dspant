#!/usr/bin/env python3
"""
Simple test script for Rerun with 10M point Polars dataframe.
Creates seconds x channels data and sends using rr.send_columns.
Push it to the limit with 40 channels!
"""

# %%
from __future__ import annotations

import numpy as np
import polars as pl
import rerun as rr
import rerun.blueprint as rrb

# %%
print("Creating 400M point dataset (10M time points × 40 channels)...")

# Initialize Rerun
rr.init("rerun_polars_40ch_test", spawn=True)

# Create 10M time points
times = np.arange(10_000_000, dtype=np.int64)  # Use int64 for sequence

print("Generating 40 channels with vectorized operations...")
channel_count = 10


# Vectorized channel generation using map
def generate_channel(n: int) -> np.ndarray:
    """Generate a single channel with unique frequency and Y-offset"""
    frequency = 0.0001 + (n * 0.0002)  # Different frequency for each channel
    y_offset = n * 4  # 4 unit spacing between channels

    # Alternate between sin and cos for variety
    if n % 2 == 0:
        signal = np.sin(2 * np.pi * frequency * times)
    else:
        signal = np.cos(2 * np.pi * frequency * times)

    # Add noise and offset
    return signal + np.random.normal(0, 0.1, len(times)) + y_offset


# Generate all channels using map (vectorized)
print("Using map to vectorize channel generation...")
channel_data = list(map(generate_channel, range(channel_count)))

# Create channel names and data dictionary
channel_names = [f"channel.{i + 1:02d}" for i in range(channel_count)]
data_dict = dict(zip(channel_names, channel_data))

# Create Polars DataFrame
print("Creating Polars DataFrame...")
df = pl.DataFrame(data_dict)

print(
    f"Created Polars DataFrame: {df.height} × {df.width} = {df.height * df.width:,} points"
)
print(f"Memory usage: ~{(df.height * df.width * 8) / (1024**3):.2f} GB (float64)")
print(f"DataFrame schema preview: {list(df.schema.keys())[:5]}... (showing first 5)")

# Set up blueprint with custom time range and scalar axis
print("Setting up blueprint...")
blueprint = rrb.Blueprint(
    rrb.TimeSeriesView(
        origin="scalars",
        # Set custom Y axis range (adjusted for 40 offset channels)
        axis_y=rrb.ScalarAxis(range=(-5.0, 160.0), zoom_lock=True),
        # Set visible time range (show first 100k points for performance)
        time_ranges=rrb.VisibleTimeRange(
            timeline="step",
            start=rrb.TimeRangeBoundary.cursor_relative(0),
            end=rrb.TimeRangeBoundary.cursor_relative(100_000),
        ),
        # Configure plot legend
        plot_legend=rrb.PlotLegend(visible=True),
    )
)

# Send the blueprint first
rr.send_blueprint(blueprint)

# Log using send_columns
print("Logging with send_columns...")

# Extract all channel data using Polars (following imu_data structure)
print("Converting to pandas for Rerun compatibility...")
accel = df.select(channel_names).to_pandas()

print("Sending 400M points to Rerun...")
rr.send_columns(
    "scalars",
    indexes=[rr.TimeColumn("step", sequence=times)],
    columns=rr.Scalars.columns(scalars=accel),
)

print("🎉 DONE! Check Rerun viewer - you're pushing the limits!")
print(f"- Using Polars DataFrame with {channel_count} channels")
print(f"- Total data points: {df.height * df.width:,}")
print(f"- Y-axis range: [-5.0, 160.0] to show all {channel_count} offset channels")
print(f"- Channels spaced 4 units apart (0, 4, 8, 12, ..., {(channel_count - 1) * 4})")
print("- Time range shows first 100,000 points for performance")
print("- Different frequencies and sin/cos alternation for visual variety")
print("- Vectorized generation using map() for efficiency")

# %%
