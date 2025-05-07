"""
Create visualization showing circular template distribution with improved layout and consistent coloring
Author: Jesus Penaloza
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv

from dspant.io.loaders.phy_kilosort_loarder import load_kilosort

sns.set_theme(style="darkgrid")
load_dotenv()

# Load data
data_dir = Path(os.getenv("DATA_DIR"))
sorter_path = data_dir.joinpath(
    r"test_/00_baseline/output_2025-04-01_18-50/testSubject-250326_00_baseline_kilosort4_output/sorter_output"
)
# %%
# Load spike data with templates
sorter_data = load_kilosort(sorter_path, load_templates=True)

# Configure units to analyze
all_unit_ids = sorter_data.unit_ids
templates_to_plot = [37, 15, 30, 40, 25, 10, 7, 8, 42]  # 9 templates for 3x3 grid

# Create a color mapping for consistent colors using colorblind-friendly palette
# Use seaborn's colorblind palette
color_palette = sns.color_palette("colorblind", n_colors=len(templates_to_plot))
# Create a dictionary mapping unit_id to color
unit_color_map = {
    unit_id: color_palette[i] for i, unit_id in enumerate(templates_to_plot)
}


def calculate_template_stats(sorter_data, unit_id):
    """Calculate SEM for templates using spike data."""
    # Get spike clusters for this unit
    spike_mask = sorter_data.spike_clusters == unit_id
    n_spikes = np.sum(spike_mask)

    # Get templates data
    templates = sorter_data.templates_data["templates"]

    # Get template for this unit
    template = templates[unit_id, :, :]

    # Simulate SEM as a percentage of the template's amplitude
    if n_spikes > 0:
        sem_factor = 0.1 / np.sqrt(n_spikes)  # Decreases with more spikes
    else:
        sem_factor = 0.1

    # Calculate SEM for each channel
    sem = np.abs(template) * sem_factor

    return template, sem, n_spikes


def create_circular_template_distribution(
    sorter_data,
    templates_to_plot,
    unit_color_map,
    max_templates_circle=50,
):
    """Create visualization with circular template distribution and 9 templates with SEM."""

    # Create figure with better spacing and adjust width ratios for 3:1 ratio
    fig = plt.figure(figsize=(24, 15))  # Reduced height further

    # Create a more compact grid layout with minimal vertical spacing
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[3, 0.8, 0.8, 0.8],
        height_ratios=[0.8, 0.8, 0.8],
        hspace=0.2,  # Significantly reduced vertical spacing
        wspace=0.2,  # Maintain horizontal spacing
    )

    # Left side: circular template distribution
    ax_circle = fig.add_subplot(gs[:, 0])

    # Get all unit IDs and limit to max_templates_circle for visibility
    circle_units = sorter_data.unit_ids[:max_templates_circle]

    # Create circular distribution
    n_templates = len(circle_units)
    if n_templates > 0:
        # Calculate positions for circular arrangement - larger radius
        radius = 0.65
        angles = np.linspace(0, 2 * np.pi, n_templates, endpoint=False)

        # Set up circle plot - larger limits
        ax_circle.set_xlim(-0.85, 0.85)
        ax_circle.set_ylim(-0.85, 0.85)
        ax_circle.set_aspect("equal")
        ax_circle.set_title(
            "All Unit Templates (Circular Distribution)", fontsize=20, pad=20
        )
        ax_circle.axis("off")

        # Draw circle outline
        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            color="gray",
            linestyle="--",
            alpha=0.9,
            linewidth=2,
        )
        ax_circle.add_artist(circle)

        # Add radial grid lines
        for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            x = [0, radius * 0.9 * np.cos(angle)]
            y = [0, radius * 0.9 * np.sin(angle)]
            ax_circle.plot(x, y, "gray", alpha=0.3, linewidth=1)

        # Add concentric circles
        for r in [radius * 0.33, radius * 0.66]:
            circle = plt.Circle(
                (0, 0), r, fill=False, color="gray", linestyle=":", alpha=0.5
            )
            ax_circle.add_artist(circle)

        # Get templates data
        templates = sorter_data.templates_data["templates"]

        # Plot each template as a small waveform around the circle
        for i, unit_id in enumerate(circle_units):
            angle = angles[i]
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)

            # Get template for this unit
            template = templates[unit_id, :, :]

            # Find the channel with maximum amplitude for visualization
            best_channel_idx = np.argmax(np.max(np.abs(template), axis=0))
            template_1d = template[:, best_channel_idx]

            # Normalize the template for visualization
            template_normalized = (template_1d - template_1d.min()) / (
                template_1d.max() - template_1d.min()
            )
            template_scaled = (template_normalized - 0.5) * 0.12

            # Create mini waveform plot
            t_samples = np.linspace(-0.04, 0.04, len(template_1d))
            waveform_x = x_pos + t_samples
            waveform_y = y_pos + template_scaled

            # Plot the mini waveform with matching color when it's a selected unit
            if unit_id in templates_to_plot:
                line_color = unit_color_map[unit_id]
                line_width = 3
            else:
                line_color = "gray"
                line_width = 1

            ax_circle.plot(
                waveform_x,
                waveform_y,
                color=line_color,
                linewidth=line_width,
                zorder=10 if unit_id in templates_to_plot else 5,
            )

            # Add unit ID text
            text_radius = radius * 0.75
            text_x = text_radius * np.cos(angle)
            text_y = text_radius * np.sin(angle)

            # Choose text color and background to match
            if unit_id in templates_to_plot:
                text_color = "white"
                bbox_color = unit_color_map[unit_id]
                edge_color = "black"
                text_weight = "bold"
                text_size = 11
                text_zorder = 20
            else:
                text_color = "black"
                bbox_color = "lightgray"
                edge_color = "gray"
                text_weight = "normal"
                text_size = 9
                text_zorder = 10

            ax_circle.text(
                text_x,
                text_y,
                str(unit_id),
                ha="center",
                va="center",
                fontsize=text_size,
                color=text_color,
                weight=text_weight,
                zorder=text_zorder,
                bbox=dict(
                    boxstyle="circle,pad=0.3",
                    facecolor=bbox_color,
                    edgecolor=edge_color,
                    alpha=0.9,
                ),
            )

    # Right side: 3x3 grid of templates with SEM (smaller and square)
    for i, unit_id in enumerate(templates_to_plot):
        # Calculate grid position
        row = i // 3
        col = (i % 3) + 1
        ax = fig.add_subplot(gs[row, col])

        # Force the aspect to be equal
        ax.set_box_aspect(1.0)  # This makes the plot square

        # Get template and SEM
        template, sem, n_spikes = calculate_template_stats(sorter_data, unit_id)

        # Find the channel with maximum amplitude
        best_channel_idx = np.argmax(np.max(np.abs(template), axis=0))
        template_1d = template[:, best_channel_idx]
        sem_1d = sem[:, best_channel_idx]

        # Create time axis (in ms)
        fs = 30000  # Typical sampling rate for Kilosort
        n_samples = template.shape[0]
        pre_samples = n_samples // 2
        time_ms = np.arange(-pre_samples, pre_samples + 1) / fs * 1000

        # Get the unit color from our mapping
        unit_color = unit_color_map[unit_id]

        # Plot template with SEM
        ax.plot(
            time_ms,
            template_1d,
            color=unit_color,
            linewidth=2,
            label=f"Unit {unit_id}",
        )

        # Fill SEM area
        ax.fill_between(
            time_ms,
            template_1d - sem_1d,
            template_1d + sem_1d,
            color=unit_color,
            alpha=0.3,
            label="SEM",
        )

        # Mark peak location
        peak_idx = np.argmin(template_1d)
        peak_time = time_ms[peak_idx]
        peak_value = template_1d[peak_idx]
        ax.plot(peak_time, peak_value, "o", color=unit_color, markersize=6)

        # Add vertical line at time 0
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)

        # Set labels and title
        ax.set_xlabel("Time (ms)", fontsize=9, labelpad=1)  # Reduced labelpad
        ax.set_ylabel("Amplitude", fontsize=9, labelpad=1)  # Reduced labelpad

        # Set title with reduced padding
        ax.set_title(
            f"Unit {unit_id}", fontsize=11, fontweight="bold", color=unit_color, pad=2
        )

        ax.grid(True, alpha=0.3)

        # Move legend to inside plot with smaller font and handle length
        ax.legend(fontsize=7, loc="upper right", handlelength=1, handleheight=1)

        # Calculate peak-to-trough width
        trough_idx = peak_idx
        try:
            peak_after_idx = np.argmax(template_1d[trough_idx:]) + trough_idx
            peak_to_trough_width = (peak_after_idx - trough_idx) / fs * 1000
        except:
            peak_to_trough_width = 0

        # Get channel information
        if "channel_map" in sorter_data.templates_data:
            channel_map = sorter_data.templates_data["channel_map"]
            physical_channel = channel_map[best_channel_idx]
        else:
            physical_channel = best_channel_idx

        # Add compact stats text box with minimal padding
        stats_text = (
            f"N:{n_spikes}\nW:{peak_to_trough_width:.1f}ms\nCh:{physical_channel}"
        )
        ax.text(
            0.03,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="wheat", alpha=0.9),
        )

        # Adjust tick parameters to make them more compact
        ax.tick_params(axis="both", which="major", labelsize=8, pad=1)

        # Reduce the number of ticks to avoid crowding
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        # Auto-adjust y limits to focus on the main part of the waveform
        y_min = min(template_1d) * 1.2  # Reduced margin
        y_max = max(template_1d) * 1.2
        ax.set_ylim(y_min, y_max)

        # Adjust margins to reduce whitespace
        ax.margins(x=0.01, y=0.01)

    # Add a legend to the circular plot - made more compact
    legend_elements = []
    for unit_id in templates_to_plot:
        color = unit_color_map[unit_id]
        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=3, label=f"Unit {unit_id}")
        )

    ax_circle.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(5, len(templates_to_plot)),
        fontsize=11,
    )

    # Add main title with less space
    fig.suptitle(
        "Circular Template Distribution with Selected Units", fontsize=22, y=0.99
    )

    # Adjust overall layout to reduce excess space
    plt.tight_layout(rect=[0, 0, 0, 0.5])  # Leave space only for the title

    return fig


# Create the visualization
circular_distribution_fig = create_circular_template_distribution(
    sorter_data=sorter_data,
    templates_to_plot=templates_to_plot,
    unit_color_map=unit_color_map,
)

plt.show()

# %%
