from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from dspant.nodes.sorter import SorterNode

from ..processors.spike_analytics.density_estimation import SpikeDensityEstimator


def plot_spike_density(
    time_bins: np.ndarray,
    spike_density: np.ndarray,
    unit_ids: List[int],
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    time_range: Optional[Tuple[float, float]] = None,
    unit_range: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    sorting: bool = False,
    sort_by: str = "activity",
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot spike density as a heatmap.

    Parameters
    ----------
    time_bins : np.ndarray
        Time bin centers in seconds
    spike_density : np.ndarray
        Smoothed spike density in Hz (shape: time_bins × units)
    unit_ids : list of int
        Unit IDs corresponding to columns in spike_density
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, creates a new figure.
    cmap : str
        Colormap to use for spike density
    vmin, vmax : float or None
        Minimum and maximum values for color scaling
    colorbar : bool
        Whether to show a colorbar
    time_range : tuple of (float, float) or None
        Time range to display (start_time_s, end_time_s)
    unit_range : tuple of (int, int) or None
        Range of units to display (start_idx, end_idx)
    title : str or None
        Title for the plot
    sorting : bool
        Whether to sort units based on a criterion
    sort_by : str
        Criterion for sorting units:
        - "activity": Sort by mean firing rate (highest to lowest)
        - "variance": Sort by variance in firing rate (highest to lowest)
        - "peak_time": Sort by time of peak firing
        - "id": Sort by unit ID (ascending)
        - "id_desc": Sort by unit ID (descending)
    **kwargs : dict
        Additional arguments passed to imshow

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object with the plotted data
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Turn off grid for this visualization
    ax.grid(False)

    # Set consistent style elements
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Apply time and unit range filters if specified
    if time_range is not None:
        time_mask = (time_bins >= time_range[0]) & (time_bins <= time_range[1])
        time_bins = time_bins[time_mask]
        spike_density = spike_density[time_mask, :]

    if unit_range is not None:
        start_idx, end_idx = unit_range
        spike_density = spike_density[:, start_idx:end_idx]
        plot_unit_ids = unit_ids[start_idx:end_idx]
    else:
        plot_unit_ids = unit_ids.copy()  # Make a copy to avoid modifying the original

    # Apply sorting if requested
    if sorting:
        # Create array of indices for sorting
        indices = np.arange(spike_density.shape[1])

        if sort_by == "activity":
            # Sort by mean activity (highest to lowest)
            sort_values = np.mean(spike_density, axis=0)
            sorted_indices = np.argsort(-sort_values)  # Negative for descending order
        elif sort_by == "variance":
            # Sort by variance (highest to lowest)
            sort_values = np.var(spike_density, axis=0)
            sorted_indices = np.argsort(-sort_values)  # Negative for descending order
        elif sort_by == "peak_time":
            # Sort by time of peak activity
            sort_values = np.argmax(spike_density, axis=0)
            sorted_indices = np.argsort(sort_values)
        elif sort_by == "id":
            # Sort by unit ID (ascending)
            sort_values = np.array(plot_unit_ids)
            sorted_indices = np.argsort(sort_values)
        elif sort_by == "id_desc":
            # Sort by unit ID (descending)
            sort_values = np.array(plot_unit_ids)
            sorted_indices = np.argsort(-sort_values)
        else:
            # Default to no sorting
            sorted_indices = indices

        # Apply the sorting
        spike_density = spike_density[:, sorted_indices]
        plot_unit_ids = [plot_unit_ids[i] for i in sorted_indices]

    # Plot the heatmap
    im = ax.imshow(
        spike_density.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[time_bins[0], time_bins[-1], -0.5, len(plot_unit_ids) - 0.5],
        **kwargs,
    )

    # Add colorbar if requested
    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Firing rate (Hz)", fontsize=14, weight="bold")
        cbar.ax.tick_params(labelsize=12)

    # Set axis labels and title with consistent formatting
    ax.set_xlabel("Time (s)", fontsize=14, weight="bold")
    ax.set_ylabel("Unit", fontsize=14, weight="bold")

    # Set yticks to show unit IDs
    ax.set_yticks(np.arange(len(plot_unit_ids)))
    ax.set_yticklabels(plot_unit_ids)

    if title:
        ax.set_title(title, fontsize=16, weight="bold")

    return fig, ax


def plot_combined_raster_density(
    sorter: SorterNode,
    bin_size_ms: float = 10.0,
    sigma_ms: float = 20.0,
    start_time_s: float = 0.0,
    end_time_s: Optional[float] = None,
    unit_ids: Optional[List[int]] = None,
    raster_alpha: float = 0.5,
    raster_color: str = "#2D3142",  # Dark navy blue
    figsize: Tuple[float, float] = (10, 8),
    density_cmap: str = "viridis",
    time_range: Optional[Tuple[float, float]] = None,
    sorting: bool = False,
    sort_by: str = "activity",
) -> Tuple[Figure, List[Axes]]:
    """
    Create a combined plot with spike raster and density estimate.

    Parameters
    ----------
    sorter : SorterNode
        SorterNode containing spike data
    bin_size_ms : float
        Size of time bins in milliseconds
    sigma_ms : float
        Standard deviation of Gaussian smoothing kernel in milliseconds
    start_time_s : float
        Start time for analysis in seconds
    end_time_s : float or None
        End time for analysis in seconds. If None, use the end of the recording.
    unit_ids : list of int or None
        Units to include. If None, use all units.
    raster_alpha : float
        Alpha transparency for raster dots
    raster_color : str
        Color for raster plot markers
    figsize : tuple of (float, float)
        Figure size in inches
    density_cmap : str
        Colormap for density plot
    time_range : tuple of (float, float) or None
        Time range to display (start_time_s, end_time_s)
    sorting : bool
        Whether to sort units based on a criterion
    sort_by : str
        Criterion for sorting units:
        - "activity": Sort by mean firing rate (highest to lowest)
        - "variance": Sort by variance in firing rate (highest to lowest)
        - "peak_time": Sort by time of peak firing
        - "id": Sort by unit ID (ascending)
        - "id_desc": Sort by unit ID (descending)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : list of matplotlib.axes.Axes
        List of axes objects [raster_ax, density_ax]
    """
    # Create estimator and get density
    estimator = SpikeDensityEstimator(bin_size_ms=bin_size_ms, sigma_ms=sigma_ms)
    time_bins, spike_density, used_unit_ids = estimator.estimate(
        sorter, start_time_s, end_time_s, unit_ids
    )

    # Create figure with two subplots
    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    raster_ax, density_ax = axes

    # Turn off grid for both axes
    raster_ax.grid(False)
    density_ax.grid(False)

    # Show spines
    for ax in axes:
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.tick_params(labelsize=12)

    # Apply sorting if requested
    if sorting:
        # Sort the units
        if unit_ids is None:
            plot_unit_ids = sorter.unit_ids.copy()
        else:
            plot_unit_ids = [u for u in unit_ids if u in sorter.unit_ids]

        # Get indices for sorting based on criterion
        if sort_by == "activity":
            # Sort by mean activity (highest to lowest)
            sort_values = np.mean(spike_density, axis=0)
            sorted_indices = np.argsort(-sort_values)  # Negative for descending order
        elif sort_by == "variance":
            # Sort by variance (highest to lowest)
            sort_values = np.var(spike_density, axis=0)
            sorted_indices = np.argsort(-sort_values)  # Negative for descending order
        elif sort_by == "peak_time":
            # Sort by time of peak activity
            sort_values = np.argmax(spike_density, axis=0)
            sorted_indices = np.argsort(sort_values)
        elif sort_by == "id":
            # Sort by unit ID (ascending)
            sort_values = np.array(used_unit_ids)
            sorted_indices = np.argsort(sort_values)
        elif sort_by == "id_desc":
            # Sort by unit ID (descending)
            sort_values = np.array(used_unit_ids)
            sorted_indices = np.argsort(-sort_values)
        else:
            # Default to no sorting
            sorted_indices = np.arange(len(used_unit_ids))

        # Apply sorting to unit IDs
        sorted_unit_ids = [used_unit_ids[i] for i in sorted_indices]
        plot_unit_ids = sorted_unit_ids
    else:
        # No sorting
        if unit_ids is None:
            plot_unit_ids = sorter.unit_ids
        else:
            plot_unit_ids = [u for u in unit_ids if u in sorter.unit_ids]

    # Convert start/end times to samples
    sampling_rate = sorter.sampling_frequency
    start_frame = int(start_time_s * sampling_rate)
    if end_time_s is not None:
        end_frame = int(end_time_s * sampling_rate)
    else:
        end_frame = None

    # Plot raster in the sorted order
    for i, unit_id in enumerate(plot_unit_ids):
        spike_train = sorter.get_unit_spike_train(
            unit_id, start_frame=start_frame, end_frame=end_frame
        )
        spike_times_s = spike_train / sampling_rate
        raster_ax.plot(
            spike_times_s,
            i * np.ones_like(spike_times_s),
            "|",
            markersize=4,
            alpha=raster_alpha,
            color=raster_color,
        )

    raster_ax.set_ylabel("Unit", fontsize=14, weight="bold")
    raster_ax.set_yticks(np.arange(len(plot_unit_ids)))
    raster_ax.set_yticklabels(plot_unit_ids)
    raster_ax.set_title("Spike Raster", fontsize=16, weight="bold")

    # Plot density with the same sorting
    plot_spike_density(
        time_bins,
        spike_density,
        used_unit_ids,
        ax=density_ax,
        cmap=density_cmap,
        time_range=time_range,
        title="Spike Density",
        sorting=sorting,
        sort_by=sort_by,
    )

    # Remove xlabel from density plot since we have a shared x-axis
    density_ax.set_xlabel("")

    # Add overall title with information about sorting if enabled
    title = f"Spike Analysis (bin={bin_size_ms}ms, σ={sigma_ms}ms)"
    if sorting:
        title += f" - Units sorted by {sort_by}"
    plt.suptitle(title, fontsize=18, weight="bold", y=0.98)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle

    return fig, axes
