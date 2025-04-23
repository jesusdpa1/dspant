# src/dspant_neuroproc/visualization/correlogram_plots.py

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_autocorrelogram(
    autocorrelogram: Dict, ax: Optional[plt.Axes] = None, title: Optional[str] = None
) -> plt.Figure:
    """
    Plot an autocorrelogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.bar(
        autocorrelogram["time_bins"],
        autocorrelogram["autocorrelogram"],
        width=autocorrelogram["time_bins"][1] - autocorrelogram["time_bins"][0],
        edgecolor="black",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike Count")
    ax.set_title(title or f"Autocorrelogram for Unit {autocorrelogram['unit_id']}")

    return fig


def plot_crosscorrelogram(
    crosscorrelogram: Dict, ax: Optional[plt.Axes] = None, title: Optional[str] = None
) -> plt.Figure:
    """
    Plot a crosscorrelogram.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.bar(
        crosscorrelogram["time_bins"],
        crosscorrelogram["crosscorrelogram"],
        width=crosscorrelogram["time_bins"][1] - crosscorrelogram["time_bins"][0],
        edgecolor="black",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spike Count")
    ax.set_title(
        title
        or f"Crosscorrelogram between Units {crosscorrelogram['unit1']} and {crosscorrelogram['unit2']}"
    )

    return fig


def plot_all_correlograms(correlogram_results: Dict):
    """
    Plot all computed autocorrelograms and crosscorrelograms.
    """
    # Plot autocorrelograms
    n_autocorr = len(correlogram_results["autocorrelograms"])
    fig_auto, axes_auto = plt.subplots(
        nrows=(n_autocorr + 1) // 2, ncols=2, figsize=(15, 5 * ((n_autocorr + 1) // 2))
    )
    axes_auto = axes_auto.flatten() if n_autocorr > 1 else [axes_auto]

    for i, autocorr in enumerate(correlogram_results["autocorrelograms"]):
        plot_autocorrelogram(autocorr, ax=axes_auto[i])

    # Hide unused subplots
    for j in range(i + 1, len(axes_auto)):
        fig_auto.delaxes(axes_auto[j])

    # Plot crosscorrelograms
    n_cross = len(correlogram_results["crosscorrelograms"])
    if n_cross > 0:
        fig_cross, axes_cross = plt.subplots(
            nrows=(n_cross + 1) // 2, ncols=2, figsize=(15, 5 * ((n_cross + 1) // 2))
        )
        axes_cross = axes_cross.flatten() if n_cross > 1 else [axes_cross]

        for i, crosscorr in enumerate(correlogram_results["crosscorrelograms"]):
            plot_crosscorrelogram(crosscorr, ax=axes_cross[i])

        # Hide unused subplots
        for j in range(i + 1, len(axes_cross)):
            fig_cross.delaxes(axes_cross[j])

    plt.tight_layout()
    return plt
