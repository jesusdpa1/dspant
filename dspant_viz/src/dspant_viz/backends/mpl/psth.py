from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_psth(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    psth_data = data["data"]
    time_bins = psth_data["time_bins"]
    firing_rates = psth_data["firing_rates"]
    sem = psth_data["sem"]

    params = data["params"]
    params.update(kwargs)

    line_color = params.get("line_color", "orange")
    line_width = params.get("line_width", 2)
    show_sem = params.get("show_sem", True)
    sem_alpha = params.get("sem_alpha", 0.3)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.plot(time_bins, firing_rates, color=line_color, linewidth=line_width)

    if sem is not None and show_sem:
        sem = np.array(sem)
        rates = np.array(firing_rates)
        t = np.array(time_bins)
        valid = np.isfinite(sem)
        ax.fill_between(
            t[valid],
            rates[valid] - sem[valid],
            rates[valid] + sem[valid],
            color=line_color,
            alpha=sem_alpha,
        )

    ax.set_xlabel("Time from event onset (s)")
    ax.set_ylabel("Firing rate (Hz)")
    if psth_data.get("unit_id") is not None:
        ax.set_title(f"Unit {psth_data['unit_id']} PSTH")

    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    if params.get("show_event_onset", True):
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    if "baseline_window" in params:
        start, end = params["baseline_window"]
        ax.axvspan(start, end, color="gray", alpha=0.2)
        if end - start > 0.05:
            ax.text(
                (start + end) / 2,
                ax.get_ylim()[1] * 0.9,
                "baseline",
                ha="center",
                va="top",
                fontsize=8,
                alpha=0.7,
            )

    if "xlim" in params:
        ax.set_xlim(params["xlim"])
    if "ylim" in params:
        ax.set_ylim(params["ylim"])

    return fig, ax
