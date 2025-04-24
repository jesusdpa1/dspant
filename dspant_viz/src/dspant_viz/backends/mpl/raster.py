from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def render_raster(
    data: Dict[str, Any], ax: Optional[Axes] = None, **kwargs
) -> Tuple[Figure, Axes]:
    spike_data = data["data"]
    spike_times = spike_data["spike_times"]
    trial_indices = spike_data["trial_indices"]
    unit_id = spike_data["unit_id"]

    params = data["params"]
    params.update(kwargs)

    marker_size = params.get("marker_size", 4)
    marker_color = params.get("marker_color", "#2D3142")
    marker_alpha = params.get("marker_alpha", 0.7)
    marker_type = params.get("marker_type", "|")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if len(spike_times) > 0:
        ax.scatter(
            spike_times,
            trial_indices,
            marker=marker_type,
            s=marker_size,
            color=marker_color,
            alpha=marker_alpha,
            linewidths=marker_size / 4 if marker_type != "|" else 1,
        )

    ax.set_ylabel("Trial")

    if unit_id is not None:
        ax.set_title(f"Unit {unit_id}")

    if params.get("show_grid", True):
        ax.grid(True, alpha=0.3)

    if params.get("show_event_onset", True):
        ax.axvline(x=0, color="r", linestyle="--", alpha=0.6)

    if "xlim" in params:
        ax.set_xlim(params["xlim"])

    if "ylim" in params:
        ax.set_ylim(params["ylim"])

    return fig, ax
