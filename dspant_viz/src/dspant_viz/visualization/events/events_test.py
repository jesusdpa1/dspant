from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.internals import public_api


@public_api(module_override="dspant_viz.visualization")
class EventAnnotator(VisualizationComponent):
    """
    Component for annotating events on time series plots across multiple backends.

    This component allows adding event markers, spans, or regions to existing time series visualizations.
    """

    def __init__(
        self,
        events: Union[pl.DataFrame, Dict[str, List[float]]],
        time_mode: str = "seconds",
        highlight_color: str = "red",
        alpha: float = 0.3,
        marker_style: str = "line",
        label_events: bool = True,
        **kwargs,
    ):
        """
        Initialize the EventAnnotator.

        Parameters
        ----------
        events : Union[pl.DataFrame, Dict[str, List[float]]]
            Event data. Can be:
            - Polars DataFrame with columns:
              * Possible start time columns: 'start', 'onset', 'time'
              * Possible end time columns: 'end', 'offset'
              * Possible label columns: 'label', 'name', 'type'
              * Possible channel columns: 'channel', 'channels'
            - Dictionary with event types as keys and lists of times/samples
        time_mode : str, optional
            'seconds' or 'samples' to indicate the time representation
        highlight_color : str, optional
            Color for event highlights
        alpha : float, optional
            Transparency of event highlights
        marker_style : str, optional
            Style of event markers ('line', 'span', 'point')
        label_events : bool, optional
            Whether to add labels to events
        **kwargs
            Additional configuration parameters
        """
        # Standardize input to DataFrame
        if isinstance(events, dict):
            # Convert dictionary to DataFrame
            events_list = []
            for event_type, times in events.items():
                events_list.extend([{"start": t, "label": event_type} for t in times])
            events = pl.DataFrame(events_list)

        # Validate input
        if not isinstance(events, pl.DataFrame):
            raise TypeError("Events must be a Polars DataFrame or dictionary")

        # Flexible column name mapping
        column_mapping = {
            "time": "start",
            "onset": "start",
            "start_time": "start",
            "duration": "end",
            "end_time": "end",
            "offset": "end",
            "name": "label",
            "event_name": "label",
            "type": "label",
            "channels": "channel",
            "channel_id": "channel",
        }

        # Try to rename columns flexibly
        new_column_names = {}
        for old_name in events.columns:
            # Check if the old column name has a standard mapping
            new_name = column_mapping.get(old_name.lower(), old_name)
            if new_name != old_name:
                new_column_names[old_name] = new_name

        # Rename columns if needed
        if new_column_names:
            events = events.rename(new_column_names)

        # Ensure required columns exist with sensible defaults
        if "start" not in events.columns:
            raise ValueError(
                "No start time column found. Columns present: {events.columns}"
            )

        # Add default columns if not present
        if "end" not in events.columns:
            events = events.with_columns(pl.lit(None).alias("end"))

        if "label" not in events.columns:
            events = events.with_columns(pl.lit("Event").alias("label"))

        if "channel" not in events.columns:
            events = events.with_columns(pl.lit(None).alias("channel"))

        super().__init__(events, **kwargs)

        self.events = events
        self.time_mode = time_mode
        self.highlight_color = highlight_color
        self.alpha = alpha
        self.marker_style = marker_style
        self.label_events = label_events

    def get_data(self) -> Dict:
        """
        Prepare event data for rendering.

        Returns
        -------
        dict
            Prepared event data and rendering parameters
        """
        return {
            "data": {
                "events": self.events.to_dict(as_series=False),
                "time_mode": self.time_mode,
            },
            "params": {
                "highlight_color": self.highlight_color,
                "alpha": self.alpha,
                "marker_style": self.marker_style,
                "label_events": self.label_events,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update event annotation parameters.

        Parameters
        ----------
        **kwargs
            Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

    def plot(
        self,
        backend: str = "mpl",
        ax: Optional[Union[plt.Axes, go.Figure]] = None,
        **kwargs,
    ):
        """
        Add event annotations to an existing plot.

        Parameters
        ----------
        backend : str, optional
            Plotting backend ('mpl' or 'plotly')
        ax : Axes or Figure, optional
            Existing plot to annotate
        **kwargs
            Additional rendering parameters

        Returns
        -------
        Annotated plot
        """
        # Import backend-specific renderers
        if backend == "mpl":
            from dspant_viz.backends.mpl.event_annotator import render_event_annotator
        elif backend == "plotly":
            from dspant_viz.backends.plotly.event_annotator import (
                render_event_annotator,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        # Get full data with updated kwargs
        plot_data = self.get_data()
        plot_data["params"].update(kwargs)

        return render_event_annotator(plot_data, ax=ax)
