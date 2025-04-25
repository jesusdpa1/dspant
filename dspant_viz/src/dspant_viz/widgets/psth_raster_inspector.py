from typing import List, Optional, Tuple, Union

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.composites.raster_psth import RasterPSTHComposite


class PSTHRasterInspector:
    """
    Interactive widget for exploring multiple neurons in a raster and PSTH plot.

    This widget allows switching between different neurons using a slider,
    with support for both Matplotlib and Plotly backends.
    """

    def __init__(
        self,
        spike_data: SpikeData,
        bin_width: float = 0.05,
        time_window: Tuple[float, float] = (-1.0, 1.0),
        backend: str = "mpl",
        **kwargs,
    ):
        self.spike_data = spike_data
        self.bin_width = bin_width
        self.time_window = time_window
        self.backend = backend
        self.composite_kwargs = kwargs

        # Ensure unit_ids are sorted and usable in slider
        self.unit_ids = sorted(spike_data.get_unit_ids())
        if not self.unit_ids:
            raise ValueError("No units found in the provided spike data")

        # Create composite with initial unit
        initial_unit_id = self.unit_ids[0]
        self.composite = RasterPSTHComposite(
            spike_data=spike_data,  # Pass entire spike_data
            bin_width=bin_width,
            time_window=time_window,
            unit_id=initial_unit_id,  # Select unit to display
            **self.composite_kwargs,
        )

        # Create interactive widgets
        self._create_widgets()

    def _create_widgets(self):
        """
        Create interactive widgets for unit selection.
        """
        self.unit_slider = widgets.SelectionSlider(
            options=self.unit_ids,
            value=self.unit_ids[0],
            description="Unit:",
            continuous_update=False,
        )

        self.output = widgets.Output()
        self.unit_slider.observe(self._update_plot, names="value")

        with self.output:
            self._render_plot()

    def _update_plot(self, change):
        """
        Update plot when unit is changed.

        Parameters
        ----------
        change : dict
            Widget change event details
        """
        self.output.clear_output(wait=True)
        new_unit_id = change["new"]

        # Update the unit_id in the composite
        self.composite.update(unit_id=new_unit_id)

        with self.output:
            self._render_plot()

    def _render_plot(self):
        """
        Render the plot using the specified backend.
        """
        if self.backend == "mpl":
            fig, _ = self.composite.plot(backend="mpl")
            plt.show()
        elif self.backend == "plotly":
            fig = self.composite.plot(backend="plotly")
            fig.show()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def display(self):
        """
        Display the interactive widget.
        """
        vbox = widgets.VBox([self.unit_slider, self.output])
        display(vbox)
