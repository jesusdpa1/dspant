from typing import Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from dspant_viz.core.internals import public_api
from dspant_viz.core.data_models import SpikeData
from dspant_viz.visualization.composites.raster_psth import RasterPSTHComposite


@public_api(module_override="dspant_viz.widgets")
class PSTHRasterInspector:
    """
    Interactive widget for exploring multiple neurons in a raster and PSTH plot.

    This widget allows switching between different neurons using a slider,
    with support for both Matplotlib and Plotly backends.
    """

    def __init__(
        self,
        spike_data: SpikeData,
        event_times,
        pre_time: float = 1.0,
        post_time: float = 1.0,
        bin_width: float = 0.05,
        backend: str = "plotly",
        **kwargs,
    ):
        self.spike_data = spike_data
        self.event_times = event_times
        self.pre_time = pre_time
        self.post_time = post_time
        self.bin_width = bin_width
        self.backend = backend
        self.composite_kwargs = kwargs

        # Ensure unit_ids are sorted and usable in slider
        self.unit_ids = sorted(spike_data.get_unit_ids())
        if not self.unit_ids:
            raise ValueError("No units found in the provided spike data")

        # Create composite with initial unit
        initial_unit_id = self.unit_ids[0]
        self.composite = RasterPSTHComposite(
            spike_data=spike_data,
            event_times=event_times,
            pre_time=pre_time,
            post_time=post_time,
            bin_width=bin_width,
            unit_id=initial_unit_id,
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
        """
        self.output.clear_output(wait=True)
        new_unit_id = change["new"]

        # Update the composite with the new unit ID
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
