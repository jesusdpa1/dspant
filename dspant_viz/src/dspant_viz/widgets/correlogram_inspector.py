# src/dspant_viz/widgets/crosscorrelogram_inspector.py
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from dspant_viz.core.data_models import SpikeData
from dspant_viz.core.internals import public_api
from dspant_viz.visualization.spike.correlogram import CorrelogramPlot


@public_api(module_override="dspant_viz.widgets")
class CorrelogramInspector:
    """
    Interactive widget for exploring crosscorrelograms between neural units.

    This widget allows switching between different neurons and toggling
    between autocorrelogram and crosscorrelogram views.
    """

    def __init__(
        self,
        spike_data: SpikeData,
        backend: str = "plotly",
        **kwargs,
    ):
        """
        Initialize the crosscorrelogram inspector.

        Parameters
        ----------
        spike_data : SpikeData
            Spike data containing spike times for different units
        backend : str, optional
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs : dict
            Additional configuration parameters for CorrelogramPlot
        """
        self.data = spike_data
        self.backend = backend
        self.composite_kwargs = kwargs

        # Ensure unit_ids are sorted and usable in slider
        self.unit_ids = sorted(spike_data.get_unit_ids())
        if not self.unit_ids:
            raise ValueError("No units found in the provided spike data")

        # Create initial crosscorrelogram with first unit
        initial_unit_id = self.unit_ids[0]
        self.crosscorrelogram = CorrelogramPlot(
            data=spike_data,
            unit01=initial_unit_id,
            unit02=None,  # Start with autocorrelogram
            **self.composite_kwargs,
        )

        # Create interactive widgets
        self._create_widgets()

    def _create_widgets(self):
        """
        Create interactive widgets for unit selection and crosscorrelogram toggle.
        """
        # Unit 1 (Reference) Slider
        self.unit1_slider = widgets.SelectionSlider(
            options=self.unit_ids,
            value=self.unit_ids[0],
            description="Unit 1:",
            continuous_update=False,
        )

        # Crosscorrelogram Toggle Checkbox
        self.cross_toggle = widgets.Checkbox(
            value=False, description="Cross-correlogram", disabled=False, indent=False
        )

        # Unit 2 (Comparison) Slider
        self.unit2_slider = widgets.SelectionSlider(
            options=self.unit_ids,
            value=self.unit_ids[0],
            description="Unit 2:",
            continuous_update=False,
            disabled=True,  # Initially disabled
            style={"description_width": "initial"},
        )

        # Style for disabled slider
        self.disabled_style = {"description_color": "gray"}
        self.enabled_style = {"description_color": "black"}

        # Output area for the plot
        self.output = widgets.Output()

        # Link widgets to update methods
        self.unit1_slider.observe(self._update_plot, names="value")
        self.cross_toggle.observe(self._toggle_crosscorrelogram, names="value")
        self.unit2_slider.observe(self._update_plot, names="value")

        # Initial plot
        with self.output:
            self._render_plot()

    def _toggle_crosscorrelogram(self, change):
        """
        Toggle between autocorrelogram and crosscorrelogram.
        """
        is_cross = change["new"]

        # Enable/disable unit2 slider
        self.unit2_slider.disabled = not is_cross

        # Update slider style for visual queue
        if is_cross:
            self.unit2_slider.style = self.enabled_style
        else:
            self.unit2_slider.style = self.disabled_style

        # Clear and rerender plot
        self.output.clear_output(wait=True)
        with self.output:
            self._render_plot()

    def _update_plot(self, change):
        """
        Update plot when units are changed.
        """
        self.output.clear_output(wait=True)
        with self.output:
            self._render_plot()

    def _render_plot(self):
        """
        Render the plot using the specified backend.
        """
        # Determine unit selection
        unit1 = self.unit1_slider.value

        # Check if crosscorrelogram is enabled
        if self.cross_toggle.value:
            unit2 = self.unit2_slider.value
        else:
            unit2 = None

        # Update crosscorrelogram
        self.crosscorrelogram = CorrelogramPlot(
            data=self.data,
            unit01=unit1,
            unit02=unit2,
            **self.composite_kwargs,
        )

        # Render based on backend
        if self.backend == "mpl":
            fig, _ = self.crosscorrelogram.plot(backend="mpl")
            plt.show()
        elif self.backend == "plotly":
            fig = self.crosscorrelogram.plot(backend="plotly")
            fig.show()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def display(self):
        """
        Display the interactive widget.
        """
        # Create vertical box layout
        vbox = widgets.VBox(
            [self.unit1_slider, self.cross_toggle, self.unit2_slider, self.output]
        )
        display(vbox)
