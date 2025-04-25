# src/dspant_viz/widgets/waveform_inspector.py
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from dspant_viz.core.internals import public_api
from dspant_viz.visualization.spike.waveforms import WaveformPlot


@public_api(module_override="dspant_viz.widgets")
class WaveformInspector:
    """
    Interactive widget for exploring neural waveforms.

    Allows selection of units, toggling template view,
    and applying different normalization methods.
    """

    def __init__(
        self,
        waveforms,
        sampling_rate: float = 1000.0,
        backend: str = "plotly",
        **kwargs,
    ):
        """
        Initialize the waveform inspector.

        Parameters
        ----------
        waveforms : dask.array.Array
            3D array of waveforms (neurons, samples, num_waveforms)
        sampling_rate : float, optional
            Sampling rate in Hz
        backend : str, optional
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs : dict
            Additional configuration parameters
        """
        # Validate input
        if len(waveforms.shape) != 3:
            raise ValueError("Waveforms must be a 3D Dask array")

        self.waveforms = waveforms
        self.sampling_rate = sampling_rate
        self.backend = backend
        self.composite_kwargs = kwargs

        # Determine available units
        self.unit_ids = list(range(waveforms.shape[0]))
        if not self.unit_ids:
            raise ValueError("No units found in the waveform data")

        # Create initial waveform plot
        initial_unit_id = self.unit_ids[0]
        self.waveform_plot = WaveformPlot(
            waveforms=waveforms,
            unit_id=initial_unit_id,
            sampling_rate=sampling_rate,
            template=True,
            normalization=None,
            **self.composite_kwargs,
        )

        # Create interactive widgets
        self._create_widgets()

    def _create_widgets(self):
        """
        Create interactive widgets for waveform visualization.
        """
        # Unit Slider
        self.unit_slider = widgets.SelectionSlider(
            options=self.unit_ids,
            value=self.unit_ids[0],
            description="Unit:",
            continuous_update=False,
        )

        # Normalization Dropdown
        self.normalization_dropdown = widgets.Dropdown(
            options=[("None", None), ("Z-Score", "zscore"), ("Min-Max", "minmax")],
            value=None,
            description="Normalization:",
            disabled=False,
        )

        # Template Checkbox (default True)
        self.template_checkbox = widgets.Checkbox(
            value=True, description="Template View", disabled=False, indent=False
        )

        # Output area for the plot
        self.output = widgets.Output()

        # Link widgets to update methods
        self.unit_slider.observe(self._update_plot, names="value")
        self.normalization_dropdown.observe(self._update_plot, names="value")
        self.template_checkbox.observe(self._update_plot, names="value")

        # Initial plot
        with self.output:
            self._render_plot()

    def _update_plot(self, change):
        """
        Update plot when parameters are changed.
        """
        self.output.clear_output(wait=True)
        with self.output:
            self._render_plot()

    def _render_plot(self):
        """
        Render the plot using the specified backend.
        """
        # Get current widget values
        unit_id = self.unit_slider.value
        normalization = self.normalization_dropdown.value
        template = self.template_checkbox.value

        # Update waveform plot
        self.waveform_plot = WaveformPlot(
            waveforms=self.waveforms,
            unit_id=unit_id,
            sampling_rate=self.sampling_rate,
            template=template,
            normalization=normalization,
            **self.composite_kwargs,
        )

        # Render based on backend
        if self.backend == "mpl":
            fig, _ = self.waveform_plot.plot(backend="mpl")
            plt.show()
        elif self.backend == "plotly":
            fig = self.waveform_plot.plot(backend="plotly")
            fig.show()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def display(self):
        """
        Display the interactive widget.
        """
        # Create vertical box layout
        vbox = widgets.VBox(
            [
                self.unit_slider,
                self.normalization_dropdown,
                self.template_checkbox,
                self.output,
            ]
        )
        display(vbox)
