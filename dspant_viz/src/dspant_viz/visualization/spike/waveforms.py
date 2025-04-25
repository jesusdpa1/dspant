# src/dspant_viz/visualization/spike/waveform.py
from typing import Any, Dict, List, Optional, Union

import dask.array as da
import numba as nb
import numpy as np
import seaborn as sns

from dspant_viz.core.base import VisualizationComponent
from dspant_viz.core.internals import public_api
from dspant_viz.utils.normalization import normalize_data


@public_api(module_override="dspant_viz.visualization")
class WaveformPlot(VisualizationComponent):
    """
    Visualization component for neural waveform data.

    Supports plotting waveforms with various visualization options.
    """

    def __init__(
        self,
        waveforms: da.Array,
        unit_id: int = 0,
        sampling_rate: float = 1000.0,
        num_waveforms: Optional[int] = None,
        template: bool = False,
        normalization: Optional[str] = None,
        color_mode: str = "colormap",
        colormap: str = "colorblind",
        line_width: float = 1.0,
        alpha: float = 0.7,
        **kwargs,
    ):
        """
        Initialize WaveformPlot.

        Parameters
        ----------
        waveforms : da.Array
            3D array of waveforms (neurons, samples, num_waveforms)
        unit_id : int, optional
            Specific unit to plot (default: 0)
        sampling_rate : float, optional
            Sampling rate in Hz (default: 1000.0)
        num_waveforms : int, optional
            Number of waveforms to plot
        template : bool, optional
            Whether to plot template (mean/median) waveform
        normalization : str, optional
            Normalization method ('zscore', 'minmax', None)
        color_mode : str, optional
            Color selection mode ('colormap', 'single')
        colormap : str, optional
            Colormap or color palette to use
        line_width : float, optional
            Width of waveform lines
        alpha : float, optional
            Transparency of waveform lines
        **kwargs
            Additional configuration parameters
        """
        # Validate input
        if len(waveforms.shape) != 3:
            raise ValueError("Waveforms must be a 3D Dask array")

        # Validate unit_id
        if unit_id is None:
            raise ValueError("unit_id must be specified")

        if unit_id < 0 or unit_id >= waveforms.shape[0]:
            raise ValueError(
                f"Invalid unit_id. Must be between 0 and {waveforms.shape[0] - 1}"
            )

        # Determine number of waveforms to plot
        max_waveforms = waveforms.shape[2]
        if num_waveforms is None:
            num_waveforms = max_waveforms
        elif num_waveforms < 1:
            raise ValueError("num_waveforms must be at least 1")
        elif num_waveforms > max_waveforms:
            print(
                f"Warning: num_waveforms ({num_waveforms}) exceeds available waveforms ({max_waveforms}). Using all available."
            )
            num_waveforms = max_waveforms

        # Initialize base class
        super().__init__(waveforms, **kwargs)

        # Store parameters
        self.waveforms = waveforms
        self.unit_id = unit_id
        self.sampling_rate = sampling_rate
        self.num_waveforms = num_waveforms
        self.template = template
        self.normalization = normalization
        self.color_mode = color_mode
        self.colormap = colormap
        self.line_width = line_width
        self.alpha = alpha

        # Prepare waveform data
        self.prepared_data = self._prepare_waveform_data()

    def _prepare_waveform_data(self) -> Dict[str, Any]:
        """
        Prepare waveform data for visualization.

        Returns
        -------
        Dict containing prepared visualization data
        """
        # Extract waveforms for specific unit
        unit_waveforms = self.waveforms[self.unit_id, :, :].compute()

        # Select subset of waveforms
        if not self.template:
            # Randomly select waveforms
            np.random.seed(42)  # For reproducibility
            selected_indices = np.random.choice(
                unit_waveforms.shape[1], size=self.num_waveforms, replace=False
            )
            selected_waveforms = unit_waveforms[:, selected_indices]
        else:
            # Compute template (mean) waveform
            selected_waveforms = np.mean(unit_waveforms, axis=1)
            selected_waveforms = selected_waveforms.reshape(-1, 1)

        # Normalize if requested
        if self.normalization:
            normalized_waveforms = np.array(
                [normalize_data(wf, self.normalization) for wf in selected_waveforms.T]
            ).T

        # Compute time array
        time_array = np.arange(selected_waveforms.shape[0]) / self.sampling_rate

        # Compute SEM if template is True
        sem = None
        if self.template:
            sem = np.std(unit_waveforms, axis=1) / np.sqrt(unit_waveforms.shape[1])

        return {
            "time": time_array,
            "waveforms": normalized_waveforms
            if self.normalization
            else selected_waveforms,
            "sem": sem,
            "unit_id": self.unit_id,
        }

    def get_data(self) -> Dict[str, Any]:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        return {
            "data": self.prepared_data,
            "params": {
                "unit_id": self.unit_id,
                "template": self.template,
                "normalization": self.normalization,
                "color_mode": self.color_mode,
                "colormap": self.colormap,
                "line_width": self.line_width,
                "alpha": self.alpha,
                **self.config,
            },
        }

    def update(self, **kwargs) -> None:
        """
        Update plot parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        # Update parameters
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.config[key] = value

        # Re-prepare data
        self.prepared_data = self._prepare_waveform_data()

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a plot using the specified backend.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl' or 'plotly')
        **kwargs : dict
            Additional parameters for the backend

        Returns
        -------
        Any
            Plot figure from the specified backends
        """
        if backend == "mpl":
            from dspant_viz.backends.mpl.waveforms import render_waveform
        elif backend == "plotly":
            from dspant_viz.backends.plotly.waveforms import render_waveform
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return render_waveform(self.get_data(), **kwargs)
