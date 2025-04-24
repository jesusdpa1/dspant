from abc import ABC
from typing import Any


class VisualizationComponent(ABC):
    """
    Base class for visualization components with multi-backend support.

    This class provides a standardized interface for creating visualizations
    that can be rendered across different backends.
    """

    def __init__(self, data, **kwargs):
        """
        Initialize the visualization component.

        Parameters
        ----------
        data : Any
            Input data for visualization
        **kwargs : dict
            Additional configuration parameters
        """
        self.data = data
        self.config = kwargs

    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a plot using the specified backend.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl', 'plotly')
        **kwargs : dict
            Additional plot-specific parameters

        Returns
        -------
        Any
            Plot figure from the specified backend
        """
        raise NotImplementedError(
            "Subclasses should implement their own plot() method that resolves the backend dynamically."
        )
