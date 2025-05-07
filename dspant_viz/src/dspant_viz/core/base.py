# src/dspant_viz/core/base.py (extension)
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from dspant_viz.core.internals import public_api


@public_api(module_override="dspant_viz.core")
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

    @abstractmethod
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

    @abstractmethod
    def get_data(self) -> Dict:
        """
        Prepare data for rendering.

        Returns
        -------
        dict
            Data and parameters for rendering
        """
        raise NotImplementedError(
            "Subclasses should implement their own get_data() method."
        )

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update component parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        raise NotImplementedError(
            "Subclasses should implement their own update() method."
        )


@public_api(module_override="dspant_viz.core")
class CompositeVisualization(ABC):
    """
    Base class for composite visualizations that combine multiple visualization components.

    This class provides a standardized interface for creating complex visualizations
    that combine multiple simple components, with consistent rendering across backends.
    """

    def __init__(self, components: List[VisualizationComponent], **kwargs):
        """
        Initialize the composite visualization.

        Parameters
        ----------
        components : List[VisualizationComponent]
            List of visualization components to combine
        **kwargs : dict
            Additional configuration parameters
        """
        self.components = components
        self.config = kwargs

    @abstractmethod
    def plot(self, backend: str = "mpl", **kwargs):
        """
        Generate a composite plot using the specified backend.

        Parameters
        ----------
        backend : str, optional
            Backend to use for plotting ('mpl', 'plotly')
        **kwargs : dict
            Additional plot-specific parameters

        Returns
        -------
        Any
            Composite figure from the specified backend
        """
        raise NotImplementedError(
            "Subclasses should implement their own plot() method that resolves the backend dynamically."
        )

    @abstractmethod
    def get_data(self) -> Dict:
        """
        Prepare data from all components for rendering.

        Returns
        -------
        dict
            Combined data and parameters for rendering
        """
        raise NotImplementedError(
            "Subclasses should implement their own get_data() method."
        )

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update component parameters.

        Parameters
        ----------
        **kwargs : dict
            Parameters to update
        """
        raise NotImplementedError(
            "Subclasses should implement their own update() method."
        )
