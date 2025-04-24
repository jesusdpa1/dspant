# core/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

class VisualizationComponent(ABC):
    """Base class for all visualization components in dspant_viz"""

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """Return component data in format ready for rendering"""
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update component parameters"""
        pass


class RenderBackend(ABC):
    """Abstract interface for rendering backends"""

    @abstractmethod
    def render(self, component_data: Dict[str, Any], **kwargs) -> Any:
        """Render the component data using this backend"""
        pass

    @abstractmethod
    def create_figure(self, component_data: Dict[str, Any], **kwargs) -> Any:
        """Create a figure containing the rendered component"""
        pass


class Widget(ABC):
    """Base class for composite visualization widgets"""

    @abstractmethod
    def get_components(self) -> Dict[str, VisualizationComponent]:
        """Return all component parts of this widget"""
        pass

    @abstractmethod
    def get_layout(self) -> Dict[str, Any]:
        """Return layout configuration for the widget"""
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update widget parameters"""
        pass
