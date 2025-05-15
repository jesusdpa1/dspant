"""
Base classes for template subtraction algorithms.

This module provides base classes and shared functionality
for different template subtraction strategies in electrophysiological data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np

from dspant.core.internals import public_api
from dspant.engine.base import BaseProcessor


@public_api
class BaseSubtractor(BaseProcessor):
    """
    Base class for all template subtraction processors.

    This class defines common functionality for template subtraction algorithms
    that remove specific patterns (like ECG artifacts) from signals.
    """

    def __init__(self, template: Optional[np.ndarray] = None):
        """
        Initialize the base subtractor.

        Args:
            template: Template array to subtract.
        """
        super().__init__()
        self.template = template
        self._overlap_samples = 0
        self._subtraction_stats = {}

        # Set overlap samples based on template size
        if template is not None:
            if template.ndim == 1:
                self._overlap_samples = len(template)
            else:
                self._overlap_samples = template.shape[0]

    def set_template(self, template: np.ndarray) -> None:
        """
        Set or update the template to be subtracted.

        Args:
            template: New template array
        """
        self.template = template
        # Update overlap samples based on template size
        if template is not None:
            if template.ndim == 1:
                self._overlap_samples = len(template)
            else:
                self._overlap_samples = template.shape[0]

    @property
    def subtraction_stats(self) -> Dict[str, Any]:
        """Get statistics from the last subtraction operation"""
        return self._subtraction_stats

    @property
    def overlap_samples(self) -> int:
        """Return number of samples needed for overlap"""
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of subtractor configuration"""
        base_summary = super().summary
        template_shape = None
        if self.template is not None:
            template_shape = self.template.shape

        base_summary.update(
            {
                "template_shape": template_shape,
                "overlap_samples": self.overlap_samples,
            }
        )
        return base_summary


@public_api
class MultiTemplateSubtractor(BaseSubtractor):
    """
    Base class for subtractors that use multiple templates.

    This class extends BaseSubtractor to support using different templates
    for different events (such as multiple ECG morphologies).
    """

    def __init__(self, templates: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize the multi-template subtractor.

        Args:
            templates: Optional dictionary of {name: template_array}
                      If None, templates must be added later.
        """
        super().__init__(None)  # No single template
        self.templates = templates or {}
        self._current_template = None

        # Calculate maximum overlap from templates
        if templates:
            self._calculate_overlap()

    def _calculate_overlap(self) -> None:
        """Calculate the required overlap based on template sizes"""
        if not self.templates:
            self._overlap_samples = 0
            return

        # Find maximum template size
        max_size = 0
        for template in self.templates.values():
            if template.ndim == 1:
                size = len(template)
            else:
                size = template.shape[0]
            max_size = max(max_size, size)

        self._overlap_samples = max_size

    def add_template(self, name: str, template: np.ndarray) -> None:
        """
        Add a new template or replace an existing one.

        Args:
            name: Template identifier
            template: Template array
        """
        self.templates[name] = template
        self._calculate_overlap()

    def remove_template(self, name: str) -> bool:
        """
        Remove a template by name.

        Args:
            name: Template identifier

        Returns:
            True if template was removed, False if not found
        """
        if name in self.templates:
            del self.templates[name]
            self._calculate_overlap()
            return True
        return False

    def select_template(self, name: str) -> bool:
        """
        Select a template as the current active template.

        Args:
            name: Template identifier

        Returns:
            True if template was selected, False if not found
        """
        if name in self.templates:
            self._current_template = name
            self.template = self.templates[name]
            return True
        return False

    @property
    def current_template(self) -> Optional[str]:
        """Get the name of the currently selected template"""
        return self._current_template

    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of multi-template subtractor configuration"""
        base_summary = super().summary
        template_info = {}

        for name, template in self.templates.items():
            template_info[name] = {
                "shape": template.shape,
                "active": name == self._current_template,
            }

        base_summary.update(
            {
                "templates": template_info,
                "num_templates": len(self.templates),
                "current_template": self._current_template,
            }
        )
        return base_summary
