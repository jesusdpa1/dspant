"""
Base abstractions for all pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

import dask.array as da
import numpy as np


class BaseProcessor(ABC):
    """Abstract base class for all processors"""

    @abstractmethod
    def process(self, data: da.Array, fs: Optional[float] = None, **kwargs) -> da.Array:
        """Process the input data"""
        pass

    @property
    @abstractmethod
    def overlap_samples(self) -> int:
        """Return the number of samples needed for overlap"""
        pass

    @property
    def summary(self) -> Dict[str, Any]:
        """
        Return a dictionary containing processor configuration details
        Override this in derived classes to provide specific details
        """
        return {"type": self.__class__.__name__, "overlap": self.overlap_samples}


class ProcessingFunction(Protocol):
    """Protocol defining the interface for processing functions"""

    def __call__(
        self, data: np.ndarray, fs: Optional[float] = None, **kwargs
    ) -> np.ndarray: ...
