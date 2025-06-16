import json
from pathlib import Path
from typing import Dict, List, Optional, Self, Union

import numpy as np

from .base import BaseNode


class BaseSorterNode(BaseNode):
    """Base class for handling sorted spike data"""

    name: Optional[str] = None
    sampling_frequency: Optional[float] = None
    unit_ids: Optional[List[int]] = None


class SorterNode(BaseSorterNode):
    """Class for loading and accessing sorted spike data"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.data = None
        self.spike_times = None
        self.spike_clusters = None
        self.unit_properties = {}
        self._spike_indices = {}  # For efficient access to spikes by unit ID
