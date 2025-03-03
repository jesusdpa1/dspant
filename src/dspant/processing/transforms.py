from typing import Any, Dict

import dask.array as da

from ..core.nodes.stream_processing import BaseProcessor


class TKEOProcessor(BaseProcessor):
    """TKEO processor implementation"""

    def __init__(self, method: str = "standard"):
        self.method = method
        self._overlap_samples = 2 if method == "standard" else 4

    def process(self, data: da.Array, **kwargs) -> da.Array:
        if self.method == "standard":

            def tkeo(x):
                return x[1:-1] ** 2 - x[:-2] * x[2:]
        else:

            def tkeo(x):
                return x[2:-2] ** 2 - x[:-4] * x[4:]

        return data.map_overlap(
            tkeo, depth=(self.overlap_samples, 0), boundary="reflect", dtype=data.dtype
        )

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary
        base_summary.update(
            {"method": self.method, "window_size": getattr(self, "window_size", None)}
        )
        return base_summary


class NormalizationProcessor(BaseProcessor):
    """Normalization processor implementation"""

    def __init__(self, method: str = "zscore"):
        self.method = method
        self._overlap_samples = 0

    def process(self, data: da.Array, **kwargs) -> da.Array:
        if self.method == "zscore":
            mean = data.mean()
            std = data.std()
            return (data - mean) / std
        elif self.method == "minmax":
            min_val = data.min()
            max_val = data.max()
            return (data - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    @property
    def overlap_samples(self) -> int:
        return self._overlap_samples

    @property
    def summary(self) -> Dict[str, Any]:
        base_summary = super().summary
        base_summary.update(
            {"method": self.method, "scale": getattr(self, "scale", None)}
        )
        return base_summary
