# backends/mpl/themes.py
from typing import Dict, Any, List, Tuple, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt

class Theme:
    """Matplotlib theme configuration for dspant_viz"""

    # Color palettes
    PALETTES = {
        "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "pastel": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b"],
        "vibrant": ["#0077bb", "#ee7733", "#009988", "#cc3311", "#33bbee", "#ee3377"],
        "muted": ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4", "#8c613c"],
        "dark": ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00"],
    }

    # Line styles
    LINE_STYLES = ['-', '--', '-.', ':']

    # Marker styles
    MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']

    # Default parameters
    DEFAULTS = {
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }

    @classmethod
    def apply(cls, style: str = "default") -> None:
        """Apply a predefined style to matplotlib"""
        if style == "neuroscience":
            # Neuroscience-specific styling
            params = {
                **cls.DEFAULTS,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.linewidth": 1.5,
                "xtick.major.width": 1.5,
                "ytick.major.width": 1.5,
                "xtick.major.size": 5,
                "ytick.major.size": 5,
                "lines.linewidth": 1.5,
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            }
            plt.rcParams.update(params)

        elif style == "publication":
            # Publication-ready styling
            params = {
                **cls.DEFAULTS,
                "figure.figsize": (8, 6),
                "figure.dpi": 300,
                "font.size": 10,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "lines.linewidth": 1.0,
                "axes.linewidth": 1.0,
                "xtick.major.width": 1.0,
                "ytick.major.width": 1.0,
                "savefig.format": "pdf",
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
            }
            plt.rcParams.update(params)

        elif style == "presentation":
            # Presentation styling
            params = {
                **cls.DEFAULTS,
                "figure.figsize": (12, 8),
                "figure.dpi": 150,
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 18,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "lines.linewidth": 2.0,
                "axes.linewidth": 2.0,
                "xtick.major.width": 2.0,
                "ytick.major.width": 2.0,
                "xtick.major.size": 6,
                "ytick.major.size": 6,
            }
            plt.rcParams.update(params)

        else:
            # Default styling
            plt.rcParams.update(cls.DEFAULTS)

    @classmethod
    def get_color_palette(cls, palette_name: str = "default", n_colors: Optional[int] = None) -> List[str]:
        """Get a list of colors from a named palette"""
        if palette_name in cls.PALETTES:
            palette = cls.PALETTES[palette_name]
            if n_colors is not None:
                # Cycle through the palette if more colors are needed
                return [palette[i % len(palette)] for i in range(n_colors)]
            return palette
        else:
            return cls.PALETTES["default"]
