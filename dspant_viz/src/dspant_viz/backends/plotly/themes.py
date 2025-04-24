# backends/plotly/themes.py
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.io as pio

class Theme:
    """Plotly theme configuration for dspant_viz"""

    # Color palettes
    PALETTES = {
        "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "pastel": ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff", "#debb9b"],
        "vibrant": ["#0077bb", "#ee7733", "#009988", "#cc3311", "#33bbee", "#ee3377"],
        "muted": ["#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#956cb4", "#8c613c"],
        "dark": ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00"],
    }

    # Line styles
    LINE_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

    # Marker symbols
    MARKER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "triangle-down",
                      "star", "x", "cross", "pentagon", "hexagon"]

    # Custom templates
    TEMPLATES = {
        "neuroscience": {
            "layout": {
                "font": {"family": "Arial, sans-serif", "size": 14},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "xaxis": {
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "linecolor": "black",
                    "linewidth": 1.5,
                    "ticks": "outside",
                    "tickwidth": 1.5,
                    "tickcolor": "black",
                },
                "yaxis": {
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "linecolor": "black",
                    "linewidth": 1.5,
                    "ticks": "outside",
                    "tickwidth": 1.5,
                    "tickcolor": "black",
                },
                "margin": {"t": 60, "b": 60, "l": 60, "r": 20},
            }
        },
        "publication": {
            "layout": {
                "font": {"family": "Arial, sans-serif", "size": 12},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "xaxis": {
                    "showgrid": False,
                    "linecolor": "black",
                    "linewidth": 1.0,
                    "ticks": "outside",
                },
                "yaxis": {
                    "showgrid": False,
                    "linecolor": "black",
                    "linewidth": 1.0,
                    "ticks": "outside",
                },
                "margin": {"t": 40, "b": 40, "l": 40, "r": 10},
            }
        },
        "presentation": {
            "layout": {
                "font": {"family": "Arial, sans-serif", "size": 16},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "xaxis": {
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "linecolor": "black",
                    "linewidth": 2.0,
                    "ticks": "outside",
                    "tickwidth": 2.0,
                    "tickcolor": "black",
                },
                "yaxis": {
                    "showgrid": True,
                    "gridcolor": "lightgray",
                    "linecolor": "black",
                    "linewidth": 2.0,
                    "ticks": "outside",
                    "tickwidth": 2.0,
                    "tickcolor": "black",
                },
                "margin": {"t": 80, "b": 80, "l": 80, "r": 40},
            }
        }
    }

    @classmethod
    def apply(cls, style: str = "default") -> None:
        """Apply a predefined style template to plotly"""
        if style in ["neuroscience", "publication", "presentation"]:
            # Create and register custom template
            custom_template = cls.TEMPLATES[style]
            pio.templates[style] = custom_template
            pio.templates.default = style
        else:
            # Use built-in templates
            if style == "default":
                pio.templates.default = "plotly_white"
            else:
                # Try to use the style name as a built-in template
                try:
                    pio.templates.default = style
                except:
                    pio.templates.default = "plotly_white"

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
