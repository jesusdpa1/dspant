import importlib.util
import sys
from typing import Any, Dict, List, Optional, Union

# Check Python version for TOML parsing
if sys.version_info >= (3, 11):
    import tomllib
else:
    # Fallback for older Python versions
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None
        print("Warning: No TOML parser available. Install 'tomli' for Python < 3.11.")

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

from dspant.core.internals import public_api


@public_api(module_override="dspant_viz.core")
class ThemeManager:
    """
    Comprehensive theme management for visualization across backends.

    Supports:
    - Seaborn-inspired default themes
    - Built-in Plotly and Matplotlib themes
    - Custom TOML-based theme loading
    """

    # Predefined theme names
    PLOTLY_THEMES = [
        "plotly",
        "plotly_white",
        "plotly_dark",
        "ggplot2",
        "seaborn",
        "simple_white",
        "none",
    ]

    MATPLOTLIB_THEMES = [
        "default",
        "seaborn",
        "seaborn-darkgrid",
        "seaborn-whitegrid",
        "ggplot",
        "dark_background",
        "classic",
    ]

    def __init__(
        self,
        mpl_theme: str = "seaborn-darkgrid",
        plotly_theme: str = "seaborn",
        custom_themes_dir: Optional[Path] = None,
    ):
        """
        Initialize ThemeManager with theme preferences.

        Args:
            mpl_theme: Matplotlib theme (default: 'seaborn-darkgrid')
            plotly_theme: Plotly theme (default: 'seaborn')
            custom_themes_dir: Optional directory for custom TOML themes
        """
        # Setup TOML parsing
        if tomllib is None and sys.version_info < (3, 11):
            print("Warning: TOML parsing requires Python 3.11+ or 'tomli' package")

        # Theme configuration
        self.mpl_theme = mpl_theme
        self.plotly_theme = plotly_theme

        # Custom themes directory
        self.custom_themes_dir = custom_themes_dir or Path(__file__).parent / "themes"
        self.custom_themes_dir.mkdir(exist_ok=True)

    def apply_matplotlib_theme(self, theme: Optional[str] = None):
        """
        Apply theme to Matplotlib.

        Args:
            theme: Theme name. Uses default if not specified.
        """
        applied_theme = theme or self.mpl_theme

        # Validate theme
        if applied_theme not in self.MATPLOTLIB_THEMES:
            print(
                f"Warning: Unknown Matplotlib theme '{applied_theme}'. Using default."
            )
            applied_theme = "seaborn-darkgrid"

        # Special handling for seaborn themes
        if applied_theme.startswith("seaborn"):
            try:
                # Use Seaborn's built-in themes
                sns.set_theme(style=applied_theme.replace("seaborn-", ""))
            except Exception as e:
                print(f"Error applying Seaborn theme: {e}. Falling back to default.")
                plt.style.use("default")
        else:
            # Use Matplotlib's style system for other themes
            try:
                plt.style.use(applied_theme)
            except Exception as e:
                print(f"Error applying Matplotlib theme: {e}. Falling back to default.")
                plt.style.use("default")

    def apply_plotly_theme(self, theme: Optional[str] = None):
        """
        Apply theme to Plotly.

        Args:
            theme: Theme name. Uses default if not specified.
        """
        applied_theme = theme or self.plotly_theme

        # Validate theme
        if applied_theme not in self.PLOTLY_THEMES:
            print(f"Warning: Unknown Plotly theme '{applied_theme}'. Using default.")
            applied_theme = "seaborn"

        # Set the template
        try:
            pio.templates.default = applied_theme
        except Exception as e:
            print(f"Error applying Plotly theme: {e}. Falling back to default.")
            pio.templates.default = "seaborn"

    def load_custom_theme(self, theme_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a custom theme from a TOML file.

        Args:
            theme_path: Path to the theme TOML file

        Returns:
            Parsed theme configuration
        """
        theme_path = Path(theme_path)
        if not theme_path.exists():
            raise FileNotFoundError(f"Theme file not found: {theme_path}")

        if tomllib is None:
            raise ImportError("TOML parsing requires Python 3.11+ or 'tomli' package")

        with open(theme_path, "rb") as f:
            return tomllib.load(f)

    def create_custom_theme(self, theme_name: str, theme_config: Dict[str, Any]):
        """
        Create and save a custom theme to a TOML file.

        Args:
            theme_name: Name of the custom theme
            theme_config: Dictionary containing theme configuration
        """
        # Ensure tomli_w is available for writing
        try:
            import tomli_w
        except ImportError:
            raise ImportError("Install 'tomli-w' to write TOML files")

        theme_file = self.custom_themes_dir / f"{theme_name}.toml"
        with open(theme_file, "wb") as f:
            tomli_w.dump(theme_config, f)

    def list_available_themes(self) -> Dict[str, List[str]]:
        """
        List available themes for different backends.

        Returns:
            Dictionary of available themes
        """
        # Get custom themes from TOML files
        custom_themes = [f.stem for f in self.custom_themes_dir.glob("*.toml")]

        return {
            "matplotlib": self.MATPLOTLIB_THEMES + custom_themes,
            "plotly": self.PLOTLY_THEMES + custom_themes,
        }

    def example_themes(self):
        """
        Generate example plots to demonstrate available themes.

        Useful for theme comparison and exploration.
        """
        import plotly.express as px

        # Gapminder dataset for Plotly themes
        df = px.data.gapminder()
        df_2007 = df.query("year==2007")

        # Plotly themes
        for template in self.PLOTLY_THEMES:
            fig = px.scatter(
                df_2007,
                x="gdpPercap",
                y="lifeExp",
                size="pop",
                color="continent",
                log_x=True,
                size_max=60,
                template=template,
                title=f"Gapminder 2007: '{template}' theme",
            )
            fig.show()

        # Matplotlib themes (example plot)
        for theme in self.MATPLOTLIB_THEMES:
            plt.style.use(theme)
            plt.figure(figsize=(10, 6))
            plt.plot([1, 2, 3, 4], [1, 4, 2, 3], label=f"{theme} theme")
            plt.title(f"{theme} Theme Example")
            plt.legend()
            plt.show()


# Singleton theme manager
theme_manager = ThemeManager()


@public_api(module_override="dspant_viz.core")
# Expose key functions for easy import
def apply_matplotlib_theme(theme: Optional[str] = None):
    """Convenience function to apply Matplotlib theme"""
    theme_manager.apply_matplotlib_theme(theme)


@public_api(module_override="dspant_viz.core")
def apply_plotly_theme(theme: Optional[str] = None):
    """Convenience function to apply Plotly theme"""
    theme_manager.apply_plotly_theme(theme)


@public_api(module_override="dspant_viz.core")
def list_available_themes():
    """Convenience function to list available themes"""
    return theme_manager.list_available_themes()


@public_api(module_override="dspant_viz.core")
def create_custom_theme(theme_name: str, theme_config: Dict[str, Any]):
    """Convenience function to create a custom theme"""
    theme_manager.create_custom_theme(theme_name, theme_config)


@public_api(module_override="dspant_viz.core")
def show_example_themes():
    """Convenience function to show example themes"""
    theme_manager.example_themes()
