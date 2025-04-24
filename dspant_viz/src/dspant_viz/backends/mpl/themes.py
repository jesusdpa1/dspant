# dspant_viz/backends/mpl/themes.py
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns


class Theme:
    """Matplotlib theme configuration for dspant_viz"""

    # Colorblind-friendly palette
    COLOR_PALETTE = sns.color_palette("colorblind")

    # Typography
    FONT_FAMILY = "Montserrat"

    # Font sizes with scaling
    SIZES = {"title": 18, "subtitle": 16, "axis_label": 14, "tick": 12, "caption": 13}

    @classmethod
    def apply(cls, style: str = "neuroscience"):
        """
        Apply a comprehensive matplotlib theme with custom styling.

        Parameters
        ----------
        style : str, optional
            Theme style to apply. Defaults to "neuroscience".
        """
        # Attempt to load Montserrat font
        try:
            font_files = fm.findSystemFonts(fontpaths=None, fontext="ttf")
            montserrat_fonts = [f for f in font_files if "Montserrat" in f]
            if montserrat_fonts:
                for font_path in montserrat_fonts:
                    fm.fontManager.addfont(font_path)
        except Exception:
            print("Montserrat font not found. Using system default.")

        # Matplotlib RC Parameters
        plt.rcParams.update(
            {
                # Font Configuration
                "font.family": "sans-serif",
                "font.sans-serif": [cls.FONT_FAMILY, "Arial", "Helvetica"],
                # Figure Properties
                "figure.figsize": (15, 8),
                "figure.dpi": 300,
                # Title and Label Sizes
                "font.size": cls.SIZES["tick"],
                "axes.titlesize": cls.SIZES["subtitle"],
                "axes.labelsize": cls.SIZES["axis_label"],
                "xtick.labelsize": cls.SIZES["tick"],
                "ytick.labelsize": cls.SIZES["tick"],
                # Line Properties
                "lines.linewidth": 2,
                # Grid
                "axes.grid": True,
                "axes.grid.linestyle": "--",
                "axes.grid.alpha": 0.3,
                # Spines
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

        # Seaborn style for additional aesthetics
        sns.set_theme(style="darkgrid", palette="colorblind", font=cls.FONT_FAMILY)

    @classmethod
    def get_color_palette(cls):
        """
        Return the colorblind-friendly color palette.

        Returns
        -------
        list
            List of color hex codes
        """
        return [
            f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            for r, g, b in cls.COLOR_PALETTE
        ]

    @classmethod
    def setup_figure(
        cls, figsize: tuple = (15, 8), title: str = None, suptitle: str = None
    ):
        """
        Create a figure with consistent styling.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size
        title : str, optional
            Subplot title
        suptitle : str, optional
            Figure-level title

        Returns
        -------
        tuple
            Figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)

        if title:
            ax.set_title(title, fontsize=cls.SIZES["subtitle"], fontweight="bold")

        if suptitle:
            fig.suptitle(
                suptitle, fontsize=cls.SIZES["title"], fontweight="bold", y=0.98
            )

        return fig, ax
