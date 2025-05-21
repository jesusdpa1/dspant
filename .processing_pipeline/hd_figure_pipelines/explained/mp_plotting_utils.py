"""
mp_plotting_utils.py
Author: jpenalozaa
Description: Utility functions for creating standardized publication-quality plots
             with colorblind-friendly palettes
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Standard colorblind-friendly color palette based on seaborn colorblind palette
# with primary color set to dark navy blue
PRIMARY_COLOR = "#2D3142"  # Dark navy blue for main trace
COLORS = {
    "primary": PRIMARY_COLOR,  # Dark navy blue (main trace)
    "blue": "#0173B2",  # Blue (colorblind-friendly)
    "orange": "#DE8F05",  # Orange (colorblind-friendly)
    "green": "#029E73",  # Green (colorblind-friendly)
    "red": "#D55E00",  # Red (colorblind-friendly)
    "purple": "#CC78BC",  # Purple (colorblind-friendly)
    "brown": "#CA9161",  # Brown (colorblind-friendly)
    "pink": "#FBAFE4",  # Pink (colorblind-friendly)
    "gray": "#949494",  # Gray (colorblind-friendly)
    "yellow": "#ECE133",  # Yellow (colorblind-friendly)
    "light_blue": "#56B4E9",  # Light blue (colorblind-friendly)
    "highlight": "#FF0000",  # Bright red (for emphasis)
}

# Standard font sizes with appropriate scaling
FONT_SIZES = {
    "title": 18,
    "subtitle": 16,
    "axis_label": 14,
    "tick_label": 12,
    "caption": 13,
    "panel_label": 20,
    "legend": 12,
    "annotation": 11,
}


def set_publication_style(
    font_family: str = "Montserrat", use_seaborn: bool = True
) -> None:
    """
    Set matplotlib parameters for publication-quality plots.

    Parameters
    ----------
    font_family : str
        Font family to use for all text elements
    use_seaborn : bool
        Whether to apply seaborn styling (includes colorblind-friendly palette)
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_family]
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.titleweight"] = "bold"
    plt.rcParams["figure.figsize"] = (14, 10)

    # Apply seaborn style for attractive grids and colors
    if use_seaborn:
        sns.set_theme(style="darkgrid")
        # Use colorblind-friendly palette
        sns.set_palette("colorblind")

    # Ensure high DPI for crisp rendering
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 600


def create_figure_grid(
    rows: int,
    cols: int,
    height_ratios: Optional[List[float]] = None,
    width_ratios: Optional[List[float]] = None,
    figsize: Tuple[float, float] = (14, 12),
) -> Tuple[plt.Figure, GridSpec]:
    """
    Create a figure with a grid specification for subplots.

    Parameters
    ----------
    rows : int
        Number of rows in the grid
    cols : int
        Number of columns in the grid
    height_ratios : list of float, optional
        Relative heights of rows
    width_ratios : list of float, optional
        Relative widths of columns
    figsize : tuple of float
        Figure size (width, height) in inches

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    gs : matplotlib.gridspec.GridSpec
        GridSpec object for creating subplots
    """
    fig = plt.figure(figsize=figsize)

    gs_kwargs = {}
    if height_ratios is not None:
        gs_kwargs["height_ratios"] = height_ratios
    if width_ratios is not None:
        gs_kwargs["width_ratios"] = width_ratios

    gs = GridSpec(rows, cols, **gs_kwargs)

    return fig, gs


def add_panel_label(
    ax: plt.Axes,
    label: str,
    position: str = "top-left",
    x_offset_factor: float = 0.1,
    y_offset_factor: float = 0.1,
    fontsize: Optional[int] = None,
    fontweight: str = "bold",
    fontfamily: Optional[str] = None,
    color: str = "black",
) -> None:
    """
    Add a panel label (A, B, C, etc.) to a subplot with adaptive positioning.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the label to
    label : str
        Label text (typically a single letter like 'A', 'B', etc.)
    position : str
        Position of the label relative to the subplot. Options:
        'top-left' (default), 'top-right', 'bottom-left', 'bottom-right'
    x_offset_factor : float
        Factor to determine the horizontal offset relative to subplot width.
        Smaller values place the label closer to the subplot horizontally.
        Typical values range from 0.05 to 0.2.
    y_offset_factor : float
        Factor to determine the vertical offset relative to subplot height.
        Smaller values place the label closer to the subplot vertically.
        Typical values range from 0.05 to 0.2.
    fontsize : int, optional
        Font size for the label. If None, uses the FONT_SIZES["panel_label"]
    fontweight : str
        Font weight for the label
    fontfamily : str, optional
        Font family for the label. If None, uses the current default
    color : str
        Color for the label text
    """
    # Get the position of the axes in figure coordinates
    bbox = ax.get_position()
    fig = plt.gcf()

    # Calculate horizontal and vertical offsets based on subplot size and offset factors
    # This will scale the offsets proportionally to the subplot dimensions
    x_offset = bbox.width * x_offset_factor
    y_offset = bbox.height * y_offset_factor

    # Set default font size if not specified
    if fontsize is None:
        fontsize = FONT_SIZES["panel_label"]

    # Determine position coordinates based on selected position
    if position == "top-left":
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset
    elif position == "top-right":
        x = bbox.x1 + x_offset
        y = bbox.y1 + y_offset
    elif position == "bottom-left":
        x = bbox.x0 - x_offset
        y = bbox.y0 - y_offset
    elif position == "bottom-right":
        x = bbox.x1 + x_offset
        y = bbox.y0 - y_offset
    else:
        # Default to top-left if invalid position
        x = bbox.x0 - x_offset
        y = bbox.y1 + y_offset

    # Determine text alignment based on position
    if "left" in position:
        ha = "right"
    else:
        ha = "left"

    if "top" in position:
        va = "bottom"
    else:
        va = "top"

    # Position the label outside the subplot
    fig.text(
        x,
        y,
        label,
        fontsize=fontsize,
        fontweight=fontweight,
        va=va,
        ha=ha,
        color=color,
        fontfamily=fontfamily,
    )


def format_axis(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    title_fontsize: Optional[int] = None,
    label_fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    grid: bool = True,
    tick_params: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Format a matplotlib axis with publication-quality settings.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to format
    title : str, optional
        Title for the axes
    xlabel : str, optional
        Label for the x-axis
    ylabel : str, optional
        Label for the y-axis
    xlim : tuple of float, optional
        Limits for the x-axis
    ylim : tuple of float, optional
        Limits for the y-axis
    xscale : str
        Scale for the x-axis ('linear', 'log', 'symlog', 'logit')
    yscale : str
        Scale for the y-axis ('linear', 'log', 'symlog', 'logit')
    title_fontsize : int, optional
        Font size for the title. If None, uses FONT_SIZES["subtitle"]
    label_fontsize : int, optional
        Font size for the axis labels. If None, uses FONT_SIZES["axis_label"]
    tick_fontsize : int, optional
        Font size for the tick labels. If None, uses FONT_SIZES["tick_label"]
    grid : bool
        Whether to display grid lines
    tick_params : dict, optional
        Additional parameters to pass to ax.tick_params()
    """
    # Use default font sizes if not specified
    if title_fontsize is None:
        title_fontsize = FONT_SIZES["subtitle"]
    if label_fontsize is None:
        label_fontsize = FONT_SIZES["axis_label"]
    if tick_fontsize is None:
        tick_fontsize = FONT_SIZES["tick_label"]

    # Set title if provided
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    # Set axis labels if provided
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize, fontweight="bold")

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set axis scales
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # Configure tick parameters
    base_tick_params = {"labelsize": tick_fontsize}
    if tick_params:
        base_tick_params.update(tick_params)
    ax.tick_params(**base_tick_params)

    # Configure grid
    ax.grid(grid, linestyle="--", alpha=0.7)


def add_legend(
    ax: plt.Axes,
    loc: str = "best",
    fontsize: Optional[int] = None,
    frameon: bool = True,
    framealpha: float = 0.8,
    title: Optional[str] = None,
    title_fontsize: Optional[int] = None,
    outside: bool = False,
    ncol: int = 1,
) -> None:
    """
    Add a formatted legend to an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the legend to
    loc : str
        Location of the legend
    fontsize : int, optional
        Font size for the legend text. If None, uses FONT_SIZES["legend"]
    frameon : bool
        Whether to draw a frame around the legend
    framealpha : float
        Alpha transparency for the legend's background
    title : str, optional
        Title for the legend
    title_fontsize : int, optional
        Font size for the legend title. If None, uses FONT_SIZES["legend"] + 2
    outside : bool
        If True, places the legend outside the axes
    ncol : int
        Number of columns in the legend
    """
    if fontsize is None:
        fontsize = FONT_SIZES["legend"]
    if title_fontsize is None and title is not None:
        title_fontsize = fontsize + 2

    legend_kwargs = {
        "loc": loc,
        "fontsize": fontsize,
        "frameon": frameon,
        "framealpha": framealpha,
        "ncol": ncol,
    }

    if title is not None:
        legend_kwargs["title"] = title
        legend_kwargs["title_fontsize"] = title_fontsize

    if outside:
        legend_kwargs["bbox_to_anchor"] = (1.05, 1)
        legend_kwargs["loc"] = "upper left"

    ax.legend(**legend_kwargs)


def add_colorbar(
    fig: plt.Figure,
    mappable,
    ax: plt.Axes,
    position: str = "right",
    size: str = "5%",
    pad: float = 0.05,
    label: Optional[str] = None,
    label_fontsize: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
) -> None:
    """
    Add a colorbar to a plot.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    mappable : matplotlib.cm.ScalarMappable
        Mappable object that the colorbar will be based on
    ax : matplotlib.axes.Axes
        Axes object that the colorbar will be attached to
    position : str
        Position of the colorbar relative to the axes ('right', 'left', 'top', 'bottom')
    size : str
        Size of the colorbar as a percentage of the axes
    pad : float
        Padding between the axes and the colorbar
    label : str, optional
        Label for the colorbar
    label_fontsize : int, optional
        Font size for the colorbar label. If None, uses FONT_SIZES["axis_label"]
    tick_fontsize : int, optional
        Font size for the colorbar tick labels. If None, uses FONT_SIZES["tick_label"]
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Use default font sizes if not specified
    if label_fontsize is None:
        label_fontsize = FONT_SIZES["axis_label"]
    if tick_fontsize is None:
        tick_fontsize = FONT_SIZES["tick_label"]

    # Create a divider for the axes
    divider = make_axes_locatable(ax)

    # Append an axes for the colorbar
    cax = divider.append_axes(position, size=size, pad=pad)

    # Create the colorbar
    cbar = fig.colorbar(mappable, cax=cax)

    # Set the label if provided
    if label is not None:
        cbar.set_label(label, fontsize=label_fontsize, fontweight="bold")

    # Set the tick label font size
    cbar.ax.tick_params(labelsize=tick_fontsize)


def add_cutoff_marker(
    ax: plt.Axes,
    x: float,
    label: Optional[str] = None,
    y_pos: Optional[float] = None,
    color: str = "red",
    linestyle: str = "--",
    alpha: float = 0.7,
    fontsize: Optional[int] = None,
    fontweight: str = "normal",
) -> None:
    """
    Add a vertical line marker for a cutoff frequency or threshold.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to add the marker to
    x : float
        x-coordinate for the vertical line
    label : str, optional
        Label for the marker
    y_pos : float, optional
        y-coordinate for the label. If None, uses a default position near the bottom
    color : str
        Color for the line and label
    linestyle : str
        Line style for the marker line
    alpha : float
        Alpha transparency for the line
    fontsize : int, optional
        Font size for the label. If None, uses FONT_SIZES["annotation"]
    fontweight : str
        Font weight for the label
    """
    # Add the vertical line
    ax.axvline(x=x, color=color, linestyle=linestyle, alpha=alpha)

    # Add the label if provided
    if label is not None:
        # Use default font size if not specified
        if fontsize is None:
            fontsize = FONT_SIZES["annotation"]

        # Use a default y-position if not specified
        if y_pos is None:
            # Calculate a position near the bottom of the plot
            y_lim = ax.get_ylim()
            y_range = y_lim[1] - y_lim[0]
            y_pos = y_lim[0] + 0.05 * y_range

        ax.text(
            x,
            y_pos,
            label,
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            horizontalalignment="center",
            verticalalignment="center",
        )


def finalize_figure(
    fig: plt.Figure,
    title: Optional[str] = None,
    title_y: float = 0.98,
    tight_layout: bool = True,
    top_margin: Optional[float] = None,
    bottom_margin: Optional[float] = None,
    left_margin: Optional[float] = None,
    right_margin: Optional[float] = None,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    title_fontsize: Optional[int] = None,
) -> None:
    """
    Finalize a figure with adjustments for publication quality.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    title : str, optional
        Overall title for the figure
    title_y : float
        y-position for the figure title
    tight_layout : bool
        Whether to apply tight_layout() to the figure
    top_margin : float, optional
        Top margin for the figure
    bottom_margin : float, optional
        Bottom margin for the figure
    left_margin : float, optional
        Left margin for the figure
    right_margin : float, optional
        Right margin for the figure
    wspace : float, optional
        Width spacing between subplots
    hspace : float, optional
        Height spacing between subplots
    title_fontsize : int, optional
        Font size for the figure title. If None, uses FONT_SIZES["title"]
    """
    # Apply tight_layout if requested
    if tight_layout:
        fig.tight_layout()

    # Build a dictionary of subplot adjustments
    adjust_params = {}
    if top_margin is not None:
        adjust_params["top"] = 1.0 - top_margin
    if bottom_margin is not None:
        adjust_params["bottom"] = bottom_margin
    if left_margin is not None:
        adjust_params["left"] = left_margin
    if right_margin is not None:
        adjust_params["right"] = 1.0 - right_margin
    if wspace is not None:
        adjust_params["wspace"] = wspace
    if hspace is not None:
        adjust_params["hspace"] = hspace

    # Apply subplot adjustments if any parameters were provided
    if adjust_params:
        fig.subplots_adjust(**adjust_params)

    # Add a title if provided
    if title is not None:
        if title_fontsize is None:
            title_fontsize = FONT_SIZES["title"]

        fig.suptitle(
            title,
            fontsize=title_fontsize,
            fontweight="bold",
            y=title_y,
        )


def save_figure(
    fig: plt.Figure,
    filename: str,
    dpi: int = 600,
    bbox_inches: str = "tight",
    pad_inches: float = 0.1,
    transparent: bool = False,
    facecolor: Optional[str] = None,
    format: Optional[str] = None,
) -> None:
    """
    Save a figure to a file with publication-quality settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    filename : str
        Filename to save the figure to
    dpi : int
        Resolution in dots per inch
    bbox_inches : str
        Bounding box in inches
    pad_inches : float
        Padding in inches
    transparent : bool
        Whether to make the background transparent
    facecolor : str, optional
        Background color for the figure
    format : str, optional
        File format to save the figure in. If None, inferred from the filename
    """
    img_path = Path("./img/")
    file_path = img_path.joinpath(filename)
    fig.savefig(
        file_path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
        transparent=transparent,
        facecolor=facecolor,
        format=format,
    )
