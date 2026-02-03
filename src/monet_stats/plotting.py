"""
Visualization module for atmospheric science data (Aero Protocol Compliant).
"""

from typing import Any, Optional

import xarray as xr

from .utils_stats import _update_history


def plot_spatial_static(
    da: xr.DataArray,
    projection: Any = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Publication-quality static spatial plot (Matplotlib + Cartopy).

    Parameters
    ----------
    da : xarray.DataArray
        2D DataArray with lat/lon coordinates.
    projection : cartopy.crs.Projection, optional
        Map projection to use. If None, defaults to PlateCarree.
    cmap : str, optional
        Colormap name. Default is 'viridis'.
    title : str, optional
        Plot title.
    **kwargs : Any
        Additional arguments passed to da.plot().

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> plot_spatial_static(da, projection=ccrs.Robinson())
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plot_spatial_static requires matplotlib. Install with 'pip install monet-stats[viz]'.")

    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    if projection is None:
        projection = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": projection})

    # Standard Aero Protocol: Always use transform=PlateCarree for lat/lon data
    da.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, **kwargs)

    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if title:
        ax.set_title(title)

    _update_history(da, "Created static spatial plot")
    return ax


def plot_spatial_interactive(
    da: xr.DataArray,
    cmap: str = "viridis",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Track B: Interactive spatial exploration plot (HvPlot + Geoviews).

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to plot.
    cmap : str, optional
        Colormap name. Default is 'viridis'.
    title : str, optional
        Plot title.
    **kwargs : Any
        Additional arguments passed to da.hvplot().

    Returns
    -------
    holoviews.core.Element
        The interactive plot object.

    Notes
    -----
    Uses rasterize=True by default for large grids (Aero Protocol Rule 3).
    """
    try:
        import geoviews as gv  # noqa: F401
        import hvplot.xarray  # noqa: F401
    except ImportError:
        raise ImportError(
            "plot_spatial_interactive requires hvplot and geoviews. Install with 'pip install monet-stats[viz]'."
        )

    # Standard Aero Protocol: Always rasterize large grids for interactivity
    plot = da.hvplot.quadmesh(
        geo=True,
        cmap=cmap,
        title=title,
        rasterize=kwargs.pop("rasterize", True),
        **kwargs,
    )

    _update_history(da, "Created interactive spatial plot")
    return plot
