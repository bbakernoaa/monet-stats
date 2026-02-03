"""
Visualization suite following the Aero Protocol (Two-Track Rule).
"""

from typing import Any, Optional

import xarray as xr

from .utils_stats import _update_history


def plot_spatial_static(
    da: xr.DataArray,
    projection: Optional[Any] = None,
    transform: Optional[Any] = None,
    ax: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """
    Track A: Publication-quality static spatial plot (Aero Protocol).

    Parameters
    ----------
    da : xarray.DataArray
        Input data with lat/lon coordinates.
    projection : cartopy.crs.Projection, optional
        The target projection for the plot. Default is ccrs.PlateCarree().
    transform : cartopy.crs.Projection, optional
        The data's own projection. Default is ccrs.PlateCarree().
    ax : Any, optional
        Existing axes to plot on (should be a GeoAxes). If None, a new figure is created.
    **kwargs : Any
        Additional keyword arguments passed to xarray.plot.pcolormesh or contourf.

    Returns
    -------
    Any
        The axes containing the plot (matplotlib.axes.Axes).

    Examples
    --------
    >>> import cartopy.crs as ccrs
    >>> import xarray as xr
    >>> da = xr.DataArray(...)
    >>> ax = plot_spatial_static(da)
    """
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plot_spatial_static requires matplotlib and cartopy.")

    if projection is None:
        projection = ccrs.PlateCarree()
    if transform is None:
        transform = ccrs.PlateCarree()

    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": projection})

    # Ensure we use pcolormesh or contourf for spatial data
    # and pass the transform
    da.plot(ax=ax, transform=transform, **kwargs)

    if hasattr(ax, "coastlines"):
        ax.coastlines()
    if hasattr(ax, "gridlines"):
        ax.gridlines(draw_labels=True)

    _update_history(da, "plot_spatial_static")
    return ax


def plot_spatial_interactive(
    da: xr.DataArray,
    rasterize: bool = True,
    width: int = 700,
    height: int = 500,
    **kwargs: Any,
) -> Any:
    """
    Track B: Exploratory interactive spatial plot (Aero Protocol).

    Parameters
    ----------
    da : xarray.DataArray
        Input data with lat/lon coordinates.
    rasterize : bool, optional
        Whether to rasterize the plot for performance on large grids.
        Default is True.
    width : int, optional
        Plot width in pixels. Default is 700.
    height : int, optional
        Plot height in pixels. Default is 500.
    **kwargs : Any
        Additional keyword arguments passed to hvplot.

    Returns
    -------
    Any
        The interactive plot object (holoviews.core.Element).
    """
    try:
        import hvplot.xarray  # noqa: F401
    except ImportError:
        raise ImportError("plot_spatial_interactive requires hvplot.")

    # hvplot handles geo=True if lat/lon are found
    plot = da.hvplot.quadmesh(
        rasterize=rasterize,
        width=width,
        height=height,
        geo=True,
        **kwargs,
    )

    _update_history(da, "plot_spatial_interactive")
    return plot
