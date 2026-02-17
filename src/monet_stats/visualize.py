"""
Visualization utilities for the MONET Stats package (Aero Protocol Compliant).

This module implements the "Two-Track Rule" for scientific visualization:
- Track A (Publication): Static quality using Matplotlib and Cartopy.
- Track B (Exploration): Interactive exploration using HvPlot and GeoViews.
"""

from typing import Any, Optional

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def plot_spatial(
    da: xr.DataArray,
    method: str = "matplotlib",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    title: Optional[str] = None,
    cmap: str = "viridis",
    **kwargs: Any,
) -> Any:
    """
    Plot spatial data following the Aero Protocol's Two-Track Rule.

    Parameters
    ----------
    da : xarray.DataArray
        The spatial data to plot. Must have latitude and longitude coordinates.
    method : str, optional
        The plotting track to use:
        - 'matplotlib' (Track A): Static publication quality.
        - 'hvplot' (Track B): Interactive exploration.
        Default is 'matplotlib'.
    lat_dim : str, optional
        Name of the latitude dimension/coordinate. Default is 'lat'.
    lon_dim : str, optional
        Name of the longitude dimension/coordinate. Default is 'lon'.
    title : str, optional
        Title for the plot.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    **kwargs : Any
        Additional keyword arguments passed to the underlying plotting function.
        For 'matplotlib', these are passed to `da.plot()`.
        For 'hvplot', these are passed to `da.hvplot.quadmesh()`.

    Returns
    -------
    Any
        The plot object (matplotlib.axes.Axes or holoviews.element.Element).

    Raises
    ------
    ValueError
        If an unknown method is specified.
    ImportError
        If the required libraries for the chosen track are missing.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(10, 10),
    ...                   coords={'lat': np.arange(10), 'lon': np.arange(10)},
    ...                   dims=('lat', 'lon'))
    >>> # Track A (Static)
    >>> ax = plot_spatial(da, method='matplotlib')
    >>> # Track B (Interactive)
    >>> plot = plot_spatial(da, method='hvplot')
    """
    if method == "matplotlib":
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Track A (matplotlib) requires 'matplotlib' and 'cartopy'. "
                "Install them with 'pip install monet-stats[viz]'."
            )

        # Ensure projection is set in kwargs or defaults to PlateCarree
        projection = kwargs.pop("projection", ccrs.PlateCarree())
        transform = kwargs.pop("transform", ccrs.PlateCarree())

        if "ax" not in kwargs:
            fig = plt.figure(figsize=kwargs.pop("figsize", (10, 6)))
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            kwargs["ax"] = ax
        else:
            ax = kwargs["ax"]

        plot = da.plot(transform=transform, cmap=cmap, **kwargs)

        if hasattr(ax, "coastlines"):
            ax.coastlines()
        if hasattr(ax, "gridlines"):
            ax.gridlines(draw_labels=True)

        if title:
            ax.set_title(title)

        _update_history(da, f"Plotted spatial data using Track A (matplotlib, projection={projection})")
        return ax

    elif method == "hvplot":
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "Track B (hvplot) requires 'hvplot' and 'geoviews'. Install them with 'pip install monet-stats[viz]'."
            )

        # Aero Protocol mandatory for large grids
        rasterize = kwargs.pop("rasterize", True)
        geo = kwargs.pop("geo", True)

        plot = da.hvplot.quadmesh(
            x=lon_dim,
            y=lat_dim,
            geo=geo,
            rasterize=rasterize,
            cmap=cmap,
            title=title,
            **kwargs,
        )

        _update_history(da, "Plotted spatial data using Track B (hvplot, rasterize=True)")
        return plot

    else:
        raise ValueError(f"Unknown plotting method: {method}. Must be 'matplotlib' or 'hvplot'.")


def plot_diurnal_cycle(
    da: xr.DataArray,
    method: str = "matplotlib",
    dim: str = "time",
    stat: str = "mean",
    show_std: bool = True,
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot the diurnal cycle following the Aero Protocol's Two-Track Rule.

    Parameters
    ----------
    da : xarray.DataArray
        The data to plot. If it doesn't have exactly 24 values along 'hour',
        it will be automatically reduced using the specified 'stat'.
    method : str, optional
        The plotting track to use:
        - 'matplotlib' (Track A): Static publication quality.
        - 'hvplot' (Track B): Interactive exploration.
        Default is 'matplotlib'.
    dim : str, optional
        Dimension along which to compute the cycle if not already reduced.
        Default is 'time'.
    stat : str, optional
        Statistical method to use for reduction if needed ('mean', 'median').
        Default is 'mean'.
    show_std : bool, optional
        If True, show the standard deviation as a shaded area (Track A).
        Only applicable if 'da' needs reduction and has a time dimension.
        Default is True.
    title : str, optional
        Title for the plot.
    **kwargs : Any
        Additional keyword arguments passed to the underlying plotting function.

    Returns
    -------
    Any
        The plot object (matplotlib.axes.Axes or holoviews.element.Element).

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> times = pd.date_range("2020-01-01", periods=24*10, freq="h")
    >>> da = xr.DataArray(np.random.rand(240), coords={"time": times}, dims="time")
    >>> ax = plot_diurnal_cycle(da, method='matplotlib', title='Diurnal Cycle')
    """
    from .analysis import climatology, diurnal_cycle

    # 1. Prepare data (Reduction if necessary)
    if "hour" in da.coords and da.sizes.get("hour") == 24 and dim not in da.dims:
        plot_da = da
        std_da = None
    else:
        plot_da = diurnal_cycle(da, method=stat, dim=dim)
        if show_std:
            std_da = climatology(da, freq="hour", method="std", dim=dim)
        else:
            std_da = None

    # Ensure we are plotting against 'hour'
    if "hour" not in plot_da.coords:
        # This shouldn't happen with diurnal_cycle, but be safe
        plot_da = plot_da.assign_coords(hour=np.arange(24))

    if method == "matplotlib":
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Track A (matplotlib) requires 'matplotlib'. Install it with 'pip install monet-stats[viz]'."
            )

        if "ax" not in kwargs:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 5)))
        else:
            ax = kwargs.pop("ax")

        color = kwargs.pop("color", "tab:blue")
        label = kwargs.pop("label", da.name or "Data")

        ax.plot(plot_da.hour, plot_da, color=color, label=label, **kwargs)

        if std_da is not None:
            ax.fill_between(
                plot_da.hour,
                plot_da - std_da,
                plot_da + std_da,
                alpha=kwargs.get("alpha", 0.2),
                color=color,
            )

        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(da.attrs.get("units", "Value"))
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_xlim(0, 23)
        ax.grid(True, linestyle="--", alpha=0.7)

        if title:
            ax.set_title(title)
        elif not ax.get_title():
            ax.set_title(f"Diurnal Cycle ({stat.capitalize()})")

        _update_history(da, f"Plotted diurnal cycle using Track A (matplotlib, stat={stat})")
        return ax

    elif method == "hvplot":
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError("Track B (hvplot) requires 'hvplot'. Install it with 'pip install monet-stats[viz]'.")

        plot = plot_da.hvplot.line(
            x="hour",
            title=title or f"Diurnal Cycle ({stat.capitalize()})",
            ylabel=da.attrs.get("units", "Value"),
            xticks=list(range(0, 25, 3)),
            **kwargs,
        )

        _update_history(da, f"Plotted diurnal cycle using Track B (hvplot, stat={stat})")
        return plot

    else:
        raise ValueError(f"Unknown plotting method: {method}. Must be 'matplotlib' or 'hvplot'.")
