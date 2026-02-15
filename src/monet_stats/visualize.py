"""
Visualization utilities for the MONET Stats package (Aero Protocol Compliant).

This module implements the "Two-Track Rule" for scientific visualization:
- Track A (Publication): Static quality using Matplotlib and Cartopy.
- Track B (Exploration): Interactive exploration using HvPlot and GeoViews.
"""

from typing import Any, Optional

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
    stat: str = "mean",
    dim: str = "time",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot diurnal cycle following the Aero Protocol's Two-Track Rule.

    Parameters
    ----------
    da : xarray.DataArray
        The input data with a time-like coordinate to compute the cycle from.
    method : str, optional
        The plotting track to use:
        - 'matplotlib' (Track A): Static publication quality.
        - 'hvplot' (Track B): Interactive exploration.
        Default is 'matplotlib'.
    stat : str, optional
        The statistical method to compute the cycle ('mean', 'median', 'std').
        Default is 'mean'.
    dim : str, optional
        The dimension along which to compute the cycle. Default is 'time'.
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
    >>> # Track A (Static)
    >>> ax = plot_diurnal_cycle(da, method='matplotlib')
    """
    from .analysis import diurnal_cycle

    # Ensure metadata is preserved for plot labels
    original_attrs = da.attrs.copy()
    original_name = da.name

    cycle = diurnal_cycle(da, method=stat, dim=dim)

    # Restore name and units if they were lost during reduction
    if not cycle.name:
        cycle.name = original_name
    for attr in ["units", "long_name"]:
        if attr in original_attrs and attr not in cycle.attrs:
            cycle.attrs[attr] = original_attrs[attr]

    if method == "matplotlib":
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Track A (matplotlib) requires 'matplotlib'. Install it with 'pip install monet-stats[viz]'."
            )

        if "ax" not in kwargs:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 5)))
            kwargs["ax"] = ax
        else:
            ax = kwargs["ax"]

        cycle.plot(**kwargs)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of Day")

        if title:
            ax.set_title(title)
        elif not ax.get_title():
            ax.set_title(f"Diurnal Cycle ({stat})")

        _update_history(da, f"Plotted diurnal cycle using Track A (matplotlib, stat={stat})")
        return ax

    elif method == "hvplot":
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError("Track B (hvplot) requires 'hvplot'. Install it with 'pip install monet-stats[viz]'.")

        plot = cycle.hvplot.line(x="hour", title=title or f"Diurnal Cycle ({stat})", **kwargs)

        _update_history(da, f"Plotted diurnal cycle using Track B (hvplot, stat={stat})")
        return plot

    else:
        raise ValueError(f"Unknown plotting method: {method}. Must be 'matplotlib' or 'hvplot'.")
