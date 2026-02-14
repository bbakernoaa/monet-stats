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
    da_in: xr.DataArray,
    method: str = "matplotlib",
    title: Optional[str] = None,
    color: str = "tab:blue",
    **kwargs: Any,
) -> Any:
    """
    Plot the diurnal cycle profile following the Aero Protocol's Two-Track Rule.

    Parameters
    ----------
    da_in : xarray.DataArray
        The data to plot. If it contains a 'time' dimension, the diurnal
        cycle will be computed first (mean). If it already contains an 'hour'
        dimension, it will be plotted directly.
    method : str, optional
        The plotting track to use:
        - 'matplotlib' (Track A): Static publication quality.
        - 'hvplot' (Track B): Interactive exploration.
        Default is 'matplotlib'.
    title : str, optional
        Title for the plot.
    color : str, optional
        Color for the line. Default is 'tab:blue'.
    **kwargs : Any
        Additional keyword arguments passed to the underlying plotting function.

    Returns
    -------
    Any
        The plot object (matplotlib.axes.Axes or holoviews.element.Element).

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> import pandas as pd
    >>> times = pd.date_range("2020-01-01", periods=24*10, freq="h")
    >>> da = xr.DataArray(np.random.rand(240), coords={"time": times}, dims="time")
    >>> # Track A
    >>> ax = plot_diurnal_cycle(da, method='matplotlib')
    """
    da = da_in
    from .analysis import diurnal_cycle

    # 1. Prepare data (Compute diurnal cycle if needed)
    if "hour" not in da.dims:
        if "time" in da.dims:
            da = diurnal_cycle(da)
        else:
            raise ValueError("Input DataArray must have 'hour' or 'time' dimension.")

    # 2. Ensure 1D for profile plotting
    # If other dims exist (lat, lon), take mean over them
    other_dims = [d for d in da.dims if d != "hour"]
    if other_dims:
        da = da.mean(dim=other_dims, keep_attrs=True)
        # Update local da history
        _update_history(da, f"Reduced {other_dims} to mean for diurnal plot")

    if method == "matplotlib":
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Track A (matplotlib) requires 'matplotlib'. " "Install it with 'pip install monet-stats[viz]'."
            )

        if "ax" not in kwargs:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 5)))
            kwargs["ax"] = ax
        else:
            ax = kwargs["ax"]

        da.plot.line(x="hour", color=color, **kwargs)

        ax.set_xticks(range(0, 24, 3))
        ax.set_xlim(0, 23)
        ax.set_xlabel("Hour of Day")
        if title:
            ax.set_title(title)

        _update_history(da_in, "Plotted diurnal cycle using Track A (matplotlib)")
        return ax

    elif method == "hvplot":
        try:
            import hvplot.xarray  # noqa: F401
        except ImportError:
            raise ImportError(
                "Track B (hvplot) requires 'hvplot'. Install it with 'pip install monet-stats[viz]'."
            )

        plot = da.hvplot.line(
            x="hour",
            color=color,
            title=title or "Diurnal Cycle",
            xticks=list(range(0, 24, 3)),
            **kwargs,
        )

        _update_history(da_in, "Plotted diurnal cycle using Track B (hvplot)")
        return plot

    else:
        raise ValueError(f"Unknown plotting method: {method}. Must be 'matplotlib' or 'hvplot'.")
