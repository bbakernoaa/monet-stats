"""
Visualization module for MONET Stats (Aero Protocol Compliant).

This module implements the 'Two-Track Rule' for geospatial visualization:
- Track A (Publication): Matplotlib + Cartopy for static, publication-quality maps.
- Track B (Exploration): HvPlot + GeoViews for interactive, scalable exploration.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def plot_spatial(
    data: Union[xr.DataArray, xr.Dataset],
    method: str = "static",
    lat_dim: str = "lat",
    lon_dim: str = "lon",
    projection: Optional[Any] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Geospatial visualization of 2D fields (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        2D data to plot. If Dataset, the first data variable is used.
    method : str, optional
        Visualization track:
        - 'static' (Track A): Matplotlib + Cartopy.
        - 'interactive' (Track B): HvPlot + GeoViews.
        Default is 'static'.
    lat_dim : str, optional
        Name of the latitude dimension/coordinate. Default is 'lat'.
    lon_dim : str, optional
        Name of the longitude dimension/coordinate. Default is 'lon'.
    projection : cartopy.crs.Projection, optional
        Geosatpatial projection for Track A. Required for geospatial correctness.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.
    title : str, optional
        Plot title.
    **kwargs : Any
        Additional arguments passed to the underlying plotting function:
        - For 'static': passed to `ax.pcolormesh` or `da.plot`.
        - For 'interactive': passed to `da.hvplot.quadmesh`.

    Returns
    -------
    Any
        The resulting plot object (Matplotlib Axes or HvPlot object).

    Notes
    -----
    - Track A (Static) mandatory: `projection` in axes and `transform` in plot calls.
    - Track B (Interactive) mandatory: `rasterize=True` for large grids.

    Examples
    --------
    >>> import xarray as xr
    >>> import cartopy.crs as ccrs
    >>> da = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> plot_spatial(da, method='static', projection=ccrs.PlateCarree())
    """
    if isinstance(data, xr.Dataset):
        var_name = list(data.data_vars)[0]
        da = data[var_name]
    else:
        da = data

    # Ensure scientific provenance
    da = _update_history(da, f"Plotted using {method} spatial plot")

    if method == "static":
        return _plot_static(da, lat_dim=lat_dim, lon_dim=lon_dim, projection=projection, cmap=cmap, title=title, **kwargs)
    elif method == "interactive":
        return _plot_interactive(da, lat_dim=lat_dim, lon_dim=lon_dim, cmap=cmap, title=title, **kwargs)
    else:
        raise ValueError(f"Unknown visualization method: {method}")


def _plot_static(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    projection: Optional[Any],
    cmap: str,
    title: Optional[str],
    **kwargs: Any,
) -> Any:
    """Track A: Static publication-quality plot."""
    try:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib and Cartopy are required for static plots. Install with 'pip install monet-stats[viz]'.")

    if projection is None:
        projection = ccrs.PlateCarree()

    fig, ax = plt.subplots(subplot_kw={"projection": projection})

    # Standard Aero Protocol: mandatory transform for geospatial data
    transform = kwargs.pop("transform", ccrs.PlateCarree())

    da.plot(ax=ax, x=lon_dim, y=lat_dim, transform=transform, cmap=cmap, **kwargs)

    ax.coastlines()
    if title:
        ax.set_title(title)

    return ax


def _plot_interactive(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    cmap: str,
    title: Optional[str],
    **kwargs: Any,
) -> Any:
    """Track B: Interactive exploration plot."""
    try:
        import hvplot.xarray  # noqa: F401
    except ImportError:
        raise ImportError("HvPlot and GeoViews are required for interactive plots. Install with 'pip install monet-stats[viz]'.")

    # Standard Aero Protocol: rasterize=True for performance on large grids
    rasterize = kwargs.pop("rasterize", True)
    geo = kwargs.pop("geo", True)

    plot = da.hvplot.quadmesh(
        x=lon_dim,
        y=lat_dim,
        cmap=cmap,
        rasterize=rasterize,
        geo=geo,
        title=title,
        **kwargs
    )

    return plot


def plot_taylor(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray, Dict[str, Union[xr.DataArray, np.ndarray]]],
    ax: Optional[Any] = None,
    title: str = "Taylor Diagram",
    **kwargs: Any,
) -> Any:
    """
    Generate a Taylor Diagram (Aero Protocol).

    A Taylor diagram provides a concise summary of how well a model matches
    observations in terms of their correlation, their root-mean-square
    difference, and the ratio of their variances.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values (reference).
    mod : xarray.DataArray, numpy.ndarray, or dict
        Modeled values. If a dict, keys are labels and values are data arrays
        for multi-model comparison.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to plot on. Must be a polar projection.
    title : str, optional
        Plot title. Default is 'Taylor Diagram'.
    **kwargs : Any
        Additional arguments passed to `ax.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the Taylor diagram.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.rand(100)
    >>> mod = obs + np.random.normal(0, 0.1, 100)
    >>> plot_taylor(obs, mod)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for Taylor diagrams.")

    # Standard Aero Protocol: Calculate statistics lazily if using Xarray
    from .correlation_metrics import pearsonr
    from .error_metrics import STDO, STDP

    # Update provenance
    if isinstance(obs, (xr.DataArray, xr.Dataset)):
        obs = _update_history(obs, "Plotted on Taylor Diagram")
    if isinstance(mod, (xr.DataArray, xr.Dataset)):
        mod = _update_history(mod, "Plotted on Taylor Diagram")
    elif isinstance(mod, dict):
        for k in mod:
            if isinstance(mod[k], (xr.DataArray, xr.Dataset)):
                mod[k] = _update_history(mod[k], "Plotted on Taylor Diagram")

    def _get_stats(o, m):
        std_o = float(STDO(o, o))  # Std of obs
        std_p = float(STDP(o, m))  # Std of mod
        r = float(pearsonr(o, m))
        return std_o, std_p, r

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")

    # Taylor Diagram Setup
    # theta = arccos(correlation)
    # radial distance = standard deviation

    if isinstance(mod, dict):
        models = mod
    else:
        models = {"Model": mod}

    # Reference (Observation)
    std_ref, _, _ = _get_stats(obs, obs)
    ax.plot(0, std_ref, "ko", label="Observations", markersize=10)

    for label, m_data in models.items():
        _, std_p, r = _get_stats(obs, m_data)
        theta = np.arccos(np.clip(r, -1, 1))
        ax.plot(theta, std_p, "o", label=label, **kwargs)

    # Styling
    ax.set_thetamax(90)
    ax.set_theta_direction(1)
    ax.set_theta_offset(0)

    # Custom ticks for correlation
    r_ticks = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
    ax.set_xticks(np.arccos(r_ticks))
    ax.set_xticklabels(r_ticks)

    ax.set_xlabel("Standard Deviation")
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))

    return ax
