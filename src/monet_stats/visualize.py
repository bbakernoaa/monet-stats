"""
Visualization utilities for the MONET Stats package (Aero Protocol Compliant).

This module implements the "Two-Track Rule" for scientific visualization:
- Track A (Publication): Static quality using Matplotlib and Cartopy.
- Track B (Exploration): Interactive exploration using HvPlot and GeoViews.
"""

from typing import Any, Dict, Optional, Tuple, Union

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


def plot_taylor(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray, Dict[str, Union[xr.DataArray, np.ndarray]]],
    ax: Optional[Any] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Taylor Diagram",
    label_obs: str = "Observations",
    **kwargs: Any,
) -> Any:
    """
    Generate a Taylor Diagram (Aero Protocol - Track A).

    The Taylor Diagram summarizes three statistics: correlation, standard
    deviation, and centered Root Mean Square Error (CRMSE) on a single plot.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values (the reference).
    mod : xarray.DataArray, numpy.ndarray, or dict
        Model values to compare. If a dictionary, keys are used as labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure with polar projection
        is created.
    figsize : Tuple[int, int], optional
        Figure size. Default is (10, 8).
    title : str, optional
        Plot title. Default is 'Taylor Diagram'.
    label_obs : str, optional
        Label for the observation point. Default is 'Observations'.
    **kwargs : Any
        Additional keyword arguments passed to the point plotting (e.g., markersize).

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.

    Notes
    -----
    - Radial distance from the origin represents standard deviation.
    - Angular position represents the Pearson correlation coefficient.
    - Distance from the observation point represents the centered RMS error.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.rand(100)
    >>> mod1 = obs + np.random.randn(100) * 0.1
    >>> mod2 = obs + np.random.randn(100) * 0.5
    >>> ax = plot_taylor(obs, {'Model A': mod1, 'Model B': mod2})
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Taylor Diagram requires 'matplotlib'. Install it with 'pip install monet-stats[viz]'.")

    # Convert single mod to dict
    if not isinstance(mod, dict):
        mod_dict = {"Model": mod}
    else:
        mod_dict = mod

    # Helper to get std and corr
    def _get_stats(o, m):
        if isinstance(o, xr.DataArray) and isinstance(m, xr.DataArray):
            # Align and compute lazily
            o, m = xr.align(o, m, join="inner")
            s_o = o.std().values.item()
            s_m = m.std().values.item()
            r = xr.corr(o, m).values.item()
        else:
            from .correlation_metrics import pearsonr

            s_o = np.nanstd(o)
            s_m = np.nanstd(m)
            r = pearsonr(o, m)
        return s_o, s_m, r

    # Collect stats for all models
    stats_data = {}
    std_obs = 0.0
    for name, m_val in mod_dict.items():
        std_obs, std_mod, corr = _get_stats(obs, m_val)
        stats_data[name] = (std_mod, corr)

    # Setup polar axes if not provided
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="polar")

    # Taylor Diagrams typically only show the first quadrant (0 to 90 deg)
    # Correlation from 0 to 1
    ax.set_thetalim(0, np.pi / 2)

    # Set radial ticks (Standard Deviation)
    max_std = max(std_obs, max([s[0] for s in stats_data.values()])) * 1.1
    ax.set_rmax(max_std)

    # Radial labels
    ax.set_xlabel("Standard Deviation")

    # Theta labels (Correlation)
    r_ticks = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])
    theta_ticks = np.arccos(r_ticks)
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels([str(r) for r in r_ticks])
    ax.set_ylabel("Correlation Coefficient", labelpad=40)

    # Plot Reference point (Observation)
    ax.plot(0, std_obs, "ko", label=label_obs, markersize=10)

    # Plot CRMSE contours (optional but helpful)
    # These are circles centered at the reference point (0, std_obs)
    rs = np.linspace(0, max_std, 100)
    thetas = np.linspace(0, np.pi / 2, 100)
    R, THETA = np.meshgrid(rs, thetas)
    # Distance from (std_obs, 0) in polar coords
    # d = sqrt(R^2 + std_obs^2 - 2*R*std_obs*cos(THETA))
    CRMSE = np.sqrt(R**2 + std_obs**2 - 2 * R * std_obs * np.cos(THETA))

    levels = np.linspace(0, max_std, 6)[1:]
    contour = ax.contour(THETA, R, CRMSE, levels=levels, colors="0.5", linestyles="--", linewidths=0.5)
    ax.clabel(contour, inline=1, fontsize=10, fmt="%.2f")

    # Add a circular arc at the observation's standard deviation
    t = np.linspace(0, np.pi / 2, 100)
    ax.plot(t, np.full_like(t, std_obs), "k--", linewidth=0.8, alpha=0.5)

    # Plot models
    markers = ["o", "s", "^", "v", "D", "*", "p", "h"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(mod_dict)))

    for i, (name, (s_m, r)) in enumerate(stats_data.items()):
        theta = np.arccos(np.clip(r, -1, 1))
        ax.plot(
            theta,
            s_m,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=name,
            linestyle="None",
            markersize=kwargs.get("markersize", 8),
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(title, pad=20)

    # Update history for provenance
    if hasattr(obs, "attrs"):
        _update_history(obs, f"Generated Taylor Diagram with models: {list(mod_dict.keys())}")

    return ax
