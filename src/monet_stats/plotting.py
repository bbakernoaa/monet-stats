"""
Visualization module for atmospheric science data (Aero Protocol Compliant).
"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .correlation_metrics import pearsonr
from .error_metrics import CRMSE
from .utils_stats import _update_history


def plot_taylor_interactive(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray, List[Union[np.ndarray, xr.DataArray]]],
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Track B: Interactive Taylor-like exploration plot (HvPlot).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray or list of these
        Model values.
    labels : list of str, optional
        Labels for the models.
    normalize : bool, optional
        Whether to normalize by observation standard deviation.
    **kwargs : Any
        Additional arguments passed to hvplot.scatter().

    Returns
    -------
    holoviews.core.Element
        The interactive plot.
    """
    try:
        import hvplot.pandas  # noqa: F401
    except ImportError:
        raise ImportError("plot_taylor_interactive requires hvplot. Install with 'pip install monet-stats[viz]'.")

    if not isinstance(mod, list):
        mod = [mod]
    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(mod))]

    std_obs = float(np.nanstd(obs))
    data = []
    for i, m in enumerate(mod):
        std_m = float(np.nanstd(m))
        corr = float(pearsonr(obs, m))
        crmse = float(CRMSE(obs, m))
        if normalize:
            data.append({"Model": labels[i], "StdDev": std_m / std_obs, "Correlation": corr, "CRMSE": crmse / std_obs})
        else:
            data.append({"Model": labels[i], "StdDev": std_m, "Correlation": corr, "CRMSE": crmse})

    df = pd.DataFrame(data)

    # Use a scatter plot for exploration
    plot = df.hvplot.scatter(
        x="StdDev",
        y="Correlation",
        by="Model",
        title="Taylor Diagram Exploration" + (" (Normalized)" if normalize else ""),
        hover_cols=["CRMSE"],
        size=100,
        **kwargs,
    )

    if isinstance(obs, xr.DataArray):
        _update_history(obs, "Created interactive Taylor exploration plot")

    return plot


def plot_taylor(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray, List[Union[np.ndarray, xr.DataArray]]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    normalize: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Create a Taylor Diagram to compare model performance (Track A).

    The Taylor Diagram summarizes the relative skill of models by plotting
    their correlation, root-mean-square error, and standard deviation
    on a single 2D plot.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray or list of these
        Model values to compare against observations.
    labels : list of str, optional
        Labels for the model points.
    colors : list of str, optional
        Colors for the model points.
    markers : list of str, optional
        Markers for the model points.
    normalize : bool, optional
        If True, normalize all statistics by the standard deviation of
        observations. Default is False.
    **kwargs : Any
        Additional keyword arguments passed to plt.plot().

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.rand(100)
    >>> mod1 = obs + np.random.rand(100) * 0.1
    >>> mod2 = obs + np.random.rand(100) * 0.5
    >>> fig = plot_taylor(obs, [mod1, mod2], labels=['Model 1', 'Model 2'])
    """
    try:
        import matplotlib.pyplot as plt
        import mpl_toolkits.axisartist.floating_axes as floating_axes
        import mpl_toolkits.axisartist.grid_finder as grid_finder
        from matplotlib.projections import PolarAxes
    except ImportError:
        raise ImportError("plot_taylor requires matplotlib. Install with 'pip install monet-stats[viz]'.")

    # Handle single model input
    if not isinstance(mod, list):
        mod = [mod]

    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(mod))]
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(mod)))
    if markers is None:
        markers = ["o"] * len(mod)

    # Compute statistics
    std_obs = float(np.nanstd(obs))
    stats = []
    for m in mod:
        std_m = float(np.nanstd(m))
        corr = float(pearsonr(obs, m))
        crmse = float(CRMSE(obs, m))
        if normalize:
            stats.append((std_m / std_obs, corr, crmse / std_obs))
        else:
            stats.append((std_m, corr, crmse))

    ref_std = 1.0 if normalize else std_obs

    # Setup Taylor Diagram geometry
    tr = PolarAxes.PolarTransform()

    # Correlation axis (angle)
    r_locs = np.concatenate((np.arange(10) / 10.0, [0.95, 0.99]))
    t_locs = np.arccos(r_locs)
    gl1 = grid_finder.FixedLocator(t_locs)
    tf1 = grid_finder.DictFormatter(dict(zip(t_locs, map(str, r_locs))))

    # Standard deviation axis (radius)
    max_std = 1.5 * max([s[0] for s in stats] + [ref_std])
    gl2 = grid_finder.MaxNLocator(5)

    gh = floating_axes.GridHelperCurveLinear(
        tr, extremes=(0, np.pi / 2, 0, max_std), grid_locator1=gl1, tick_formatter1=tf1, grid_locator2=gl2
    )

    fig = plt.figure(figsize=(8, 8))
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=gh)
    fig.add_subplot(ax)

    # Adjust axis labels
    ax.axis["top"].set_axis_direction("bottom")
    ax.axis["top"].toggle(ticklabels=True, label=True)
    ax.axis["top"].major_ticklabels.set_axis_direction("top")
    ax.axis["top"].label.set_axis_direction("top")
    ax.axis["top"].label.set_text("Correlation")

    ax.axis["left"].set_axis_direction("bottom")
    ax.axis["left"].label.set_text("Standard Deviation" + (" (Normalized)" if normalize else ""))

    ax.axis["right"].set_axis_direction("top")
    ax.axis["right"].toggle(ticklabels=True)
    ax.axis["right"].major_ticklabels.set_axis_direction("left")

    ax.axis["bottom"].set_visible(False)

    # Create auxiliary polar axis for the RMSE contours
    ax2 = ax.get_aux_axes(tr)

    # Add RMSE contours
    rs, ts = np.meshgrid(np.linspace(0, max_std, 100), np.linspace(0, np.pi / 2, 100))
    # Distance formula in polar coordinates: d^2 = r1^2 + r2^2 - 2*r1*r2*cos(t1-t2)
    # Here r2 = ref_std, t2 = 0
    rms = np.sqrt(ref_std**2 + rs**2 - 2 * ref_std * rs * np.cos(ts))
    contours = ax2.contour(ts, rs, rms, colors="0.5", linestyles="--", alpha=0.5)
    ax2.clabel(contours, inline=1, fontsize=10, fmt="%.2f")

    # Plot reference (Observations)
    ax2.plot(0, ref_std, "ko", label="Observations", markersize=8)

    # Plot models
    for i, (std, corr, _) in enumerate(stats):
        theta = np.arccos(corr)
        ax2.plot(theta, std, color=colors[i], marker=markers[i], ls="", label=labels[i], markersize=8)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    # Provenance
    if isinstance(obs, xr.DataArray):
        _update_history(obs, "Taylor Diagram comparison")

    return fig


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
