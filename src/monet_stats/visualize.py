"""
Visualization utilities for the MONET Stats package (Aero Protocol Compliant).

This module implements the "Two-Track Rule" for scientific visualization:
- Track A (Publication): Static quality using Matplotlib and Cartopy.
- Track B (Exploration): Interactive exploration using HvPlot and GeoViews.
"""

from typing import Any, Optional, Union

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


def plot_performance_diagram(
    sr: Union[np.ndarray, xr.DataArray],
    pod: Union[np.ndarray, xr.DataArray],
    ax: Optional[Any] = None,
    title: Optional[str] = "Performance Diagram",
    label: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot a Performance Diagram (Roebber Diagram) (Aero Protocol).

    A performance diagram maps Success Ratio (1 - FAR) and Probability of
    Detection (POD), with isolines for Critical Success Index (CSI) and
    Frequency Bias.

    Parameters
    ----------
    sr : numpy.ndarray or xarray.DataArray
        Success Ratio (1 - FAR). Values must be between 0 and 1.
    pod : numpy.ndarray or xarray.DataArray
        Probability of Detection. Values must be between 0 and 1.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    title : str, optional
        Title for the plot. Default is "Performance Diagram".
    label : str, optional
        Label for the data points (for legend).
    **kwargs : Any
        Additional keyword arguments passed to `ax.scatter()`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the performance diagram.

    Notes
    -----
    The performance diagram is a standard tool for meteorological and air quality
    model verification (Roebber, 2009).

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> sr = np.array([0.6, 0.7, 0.8])
    >>> pod = np.array([0.5, 0.6, 0.7])
    >>> ax = plot_performance_diagram(sr, pod, label="Model A")
    >>> plt.legend()
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("plot_performance_diagram requires 'matplotlib'. Install it with 'pip install matplotlib'.")

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 8)))

    # 1. Create CSI isolines
    # CSI = 1 / (1/SR + 1/POD - 1)
    sr_grid, pod_grid = np.meshgrid(np.linspace(0.01, 1.0, 100), np.linspace(0.01, 1.0, 100))
    csi = 1.0 / (1.0 / sr_grid + 1.0 / pod_grid - 1.0)
    csi_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    csi_contour = ax.contour(sr_grid, pod_grid, csi, levels=csi_levels, colors="lightgrey", linestyles="--", alpha=0.8)
    ax.clabel(csi_contour, inline=True, fontsize=10, fmt="%.1f")

    # 2. Create Frequency Bias lines
    # Bias = POD / SR => POD = Bias * SR
    bias_levels = [0.3, 0.5, 0.8, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]
    for b in bias_levels:
        ax.plot([0, 1], [0, b], color="lightgrey", linestyle="-", linewidth=0.8, alpha=0.5)
        # Position label at the edge
        if b <= 1.0:
            ax.text(1.0, b, f" {b}", verticalalignment="center", color="grey", fontsize=9)
        else:
            ax.text(
                1.0 / b,
                1.0,
                f" {b}",
                horizontalalignment="center",
                verticalalignment="bottom",
                color="grey",
                fontsize=9,
            )

    # 3. Plot data points
    marker = kwargs.pop("marker", "o")
    s = kwargs.pop("s", 50)
    ax.scatter(sr, pod, marker=marker, s=s, label=label, **kwargs)

    # 4. Formatting
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Success Ratio (1 - FAR)")
    ax.set_ylabel("Probability of Detection (POD)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.2)

    # Add labels for corners
    ax.text(0.95, 0.05, "Rare Events", ha="right", va="bottom", alpha=0.3, fontsize=10)
    ax.text(0.05, 0.95, "Frequent Events", ha="left", va="top", alpha=0.3, fontsize=10)

    if isinstance(sr, xr.DataArray):
        _update_history(sr, "plot_performance_diagram")
    if isinstance(pod, xr.DataArray):
        _update_history(pod, "plot_performance_diagram")

    return ax
