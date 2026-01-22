"""
Aero Protocol Visualization Example (Two-Track Rule).

This script demonstrates the dual visualization approach:
Track A: Static Publication Quality (Matplotlib + Cartopy)
Track B: Interactive Exploration (HvPlot + Geoviews)
"""

import cartopy.crs as ccrs
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def create_sample_geodata() -> xr.DataArray:
    """
    Create a sample geospatial DataArray for demonstration.

    Returns
    -------
    xarray.DataArray
        Sample ozone anomaly data.
    """
    lon = np.linspace(-130, -60, 100)
    lat = np.linspace(20, 50, 80)
    data = np.sin(lon[None, :] / 10) * np.cos(lat[:, None] / 10)

    da = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="ozone_anomaly",
        attrs={"units": "ppb", "history": "Generated for visualization example."},
    )
    return da


def track_a_publication(da: xr.DataArray) -> None:
    """
    Track A: Static Publication Quality (Matplotlib + Cartopy).

    Parameters
    ----------
    da : xarray.DataArray
        Data to plot.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        cbar_kwargs={"label": da.attrs.get("units", "")},
    )

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.title("Ozone Anomaly (Track A - Publication)")
    import os

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/track_a_viz.png", dpi=300, bbox_inches="tight")
    print("Track A plot saved to outputs/track_a_viz.png")


def track_b_exploration(da: xr.DataArray) -> None:
    """
    Track B: Interactive Exploration (HvPlot + Geoviews).

    Parameters
    ----------
    da : xarray.DataArray
        Data to plot.
    """
    # Note: In a live environment, this would display interactively.
    # Here we save to HTML as proof of concept.
    plot = da.hvplot.quadmesh(
        x="lon",
        y="lat",
        geo=True,
        rasterize=True,
        cmap="RdBu_r",
        title="Ozone Anomaly (Track B - Interactive)",
        clabel=da.attrs.get("units", ""),
    )
    import os

    import holoviews as hv

    os.makedirs("outputs", exist_ok=True)
    hv.save(plot, "outputs/track_b_viz.html")
    print("Track B plot saved to outputs/track_b_viz.html")


if __name__ == "__main__":
    data = create_sample_geodata()

    print("Generating Track A (Static)...")
    try:
        track_a_publication(data)
    except Exception as e:
        print(f"Track A failed: {e}")

    print("Generating Track B (Interactive)...")
    try:
        track_b_exploration(data)
    except Exception as e:
        print(f"Track B failed: {e}")
