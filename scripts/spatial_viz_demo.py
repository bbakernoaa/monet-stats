"""
Demonstration of Two-Track Visualization for Spatial Metrics (Aero Protocol).

Track A: Static (Publication) - Matplotlib + Cartopy
Track B: Interactive (Exploration) - HvPlot + GeoViews
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import holoviews as hv
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.spatial_skill_metrics import FSS


def create_demo_data():
    """Create synthetic spatial data for demonstration."""
    lon = np.linspace(-125, -65, 100)
    lat = np.linspace(25, 50, 50)

    # Create a 'storm' in the observations
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    obs_data = np.exp(-((lon_grid + 95) ** 2 / 10 + (lat_grid - 38) ** 2 / 5))

    # Model is slightly shifted and weaker
    mod_data = 0.8 * np.exp(-((lon_grid + 92) ** 2 / 12 + (lat_grid - 40) ** 2 / 6))

    obs = xr.DataArray(obs_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="Obs")
    mod = xr.DataArray(mod_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="Mod")

    # Add units and long names for better plotting
    obs.attrs["units"] = "unit"
    obs.attrs["long_name"] = "Observations"
    mod.attrs["units"] = "unit"
    mod.attrs["long_name"] = "Model"

    return obs, mod


def track_a_publication_plot(obs, mod):
    """Track A: Static plot for publication using Cartopy."""
    print("Generating Track A Plot (Static with Cartopy)...")

    fig = plt.figure(figsize=(18, 6))
    proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=37)

    # Plot Observations
    ax1 = fig.add_subplot(1, 3, 1, projection=proj)
    ax1.add_feature(cfeature.STATES, edgecolor="gray", alpha=0.5)
    ax1.add_feature(cfeature.COASTLINE)
    obs.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap="YlGnBu", cbar_kwargs={"shrink": 0.6})
    ax1.set_title("Observations")

    # Plot Model
    ax2 = fig.add_subplot(1, 3, 2, projection=proj)
    ax2.add_feature(cfeature.STATES, edgecolor="gray", alpha=0.5)
    ax2.add_feature(cfeature.COASTLINE)
    mod.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap="YlGnBu", cbar_kwargs={"shrink": 0.6})
    ax2.set_title("Model")

    # Plot FSS vs Scale
    ax3 = fig.add_subplot(1, 3, 3)
    scales = [3, 5, 9, 15, 25, 35, 45]
    scores = [FSS(obs, mod, window_size=s, threshold=0.1).values.item() for s in scales]

    ax3.plot(scales, scores, "o-", linewidth=2, markersize=8)
    ax3.set_xlabel("Window Size (grid cells)")
    ax3.set_ylabel("FSS")
    ax3.set_title("Fractions Skill Score vs Scale")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("fss_demo_track_a.png", dpi=300)
    print("Saved Track A plot to fss_demo_track_a.png")


def track_b_interactive_plot(obs, mod):
    """Track B: Interactive plot for exploration using HvPlot."""
    print("Generating Track B Plot (Interactive with HvPlot)...")

    # Create interactive quadmesh plots with rasterization for performance
    # Mandatory: rasterize=True for large grids
    plot_obs = obs.hvplot.quadmesh(
        x="lon", y="lat", title="Observations", geo=True, coastline=True, cmap="YlGnBu", rasterize=True
    )
    plot_mod = mod.hvplot.quadmesh(
        x="lon", y="lat", title="Model", geo=True, coastline=True, cmap="YlGnBu", rasterize=True
    )

    layout = (plot_obs + plot_mod).cols(1)

    # Save as HTML for demonstration
    hv.save(layout, "fss_demo_track_b.html")
    print("Saved Track B plot to fss_demo_track_b.html")


if __name__ == "__main__":
    obs, mod = create_demo_data()

    track_a_publication_plot(obs, mod)
    track_b_interactive_plot(obs, mod)
