"""
Visualization example for COE (Center of Mass Error) using the Aero Protocol.
Two-Track Rule:
Track A: Static publication quality (Matplotlib + Cartopy).
Track B: Interactive exploration (HvPlot + GeoViews).
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.error_metrics import COE

# 1. Create synthetic spatial data with a shift
lats = np.linspace(30, 50, 100)
lons = np.linspace(-120, -70, 100)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Gaussian peak for observations
obs_data = np.exp(-((lat_grid - 40) ** 2 / 4 + (lon_grid - -100) ** 2 / 16))
obs = xr.DataArray(obs_data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"), name="Obs")

# Shifted Gaussian peak for model
mod_data = np.exp(-((lat_grid - 42) ** 2 / 4 + (lon_grid - -95) ** 2 / 16))
mod = xr.DataArray(mod_data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"), name="Mod")

# Calculate COE
coe_val = COE(obs, mod).values.item()
print(f"Calculated COE (displacement between centroids): {coe_val:.4f} degrees")


# 2. Track A: Static Publication Plot (Matplotlib + Cartopy)
def plot_track_a(obs, mod, coe_val):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    # Obs plot
    obs.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
    axes[0].coastlines()
    axes[0].set_title("Observations (Centroid at 40N, 100W)")

    # Mod plot
    mod.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap="viridis", add_colorbar=True)
    axes[1].coastlines()
    axes[1].set_title(f"Model (Centroid at 42N, 95W)\nCOE = {coe_val:.4f}")

    plt.tight_layout()
    plt.savefig("coe_visualization_track_a.png")
    print("Track A plot saved as 'coe_visualization_track_a.png'.")


# 3. Track B: Interactive Exploration (HvPlot + GeoViews)
def plot_track_b(obs, mod):
    # Combine into dataset for easier hvplotting
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    # Use rasterize=True for large grids (Aero Protocol requirement)
    plot = ds.hvplot.quadmesh(x="lon", y="lat", geo=True, rasterize=True, cmap="viridis", subplots=True, width=500)

    # In a real environment, this would display in a notebook
    # hvplot.save(plot, "coe_visualization_track_b.html")
    print("Track B interactive plot code prepared (hvplot + geoviews).")
    return plot


if __name__ == "__main__":
    plot_track_a(obs, mod, coe_val)
    plot_track_b(obs, mod)
