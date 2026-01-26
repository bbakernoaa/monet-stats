import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.error_metrics import COE

# 1. Generate Synthetic Data
lat = np.linspace(30, 50, 50)
lon = np.linspace(-120, -70, 50)
lon_grid, lat_grid = np.meshgrid(lon, lat)


def gaussian_2d(x, y, x0, y0, sigma):
    """
    Generate a 2D Gaussian distribution.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates.
    y : numpy.ndarray
        Y coordinates.
    x0 : float
        Center X coordinate.
    y0 : float
        Center Y coordinate.
    sigma : float
        Standard deviation.

    Returns
    -------
    numpy.ndarray
        2D Gaussian distribution.
    """
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


# Observation: Peak at (-100, 40)
obs_data = gaussian_2d(lon_grid, lat_grid, -100, 40, 5)
obs = xr.DataArray(obs_data, coords=[lat, lon], dims=["lat", "lon"], name="obs")

# Model: Peak at (-90, 42) - Shifted
mod_data = gaussian_2d(lon_grid, lat_grid, -90, 42, 5)
mod = xr.DataArray(mod_data, coords=[lat, lon], dims=["lat", "lon"], name="mod")

# Calculate COE
coe_val = COE(obs, mod)
print(f"Center of Mass Error: {coe_val.values:.4f} degrees")

# 2. Track A: Publication-Quality Static Plot (Matplotlib + Cartopy)
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
obs.plot(ax=ax1, transform=ccrs.PlateCarree(), cmap="Blues", add_colorbar=True)
ax1.coastlines()
ax1.set_title("Observations")

ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
mod.plot(ax=ax2, transform=ccrs.PlateCarree(), cmap="Reds", add_colorbar=True)
ax2.coastlines()
ax2.set_title(f"Model (COE={coe_val.values:.2f})")

plt.tight_layout()
plt.savefig("coe_viz_static.png")
print("Saved static plot to coe_viz_static.png")

# 3. Track B: Interactive Exploration (HvPlot)
# Note: In a real environment, this would open in a browser.
# We use rasterize=True for larger grids as per Aero Protocol.
interactive_plot = (
    obs.hvplot.image(x="lon", y="lat", title="Observations", cmap="Blues", rasterize=True)
    + mod.hvplot.image(x="lon", y="lat", title="Model", cmap="Reds", rasterize=True)
).cols(1)

# To 'provide' the code, we just show it here.
# hvplot.save(interactive_plot, "coe_viz_interactive.html")
