
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import hvplot.xarray  # noqa: F401
from monet_stats.error_metrics import STDO

# 1. Setup Sample Data (Aero-style: Lazy/Dask)
lon = np.linspace(-130, -60, 100)
lat = np.linspace(20, 50, 100)
obs_data = xr.DataArray(
    np.random.rand(100, 100),
    coords=[lat, lon],
    dims=["lat", "lon"],
    name="obs"
).chunk({"lat": 50, "lon": 50})

mod_data = obs_data + 0.1 * np.random.randn(100, 100)
mod_data.name = "mod"

# 2. Compute Metric
# Aero Note: This stays lazy until plot call
stdo_field = STDO(obs_data, mod_data, axis=None) # Scalar in this case, let's do it over time if we had it
# For viz, let's assume we want a spatial map of errors (not possible with STDO over all axes,
# so let's compute it over a 'time' dimension if we had one)

# Mocking time dimension
obs_t = xr.concat([obs_data for _ in range(10)], dim="time").chunk({"time": 1})
mod_t = xr.concat([mod_data for _ in range(10)], dim="time").chunk({"time": 1})
stdo_map = STDO(obs_t, mod_t, axis="time")

# --- TRACK A: Publication (Static) ---
fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
stdo_map.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    cbar_kwargs={'label': 'STDO'}
)
ax.coastlines()
ax.set_title("Observation Error Standard Deviation (Static)")
plt.savefig("stdo_static_plot.png")

# --- TRACK B: Exploration (Interactive) ---
# Aero Note: rasterize=True for performance on large grids
interactive_plot = stdo_map.hvplot.quadmesh(
    x='lon', y='lat',
    geo=True,
    rasterize=True,
    cmap='viridis',
    title="STDO (Interactive)"
)
# hvplot.save(interactive_plot, "stdo_interactive.html")
print("Aero Visualization Protocol: Static plot saved, Interactive plot object created.")
