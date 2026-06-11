import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.spatial_ensemble_metrics import SAL
from monet_stats.visualize import plot_spatial


# 1. Create synthetic spatial data with objects
def create_data():
    lat = np.linspace(30, 50, 100)
    lon = np.linspace(-120, -70, 100)
    obs_data = np.zeros((100, 100))
    mod_data = np.zeros((100, 100))

    # Obs: one large object
    obs_data[30:50, 30:50] = 5.0
    # Mod: shifted and slightly smaller object
    mod_data[40:55, 45:60] = 4.0

    obs = xr.DataArray(obs_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])
    mod = xr.DataArray(mod_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])

    # Add a time dimension to demonstrate vectorization
    obs_3d = xr.concat([obs, obs], dim="time").assign_coords(time=[0, 1])
    mod_3d = xr.concat([mod, mod * 0.8], dim="time").assign_coords(time=[0, 1])

    return obs_3d, mod_3d


obs, mod = create_data()

# 2. Compute SAL
# Since SAL is vectorized, S, A, L will have a 'time' dimension
S, A, L = SAL(obs, mod, threshold=1.0)

print("SAL Results:")
print(f"S: {S.values}")
print(f"A: {A.values}")
print(f"L: {L.values}")

# 3. Visualize a component (Track A - Static)
# We'll plot the S component at time 0
# plot_spatial expects a DataArray with lat/lon
ax = plot_spatial(obs.isel(time=0), title="Observations (Time 0)")
plt.savefig("obs_demo.png")
plt.close()

ax = plot_spatial(mod.isel(time=0), title="Model (Time 0)")
plt.savefig("mod_demo.png")
plt.close()

print("\nSaved demo plots to 'obs_demo.png' and 'mod_demo.png'.")
