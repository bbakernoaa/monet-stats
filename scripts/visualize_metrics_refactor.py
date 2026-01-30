import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.error_metrics import RMSE
from monet_stats.visualize import plot_spatial


def run_visualization_demo():
    print("🍃 Aero Protocol: Visualization Track A (Publication)")

    # Create synthetic spatial data
    lat = np.linspace(-90, 90, 100)
    lon = np.linspace(-180, 180, 100)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    obs_data = np.sin(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
    mod_data = obs_data + 0.1 * np.random.randn(*obs_data.shape)

    obs = xr.DataArray(obs_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="obs")
    mod = xr.DataArray(mod_data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"], name="mod")

    # Calculate spatial RMSE
    # Using the refactored RMSE which is now in error_metrics and imported via correlation_metrics if needed
    spatial_rmse = RMSE(obs, mod, axis=None)  # Scalar
    print(f"Calculated RMSE: {spatial_rmse.values if hasattr(spatial_rmse, 'values') else spatial_rmse:.4f}")

    # Plotting Track A (Static)
    fig = plt.figure(figsize=(12, 6))

    # Subplot 1: Observations
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    plot_spatial(obs, ax=ax1, title="Observations", cmap="RdBu_r", cbar_kwargs={"shrink": 0.7})

    # Subplot 2: Local Error (Absolute Difference)
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    error = abs(mod - obs)
    plot_spatial(error, ax=ax2, title="Absolute Error", cmap="Reds", cbar_kwargs={"shrink": 0.7})

    plt.tight_layout()
    plt.savefig("refactor_visualization_demo.png")
    print("✅ Visualization saved to 'refactor_visualization_demo.png'")

    print("\n⚡ Aero Protocol: Visualization Track B (Interactive)")
    print("Use `plot_spatial(da, method='hvplot')` for interactive exploration.")


if __name__ == "__main__":
    run_visualization_demo()
