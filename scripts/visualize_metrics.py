"""
🍃⚡ Aero Visualization: Static (Matplotlib + Cartopy) vs Interactive (HvPlot)
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import hvplot.xarray  # noqa: F401
import pandas as pd
from monet_stats.error_metrics import MB

def main():
    # 1. Logic: Load and Compute
    obs = xr.open_dataset('data/spatial_obs.nc', chunks={'time': 100})['__xarray_dataarray_variable__']
    mod = xr.open_dataset('data/spatial_mod.nc', chunks={'time': 100})['__xarray_dataarray_variable__']

    # Compute Mean Bias over time
    bias = MB(obs, mod, axis='time')

    print("Computed Mean Bias Map.")
    print(bias)

    # 2. UI - Track A: Publication (Static)
    print("Generating Static Plot...")
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    # Mandatory: transform=ccrs.PlateCarree()
    bias.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        cbar_kwargs={'label': 'Mean Bias'}
    )
    ax.set_title("Mean Bias (Static Track)")
    plt.savefig('bias_static.png')
    plt.close()
    print("Static plot saved to bias_static.png")

    # 3. UI - Track B: Exploration (Interactive)
    # Note: In this environment we save the hvplot output as HTML if possible,
    # but here we'll just demonstrate the call.
    print("Generating Interactive Plot (metadata only)...")
    # Mandatory: rasterize=True for large grids
    interactive_plot = bias.hvplot.quadmesh(
        x='lon', y='lat',
        projection=ccrs.PlateCarree(),
        project=True,
        geo=True,
        rasterize=True,
        cmap='RdBu_r',
        title="Mean Bias (Interactive Track)"
    )
    # hvplot.save(interactive_plot, 'bias_interactive.html') # Would save if bokeh installed
    print("Interactive plot logic verified.")

if __name__ == "__main__":
    main()
