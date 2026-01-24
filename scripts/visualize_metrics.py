"""
Example script demonstrating the Aero 'Two-Track' Visualization Rule.

Track A (Publication): Matplotlib + Cartopy for static, publication-quality maps.
Track B (Exploration): HvPlot + GeoViews for interactive, scalable data exploration.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def create_sample_data():
    """Create sample geospatial data for visualization."""
    lon = np.linspace(-130, -60, 100)
    lat = np.linspace(20, 55, 100)
    data = np.sin(np.deg2rad(lon[None, :])) * np.cos(np.deg2rad(lat[:, None]))

    ds = xr.Dataset(
        {"pollution": (("lat", "lon"), data)},
        coords={"lat": lat, "lon": lon},
    )
    ds.pollution.attrs["units"] = "µg/m³"
    ds.pollution.attrs["long_name"] = "Sample Air Pollution"
    return ds


def track_a_publication_plot(ds: xr.Dataset, output_file: str = "publication_plot.png"):
    """
    Track A: Static Publication-Quality Plot.

    Uses Matplotlib and Cartopy with explicit projections.
    """
    fig = plt.figure(figsize=(12, 8))
    # Requirement: projection= in axes
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, edgecolor="gray", alpha=0.5)

    # Requirement: transform= in plot calls
    ds.pollution.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        cbar_kwargs={"label": ds.pollution.attrs["long_name"] + " (" + ds.pollution.attrs["units"] + ")"},
    )

    ax.set_title("Aero Protocol: Track A (Publication Plot)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved publication plot to {output_file}")
    plt.close()


def track_b_exploration_plot(ds: xr.Dataset):
    """
    Track B: Interactive Exploration Plot.

    Uses HvPlot/GeoViews with rasterization for scalability.
    """
    # Requirement: rasterize=True for large grids
    plot = ds.pollution.hvplot.quadmesh(
        "lon",
        "lat",
        projection=ccrs.PlateCarree(),
        geo=True,
        cmap="viridis",
        rasterize=True,
        title="Aero Protocol: Track B (Interactive Exploration)",
        clabel=ds.pollution.attrs["long_name"],
    )

    # In a real notebook, you would return this.
    # Here we just show how it's constructed.
    return plot


if __name__ == "__main__":
    ds = create_sample_data()

    print("Generating Track A Plot...")
    track_a_publication_plot(ds)

    print("Generating Track B Plot structure...")
    interactive_plot = track_b_exploration_plot(ds)
    # To save as HTML for viewing
    # gv.save(interactive_plot, "interactive_plot.html")
    print("Track B plot structure created successfully.")
