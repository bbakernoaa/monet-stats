import numpy as np
import xarray as xr

from monet_stats.analysis import calculate_grid_area, weighted_spatial_mean


def test_calculate_grid_area_regular():
    # Simple 1x1 degree grid
    lats = np.arange(-89.5, 90, 1.0)
    lons = np.arange(-179.5, 180, 1.0)
    ds = xr.Dataset(coords={"lat": lats, "lon": lons})

    areas = calculate_grid_area(ds)
    assert areas.ndim == 2
    assert areas.shape == (180, 360)

    # Area should be largest at equator and smallest at poles
    # lat index 90 is 0.5 degrees
    # lat index 0 is -89.5 degrees
    assert areas.isel(lat=90).mean() > areas.isel(lat=0).mean()
    assert areas.isel(lat=90).mean() > areas.isel(lat=179).mean()


def test_calculate_grid_area_curvilinear():
    # Create a dummy 2D curvilinear grid (slanted)
    y = np.linspace(0, 1, 10)
    x = np.linspace(0, 1, 10)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Simple linear transformation to lat/lon
    lat = xr.DataArray(yy * 10, dims=("y", "x"), coords={"y": y, "x": x})
    lon = xr.DataArray(xx * 10, dims=("y", "x"), coords={"y": y, "x": x})

    ds = xr.Dataset({"foo": (("y", "x"), np.ones((10, 10)))}, coords={"lat": lat, "lon": lon})

    areas = calculate_grid_area(ds)
    assert areas.dims == ("y", "x")
    assert areas.shape == (10, 10)
    assert np.all(areas > 0)


def test_weighted_spatial_mean_fallback():
    lats = np.array([-45, 0, 45])
    lons = np.array([-10, 0, 10])
    # Data is all 1s, but we want to see if it uses the helper
    da = xr.DataArray(
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), coords={"lat": lats, "lon": lons}, dims=("lat", "lon")
    )

    # This should trigger calculate_grid_area internally
    res = weighted_spatial_mean(da)

    # Expected: 0.5 degree of weighted average.
    # At 45 deg, cos is ~0.707. At 0 deg, cos is 1.
    # weights approx [0.7, 1, 0.7]
    # mean should be closer to the equator value (4) than a simple average
    simple_mean = da.mean()
    assert res > simple_mean
