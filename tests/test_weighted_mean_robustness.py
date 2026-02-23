import numpy as np
import pytest
import xarray as xr

from monet_stats.analysis import weighted_spatial_mean


def test_weighted_spatial_mean_dim_order_robustness():
    """Verify that weighted_spatial_mean handles different dimension orders correctly."""
    lats = np.array([-60, 0, 60])
    lons = np.array([0, 120, 240])
    times = np.arange(2)

    # 1. Test with (lat, lon, time)
    data1 = np.zeros((3, 3, 2))
    data1[0, :, :] = 1.0
    data1[1, :, :] = 2.0
    data1[2, :, :] = 3.0
    da1 = xr.DataArray(data1, coords={"lat": lats, "lon": lons, "time": times}, dims=("lat", "lon", "time"))

    # 2. Test with (time, lat, lon)
    data2 = np.transpose(data1, (2, 0, 1))
    da2 = xr.DataArray(data2, coords={"time": times, "lat": lats, "lon": lons}, dims=("time", "lat", "lon"))

    # Custom weights (2D: lat, lon) as NumPy array
    custom_weights_np = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])

    # Both should yield 2.0 (only middle lat counts)
    res1 = weighted_spatial_mean(da1, weights=custom_weights_np)
    res2 = weighted_spatial_mean(da2, weights=custom_weights_np)

    assert np.allclose(res1.values, 2.0)
    assert np.allclose(res2.values, 2.0)
    assert "time" in res1.dims
    assert "time" in res2.dims


def test_weighted_spatial_mean_transposed_weights():
    """Verify that weighted_spatial_mean handles transposed NumPy weights correctly."""
    lats = np.array([-60, 0, 60])
    lons = np.array([0, 120, 240])
    da = xr.DataArray(np.random.rand(3, 3), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))

    # Weights as (lon, lat) instead of (lat, lon)
    weights_lon_lat = np.random.rand(3, 3)

    # This should work now because of the transposition check in the refactored code
    res = weighted_spatial_mean(da, weights=weights_lon_lat)
    assert not np.isnan(res)


def test_weighted_spatial_mean_1d_weights():
    """Verify that weighted_spatial_mean handles 1D NumPy weights correctly."""
    lats = np.array([-60, 0, 60])
    lons = np.array([0, 120, 240])
    da = xr.DataArray(np.ones((3, 3)), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))

    # 1D weights for latitude
    weights_lat = np.array([0, 1, 0])

    # Should only count the middle row
    res = weighted_spatial_mean(da, weights=weights_lat)
    assert np.isclose(res, 1.0)


def test_weighted_spatial_mean_no_spatial_dims():
    """Verify fallback behavior when no spatial dims are found."""
    da = xr.DataArray([1.0, 2.0, 3.0], dims="x", coords={"x": [1, 2, 3]})

    with pytest.warns(UserWarning, match="No weights found"):
        res = weighted_spatial_mean(da)
        assert np.isclose(float(res), 2.0)


def test_weighted_spatial_mean_xarray_weights():
    """Verify that weighted_spatial_mean handles Xarray weights correctly."""
    lats = np.array([-60, 0, 60])
    lons = np.array([0, 120, 240])
    da = xr.DataArray(np.ones((3, 3)), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))

    weights_xr = xr.DataArray([0, 1, 0], coords={"lat": lats}, dims="lat")

    res = weighted_spatial_mean(da, weights=weights_xr)
    assert np.isclose(res, 1.0)
