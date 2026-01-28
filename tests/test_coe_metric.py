import numpy as np
import pytest
import xarray as xr

from monet_stats.error_metrics import COE


def test_coe_numpy_basic():
    """Test COE with simple numpy arrays."""
    obs = np.zeros((5, 5))
    obs[2, 2] = 1.0  # Centroid at (2, 2)
    mod = np.zeros((5, 5))
    mod[3, 3] = 1.0  # Centroid at (3, 3)

    # Distance between (2, 2) and (3, 3) is sqrt(1^2 + 1^2) = sqrt(2)
    res = COE(obs, mod)
    assert np.isclose(res, np.sqrt(2))


def test_coe_numpy_multi_peak():
    """Test COE with multiple peaks in numpy arrays."""
    obs = np.zeros((5, 5))
    obs[0, 0] = 1.0
    obs[4, 4] = 1.0  # Centroid at (2, 2)

    mod = np.zeros((5, 5))
    mod[1, 1] = 1.0
    mod[3, 3] = 1.0  # Centroid at (2, 2)

    res = COE(obs, mod)
    assert np.isclose(res, 0.0)


def test_coe_xarray_basic():
    """Test COE with xarray DataArrays using dimension indices."""
    obs = xr.DataArray(np.zeros((5, 5)), dims=("y", "x"))
    obs.values[2, 2] = 1.0
    mod = xr.DataArray(np.zeros((5, 5)), dims=("y", "x"))
    mod.values[3, 3] = 1.0

    res = COE(obs, mod)
    assert np.isclose(res, np.sqrt(2))
    assert "history" in res.attrs


def test_coe_xarray_coords():
    """Test COE with xarray DataArrays using explicit coordinates."""
    lats = np.array([10, 20, 30, 40, 50])
    lons = np.array([100, 110, 120, 130, 140])

    obs = xr.DataArray(np.zeros((5, 5)), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    obs.loc[30, 120] = 1.0  # Centroid at (30, 120)

    mod = xr.DataArray(np.zeros((5, 5)), coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    mod.loc[40, 130] = 1.0  # Centroid at (40, 130)

    # Distance between (30, 120) and (40, 130) is sqrt(10^2 + 10^2) = sqrt(200) approx 14.14
    res = COE(obs, mod)
    assert np.isclose(res, np.sqrt(200))


def test_coe_lazy_dask():
    """Test COE with Dask-backed xarray arrays to ensure laziness."""
    pytest.importorskip("dask")
    import dask.array as da

    obs_data = da.zeros((10, 10), chunks=(5, 5))
    obs_data[2, 2] = 1.0
    obs = xr.DataArray(obs_data, dims=("y", "x"))

    mod_data = da.zeros((10, 10), chunks=(5, 5))
    mod_data[4, 4] = 1.0
    mod = xr.DataArray(mod_data, dims=("y", "x"))

    res = COE(obs, mod)

    # Check if result is still lazy (has dask graph)
    assert hasattr(res.data, "dask")

    # Compute and verify
    computed_res = res.compute()
    assert np.isclose(computed_res, np.sqrt(8))


def test_coe_zero_sum():
    """Test COE with zero-sum fields."""
    obs = np.zeros((5, 5))
    mod = np.zeros((5, 5))

    # Should not crash and return 0
    res = COE(obs, mod)
    assert np.isclose(res, 0.0)
