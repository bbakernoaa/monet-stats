"""
Tests for Aero Protocol optimization and accessor enhancements.
"""

import numpy as np
import xarray as xr


def test_da_optimize():
    """Test optimize() method on DataArray."""
    data = np.random.rand(100, 100)
    da = xr.DataArray(data, dims=["x", "y"], name="test")

    # Not lazy initially
    assert not hasattr(da.data, "chunks")

    # Optimize it (force laziness by using small target)
    da_opt = da.monet_stats.optimize(target_mb=0.01)

    # Should be lazy now
    assert hasattr(da_opt.data, "chunks")
    assert "Optimized for performance" in da_opt.attrs.get("history", "")


def test_ds_optimize():
    """Test optimize() method on Dataset."""
    data = np.random.rand(100, 100)
    ds = xr.Dataset({"v1": (["x", "y"], data)})

    # Optimize it
    ds_opt = ds.monet_stats.optimize(target_mb=0.01)

    # Should be lazy now
    assert hasattr(ds_opt.v1.data, "chunks")
    assert "Optimized for performance" in ds_opt.attrs.get("history", "")


def test_da_rechunk():
    """Test rechunk() method on DataArray."""
    data = np.random.rand(100, 100)
    da = xr.DataArray(data, dims=["x", "y"], name="test").chunk({"x": 50, "y": 50})

    # Rechunk it
    da_rechunked = da.monet_stats.rechunk({"x": 10, "y": 10})

    assert da_rechunked.chunks == ((10,) * 10, (10,) * 10)
    assert "Rechunked" in da_rechunked.attrs.get("history", "")


def test_taylor_statistics():
    """Test taylor_statistics() method."""
    obs_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mod_data = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    obs = xr.DataArray(obs_data, dims="time", name="obs")
    mod = xr.DataArray(mod_data, dims="time", name="mod")

    stats = mod.monet_stats.taylor_statistics(obs)

    assert isinstance(stats, xr.Dataset)
    assert "std_obs" in stats
    assert "std_mod" in stats
    assert "correlation" in stats

    assert np.isclose(stats.std_obs, np.std(obs_data), atol=1e-5)
    assert np.isclose(stats.std_mod, np.std(mod_data), atol=1e-5)
    assert "Taylor statistics components" in stats.attrs.get("history", "")


def test_error_metrics_resolution():
    """Test that refactored error metrics handle dimension resolution correctly."""
    from monet_stats.error_metrics import MAE

    data1 = np.random.rand(10, 5)
    data2 = np.random.rand(10, 5)

    da1 = xr.DataArray(data1, dims=["lat", "lon"], coords={"lat": np.arange(10), "lon": np.arange(5)})
    da2 = xr.DataArray(data2, dims=["lat", "lon"], coords={"lat": np.arange(10), "lon": np.arange(5)})

    # Test with string dim
    res_str = MAE(da1, da2, axis="lat")
    assert res_str.dims == ("lon",)

    # Test with integer axis
    res_int = MAE(da1, da2, axis=0)
    xr.testing.assert_allclose(res_str, res_int)

    # Test with multiple dims
    res_multi = MAE(da1, da2, axis=["lat", "lon"])
    assert res_multi.ndim == 0
