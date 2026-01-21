import numpy as np
import xarray as xr

from monet_stats.performance import fast_mae, fast_rmse, memory_efficient_correlation, parallel_compute
from monet_stats.utils_stats import angular_difference, correlation, mae, rmse

try:
    import dask.array  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def test_rmse_aero():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])

    # Eager (NumPy)
    # New signature: rmse(obs, mod)
    res_np = rmse(obs, mod)

    # Eager (Xarray)
    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    res_xr = rmse(obs_xr, mod_xr)

    np.testing.assert_allclose(res_np, res_xr)
    assert "history" in res_xr.attrs

    # Lazy (Dask)
    if HAS_DASK:
        obs_lazy = obs_xr.chunk({"x": 1})
        mod_lazy = mod_xr.chunk({"x": 1})
        res_lazy = rmse(obs_lazy, mod_lazy)
        assert res_lazy.chunks is not None
        np.testing.assert_allclose(res_np, res_lazy.compute())
        assert "history" in res_lazy.attrs


def test_mae_aero():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])

    # Eager
    res_np = mae(obs, mod)
    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    res_xr = mae(obs_xr, mod_xr)
    np.testing.assert_allclose(res_np, res_xr)
    assert "history" in res_xr.attrs

    # Lazy
    if HAS_DASK:
        res_lazy = mae(obs_xr.chunk({"x": 1}), mod_xr.chunk({"x": 1}))
        np.testing.assert_allclose(res_np, res_lazy.compute())
        assert "history" in res_lazy.attrs


def test_correlation_aero():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])

    # Eager
    res_np = correlation(obs, mod)
    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    res_xr = correlation(obs_xr, mod_xr)
    np.testing.assert_allclose(res_np, res_xr)
    assert "history" in res_xr.attrs

    # Lazy
    if HAS_DASK:
        res_lazy = correlation(obs_xr.chunk({"x": 3}), mod_xr.chunk({"x": 3}))
        np.testing.assert_allclose(res_np, res_lazy.compute())
        assert "history" in res_lazy.attrs


def test_angular_difference_aero():
    a1 = np.array([10, 350, 180])
    a2 = np.array([350, 10, 170])

    # Eager
    res_np = angular_difference(a1, a2)
    expected = np.array([20, 20, 10])
    np.testing.assert_allclose(res_np, expected)

    # Xarray
    a1_xr = xr.DataArray(a1, dims="x")
    a2_xr = xr.DataArray(a2, dims="x")
    res_xr = angular_difference(a1_xr, a2_xr)
    np.testing.assert_allclose(res_xr, expected)
    assert "history" in res_xr.attrs

    # Lazy
    if HAS_DASK:
        res_lazy = angular_difference(a1_xr.chunk({"x": 1}), a2_xr.chunk({"x": 1}))
        np.testing.assert_allclose(res_lazy.compute(), expected)
        assert "history" in res_lazy.attrs


def test_performance_aliases():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])

    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")

    np.testing.assert_allclose(fast_rmse(obs, mod), rmse(obs, mod))
    np.testing.assert_allclose(fast_mae(obs, mod), mae(obs, mod))
    np.testing.assert_allclose(memory_efficient_correlation(obs, mod), correlation(obs, mod))

    res_rmse_xr = fast_rmse(obs_xr, mod_xr)
    assert "history" in res_rmse_xr.attrs


def test_parallel_compute():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    # NumPy path
    res = parallel_compute(np.mean, data, chunk_size=2)
    assert res == 2.5

    # Xarray path
    data_xr = xr.DataArray(data, dims="x")
    res_xr = parallel_compute(lambda d, axis: d.mean(dim=axis), data_xr, axis="x")
    assert res_xr == 2.5
    assert "history" in res_xr.attrs
