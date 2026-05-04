import numpy as np
import pytest
import xarray as xr

from monet_stats.analysis import exceedance_count, percentile
from monet_stats.data_processing import compute_anomalies, detrend_data, handle_missing_values
from monet_stats.interfaces import DataProcessor, PerformanceOptimizer
from monet_stats.performance import (
    chunk_array,
    fast_mae,
    fast_rmse,
    memory_efficient_correlation,
    parallel_compute,
)
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


def test_handle_missing_values_numpy():
    obs = np.array([1.0, np.nan, 3.0])
    mod = np.array([1.1, 2.0, np.nan])

    # Default (flattening)
    o, m = handle_missing_values(obs, mod)
    assert len(o) == 1
    assert len(m) == 1
    assert o[0] == 1.0
    assert m[0] == 1.1

    # preserve_shape
    o, m = handle_missing_values(obs, mod, preserve_shape=True)
    assert isinstance(o, np.ma.MaskedArray)
    assert o.shape == (3,)
    assert not o.mask[0]
    assert o.mask[1]
    assert o.mask[2]


def test_detrend_data_delegation():
    times = np.arange(10)
    obs = xr.DataArray(times + np.random.randn(10), dims="time", coords={"time": times})
    mod = xr.DataArray(times + 0.5 + np.random.randn(10), dims="time", coords={"time": times})

    obs_d, mod_d = detrend_data(obs, mod, method="linear")
    assert isinstance(obs_d, xr.DataArray)
    assert "history" in obs_d.attrs
    assert "Detrended (linear)" in obs_d.attrs["history"]


def test_compute_anomalies_delegation():
    times = xr.date_range("2020-01-01", periods=12, freq="MS")
    obs = xr.DataArray(np.random.rand(12), dims="time", coords={"time": times})
    mod = xr.DataArray(np.random.rand(12), dims="time", coords={"time": times})

    obs_a, mod_a = compute_anomalies(obs, mod, freq="month")
    assert isinstance(obs_a, xr.DataArray)
    assert "Anomalies (month)" in obs_a.attrs["history"]


def test_exceedance_count_numpy():
    data = np.array([1, 5, 2, 6, 3])
    count = exceedance_count(data, threshold=4)
    assert count == 2


def test_percentile_numpy():
    data = np.arange(101)
    p = percentile(data, q=50)
    assert p == 50


def test_chunk_array_optimized():
    arr = np.arange(10)
    chunks = chunk_array(arr, chunk_size=3)
    assert len(chunks) == 4  # Balanced by np.array_split: 3, 3, 2, 2
    assert chunks[0].size == 3
    assert chunks[3].size == 2


def test_legacy_deprecation():
    with pytest.warns(DeprecationWarning, match="DataProcessor is deprecated"):
        _ = DataProcessor()

    with pytest.warns(DeprecationWarning, match="PerformanceOptimizer is deprecated"):
        _ = PerformanceOptimizer()
