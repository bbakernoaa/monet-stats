"""
Tests for the plugin system (Aero Protocol Compliant).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.plugin_system import CustomMetric, plugin_manager


def test_custom_metric_registration():
    """Test that a custom metric can be registered and retrieved."""

    def my_metric(obs, mod):
        return np.mean(obs + mod)

    metric = CustomMetric(name="MyMetric", description="A simple test metric", func=my_metric)

    plugin_manager.register_plugin(metric)
    assert "MyMetric" in plugin_manager.list_plugins()
    assert plugin_manager.get_plugin("MyMetric") == metric

    plugin_manager.unregister_plugin("MyMetric")
    assert "MyMetric" not in plugin_manager.list_plugins()


def test_custom_metric_compute_numpy():
    """Test custom metric computation with NumPy arrays."""

    def bias_func(obs, mod):
        return np.mean(mod - obs)

    bias_metric = CustomMetric(name="TestBias", description="Test Bias Metric", func=bias_func)

    obs = np.array([1, 2, 3])
    mod = np.array([2, 3, 4])

    res = bias_metric.compute(obs, mod)
    assert res == 1.0


def test_custom_metric_compute_xarray():
    """Test custom metric computation with Xarray DataArrays and history tracking."""

    def bias_func(obs, mod):
        return (mod - obs).mean()

    bias_metric = CustomMetric(name="TestBiasXR", description="Test Bias Metric with Xarray", func=bias_func)

    obs = xr.DataArray([1, 2, 3], dims="x", attrs={"history": "original"})
    mod = xr.DataArray([2, 3, 4], dims="x")

    res = bias_metric.compute(obs, mod)
    assert float(res) == 1.0
    assert "history" in res.attrs
    assert "Calculated TestBiasXR" in res.attrs["history"]


def test_builtin_plugins():
    """Test that built-in plugins (WMAPE, MAPE_Bias) work correctly."""
    obs = np.array([10, 20, 30])
    mod = np.array([11, 18, 33])

    # WMAPE = (sum(|11-10| + |18-20| + |33-30|) / sum(10+20+30)) * 100
    # WMAPE = ( (1 + 2 + 3) / 60 ) * 100 = (6 / 60) * 100 = 10%
    wmape_res = plugin_manager.compute_metric("WMAPE", obs, mod)
    assert pytest.approx(wmape_res) == 10.0

    # MAPE_Bias test
    # pe = (mod - obs) / |obs|
    # pe = [(11-10)/10, (18-20)/20, (33-30)/30] = [0.1, -0.1, 0.1]
    # pos_errors = (0.1 + 0 + 0.1) / 3 = 0.0666...
    # neg_errors = (0 + 0.1 + 0) / 3 = 0.0333...
    # bias = pos_errors - neg_errors = 0.0333...
    mape_bias_res = plugin_manager.compute_metric("MAPE_Bias", obs, mod)
    assert pytest.approx(mape_bias_res) == 1 / 30


try:
    import dask.array as da  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


@pytest.mark.skipif(not HAS_DASK, reason="Dask not available")
def test_custom_metric_dask():
    """Test custom metric with Dask-backed Xarray DataArrays (Lazy by Default)."""

    def sum_func(obs, mod):
        return (obs + mod).sum()

    sum_metric = CustomMetric(name="LazySum", description="Lazy sum", func=sum_func)

    obs = xr.DataArray(np.random.rand(10), dims="x").chunk({"x": 5})
    mod = xr.DataArray(np.random.rand(10), dims="x").chunk({"x": 5})

    res = sum_metric.compute(obs, mod)

    # Check that it's still a dask array (lazy)
    assert hasattr(res, "chunks")
    # Compute and verify
    assert res.compute() > 0
    assert "Calculated LazySum" in res.attrs["history"]
