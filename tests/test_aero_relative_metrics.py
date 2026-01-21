import numpy as np
import pytest
import xarray as xr

from monet_stats import relative_metrics


def test_relative_metrics_dask_compliance():
    """Verify that relative metrics work identically for NumPy and Dask backends."""
    dask = pytest.importorskip("dask.array")

    # Create synthetic data
    data_obs = np.random.rand(10, 10)
    data_mod = np.random.rand(10, 10)

    da_obs_eager = xr.DataArray(data_obs, dims=("x", "y"), name="obs")
    da_mod_eager = xr.DataArray(data_mod, dims=("x", "y"), name="mod")

    da_obs_lazy = da_obs_eager.chunk({"x": 5})
    da_mod_lazy = da_mod_eager.chunk({"x": 5})

    # Test NMB
    res_eager = relative_metrics.NMB(da_obs_eager, da_mod_eager)
    res_lazy = relative_metrics.NMB(da_obs_lazy, da_mod_lazy)

    assert isinstance(res_lazy.data, dask.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated Normalized Mean Bias (NMB) using monet-stats." in res_lazy.attrs["history"]

    # Test ME
    res_eager = relative_metrics.ME(da_obs_eager, da_mod_eager, axis=0)
    res_lazy = relative_metrics.ME(da_obs_lazy, da_mod_lazy, axis=0)

    assert isinstance(res_lazy.data, dask.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated Mean Gross Error (ME) using monet-stats." in res_lazy.attrs["history"]

    # Test USUTPB (Peak Bias)
    res_eager = relative_metrics.USUTPB(da_obs_eager, da_mod_eager)
    res_lazy = relative_metrics.USUTPB(da_obs_lazy, da_mod_lazy)

    assert isinstance(res_lazy.data, dask.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated Unpaired Space/Unpaired Time Peak Bias (USUTPB) using monet-stats." in res_lazy.attrs["history"]


def test_relative_metrics_numpy_fallback():
    """Verify that relative metrics still work with raw NumPy arrays."""
    obs = np.array([1, 2, 3])
    mod = np.array([1.1, 2.2, 3.3])

    res = relative_metrics.NMB(obs, mod)
    assert np.isclose(res, 10.0)


def test_relative_metrics_axis_string():
    """Verify that dimension names can be used for axis."""
    data_obs = np.random.rand(10, 10)
    data_mod = np.random.rand(10, 10)

    da_obs = xr.DataArray(data_obs, dims=("time", "site"), name="obs")
    da_mod = xr.DataArray(data_mod, dims=("time", "site"), name="mod")

    res = relative_metrics.NMB(da_obs, da_mod, axis="time")
    assert res.dims == ("site",)
    assert "Calculated Normalized Mean Bias (NMB) using monet-stats." in res.attrs["history"]
