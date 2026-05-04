import numpy as np
import pytest
import xarray as xr

from monet_stats.correlation_metrics import RMSEs, RMSEu


def test_regression_metrics_laziness():
    """Verify that RMSEs and RMSEu preserve Dask laziness."""
    pytest.importorskip("dask.array")

    # Create Dask-backed DataArrays
    data_obs = np.random.rand(10, 10)
    data_mod = np.random.rand(10, 10)
    obs = xr.DataArray(data_obs, dims=["x", "y"]).chunk({"x": 5})
    mod = xr.DataArray(data_mod, dims=["x", "y"]).chunk({"x": 5})

    # Compute RMSEs (should be lazy)
    res_s = RMSEs(obs, mod, axis="x")
    assert hasattr(res_s.data, "chunks")

    # Compute RMSEu (should be lazy)
    res_u = RMSEu(obs, mod, axis="x")
    assert hasattr(res_u.data, "chunks")

    # Verify values match NumPy (eager) path
    res_s_computed = res_s.compute()
    res_s_expected = RMSEs(data_obs, data_mod, axis=0)
    np.testing.assert_allclose(res_s_computed.values, res_s_expected)

    res_u_computed = res_u.compute()
    res_u_expected = RMSEu(data_obs, data_mod, axis=0)
    np.testing.assert_allclose(res_u_computed.values, res_u_expected)

    # Verify history tracking
    assert "Calculated RMSEs" in res_s.attrs["history"]
    assert "Calculated RMSEu" in res_u.attrs["history"]


def test_regression_metrics_mixed_types():
    """Verify that mixed DataArray/ndarray inputs return DataArray and work correctly."""
    data_obs = np.random.rand(10)
    obs = xr.DataArray(data_obs, dims=["t"], coords={"t": np.arange(10)})
    data_mod = np.random.rand(10)

    # Case 1: obs is DataArray, mod is ndarray
    res = RMSEs(obs, data_mod, axis="t")
    assert isinstance(res, xr.DataArray)
    expected = RMSEs(data_obs, data_mod)
    np.testing.assert_allclose(res.values, expected)

    # Case 2: mod is DataArray, obs is ndarray
    mod = xr.DataArray(data_mod, dims=["t"], coords={"t": np.arange(10)})
    res2 = RMSEu(data_obs, mod, axis="t")
    assert isinstance(res2, xr.DataArray)
    expected2 = RMSEu(data_obs, data_mod)
    np.testing.assert_allclose(res2.values, expected2)


def test_regression_metrics_nan_handling():
    """Verify that RMSEs and RMSEu handle NaNs consistently."""
    obs = np.array([1.0, 2.0, np.nan, 4.0])
    mod = np.array([1.1, np.nan, 3.1, 3.9])

    # Expected: only points [0, 3] used
    # x = [1, 4], y = [1.1, 3.9]
    # m = (3.9-1.1)/(4-1) = 2.8/3 = 0.9333...
    # b = 1.1 - m*1 = 1.1 - 0.9333... = 0.1666...
    # Regression fit: y_hat = m*x + b
    # x=1 -> y_hat = 1.1
    # x=4 -> y_hat = 3.9
    # RMSEs = sqrt(mean((y_hat - x)^2)) = sqrt(((1.1-1)^2 + (3.9-4)^2)/2) = sqrt((0.1^2 + (-0.1)^2)/2) = 0.1

    res_s = RMSEs(obs, mod)
    np.testing.assert_allclose(res_s, 0.1)

    # Xarray path
    obs_da = xr.DataArray(obs, dims="t")
    mod_da = xr.DataArray(mod, dims="t")
    res_s_da = RMSEs(obs_da, mod_da, axis="t")
    np.testing.assert_allclose(res_s_da.values, 0.1)
