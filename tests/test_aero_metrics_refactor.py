import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

from monet_stats.efficiency_metrics import NSE, PC, NSElog
from monet_stats.error_metrics import IOA, MSE, RMSE


def test_mse_consistency():
    obs = np.array([1, 2, 3])
    mod = np.array([2, 2, 4])
    # MSE = ((2-1)^2 + (2-2)^2 + (4-3)^2) / 3 = (1 + 0 + 1) / 3 = 0.666...
    expected = 2 / 3
    assert np.allclose(MSE(obs, mod), expected)

    # Xarray
    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    assert np.allclose(MSE(obs_xr, mod_xr), expected)
    assert "history" in MSE(obs_xr, mod_xr).attrs


def test_rmse_consistency():
    obs = np.array([1, 2, 3])
    mod = np.array([2, 2, 4])
    expected = np.sqrt(2 / 3)
    assert np.allclose(RMSE(obs, mod), expected)

    # Xarray
    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    assert np.allclose(RMSE(obs_xr, mod_xr), expected)


def test_ioa_consistency():
    obs = np.array([1, 2, 3])
    mod = np.array([2, 2, 4])
    # mean_obs = 2
    # num = (1-2)^2 + (2-2)^2 + (3-4)^2 = 1 + 0 + 1 = 2
    # denom = (|2-2| + |1-2|)^2 + (|2-2| + |2-2|)^2 + (|4-2| + |3-2|)^2
    #       = (0+1)^2 + (0+0)^2 + (2+1)^2 = 1 + 0 + 9 = 10
    # IOA = 1 - 2/10 = 0.8
    expected = 0.8
    assert np.allclose(IOA(obs, mod), expected)

    # Xarray
    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    assert np.allclose(IOA(obs_xr, mod_xr), expected)


def test_nselog_backend():
    obs = np.array([1, 10, 100])
    mod = np.array([1.1, 9.0, 110])

    res_np = NSElog(obs, mod)

    obs_xr = xr.DataArray(obs, dims="x")
    mod_xr = xr.DataArray(mod, dims="x")
    res_xr = NSElog(obs_xr, mod_xr)

    assert np.allclose(res_np, res_xr)
    assert isinstance(res_xr, xr.DataArray)
    assert "history" in res_xr.attrs


def test_pc_nan_handling():
    obs = np.array([1, 2, np.nan, 4])
    mod = np.array([1.05, 2.5, 3.0, 4.05])
    # Valid pairs: (1, 1.05), (2, 2.5), (4, 4.05)
    # Correct (tol=0.1):
    # 1: |1-1.05|=0.05 <= 0.1*1=0.1 (Correct)
    # 2: |2-2.5|=0.5 > 0.1*2=0.2 (Incorrect)
    # 4: |4-4.05|=0.05 <= 0.1*4=0.4 (Correct)
    # PC = 2/3 * 100 = 66.666...
    expected = 200 / 3
    assert np.allclose(PC(obs, mod), expected)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_dask_compatibility():
    obs = xr.DataArray(da.random.random((10, 10), chunks=5), dims=("x", "y"))
    mod = xr.DataArray(da.random.random((10, 10), chunks=5), dims=("x", "y"))

    res = NSE(obs, mod)
    assert hasattr(res.data, "chunks")
    computed = res.compute()
    assert not np.isnan(computed)
