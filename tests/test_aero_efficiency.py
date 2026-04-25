"""
Tests for efficiency metrics (NSE, rNSE, mNSE, PC) under Aero Protocol.
"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from monet_stats.efficiency_metrics import NSE, PC, NSElog, mNSE, rNSE


def test_nse_mixed_types():
    """Test NSE handles mixed DataArray and ndarray inputs lazily."""
    obs_data = np.array([1.0, 2.0, 3.0, 4.0])
    mod_data = np.array([1.1, 2.1, 2.9, 4.1])

    obs = xr.DataArray(obs_data, dims="x", name="obs")
    mod = mod_data  # ndarray

    # Should promote mod to DataArray and stay in Xarray path
    res = NSE(obs, mod)
    assert isinstance(res, xr.DataArray)
    assert np.isclose(res.item(), 0.992)
    assert "NSE" in res.attrs.get("history", "")

    # Reverse types
    res2 = NSE(obs_data, xr.DataArray(mod_data, dims="x"))
    assert isinstance(res2, xr.DataArray)
    assert np.isclose(res2.item(), 0.992)


def test_nse_weighted():
    """Test NSE with area weighting."""
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="x")
    mod = xr.DataArray([1.2, 1.8, 3.2], dims="x")
    weights = xr.DataArray([1.0, 2.0, 1.0], dims="x")

    # Manual calculation
    # obs_mean_w = (1*1 + 2*2 + 3*1) / 4 = 8/4 = 2.0
    # numerator_w = (0.2^2 * 1) + (-0.2^2 * 2) + (0.2^2 * 1) = 0.04 + 0.08 + 0.04 = 0.16
    # denominator_w = (1-2)^2 * 1 + (2-2)^2 * 2 + (3-2)^2 * 1 = 1 + 0 + 1 = 2.0
    # NSE_w = 1 - 0.16 / 2.0 = 1 - 0.08 = 0.92

    res = NSE(obs, mod, weights=weights)
    assert np.isclose(res.item(), 0.92)
    assert "Weighted NSE" in res.attrs["history"]


def test_nse_laziness():
    """Verify NSE preserves Dask laziness."""
    pytest.importorskip("dask")
    obs = xr.DataArray(np.random.rand(10), dims="x").chunk({"x": 5})
    mod = xr.DataArray(np.random.rand(10), dims="x").chunk({"x": 5})

    res = NSE(obs, mod)
    assert hasattr(res.data, "dask")

    val = res.compute()
    assert not np.isnan(val)


def test_nselog():
    """Test NSElog with positive data."""
    obs = np.array([1, 10, 100])
    mod = np.array([1.1, 9.0, 110])

    # Just check if it runs and returns expected type
    res = NSElog(obs, mod)
    assert isinstance(res, (float, np.floating))
    assert res > 0.9

    # Xarray path
    obs_da = xr.DataArray(obs, dims="x")
    res_da = NSElog(obs_da, mod)
    assert isinstance(res_da, xr.DataArray)
    assert "NSElog" in res_da.attrs["history"]


def test_rnse_edge_cases():
    """Test rNSE with zeros and near-zeros."""
    obs = np.array([0.0, 1.0, 2.0])
    mod = np.array([0.0, 1.1, 1.9])

    # rNSE handles zero obs via epsilon
    # NOTE: With obs=0, denominator becomes huge (1/eps^2), making rNSE approach 1.0
    res = rNSE(obs, mod)
    assert not np.isnan(res)
    assert np.isclose(res, 1.0)

    # Perfect agreement
    res_perf = rNSE(obs, obs)
    assert np.isclose(res_perf, 1.0)


def test_mnse_weighted():
    """Test modified NSE (absolute differences) with weights."""
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="x")
    mod = xr.DataArray([1.2, 1.8, 3.2], dims="x")
    weights = xr.DataArray([1.0, 1.0, 1.0], dims="x")

    # mNSE = 1 - sum(abs(obs-mod)) / sum(abs(obs-mean))
    # mean = 2.0
    # sum_abs_diff = 0.2 + 0.2 + 0.2 = 0.6
    # sum_abs_dev = 1.0 + 0.0 + 1.0 = 2.0
    # mNSE = 1 - 0.6 / 2.0 = 0.7

    res = mNSE(obs, mod, weights=weights)
    assert np.isclose(res.item(), 0.7)


def test_pc_vectorized():
    """Test Percent Correct (PC) with vectorization and NaNs."""
    obs = xr.DataArray([[1.0, 2.0], [3.0, np.nan]], dims=("x", "y"))
    mod = xr.DataArray([[1.05, 2.5], [3.1, 4.0]], dims=("x", "y"))

    # Tolerance 0.1
    # [0,0]: abs(1.0-1.05)=0.05, tol=0.1. Correct.
    # [0,1]: abs(2.0-2.5)=0.5, tol=0.2. Incorrect.
    # [1,0]: abs(3.0-3.1)=0.1, tol=0.3. Correct.
    # [1,1]: NaN in obs. Ignored.

    # Global PC: 2 correct out of 3 valid = 66.66%
    res = PC(obs, mod)
    assert np.isclose(res.item(), 66.66666666666666)

    # Vectorized over y (dim x remains)
    # x=0: 1 correct, 2 total -> 50%
    # x=1: 1 correct, 1 total -> 100%
    res_v = PC(obs, mod, axis="y")
    assert res_v.shape == (2,)
    assert_allclose(res_v, xr.DataArray([50.0, 100.0], dims="x"))
