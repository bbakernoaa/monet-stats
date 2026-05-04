"""
Tests for optimized correlation metrics (Aero Protocol).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.correlation_metrics import R2, pearsonr


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    size = 100
    np.random.seed(42)
    obs = np.random.normal(10, 2, size)
    mod = obs + np.random.normal(0, 0.5, size)

    # Add some NaNs
    obs[0] = np.nan
    mod[1] = np.nan

    return obs, mod


def test_pearsonr_consistency(sample_data):
    """Test pearsonr consistency across NumPy and Xarray."""
    obs_np, mod_np = sample_data

    # NumPy Path
    r_np = pearsonr(obs_np, mod_np)

    # Xarray Path
    obs_xr = xr.DataArray(obs_np, dims="time")
    mod_xr = xr.DataArray(mod_np, dims="time")
    r_xr = pearsonr(obs_xr, mod_xr)

    assert isinstance(r_xr, xr.DataArray)
    np.testing.assert_allclose(r_np, r_xr.values, atol=1e-10)
    assert "pearsonr" in r_xr.attrs["history"]


def test_r2_consistency(sample_data):
    """Test R2 consistency across NumPy and Xarray."""
    obs_np, mod_np = sample_data

    # NumPy Path
    r2_np = R2(obs_np, mod_np)

    # Xarray Path
    obs_xr = xr.DataArray(obs_np, dims="time")
    mod_xr = xr.DataArray(mod_np, dims="time")
    r2_xr = R2(obs_xr, mod_xr)

    assert isinstance(r2_xr, xr.DataArray)
    np.testing.assert_allclose(r2_np, r2_xr.values, atol=1e-10)
    assert "R2" in r2_xr.attrs["history"]


def test_pearsonr_dask_laziness(sample_data):
    """Test pearsonr with Dask-backed Xarray objects."""
    pytest.importorskip("dask")
    obs_np, mod_np = sample_data

    obs_xr = xr.DataArray(obs_np, dims="time").chunk({"time": 50})
    mod_xr = xr.DataArray(mod_np, dims="time").chunk({"time": 50})

    r_xr = pearsonr(obs_xr, mod_xr)

    # Verify it is still a dask array
    assert hasattr(r_xr.data, "chunks")

    # Compute and check result
    r_computed = r_xr.compute()
    r_expected = pearsonr(obs_np, mod_np)
    np.testing.assert_allclose(r_computed.values, r_expected, atol=1e-10)


def test_pearsonr_multidim(sample_data):
    """Test multidimensional reduction."""
    obs_np = np.random.rand(10, 10, 10)
    mod_np = obs_np + np.random.rand(10, 10, 10) * 0.1

    obs_xr = xr.DataArray(obs_np, dims=["time", "lat", "lon"])
    mod_xr = xr.DataArray(mod_np, dims=["time", "lat", "lon"])

    # Reduce over lat and lon
    r_xr = pearsonr(obs_xr, mod_xr, axis=["lat", "lon"])

    assert r_xr.dims == ("time",)
    assert r_xr.shape == (10,)

    # Check one value manually
    r_expected_0 = pearsonr(obs_np[0, :, :].ravel(), mod_np[0, :, :].ravel())
    np.testing.assert_allclose(r_xr.values[0], r_expected_0, atol=1e-10)
