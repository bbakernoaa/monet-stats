"""
Tests for data_processing.py module.
"""

import numpy as np
import pandas as pd
import xarray as xr
import pytest
import dask.array as da

from monet_stats.data_processing import (
    align_arrays,
    compute_anomalies,
    detrend_data,
    handle_missing_values,
    normalize_data,
)


@pytest.fixture
def sample_data():
    """Create sample xarray DataArrays for testing."""
    obs_da = xr.DataArray(
        [1, 2, np.nan, 4, 5],
        dims=["time"],
        coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])},
    )
    mod_da = xr.DataArray(
        [1.1, 2.2, 3.3, np.nan, 5.5],
        dims=["time"],
        coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])},
    )
    return obs_da, mod_da


@pytest.fixture
def dask_sample_data(sample_data):
    """Create sample Dask-backed xarray DataArrays for testing."""
    obs_da, mod_da = sample_data
    obs_dask = obs_da.chunk({"time": 2})
    mod_dask = mod_da.chunk({"time": 2})
    return obs_dask, mod_dask


class TestDataProcessing:
    """Test suite for data processing functions."""

    def test_align_arrays(self):
        """Test that align_arrays aligns xarray arrays."""
        obs_xr = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": [0, 1, 2]})
        mod_xr = xr.DataArray([4, 5, 6], dims=["time"], coords={"time": [1, 2, 3]})
        obs_aligned, mod_aligned = align_arrays(obs_xr, mod_xr)
        assert obs_aligned.shape == (2,)
        assert mod_aligned.shape == (2,)
        xr.testing.assert_equal(obs_aligned, xr.DataArray([2, 3], dims=["time"], coords={"time": [1, 2]}))

    def test_handle_missing_values(self, sample_data):
        """Test that handle_missing_values removes NaNs."""
        obs, mod = sample_data
        obs_clean, mod_clean = handle_missing_values(obs, mod)
        assert obs_clean.notnull().all()
        assert mod_clean.notnull().all()
        assert len(obs_clean) == 3

    def test_normalize_data(self, sample_data):
        """Test that normalize_data normalizes data."""
        obs, mod = handle_missing_values(*sample_data)  # Use clean data
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        assert np.isclose(obs_norm.mean(), 0)
        assert np.isclose(obs_norm.std(), 1)

    def test_detrend_data(self):
        """Test that detrend_data removes trends."""
        time = pd.date_range("2024-01-01", periods=5)
        obs = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={"time": time})
        mod = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={"time": time})
        obs_detrended, _ = detrend_data(obs, mod, method="linear", dim="time")
        assert np.allclose(obs_detrended.values, 0, atol=1e-9)

    def test_compute_anomalies(self):
        """Test that compute_anomalies computes anomalies."""
        time = pd.date_range("2024-01-01", periods=365, freq="D")
        data = xr.DataArray(np.random.rand(365) + np.sin(np.linspace(0, 2 * np.pi, 365)), dims=["time"], coords={"time": time})
        anomalies = compute_anomalies(data, groupby="time.month")
        # The mean of monthly anomalies should be close to zero
        assert np.isclose(anomalies.mean(), 0, atol=1e-9)


class TestDataProcessingDask:
    """Test suite for data processing functions with Dask-backed arrays."""

    def test_align_arrays_dask(self):
        """Test align_arrays with Dask arrays."""
        obs_xr = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": [0, 1, 2]}).chunk()
        mod_xr = xr.DataArray([4, 5, 6], dims=["time"], coords={"time": [1, 2, 3]}).chunk()
        obs_aligned, mod_aligned = align_arrays(obs_xr, mod_xr)
        assert isinstance(obs_aligned.data, da.Array)
        xr.testing.assert_equal(obs_aligned.compute(), xr.DataArray([2, 3], dims=["time"], coords={"time": [1, 2]}))

    def test_handle_missing_values_dask(self, dask_sample_data):
        """Test handle_missing_values with Dask arrays."""
        obs, mod = dask_sample_data
        obs_clean, mod_clean = handle_missing_values(obs, mod)
        assert isinstance(obs_clean.data, da.Array)
        computed_obs = obs_clean.compute()
        assert computed_obs.notnull().all()
        assert len(computed_obs) == 3

    def test_normalize_data_dask(self, dask_sample_data):
        """Test normalize_data with Dask arrays."""
        obs, mod = handle_missing_values(*dask_sample_data)
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        assert isinstance(obs_norm.data, da.Array)
        assert np.isclose(obs_norm.mean().compute(), 0)
        assert np.isclose(obs_norm.std().compute(), 1)

    def test_detrend_data_dask(self):
        """Test detrend_data with Dask arrays."""
        time = pd.date_range("2024-01-01", periods=5)
        obs = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={"time": time}).chunk()
        mod = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], coords={"time": time}).chunk()
        obs_detrended, _ = detrend_data(obs, mod, method="linear", dim="time")
        assert isinstance(obs_detrended.data, da.Array)
        assert np.allclose(obs_detrended.compute(), 0, atol=1e-9)

    def test_compute_anomalies_dask(self):
        """Test compute_anomalies with Dask arrays."""
        time = pd.date_range("2024-01-01", periods=365, freq="D")
        data = xr.DataArray(np.random.rand(365) + np.sin(np.linspace(0, 2 * np.pi, 365)), dims=["time"], coords={"time": time}).chunk()
        anomalies = compute_anomalies(data, groupby="time.month")
        assert isinstance(anomalies.data, da.Array)
        assert np.isclose(anomalies.mean().compute(), 0, atol=1e-9)
