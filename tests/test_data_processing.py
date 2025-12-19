"""
Tests for data_processing.py module.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats.data_processing import (
    align_arrays,
    compute_anomalies,
    detrend_data,
    handle_missing_values,
    normalize_data,
    to_numpy,
)


@pytest.fixture(params=["numpy", "xarray"])
def factory(request):
    """Provide a factory for creating numpy or xarray arrays."""
    if request.param == "numpy":
        return np.array
    else:
        return xr.DataArray


class TestDataProcessing:
    """Test suite for data processing functions."""

    def test_to_numpy(self):
        """Test that to_numpy converts various data types to numpy arrays."""
        # Test with numpy array
        data_np = np.array([1, 2, 3])
        assert isinstance(to_numpy(data_np), np.ndarray)

        # Test with xarray DataArray
        data_xr = xr.DataArray([1, 2, 3])
        assert isinstance(to_numpy(data_xr), np.ndarray)

        # Test with list
        data_list = [1, 2, 3]
        assert isinstance(to_numpy(data_list), np.ndarray)

    def test_align_arrays(self, factory):
        """Test that align_arrays aligns numpy and xarray arrays."""
        obs = factory([1, 2, 3])
        mod = factory([4, 5, 6])
        if isinstance(obs, np.ndarray):
            obs_aligned, mod_aligned = align_arrays(obs, mod)
            assert np.array_equal(obs_aligned, obs)
            assert np.array_equal(mod_aligned, mod)
        else:
            obs = xr.DataArray(obs, dims=["time"], coords={"time": [0, 1, 2]})
            mod = xr.DataArray(mod, dims=["time"], coords={"time": [1, 2, 3]})
            obs_aligned, mod_aligned = align_arrays(obs, mod)
            assert obs_aligned.shape == (2,)
            assert mod_aligned.shape == (2,)

    def test_handle_missing_values(self, factory):
        """Test that handle_missing_values removes NaNs."""
        obs = factory([1, 2, np.nan, 4])
        mod = factory([5, np.nan, 7, 8])
        obs_clean, mod_clean = handle_missing_values(obs, mod)
        assert len(obs_clean) == 2
        assert len(mod_clean) == 2

    def test_normalize_data(self, factory):
        """Test that normalize_data normalizes data."""
        obs = factory([1, 2, 3, 4, 5])
        mod = factory([1, 2, 3, 4, 5])
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        assert np.isclose(np.mean(obs_norm), 0)
        assert np.isclose(np.std(obs_norm), 1)

    def test_detrend_data(self, factory):
        """Test that detrend_data removes trends."""
        obs = factory([1, 2, 3, 4, 5])
        mod = factory([1, 2, 3, 4, 5])
        if isinstance(obs, xr.DataArray):
            obs = xr.DataArray(
                obs,
                dims=["time"],
                coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])},
            )
            mod = xr.DataArray(
                mod,
                dims=["time"],
                coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])},
            )
        obs_detrended, _ = detrend_data(obs, mod, method="linear")
        assert np.allclose(obs_detrended, 0, atol=1e-9)

    def test_compute_anomalies(self, factory):
        """Test that compute_anomalies computes anomalies."""
        obs = factory([1, 2, 3, 4, 5])
        mod = factory([1, 2, 3, 4, 5])
        if isinstance(obs, xr.DataArray):
            obs = xr.DataArray(
                obs,
                dims=["time"],
                coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])},
            )
            mod = xr.DataArray(
                mod,
                dims=["time"],
                coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])},
            )
        obs_anom, _ = compute_anomalies(obs, mod)
        assert np.isclose(np.mean(obs_anom), 0)
