"""
Tests for data_processing.py module.
"""

import numpy as np
import pandas as pd
import xarray as xr

from monet_stats.data_processing import (
    align_arrays,
    compute_anomalies,
    detrend_data,
    handle_missing_values,
    normalize_data,
    to_numpy,
)


class TestDataProcessing:
    """Test suite for data processing functions."""

    def test_to_numpy(self) -> None:
        """Test that to_numpy converts various data types to numpy arrays."""
        # Test with numpy array
        data_np = np.array([1, 2, 3])
        assert isinstance(to_numpy(data_np), np.ndarray)

        # Test with xarray DataArray
        data_xr = xr.DataArray([1, 2, 3])
        assert isinstance(to_numpy(data_xr), np.ndarray)

        # Test with pandas Series
        data_pd_series = pd.Series([1, 2, 3])
        assert isinstance(to_numpy(data_pd_series), np.ndarray)

        # Test with pandas DataFrame
        data_pd_df = pd.DataFrame({"a": [1, 2, 3]})
        assert isinstance(to_numpy(data_pd_df), np.ndarray)

        # Test with list
        data_list = [1, 2, 3]
        assert isinstance(to_numpy(data_list), np.ndarray)

    def test_align_arrays(self) -> None:
        """Test that align_arrays aligns numpy and xarray arrays."""
        # Test with numpy arrays
        obs_np = np.array([1, 2, 3])
        mod_np = np.array([4, 5, 6])
        obs_aligned, mod_aligned = align_arrays(obs_np, mod_np)
        assert np.array_equal(obs_aligned, obs_np)
        assert np.array_equal(mod_aligned, mod_np)

        # Test with xarray DataArrays
        obs_xr = xr.DataArray([1, 2, 3], dims=["time"], coords={"time": [0, 1, 2]})
        mod_xr = xr.DataArray([4, 5, 6], dims=["time"], coords={"time": [1, 2, 3]})
        obs_aligned, mod_aligned = align_arrays(obs_xr, mod_xr)
        assert obs_aligned.shape == (2,)
        assert mod_aligned.shape == (2,)

    def test_handle_missing_values(self) -> None:
        """Test that handle_missing_values removes NaNs."""
        obs = np.array([1, 2, np.nan, 4])
        mod = np.array([5, np.nan, 7, 8])
        obs_clean, mod_clean = handle_missing_values(obs, mod)
        assert len(obs_clean) == 2
        assert len(mod_clean) == 2

    def test_normalize_data(self) -> None:
        """Test that normalize_data normalizes data."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1, 2, 3, 4, 5])
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        assert np.isclose(np.mean(obs_norm), 0)
        assert np.isclose(np.std(obs_norm), 1)

    def test_detrend_data(self) -> None:
        """Test that detrend_data removes trends."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1, 2, 3, 4, 5])
        obs_detrended, _ = detrend_data(obs, mod, method="linear")
        # A linear trend should result in near-zero values after detrending
        assert np.allclose(obs_detrended, 0, atol=1e-9)

    def test_compute_anomalies(self) -> None:
        """Test that compute_anomalies computes anomalies."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1, 2, 3, 4, 5])
        obs_anom, _ = compute_anomalies(obs, mod)
        assert np.isclose(np.mean(obs_anom), 0)

    def test_normalize_data_minmax(self) -> None:
        """Test that normalize_data handles minmax normalization."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([2, 3, 4, 5, 6])
        obs_norm, mod_norm = normalize_data(obs, mod, method="minmax")
        assert np.isclose(np.min(obs_norm), 0)
        assert np.isclose(np.max(obs_norm), 1)
        assert np.isclose(np.min(mod_norm), 0)
        assert np.isclose(np.max(mod_norm), 1)

    def test_normalize_data_robust(self) -> None:
        """Test that normalize_data handles robust normalization."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([2, 3, 4, 5, 6])
        obs_norm, mod_norm = normalize_data(obs, mod, method="robust")
        assert np.isclose(np.median(obs_norm), 0)
        assert np.isclose(np.median(mod_norm), 0)

    def test_normalize_data_invalid_method(self) -> None:
        """Test that normalize_data raises on invalid method."""
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2, 3])
        try:
            normalize_data(obs, mod, method="invalid")
        except ValueError as e:
            assert "Unknown normalization method" in str(e)
        else:
            raise AssertionError("Expected ValueError for invalid normalization method")

    def test_handle_missing_values_listwise(self) -> None:
        """Test handle_missing_values with 'listwise' strategy."""
        obs = np.array([1, 2, np.nan, 4])
        mod = np.array([5, np.nan, 7, 8])
        obs_clean, mod_clean = handle_missing_values(obs, mod, strategy="listwise")
        assert len(obs_clean) == 2
        assert len(mod_clean) == 2

    def test_handle_missing_values_invalid_strategy(self) -> None:
        """Test handle_missing_values raises on invalid strategy."""
        obs = np.array([1, 2, 3])
        mod = np.array([4, 5, 6])
        try:
            handle_missing_values(obs, mod, strategy="invalid")
        except ValueError as e:
            assert "Unknown strategy" in str(e)
        else:
            raise AssertionError("Expected ValueError for invalid strategy")

    def test_align_arrays_shape_mismatch(self) -> None:
        """Test align_arrays raises on shape mismatch for numpy arrays."""
        obs = np.array([1, 2, 3])
        mod = np.array([4, 5])
        try:
            align_arrays(obs, mod)
        except ValueError as e:
            assert "Arrays must have the same shape" in str(e)
        else:
            raise AssertionError("Expected ValueError for shape mismatch")
