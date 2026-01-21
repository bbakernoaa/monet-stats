"""
Tests for data_processing.py module (Aero Protocol Compliant).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

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

    def test_to_numpy(self):
        """Test that to_numpy converts various data types to numpy arrays."""
        # Test with numpy array
        data_np = np.array([1, 2, 3])
        assert isinstance(to_numpy(data_np), np.ndarray)

        # Test with xarray DataArray
        data_xr = xr.DataArray([1, 2, 3])
        assert isinstance(to_numpy(data_xr), np.ndarray)

        # Test with xarray Dataset
        data_ds = xr.Dataset({"a": (("x"), [1, 2, 3])})
        assert isinstance(to_numpy(data_ds), np.ndarray)

        # Test with pandas Series
        data_pd_series = pd.Series([1, 2, 3])
        assert isinstance(to_numpy(data_pd_series), np.ndarray)

        # Test with pandas DataFrame
        data_pd_df = pd.DataFrame({"a": [1, 2, 3]})
        assert isinstance(to_numpy(data_pd_df), np.ndarray)

        # Test with list
        data_list = [1, 2, 3]
        assert isinstance(to_numpy(data_list), np.ndarray)

    def test_align_arrays(self):
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
        assert list(obs_aligned.time.values) == [1, 2]

    def test_handle_missing_values(self):
        """Test that handle_missing_values handles NaNs."""
        # Numpy case (eager)
        obs = np.array([1, 2, np.nan, 4])
        mod = np.array([5, np.nan, 7, 8])
        obs_clean, mod_clean = handle_missing_values(obs, mod)
        assert len(obs_clean) == 2
        assert len(mod_clean) == 2
        assert np.array_equal(obs_clean, [1, 4])
        assert np.array_equal(mod_clean, [5, 8])

        # Xarray case (lazy-friendly)
        obs_xr = xr.DataArray([1, 2, np.nan, 4], dims=["x"])
        mod_xr = xr.DataArray([5, np.nan, 7, 8], dims=["x"])
        obs_clean_xr, mod_clean_xr = handle_missing_values(obs_xr, mod_xr)
        assert obs_clean_xr.isnull().sum() == 2
        assert mod_clean_xr.isnull().sum() == 2
        assert np.isnan(obs_clean_xr.values[1])
        assert np.isnan(obs_clean_xr.values[2])

    def test_normalize_data(self):
        """Test that normalize_data normalizes data."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1, 2, 3, 4, 5])

        # Z-score
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        assert np.isclose(np.mean(obs_norm), 0)
        assert np.isclose(np.std(obs_norm), 1)

        # Min-Max
        obs_norm, mod_norm = normalize_data(obs, mod, method="minmax")
        assert np.min(obs_norm) == 0
        assert np.max(obs_norm) == 1

        # Robust
        obs_norm, mod_norm = normalize_data(obs, mod, method="robust")
        assert np.isclose(np.median(obs_norm), 0)

    def test_detrend_data(self):
        """Test that detrend_data removes trends."""
        obs = np.array([1, 2, 3, 4, 5], dtype=float)
        mod = np.array([1, 2, 3, 4, 5], dtype=float)
        obs_detrended, _ = detrend_data(obs, mod, method="linear")
        # A linear trend should result in near-zero values after detrending
        # Actually scipy.signal.detrend removes the best-fit line.
        # For [1, 2, 3, 4, 5], the best fit line is [1, 2, 3, 4, 5], so result should be 0.
        assert np.allclose(obs_detrended, 0, atol=1e-9)

        # Constant detrend
        obs_detrended, _ = detrend_data(obs, mod, method="constant")
        assert np.isclose(np.mean(obs_detrended), 0)

    def test_compute_anomalies(self):
        """Test that compute_anomalies computes anomalies."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1, 2, 3, 4, 5])
        obs_anom, _ = compute_anomalies(obs, mod)
        assert np.isclose(np.mean(obs_anom), 0)

        # With climatology
        obs_anom, _ = compute_anomalies(obs, mod, climatology=3)
        assert np.array_equal(obs_anom, [-2, -1, 0, 1, 2])

    @pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
    def test_dask_laziness(self):
        """Test that operations remain lazy for dask-backed arrays."""
        obs = xr.DataArray(da.from_array(np.random.rand(10, 10), chunks=(5, 5)), dims=["x", "y"])
        mod = xr.DataArray(da.from_array(np.random.rand(10, 10), chunks=(5, 5)), dims=["x", "y"])

        # Normalization
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        assert isinstance(obs_norm.data, da.Array)

        # Anomalies
        obs_anom, _ = compute_anomalies(obs, mod)
        assert isinstance(obs_anom.data, da.Array)

        # Detrend (linear)
        obs_detrended, _ = detrend_data(obs, mod, method="linear", dim="y")
        assert isinstance(obs_detrended.data, da.Array)

    def test_history_tracking(self):
        """Test that history attribute is updated."""
        obs = xr.DataArray([1, 2, 3], dims=["x"])
        mod = xr.DataArray([1, 2, 3], dims=["x"])

        obs_norm, _ = normalize_data(obs, mod, method="zscore")
        assert "history" in obs_norm.attrs
        assert "Normalization (zscore)" in obs_norm.attrs["history"]

        obs_detrend, _ = detrend_data(obs, mod, method="linear")
        assert "Detrending (linear)" in obs_detrend.attrs["history"]

        obs_anom, _ = compute_anomalies(obs, mod)
        assert "Anomaly computation" in obs_anom.attrs["history"]
