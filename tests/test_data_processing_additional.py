#!/usr/bin/env python3
"""
Additional tests for data_processing.py to improve coverage.
Tests focus on functions that are not well-covered by existing tests.
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


class TestToNumpy:
    """Test to_numpy conversion function."""

    def test_to_numpy_array(self):
        """Test to_numpy with numpy array."""
        data = np.array([1.0, 2.0, 3.0])
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, data)

    def test_to_numpy_list(self):
        """Test to_numpy with list."""
        data = [1.0, 2.0, 3.0]
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array(data))

    def test_to_numpy_xarray(self):
        """Test to_numpy with xarray DataArray."""
        data = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_to_numpy_pandas_series(self):
        """Test to_numpy with pandas Series."""
        data = pd.Series([1.0, 2.0, 3.0])
        result = to_numpy(data)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


class TestAlignArrays:
    """Test align_arrays function."""

    def test_align_arrays_numpy(self):
        """Test align_arrays with numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([1.1, 2.1, 3.1])
        obs_aligned, mod_aligned = align_arrays(obs, mod)
        assert np.array_equal(obs_aligned, obs)
        assert np.array_equal(mod_aligned, mod)

    def test_align_arrays_xarray(self):
        """Test align_arrays with xarray DataArrays."""
        obs = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [0, 1, 2]})
        mod = xr.DataArray([1.1, 2.1, 3.1], dims=["x"], coords={"x": [0, 1, 2]})
        obs_aligned, mod_aligned = align_arrays(obs, mod)
        assert isinstance(obs_aligned, xr.DataArray)
        assert isinstance(mod_aligned, xr.DataArray)
        assert np.array_equal(obs_aligned.values, obs.values)
        assert np.array_equal(mod_aligned.values, mod.values)

    def test_align_arrays_shape_mismatch(self):
        """Test align_arrays with shape mismatch raises error."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([1.1, 2.1])  # Different shape
        with pytest.raises(ValueError, match="Arrays must have compatible shapes"):
            align_arrays(obs, mod)


class TestHandleMissingValues:
    """Test handle_missing_values function."""

    def test_handle_missing_values_pairwise(self):
        """Test handle_missing_values with pairwise strategy."""
        obs = np.array([1.0, 2.0, np.nan, 4.0])
        mod = np.array([1.1, np.nan, 3.1, 4.1])
        obs_clean, mod_clean = handle_missing_values(obs, mod, strategy="pairwise")
        assert len(obs_clean) == 2  # Only positions 0 and 3 should remain
        assert len(mod_clean) == 2
        assert np.array_equal(obs_clean, np.array([1.0, 4.0]))
        assert np.array_equal(mod_clean, np.array([1.1, 4.1]))

    def test_handle_missing_values_listwise(self):
        """Test handle_missing_values with listwise strategy."""
        obs = np.array([1.0, 2.0, np.nan, 4.0])
        mod = np.array([1.1, np.nan, 3.1, 4.1])
        obs_clean, mod_clean = handle_missing_values(obs, mod, strategy="listwise")
        assert len(obs_clean) == 2  # Only positions 0 and 3 should remain
        assert len(mod_clean) == 2
        assert np.array_equal(obs_clean, np.array([1.0, 4.0]))
        assert np.array_equal(mod_clean, np.array([1.1, 4.1]))

    def test_handle_missing_values_invalid_strategy(self):
        """Test handle_missing_values with invalid strategy raises error."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([1.1, 2.1, 3.1])
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing_values(obs, mod, strategy="invalid")


class TestNormalizeData:
    """Test normalize_data function."""

    def test_normalize_data_zscore(self):
        """Test normalize_data with zscore method."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs_norm, mod_norm = normalize_data(obs, mod, method="zscore")
        # Check that means are approximately 0 and stds are approximately 1
        assert np.isclose(np.mean(obs_norm), 0.0, atol=1e-10)
        assert np.isclose(np.mean(mod_norm), 0.0, atol=1e-10)
        assert np.isclose(np.std(obs_norm), 1.0, atol=1e-10)
        assert np.isclose(np.std(mod_norm), 1.0, atol=1e-10)

    def test_normalize_data_minmax(self):
        """Test normalize_data with minmax method."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs_norm, mod_norm = normalize_data(obs, mod, method="minmax")
        # Check that values are in [0, 1] range
        assert np.all(obs_norm >= 0.0) and np.all(obs_norm <= 1.0)
        assert np.all(mod_norm >= 0.0) and np.all(mod_norm <= 1.0)
        # Check that min values are 0 and max values are 1
        assert np.isclose(np.min(obs_norm), 0.0, atol=1e-10)
        assert np.isclose(np.min(mod_norm), 0.0, atol=1e-10)
        assert np.isclose(np.max(obs_norm), 1.0, atol=1e-10)
        assert np.isclose(np.max(mod_norm), 1.0, atol=1e-10)

    def test_normalize_data_robust(self):
        """Test normalize_data with robust method."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs_norm, mod_norm = normalize_data(obs, mod, method="robust")
        # Check that medians are approximately 0
        assert np.isclose(np.median(obs_norm), 0.0, atol=1e-10)
        assert np.isclose(np.median(mod_norm), 0.0, atol=1e-10)

    def test_normalize_data_invalid_method(self):
        """Test normalize_data with invalid method raises error."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([1.1, 2.1, 3.1])
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_data(obs, mod, method="invalid")


class TestDetrendData:
    """Test detrend_data function."""

    def test_detrend_data_linear(self):
        """Test detrend_data with linear method."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs_detrended, mod_detrended = detrend_data(obs, mod, method="linear")
        # Check that results are arrays
        assert isinstance(obs_detrended, np.ndarray)
        assert isinstance(mod_detrended, np.ndarray)
        assert obs_detrended.shape == obs.shape
        assert mod_detrended.shape == mod.shape

    def test_detrend_data_constant(self):
        """Test detrend_data with constant method."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs_detrended, mod_detrended = detrend_data(obs, mod, method="constant")
        # Check that means are approximately 0
        assert np.isclose(np.mean(obs_detrended), 0.0, atol=1e-10)
        assert np.isclose(np.mean(mod_detrended), 0.0, atol=1e-10)

    def test_detrend_data_invalid_method(self):
        """Test detrend_data with invalid method raises error."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([1.1, 2.1, 3.1])
        with pytest.raises(ValueError, match="Unknown detrending method"):
            detrend_data(obs, mod, method="invalid")


class TestComputeAnomalies:
    """Test compute_anomalies function."""

    def test_compute_anomalies_basic(self):
        """Test compute_anomalies with basic arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        obs_anom, mod_anom = compute_anomalies(obs, mod)
        # Check that anomalies have mean approximately 0
        assert np.isclose(np.mean(obs_anom), 0.0, atol=1e-10)
        assert np.isclose(np.mean(mod_anom), 0.0, atol=1e-10)

    def test_compute_anomalies_with_climatology(self):
        """Test compute_anomalies with climatology parameter."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        climatology = np.array([2.5, 2.5, 2.5, 2.5, 2.5])  # Fixed climatology
        obs_anom, mod_anom = compute_anomalies(obs, mod, climatology=climatology)
        # Check that anomalies are computed relative to the provided climatology
        expected_obs_anom = obs - climatology
        expected_mod_anom = mod - climatology
        assert np.allclose(obs_anom, expected_obs_anom)
        assert np.allclose(mod_anom, expected_mod_anom)
