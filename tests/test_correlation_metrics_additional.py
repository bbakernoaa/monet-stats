"""
Additional tests for correlation_metrics.py module to improve coverage.
"""

import numpy as np
import xarray as xr

from monet_stats.correlation_metrics import (
    AC,
    IOA,
    R2,
    RMSE,
    matchedcompressed,
    pearsonr,
    spearmanr,
)
from monet_stats.utils_stats import matchmasks


class TestCorrelationMetricsAdditional:
    """Additional test suite for correlation metrics to improve coverage."""

    def test_ac_helper_function(self) -> None:
        """Test AC (Anomaly Correlation) function."""
        # Test with perfect correlation
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 2.0, 3.0, 4.0])
        result = AC(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test with inverted correlation
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([4.0, 3.0, 2.0, 1.0])
        result = AC(obs, mod)
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_ioa_helper_function(self) -> None:
        """Test IOA (Index of Agreement) function."""
        # Test with perfect agreement
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 2.0, 3.0, 4.0])
        result = IOA(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test with no agreement (worst case)
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([4.0, 3.0, 2.0, 1.0])
        result = IOA(obs, mod)
        # IOA should be between 0 and 1
        assert 0 <= result <= 1

    def test_r2_function(self) -> None:
        """Test R2 (coefficient of determination) function."""
        # Test with perfect correlation
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 2.0, 3.0, 4.0])
        result = R2(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test with linear relationship
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([2.0, 4.0, 6.0, 8.0])  # y = 2x
        result = R2(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_rmse_function(self) -> None:
        """Test RMSE function."""
        # Test with perfect agreement
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 2.0, 3.0, 4.0])
        result = RMSE(obs, mod)
        assert np.isclose(result, 0.0, atol=1e-10)

        # Test with known RMSE
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = RMSE(obs, mod)
        expected = np.sqrt(np.mean((obs - mod) ** 2))
        assert np.isclose(result, expected, atol=1e-10)

    def test_matchedcompressed_function(self) -> None:
        """Test matchedcompressed function."""
        # Test with numpy arrays
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1])
        obs_compressed, mod_compressed = matchedcompressed(obs, mod)
        assert len(obs_compressed) == len(mod_compressed)
        assert len(obs_compressed) == len(obs)

        # Test with masked arrays containing NaN
        obs = np.array([1.0, 2.0, np.nan, 4.0])
        mod = np.array([1.1, np.nan, 3.1, 4.1])
        obs_compressed, mod_compressed = matchedcompressed(obs, mod)
        # Should only have valid paired data - positions 0 and 3
        assert len(obs_compressed) == 2  # Positions 0 and 3 have valid data in both
        assert obs_compressed[0] == 1.0
        assert mod_compressed[0] == 1.1
        assert obs_compressed[1] == 4.0
        assert mod_compressed[1] == 4.1

    def test_matchmasks_function(self) -> None:
        """Test matchmasks function from utils_stats."""
        # Test with numpy arrays
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])
        result1, result2 = matchmasks(a1, a2)
        assert np.array_equal(result1, a1)
        assert np.array_equal(result2, a2)

        # Test with masked arrays
        a1 = np.ma.array([1, 2, 3], mask=[0, 1, 0])
        a2 = np.ma.array([4, 5, 6], mask=[0, 0, 1])
        result1, result2 = matchmasks(a1, a2)
        # Combined mask should mask positions 1 and 2
        expected_mask = [False, True, True]
        assert np.array_equal(result1.mask, expected_mask)
        assert np.array_equal(result2.mask, expected_mask)

    def test_pearsonr_function(self) -> None:
        """Test pearsonr function."""
        # Test with perfect correlation
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 2.0, 3.0, 4.0])
        result = pearsonr(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test with anti-correlation
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([4.0, 3.0, 2.0, 1.0])
        result = pearsonr(obs, mod)
        assert np.isclose(result, -1.0, atol=1e-10)

    def test_spearmanr_function(self) -> None:
        """Test spearmanr function."""
        # Test with perfect rank correlation
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 2.0, 3.0, 4.0])
        result = spearmanr(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

        # Test with monotonic but non-linear relationship
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.0, 4.0, 9.0, 16.0])  # y = x^2
        result = spearmanr(obs, mod)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_edge_cases_with_xarray(self) -> None:
        """Test functions with xarray inputs."""
        # Test R2 with xarray
        obs = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["time"])
        mod = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["time"])
        result = R2(obs, mod)
        assert np.isclose(float(result), 1.0, atol=1e-10)

        # Test IOA with xarray
        result = IOA(obs, mod)
        assert np.isclose(float(result), 1.0, atol=1e-10)
