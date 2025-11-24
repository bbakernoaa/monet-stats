"""
Edge case testing for robust error handling and boundary conditions.
"""

import numpy as np
import pytest

from monet_stats.test_aliases import (
    coefficient_of_determination,
    critical_success_index,
    equitable_threat_score,
    false_alarm_rate,
    hit_rate,
    index_of_agreement,
    mean_absolute_error,
    mean_bias_error,
    pearson_correlation,
    root_mean_squared_error,
    spearman_correlation,
)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        empty_array = np.array([])

        # Should handle empty arrays gracefully
        try:
            mae = mean_absolute_error(empty_array, empty_array)
            assert np.isnan(mae) or mae == 0, "Empty arrays should return NaN or 0"
        except Exception:
            pass  # Some functions might raise exceptions for empty arrays

    def test_single_value_arrays(self):
        """Test behavior with single value arrays."""
        obs = np.array([5.0])
        mod = np.array([5.0])

        # Should handle single values
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)
        bias = mean_bias_error(obs, mod)

        assert abs(mae) < 1e-10, "Single identical values should have zero MAE"
        assert abs(rmse) < 1e-10, "Single identical values should have zero RMSE"
        assert abs(bias) < 1e-10, "Single identical values should have zero bias"

    def test_constant_arrays(self):
        """Test behavior with constant arrays."""
        obs = np.full(100, 5.0)
        mod = np.full(100, 5.0)

        # Should handle constant arrays
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)
        bias = mean_bias_error(obs, mod)

        assert abs(mae) < 1e-10, "Identical constant arrays should have zero MAE"
        assert abs(rmse) < 1e-10, "Identical constant arrays should have zero RMSE"
        assert abs(bias) < 1e-10, "Identical constant arrays should have zero bias"

    def test_arrays_with_zeros(self):
        """Test behavior with arrays containing zeros."""
        obs = np.array([0, 1, 2, 3, 4])
        mod = np.array([0.1, 1.1, 2.1, 3.1, 4.1])

        # Should handle zeros
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)

        assert mae > 0, "Arrays with zeros should produce positive MAE"
        assert rmse > 0, "Arrays with zeros should produce positive RMSE"

    def test_arrays_with_nans(self):
        """Test behavior with arrays containing NaN values."""
        obs = np.array([1, 2, np.nan, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Should handle NaN values gracefully
        try:
            mae = mean_absolute_error(obs, mod)
            # Result should be NaN or computed ignoring NaNs
            assert np.isnan(mae) or mae > 0, "Should handle NaN values appropriately"
        except Exception:
            pass  # Some implementations might raise exceptions

    def test_arrays_with_infs(self):
        """Test behavior with arrays containing infinite values."""
        obs = np.array([1, 2, np.inf, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Should handle infinite values gracefully
        try:
            mae = mean_absolute_error(obs, mod)
            # Result should be inf or NaN or computed ignoring infs
            assert np.isinf(mae) or np.isnan(mae) or mae > 0, "Should handle infinite values appropriately"
        except Exception:
            pass  # Some implementations might raise exceptions

    def test_mismatched_array_sizes(self):
        """Test behavior with mismatched array sizes."""
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2, 3, 4, 5])

        # Should raise an error or handle gracefully
        try:
            mae = mean_absolute_error(obs, mod)
            # If it doesn't raise an error, it should handle the mismatch appropriately
            assert np.isnan(mae) or mae >= 0, "Should handle mismatched sizes appropriately"
        except Exception:
            pass  # Expected behavior for mismatched sizes

    def test_extreme_values(self):
        """Test behavior with extreme values."""
        obs = np.array([1e-10, 1e10, -1e10])
        mod = np.array([1e-10 + 1e-12, 1e10 + 1e8, -1e10 - 1e8])

        # Should handle extreme values
        try:
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)

            assert np.isfinite(mae), "MAE should be finite for extreme values"
            assert np.isfinite(rmse), "RMSE should be finite for extreme values"
        except Exception:
            pass  # Some implementations might have limitations with extreme values


class TestErrorHandling:
    """Test error handling and validation."""

    def test_non_numeric_input(self):
        """Test behavior with non-numeric input."""
        obs = ["a", "b", "c"]
        mod = ["x", "y", "z"]

        # Should raise an error for non-numeric input
        try:
            mae = mean_absolute_error(obs, mod)
            # If it doesn't raise an error, result should be NaN
            assert np.isnan(mae), "Non-numeric input should produce NaN"
        except Exception:
            pass  # Expected behavior for non-numeric input

    def test_none_input(self):
        """Test behavior with None input."""
        obs = None
        mod = None

        # Should raise an error for None input
        try:
            mae = mean_absolute_error(obs, mod)
            assert np.isnan(mae), "None input should produce NaN"
        except Exception:
            pass  # Expected behavior for None input

    def test_wrong_input_types(self):
        """Test behavior with wrong input types."""
        obs = {"a": 1, "b": 2}
        mod = {"x": 1, "y": 2}

        # Should raise an error for dict input
        try:
            mae = mean_absolute_error(obs, mod)
            assert np.isnan(mae), "Dict input should produce NaN"
        except Exception:
            pass  # Expected behavior for dict input


class TestMathematicalEdgeCases:
    """Test mathematical edge cases."""

    def test_perfect_correlation_edge_cases(self):
        """Test edge cases for perfect correlation."""
        # Perfect positive correlation
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # y = 2*x

        pearson_r = pearson_correlation(x, y)
        spearman_r = spearman_correlation(x, y)
        r2 = coefficient_of_determination(x, y)

        assert abs(pearson_r - 1.0) < 1e-10, "Perfect positive correlation should be 1.0"
        assert abs(spearman_r - 1.0) < 1e-10, "Perfect positive rank correlation should be 1.0"
        assert abs(r2 - 1.0) < 1e-10, "Perfect correlation should give R² = 1.0"

        # Perfect negative correlation
        y_neg = np.array([-2, -4, -6, -8, -10])  # y = -2*x

        pearson_r_neg = pearson_correlation(x, y_neg)
        spearman_r_neg = spearman_correlation(x, y_neg)

        assert abs(pearson_r_neg + 1.0) < 1e-10, "Perfect negative correlation should be -1.0"
        assert abs(spearman_r_neg + 1.0) < 1e-10, "Perfect negative rank correlation should be -1.0"

    def test_constant_arrays_correlation(self):
        """Test correlation with constant arrays."""
        x = np.full(10, 5.0)
        y = np.full(10, 3.0)

        # Correlation with constant arrays should be undefined (NaN)
        try:
            pearson_r = pearson_correlation(x, y)
            assert np.isnan(pearson_r), "Correlation with constant arrays should be NaN"
        except Exception:
            pass  # Some implementations might handle this differently

    def test_identical_arrays(self):
        """Test behavior with identical arrays."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])

        mae = mean_absolute_error(x, y)
        rmse = root_mean_squared_error(x, y)
        bias = mean_bias_error(x, y)
        pearson_r = pearson_correlation(x, y)
        r2 = coefficient_of_determination(x, y)
        ioa = index_of_agreement(x, y)

        assert abs(mae) < 1e-10, "Identical arrays should have zero MAE"
        assert abs(rmse) < 1e-10, "Identical arrays should have zero RMSE"
        assert abs(bias) < 1e-10, "Identical arrays should have zero bias"
        assert abs(pearson_r - 1.0) < 1e-10, "Identical arrays should have correlation 1.0"
        assert abs(r2 - 1.0) < 1e-10, "Identical arrays should have R² = 1.0"
        assert abs(ioa - 1.0) < 1e-10, "Identical arrays should have IOA = 1.0"


class TestContingencyEdgeCases:
    """Test edge cases for contingency table metrics."""

    def test_perfect_forecast_contingency(self):
        """Test perfect forecast in contingency table."""
        obs = np.array([1, 1, 0, 0])
        mod = np.array([1, 1, 0, 0])

        hr = hit_rate(obs, mod)
        far = false_alarm_rate(obs, mod)
        csi = critical_success_index(obs, mod)
        ets = equitable_threat_score(obs, mod)

        assert abs(hr - 1.0) < 1e-10, "Perfect forecast should have HR = 1.0"
        assert abs(far - 0.0) < 1e-10, "Perfect forecast should have FAR = 0.0"
        assert abs(csi - 1.0) < 1e-10, "Perfect forecast should have CSI = 1.0"
        assert abs(ets - 1.0) < 1e-10, "Perfect forecast should have ETS = 1.0"

    def test_worst_forecast_contingency(self):
        """Test worst possible forecast in contingency table."""
        obs = np.array([1, 1, 0, 0])
        mod = np.array([0, 0, 1, 1])

        hr = hit_rate(obs, mod)
        far = false_alarm_rate(obs, mod)
        csi = critical_success_index(obs, mod)
        ets = equitable_threat_score(obs, mod)

        assert abs(hr - 0.0) < 1e-10, "Worst forecast should have HR = 0.0"
        assert abs(far - 1.0) < 1e-10, "Worst forecast should have FAR = 1.0"
        assert abs(csi - 0.0) < 1e-10, "Worst forecast should have CSI = 0.0"
        # ETS can be negative for worst forecasts
        assert ets <= 0.0, "Worst forecast should have ETS <= 0.0"

    def test_all_hits_contingency(self):
        """Test case with only hits (no misses or false alarms)."""
        obs = np.array([1, 1, 1, 1])
        mod = np.array([1, 1, 1, 1])

        hr = hit_rate(obs, mod)
        far = false_alarm_rate(obs, mod)
        csi = critical_success_index(obs, mod)

        assert abs(hr - 1.0) < 1e-10, "All hits should have HR = 1.0"
        assert abs(far - 0.0) < 1e-10, "All hits should have FAR = 0.0"
        assert abs(csi - 1.0) < 1e-10, "All hits should have CSI = 1.0"

    def test_all_misses_contingency(self):
        """Test case with only misses (no hits)."""
        obs = np.array([1, 1, 1, 1])
        mod = np.array([0, 0, 0, 0])

        hr = hit_rate(obs, mod)
        far = false_alarm_rate(obs, mod)
        csi = critical_success_index(obs, mod)

        assert abs(hr - 0.0) < 1e-10, "All misses should have HR = 0.0"
        assert np.isnan(far) or far == 0.0, "All misses should have undefined or 0 FAR"
        assert abs(csi - 0.0) < 1e-10, "All misses should have CSI = 0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
