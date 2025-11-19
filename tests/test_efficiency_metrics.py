"""
Comprehensive test suite for efficiency_metrics.py module.

Tests all functions in src/monet_stats/efficiency_metrics.py including:
- NSE, NSEm, NSElog, rNSE, mNSE
- PC, MAE, MSE, MAPE, MASE

Test categories:
- Perfect agreement tests (should return expected values)
- Mathematical correctness tests
- Edge case tests (NaN, infinity, empty arrays)
- Error handling tests
- Xarray compatibility tests
- Performance tests
- Property-based tests with Hypothesis
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Import all efficiency metrics functions
from src.monet_stats.efficiency_metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    NSE,
    PC,
    NSElog,
    NSEm,
    mNSE,
    rNSE,
)


class TestEfficiencyMetrics:
    """Test suite for efficiency metrics functions."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Perfect agreement data
        self.obs_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.mod_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Good agreement data (small errors)
        self.obs_good = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.mod_good = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        # Poor agreement data (large errors)
        self.obs_poor = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.mod_poor = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

        # Random data
        np.random.seed(42)
        self.obs_random = np.random.normal(10, 2, 100)
        self.mod_random = self.obs_random + np.random.normal(0, 1, 100)

        # Xarray test data
        self.obs_xr = xr.DataArray(
            self.obs_good,
            coords={"time": range(len(self.obs_good))},
            dims=["time"],
            name="observation",
        )
        self.mod_xr = xr.DataArray(
            self.mod_good,
            coords={"time": range(len(self.mod_good))},
            dims=["time"],
            name="model",
        )

    @pytest.mark.unit
    def test_nse_perfect_agreement(self):
        """Test NSE with perfect agreement (should return 1.0)."""
        result = NSE(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 1.0
        ), f"Perfect agreement NSE should be 1.0, got {result}"

    @pytest.mark.unit
    def test_nse_good_agreement(self):
        """Test NSE with good agreement (should be positive)."""
        result = NSE(self.obs_good, self.mod_good)
        assert result > 0.5, f"Good agreement NSE should be > 0.5, got {result}"
        assert (
            result < 1.0
        ), f"NSE should be < 1.0 for imperfect agreement, got {result}"

    @pytest.mark.unit
    def test_nse_poor_agreement(self):
        """Test NSE with poor agreement (should be negative or low)."""
        result = NSE(self.obs_poor, self.mod_poor)
        assert result < 0.5, f"Poor agreement NSE should be < 0.5, got {result}"

    @pytest.mark.unit
    def test_nse_constant_obs(self):
        """Test NSE with constant observations."""
        obs_const = np.ones(10) * 5.0
        mod_const = np.ones(10) * 5.0
        result = NSE(obs_const, mod_const)
        assert np.isclose(
            result, 1.0
        ), f"Constant perfect agreement should give NSE=1.0, got {result}"

    @pytest.mark.unit
    def test_nse_zero_denominator(self):
        """Test NSE when denominator is zero (should return -inf)."""
        obs_const = np.ones(10) * 5.0
        mod_diff = obs_const + 1.0
        result = NSE(obs_const, mod_diff)
        assert (
            np.isinf(result) and result < 0
        ), f"Zero denominator should give -inf, got {result}"

    @pytest.mark.unit
    def test_nsem_robust_to_masked_arrays(self):
        """Test NSEm with masked arrays."""
        obs_masked = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
        mod_masked = np.ma.array([1, 2, 3, 4, 5], mask=[0, 0, 1, 0, 0])
        result = NSEm(obs_masked, mod_masked)
        assert np.isclose(
            result, 1.0
        ), f"Masked perfect agreement should give NSEm=1.0, got {result}"

    @pytest.mark.unit
    def test_nselog_with_log_transform(self):
        """Test NSElog with logarithmic transformation."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = NSElog(obs, mod)
        assert isinstance(
            result, (float, np.floating)
        ), f"NSElog should return float, got {type(result)}"
        assert (
            result < 1.0
        ), f"NSElog should be < 1.0 for imperfect agreement, got {result}"

    @pytest.mark.unit
    def test_rnse_relative_normalization(self):
        """Test rNSE relative efficiency."""
        result = rNSE(self.obs_good, self.mod_good)
        assert isinstance(
            result, (float, np.floating)
        ), f"rNSE should return float, got {type(result)}"
        assert (
            result < 1.0
        ), f"rNSE should be < 1.0 for imperfect agreement, got {result}"

    @pytest.mark.unit
    def test_mnse_modified_calculation(self):
        """Test mNSE modified efficiency."""
        result = mNSE(self.obs_good, self.mod_good)
        assert isinstance(
            result, (float, np.floating)
        ), f"mNSE should return float, got {type(result)}"
        assert (
            result < 1.0
        ), f"mNSE should be < 1.0 for imperfect agreement, got {result}"

    @pytest.mark.unit
    def test_pc_percent_correct(self):
        """Test PC (Percent Correct) metric."""
        # Perfect agreement should give 100%
        result_perfect = PC(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result_perfect, 100.0
        ), f"Perfect agreement PC should be 100%, got {result_perfect}"

        # Good agreement should give high percentage
        result_good = PC(self.obs_good, self.mod_good)
        assert (
            result_good > 50.0
        ), f"Good agreement PC should be > 50%, got {result_good}"

    @pytest.mark.unit
    def test_mae_mean_absolute_error(self):
        """Test MAE (Mean Absolute Error)."""
        result = MAE(self.obs_good, self.mod_good)
        expected = np.mean(np.abs(self.obs_good - self.mod_good))
        assert np.isclose(
            result, expected
        ), f"MAE calculation incorrect. Expected {expected}, got {result}"
        assert result >= 0, f"MAE should be non-negative, got {result}"

    @pytest.mark.unit
    def test_mse_mean_squared_error(self):
        """Test MSE (Mean Squared Error)."""
        result = MSE(self.obs_good, self.mod_good)
        expected = np.mean((self.obs_good - self.mod_good) ** 2)
        assert np.isclose(
            result, expected
        ), f"MSE calculation incorrect. Expected {expected}, got {result}"
        assert result >= 0, f"MSE should be non-negative, got {result}"

    @pytest.mark.unit
    def test_mape_mean_absolute_percentage_error(self):
        """Test MAPE (Mean Absolute Percentage Error)."""
        result = MAPE(self.obs_good, self.mod_good)
        expected = (
            np.mean(np.abs((self.obs_good - self.mod_good) / self.obs_good)) * 100
        )
        assert np.isclose(
            result, expected
        ), f"MAPE calculation incorrect. Expected {expected}, got {result}"
        assert result >= 0, f"MAPE should be non-negative, got {result}"

    @pytest.mark.unit
    def test_mase_mean_absolute_scaled_error(self):
        """Test MASE (Mean Absolute Scaled Error)."""
        result = MASE(self.obs_good, self.mod_good)
        assert isinstance(
            result, (float, np.floating)
        ), f"MASE should return float, got {type(result)}"
        assert result >= 0, f"MASE should be non-negative, got {result}"

    @pytest.mark.parametrize(
        "metric_func", [NSE, NSEm, NSElog, MSE, MAPE, MASE, PC, mNSE, rNSE]
    )
    def test_efficiency_metrics_output_type_parametrized(self, metric_func):
        """Test that efficiency metrics return appropriate values."""
        result = metric_func(self.obs_random, self.mod_random)
        assert isinstance(
            result, (float, np.floating, int, np.integer)
        ), f"{metric_func.__name__} should return a numeric value, got {type(result)}"

    @pytest.mark.unit
    def test_efficiency_metrics_mathematical_correctness(self):
        """Test mathematical correctness of efficiency metrics."""
        # Create data with known properties
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Small errors

        # MSE should be mean of squared errors: mean([0.1^2, 0.1^2, 0.1^2, 0.1^2, 0.1^2]) = 0.01
        expected_mse = 0.01
        mse_result = MSE(obs, mod)
        assert (
            abs(mse_result - expected_mse) < 1e-10
        ), f"Expected MSE={expected_mse}, got {mse_result}"

        # NSE calculation: 1 - (sum of squared errors) / (sum of squared deviations from mean)
        # SSE = 5 * 0.01 = 0.05
        # SS_mean = sum((obs - mean)^2) = sum([4, 1, 0, 1, 4]) = 10
        # NSE = 1 - 0.05/10 = 1 - 0.005 = 0.995
        expected_nse = 1 - (0.05 / 10)
        nse_result = NSE(obs, mod)
        assert (
            abs(nse_result - expected_nse) < 1e-10
        ), f"Expected NSE={expected_nse}, got {nse_result}"

    @pytest.mark.unit
    def test_nan_handling(self):
        """Test handling of NaN values."""
        obs_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        mod_nan = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Should handle NaN gracefully (depends on implementation)
        for metric_func in [NSE, MSE, MAE, MAPE]:
            result = metric_func(obs_nan, mod_nan)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle NaN gracefully, got {type(result)}"

    @pytest.mark.unit
    def test_inf_handling(self):
        """Test handling of infinity values."""
        obs_inf = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        mod_inf = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Should handle infinity gracefully
        for metric_func in [NSE, MAE]:
            result = metric_func(obs_inf, mod_inf)
            assert isinstance(
                result, (float, np.floating, np.ma.core.MaskedConstant)
            ), f"{metric_func.__name__} should handle infinity gracefully, got {type(result)}"

    @pytest.mark.unit
    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        obs_empty = np.array([])
        mod_empty = np.array([])

        # Test that empty arrays return appropriate results
        for metric_func in [MSE, MAE]:
            result = metric_func(obs_empty, mod_empty)
            # Empty arrays should return NaN or similar indicator
            assert np.isnan(result) or np.ma.is_masked(
                result
            ), f"{metric_func.__name__} should handle empty arrays gracefully, got {result}"

    @pytest.mark.unit
    def test_single_value_arrays(self):
        """Test handling of single value arrays."""
        obs_single = np.array([5.0])
        mod_single = np.array([5.0])

        for metric_func in [NSE, MSE, MAE]:
            result = metric_func(obs_single, mod_single)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle single values, got {type(result)}"

    @pytest.mark.xarray
    def test_xarray_dataarray_input(self):
        """Test that functions work with xarray DataArray inputs."""
        # Test NSE with xarray
        result = NSE(self.obs_xr, self.mod_xr)
        assert isinstance(
            result, (float, np.floating, xr.DataArray)
        ), f"NSE should work with xarray inputs, got {type(result)}"

        # Test MSE with xarray
        result = MSE(self.obs_xr, self.mod_xr)
        assert isinstance(
            result, (float, np.floating, xr.DataArray)
        ), f"MSE should work with xarray inputs, got {type(result)}"

    @pytest.mark.xarray
    def test_xarray_alignment(self):
        """Test that xarray DataArrays are properly aligned."""
        # Create misaligned DataArrays
        obs_misaligned = xr.DataArray(
            [1, 2, 3, 4, 5], coords={"x": [0, 1, 2, 3, 4]}, dims=["x"]
        )
        mod_misaligned = xr.DataArray(
            [1.1, 2.1, 3.1, 4.1, 5.1], coords={"x": [1, 2, 3, 4, 5]}, dims=["x"]
        )

        # Should handle misaligned arrays gracefully
        result = NSE(obs_misaligned, mod_misaligned)
        assert isinstance(result, (float, np.floating, xr.DataArray))

    @pytest.mark.slow
    def test_performance_large_arrays(self):
        """Test performance with large arrays."""
        # Create large test arrays
        np.random.seed(42)
        large_obs = np.random.normal(10, 2, 10000)
        large_mod = large_obs + np.random.normal(0, 0.5, 10000)

        import time

        start_time = time.time()

        # Test multiple metrics
        nse_result = NSE(large_obs, large_mod)
        mse_result = MSE(large_obs, large_mod)
        mae_result = MAE(large_obs, large_mod)

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 1 second)
        assert (
            elapsed_time < 1.0
        ), f"Performance test took too long: {elapsed_time:.3f}s"

        # Results should be valid
        assert isinstance(nse_result, (float, np.floating))
        assert isinstance(mse_result, (float, np.floating))
        assert isinstance(mae_result, (float, np.floating))

    @pytest.mark.mathematical
    def test_nse_bounds(self):
        """Test that NSE values are within expected bounds."""
        # Perfect agreement
        nse_perfect = NSE(self.obs_perfect, self.mod_perfect)
        assert np.isclose(nse_perfect, 1.0)

        # Good agreement should be positive
        nse_good = NSE(self.obs_good, self.mod_good)
        assert nse_good > 0.0

        # Can be negative for poor models
        nse_poor = NSE(self.obs_poor, self.mod_poor)
        assert isinstance(nse_poor, (float, np.floating))

    @pytest.mark.mathematical
    def test_error_metric_relationships(self):
        """Test mathematical relationships between error metrics."""
        obs = self.obs_good
        mod = self.mod_good

        mae = MAE(obs, mod)
        mse = MSE(obs, mod)

        # MSE should be >= MAE^2 (by Jensen's inequality)
        assert mse >= mae**2, f"MSE should be >= MAE^2: {mse} >= {mae**2}"

    @pytest.mark.parametrize("axis", [None, 0])
    def test_axis_parameter(self, axis):
        """Test axis parameter for functions that support it."""
        # Create 2D data
        obs_2d = np.array([[1, 2, 3], [4, 5, 6]])
        mod_2d = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])

        # Test functions that support axis parameter
        for metric_func in [MAE, MSE]:
            result = metric_func(obs_2d, mod_2d, axis=axis)
            if axis is None:
                assert (
                    np.isscalar(result) or result.shape == ()
                ), f"{metric_func.__name__} with axis=None should return scalar"
            else:
                expected_shape = (
                    obs_2d.shape[:axis] + obs_2d.shape[axis + 1 :]  # noqa: E203
                )  # noqa: E203
                assert (
                    result.shape == expected_shape
                ), f"{metric_func.__name__} result shape mismatch"

    @pytest.mark.unit
    def test_tolerance_in_pc_metric(self):
        """Test tolerance parameter in PC metric."""
        obs = np.array([10, 20, 30, 40, 50])
        mod = np.array([10.5, 20.5, 30.5, 40.5, 50.5])  # 5% error

        # With default tolerance (10%), should be 100% correct
        pc_result = PC(obs, mod)
        assert (
            pc_result == 100.0
        ), f"5% error should be within 10% tolerance, got {pc_result}%"

        # Test with smaller tolerance manually
        tolerance = 0.03 * np.abs(obs)  # 3% tolerance
        correct = np.abs(obs - mod) <= tolerance
        expected_pc = (np.sum(correct) / len(correct)) * 100.0
        assert pc_result >= expected_pc, "PC calculation inconsistent"


class TestEfficiencyMetricsHypothesis:
    """Property-based tests using Hypothesis."""

    @given(
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_nse_perfect_agreement_property(self, data):
        """Test that NSE returns 1.0 for perfect agreement."""
        assume(np.std(data) > 0)  # Avoid constant arrays
        result = NSE(data, data)
        assert np.isclose(
            result, 1.0, rtol=1e-10
        ), f"Perfect agreement should give NSE=1.0, got {result}"

    @given(
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def test_mape_non_negative_property(self, obs, mod):
        """Test that MAPE is always non-negative."""
        assume(np.all(obs != 0))  # Avoid division by zero
        result = MAPE(obs, mod)
        assert result >= 0, f"MAPE should be non-negative, got {result}"

    @given(
        arrays(
            np.float64,
            5,
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
        arrays(
            np.float64,
            5,
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def test_mse_mae_relationship_property(self, obs, mod):
        """Test that MSE >= MAE^2."""
        assume(len(obs) > 0)
        mae = MAE(obs, mod)
        mse = MSE(obs, mod)
        assert mse >= mae**2 - 1e-10, f"MSE should be >= MAE^2: {mse} >= {mae**2}"


class TestEfficiencyMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_observation_mape(self):
        """Test MAPE with zero observations (should handle division by zero)."""
        obs = np.array([0, 1, 2, 3, 4])
        mod = np.array([0.1, 1.1, 2.1, 3.1, 4.1])

        # Should handle zero observations gracefully
        result = MAPE(obs, mod)
        assert isinstance(result, (float, np.floating))

    def test_constant_arrays_nse(self):
        """Test NSE with constant arrays."""
        obs_const = np.ones(10) * 5.0
        mod_const = np.ones(10) * 5.0
        mod_diff = np.ones(10) * 6.0

        # Perfect constant agreement
        result_perfect = NSE(obs_const, mod_const)
        assert np.isclose(result_perfect, 1.0)

        # Imperfect constant agreement
        result_imperfect = NSE(obs_const, mod_diff)
        assert np.isinf(result_imperfect) and result_imperfect < 0

    def test_negative_values(self):
        """Test metrics with negative input values."""
        obs = np.array([-5, -3, -1, 1, 3])
        mod = np.array([-4.9, -2.9, -0.9, 1.1, 3.1])

        for metric_func in [NSE, MSE, MAE]:
            result = metric_func(obs, mod)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle negative values, got {type(result)}"

    def test_large_arrays_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        # Create moderately large arrays
        np.random.seed(42)
        large_obs = np.random.normal(0, 1, 50000)
        large_mod = large_obs + np.random.normal(0, 0.1, 50000)

        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate metrics
        nse_result = NSE(large_obs, large_mod)
        mse_result = MSE(large_obs, large_mod)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 100MB)
        assert (
            memory_increase < 100
        ), f"Memory increase too large: {memory_increase:.1f}MB"

        # Results should be valid
        assert isinstance(nse_result, (float, np.floating))
        assert isinstance(mse_result, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
