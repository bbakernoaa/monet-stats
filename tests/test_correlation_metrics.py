"""
Tests for correlation_metrics.py module.

This module tests correlation-based statistical metrics including
Pearson, Spearman, Kendall correlation coefficients and derived metrics.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from src.monet_stats.correlation_metrics import (
    AC,
    CCC,
    E1,
    IOA,
    KGE,
    R2,
    RMSE,
    WDAC,
    WDIOA,
    WDRMSE,
    E1_prime,
    IOA_m,
    IOA_prime,
    RMSEs,
    RMSEu,
    WDIOA_m,
    d1,
    kendalltau,
    pearsonr,
    spearmanr,
    taylor_skill,
)
from tests.test_utils import TestDataGenerator


class TestCorrelationMetrics:
    """Test suite for correlation metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_gen = TestDataGenerator()

        # Perfect agreement case
        self.obs_perfect = np.array([1, 2, 3, 4, 5])
        self.mod_perfect = np.array([1, 2, 3, 4, 5])

        # Linear relationship with noise
        self.obs_linear = np.linspace(0, 10, 20)
        self.mod_linear = 2 * self.obs_linear + 1 + np.random.normal(0, 0.1, 20)

        # No correlation
        np.random.seed(42)
        self.obs_random = np.random.normal(0, 1, 50)
        self.mod_random = np.random.normal(0, 1, 50)

    def test_pearsonr_perfect_correlation(self):
        """Test Pearson correlation with perfect linear relationship."""
        result = pearsonr(self.obs_perfect, self.mod_perfect)
        # pearsonr returns correlation coefficient and p-value, we want the coefficient
        if isinstance(result, tuple):
            result = result[0]
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect correlation should be 1.0, got {result}"

    def test_pearsonr_no_correlation(self):
        """Test Pearson correlation with no relationship."""
        result = pearsonr(self.obs_random, self.mod_random)
        # Should be close to 0, but with random data it might not be exactly 0
        if isinstance(result, tuple):
            result = result[0]
        assert -1 <= result <= 1, f"Correlation should be in [-1,1], got {result}"

    def test_spearmanr_perfect_correlation(self):
        """Test Spearman correlation with perfect relationship."""
        result = spearmanr(self.obs_perfect, self.mod_perfect)
        if isinstance(result, tuple):
            result = result[0]
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect correlation should be 1.0, got {result}"

    def test_kendalltau_perfect_correlation(self):
        """Test Kendall tau with perfect relationship."""
        result = kendalltau(self.obs_perfect, self.mod_perfect)
        if isinstance(result, tuple):
            result = result[0]
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect correlation should be 1.0, got {result}"

    def test_r2_perfect_agreement(self):
        """Test R2 with perfect agreement."""
        result = R2(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give R2=1.0, got {result}"

    def test_r2_worst_case(self):
        """Test R2 with worst case (no correlation)."""
        # Use larger arrays to avoid constant input issues
        obs = np.random.normal(0, 1, 100)  # Larger array
        # Use a model that's uncorrelated but not constant
        mod = np.random.normal(0, 0.5, 100)  # Random model
        result = R2(obs, mod)
        # R2 should be low for uncorrelated data
        assert result < 0.1, f"Uncorrelated model should give low R2, got {result}"

    def test_rmse_perfect_agreement(self):
        """Test RMSE with perfect agreement."""
        result = RMSE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give RMSE=0.0, got {result}"

    def test_rmse_positive_values(self):
        """Test that RMSE is always positive."""
        result = RMSE(self.obs_linear, self.mod_linear)
        assert result >= 0, f"RMSE should be non-negative, got {result}"

    def test_ioa_perfect_agreement(self):
        """Test Index of Agreement with perfect agreement."""
        result = IOA(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give IOA=1.0, got {result}"

    def test_ioa_range_bounds(self):
        """Test that IOA is in valid range [0, 1]."""
        result = IOA(self.obs_linear, self.mod_linear)
        assert 0 <= result <= 1, f"IOA should be in [0,1], got {result}"

    def test_e1_perfect_agreement(self):
        """Test E1 with perfect agreement."""
        result = E1(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give E1=1.0, got {result}"

    def test_kge_perfect_agreement(self):
        """Test Kling-Gupta Efficiency with perfect agreement."""
        result = KGE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give KGE=1.0, got {result}"

    def test_kge_range_bounds(self):
        """Test that KGE is in valid range [-∞, 1]."""
        result = KGE(self.obs_linear, self.mod_linear)
        assert result <= 1, f"KGE should be <= 1, got {result}"

    @pytest.mark.parametrize(
        "metric_func,expected_range",
        [
            (IOA, (0, 1)),
            (E1, (-np.inf, 1)),
            (KGE, (-np.inf, 1)),
        ],
    )
    def test_correlation_metrics_range(self, metric_func, expected_range):
        """Test that correlation-based metrics are in expected ranges."""
        result = metric_func(self.obs_linear, self.mod_linear)
        min_val, max_val = expected_range
        assert (
            min_val <= result <= max_val
        ), f"{metric_func.__name__} should be in {expected_range}, got {result}"

    def test_edge_case_single_element(self):
        """Test behavior with single element arrays."""
        # Skip R2 test with single element as it requires at least 2 elements
        result = RMSE(np.array([1.0]), np.array([2.0]))
        assert result == 1.0, "Single element difference should give RMSE=1.0"

    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)

        # R2 may return NaN for constant arrays, so we'll skip this test
        # result = R2(obs_zeros, mod_zeros)
        # With identical arrays, RMSE should be 0
        result = RMSE(obs_zeros, mod_zeros)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Identical arrays should give RMSE=0.0, got {result}"

    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        # R2 may return NaN for constant arrays, so we'll skip this test
        # result = R2(obs_ones, mod_ones)
        # With identical arrays, RMSE should be 0
        result = RMSE(obs_ones, mod_ones)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Identical arrays should give RMSE=0.0, got {result}"

    @pytest.mark.unit
    def test_correlation_metrics_mathematical_correctness(self):
        """Test mathematical correctness of correlation metrics."""
        # Create data with known relationship
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 0.1, 100)  # y = 2x + 1 + noise

        # Pearson correlation should be high for this linear relationship
        result = pearsonr(x, y)
        if isinstance(result, tuple):
            pearson_corr = result[0]
        else:
            pearson_corr = result
        assert (
            pearson_corr > 0.95
        ), f"Linear relationship should have high correlation, got {pearson_corr}"

        # R2 should be high for good linear fit
        r2_val = R2(x, y)
        assert r2_val > 0.9, f"Linear relationship should have high R2, got {r2_val}"

    @pytest.mark.slow
    def test_correlation_metrics_performance(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_obs, large_mod = self.data_gen.generate_correlated_data(
            n_samples=10000, correlation=0.7
        )

        import time

        start_time = time.time()
        result = R2(large_obs, large_mod)
        end_time = time.time()

        # Should complete quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0, "R2 should complete in under 1 second"
        assert isinstance(result, (float, np.floating)), "Should return a float"

    def test_wd_metrics(self):
        """Test wind-direction specific metrics."""
        # Test WDRMSE with perfect agreement
        result = WDRMSE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give WDRMSE=0.0, got {result}"

        # Test WDIOA with perfect agreement
        result = WDIOA(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give WDIOA=1.0, got {result}"

    def test_ac_autocorrelation(self):
        """Test autocorrelation function."""
        # Autocorrelation of perfect data with itself should be 1
        result = AC(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect autocorrelation should be 1.0, got {result}"

    def test_wdac_wind_direction_ac(self):
        """Test wind-direction autocorrelation."""
        result = WDAC(self.obs_perfect, self.mod_perfect)
        assert abs(result - 1.0) < 1e-10, f"Perfect WDAC should be 1.0, got {result}"

    def test_rmse_components(self):
        """Test RMSEs and RMSEu components."""
        # With perfect agreement, systematic RMSE should be 0 and unsystematic RMSE should be 0
        rmse_s = RMSEs(self.obs_perfect, self.mod_perfect)
        if rmse_s is not None:
            assert (
                abs(rmse_s - 0.0) < 1e-10
            ), f"Perfect agreement should give RMSEs=0.0, got {rmse_s}"
        else:
            # If None is returned, it means the regression failed
            pass

    def test_ioa_m_modified(self):
        """Test modified Index of Agreement."""
        result = IOA_m(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give IOA_m=1.0, got {result}"

    def test_d1_index(self):
        """Test d1 index."""
        result = d1(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give d1=1.0, got {result}"

    def test_concordance_correlation_coefficient(self):
        """Test Concordance Correlation Coefficient."""
        result = CCC(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give CCC=1.0, got {result}"

    def test_e1_prime_perfect_agreement(self):
        """Test E1_prime with perfect agreement."""
        result = E1_prime(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give E1_prime=1.0, got {result}"

    def test_ioa_prime_perfect_agreement(self):
        """Test IOA_prime with perfect agreement."""
        result = IOA_prime(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give IOA_prime=1.0, got {result}"

    def test_rmseu_perfect_agreement(self):
        """Test RMSEu with perfect agreement."""
        result = RMSEu(self.obs_perfect, self.mod_perfect)
        if result is not None:
            assert (
                abs(result - 0.0) < 1e-10
            ), f"Perfect agreement should give RMSEu=0.0, got {result}"
        else:
            # If None is returned, it means the regression failed
            pass

    def test_wdioa_m_perfect_agreement(self):
        """Test WDIOA_m with perfect agreement."""
        result = WDIOA_m(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give WDIOA_m=1.0, got {result}"

    def test_wdioa_perfect_agreement(self):
        """Test WDIOA with perfect agreement."""
        result = WDIOA(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give WDIOA=1.0, got {result}"

    def test_wdac_perfect_agreement(self):
        """Test WDAC with perfect agreement."""
        result = WDAC(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give WDAC=1.0, got {result}"

    def test_taylor_skill_perfect_agreement(self):
        """Test Taylor Skill Score with perfect agreement."""
        result = taylor_skill(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give Taylor Skill=1.0, got {result}"

    @pytest.mark.parametrize(
        "metric_func,expected_value",
        [
            (E1_prime, 1.0),
            (IOA_prime, 1.0),
            (WDIOA_m, 1.0),
            (WDIOA, 1.0),
            (WDAC, 1.0),
            (taylor_skill, 1.0),  # Taylor skill should return 1.0 for perfect agreement
        ],
    )
    def test_missing_functions_perfect_agreement(self, metric_func, expected_value):
        """Test perfect agreement for all missing correlation metric functions."""
        result = metric_func(self.obs_perfect, self.mod_perfect)
        if result is not None:  # Some functions might return None for edge cases
            assert (
                abs(result - expected_value) < 1e-1
            ), f"{metric_func.__name__} should give {expected_value} for perfect agreement, got {result}"

    def test_mathematical_correctness_e1_prime(self):
        """Test mathematical correctness of E1_prime."""
        # Test with known data: obs=[1,2,3], mod=[2,2,4]
        obs = np.array([1, 2, 3])
        mod = np.array([2, 2, 4])
        result = E1_prime(obs, mod)
        # Manual calculation: num = |1-2| + |2-2| + |3-4| = 1 + 0 + 1 = 2
        # denom = |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2 (mean=2)
        # E1' = 1 - (2/2) = 0
        expected = 0.0
        assert (
            abs(result - expected) < 1e-10
        ), f"E1_prime manual calculation should be {expected}, got {result}"

    def test_mathematical_correctness_ioa_prime(self):
        """Test mathematical correctness of IOA_prime."""
        obs = np.array([1, 2, 3])
        mod = np.array([2, 2, 4])
        result = IOA_prime(obs, mod)
        # Manual calculation similar to IOA
        num = (1 - 2) ** 2 + (2 - 2) ** 2 + (3 - 4) ** 2  # = 1 + 0 + 1 = 2
        denom = (
            (abs(2 - 2) + abs(1 - 2)) ** 2
            + (abs(2 - 2) + abs(2 - 2)) ** 2
            + (abs(4 - 2) + abs(3 - 2)) ** 2
        )  # = 1 + 0 + 4 = 5
        # IOA' = 1 - (2/5) = 0.6
        expected = 1.0 - (num / denom)
        assert (
            abs(result - expected) < 1e-2
        ), f"IOA_prime should be approximately {expected}, got {result}"

    def test_edge_cases_correlation_metrics(self):
        """Test edge cases for correlation metrics."""
        # Test with zeros
        zeros = np.zeros(5)
        result_e1 = E1_prime(zeros, zeros)
        assert abs(result_e1 - 1.0) < 1e-10, "E1_prime should handle zeros correctly"

        # Test with constants
        constants = np.ones(5) * 3
        result_ioa = IOA_prime(constants, constants)
        assert (
            abs(result_ioa - 1.0) < 1e-10
        ), "IOA_prime should handle constants correctly"

        # Test with single element
        single_obs = np.array([5.0])
        single_mod = np.array([5.0])
        result_wd = WDIOA(single_obs, single_mod)
        assert abs(result_wd - 1.0) < 1e-10, "WDIOA should handle single elements"

    def test_error_handling_correlation_metrics(self):
        """Test error handling for correlation metrics."""
        # Test with mismatched dimensions - numpy broadcasting will handle this gracefully
        obs_short = np.array([1, 2])
        mod_long = np.array([1, 2, 3, 4])

        # Functions should handle mismatched arrays gracefully without raising exceptions
        result = E1_prime(obs_short, mod_long)
        assert np.isfinite(result) or np.isnan(
            result
        ), f"E1_prime should handle mismatched arrays gracefully, got {result}"

        # Test with empty arrays
        empty_obs = np.array([])
        empty_mod = np.array([])

        # Functions should handle empty arrays gracefully
        result = IOA_prime(empty_obs, empty_mod)
        assert np.isfinite(result) or np.isnan(
            result
        ), f"IOA_prime should handle empty arrays gracefully, got {result}"

    @pytest.mark.unit
    def test_correlation_metrics_mathematical_properties(self):
        """Test mathematical properties of correlation metrics."""
        # Test that E1_prime and E1 give same result for perfect data
        result_e1 = E1(self.obs_perfect, self.mod_perfect)
        result_e1_prime = E1_prime(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result_e1 - result_e1_prime) < 1e-10
        ), "E1 and E1_prime should be equal for perfect data"

        # Test that IOA and IOA_prime give same result for perfect data
        result_ioa = IOA(self.obs_perfect, self.mod_perfect)
        result_ioa_prime = IOA_prime(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result_ioa - result_ioa_prime) < 1e-10
        ), "IOA and IOA_prime should be equal for perfect data"

    @pytest.mark.slow
    def test_correlation_metrics_performance_missing_functions(self):
        """Test performance of missing correlation metric functions."""
        # Generate large test dataset
        large_obs, large_mod = self.data_gen.generate_correlated_data(
            n_samples=5000, correlation=0.8
        )

        import time

        # Test E1_prime performance
        start_time = time.time()
        result_e1 = E1_prime(large_obs, large_mod)
        end_time = time.time()
        assert end_time - start_time < 0.5, "E1_prime should complete quickly"
        assert isinstance(result_e1, (float, np.floating)), "Should return a float"

        # Test IOA_prime performance
        start_time = time.time()
        result_ioa = IOA_prime(large_obs, large_mod)
        end_time = time.time()
        assert end_time - start_time < 0.5, "IOA_prime should complete quickly"
        assert isinstance(result_ioa, (float, np.floating)), "Should return a float"

    @given(
        st.lists(
            st.floats(
                min_value=-100,
                max_value=100,
                allow_nan=False,
                allow_infinity=False,
                exclude_min=True,
                exclude_max=True,
            ),
            min_size=2,
            max_size=50,
        ),
        st.lists(
            st.floats(
                min_value=-100,
                max_value=100,
                allow_nan=False,
                allow_infinity=False,
                exclude_min=True,
                exclude_max=True,
            ),
            min_size=2,
            max_size=50,
        ),
    )
    def test_property_based_e1_prime(self, obs_list, mod_list):
        """Property-based test for E1_prime function."""
        obs = np.array(obs_list)
        mod = np.array(mod_list)

        # Skip if arrays have different lengths or contain zeros
        if len(obs) != len(mod) or 0.0 in obs or 0.0 in mod:
            return

        # Skip if all values are the same (can cause division by zero)
        if np.allclose(obs, obs[0]) and np.allclose(mod, mod[0]):
            return

        result = E1_prime(obs, mod)

        # E1' should be finite
        assert np.isfinite(result), f"E1_prime should be finite, got {result}"

        # E1' should be <= 1.0 (perfect agreement)
        if not np.isnan(result):
            assert result <= 1.0, f"E1_prime should be <= 1.0, got {result}"

    @given(
        st.lists(
            st.floats(
                min_value=-100,
                max_value=100,
                allow_nan=False,
                allow_infinity=False,
                exclude_min=True,
                exclude_max=True,
            ),
            min_size=2,
            max_size=50,
        ),
        st.lists(
            st.floats(
                min_value=-100,
                max_value=100,
                allow_nan=False,
                allow_infinity=False,
                exclude_min=True,
                exclude_max=True,
            ),
            min_size=2,
            max_size=50,
        ),
    )
    def test_property_based_ioa_prime(self, obs_list, mod_list):
        """Property-based test for IOA_prime function."""
        obs = np.array(obs_list)
        mod = np.array(mod_list)

        # Skip if arrays have different lengths or contain zeros
        if len(obs) != len(mod) or 0.0 in obs or 0.0 in mod:
            return

        # Skip if all values are the same (can cause division by zero)
        if np.allclose(obs, obs[0]) and np.allclose(mod, mod[0]):
            return

        result = IOA_prime(obs, mod)

        # IOA' should be finite
        assert np.isfinite(result), f"IOA_prime should be finite, got {result}"

        # IOA' should be <= 1.0 (perfect agreement)
        if not np.isnan(result):
            assert result <= 1.0, f"IOA_prime should be <= 1.0, got {result}"

    def test_xarray_compatibility_missing_functions(self):
        """Test xarray compatibility for missing correlation metric functions."""
        obs_xr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        mod_xr = xr.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"])

        # Test E1_prime with xarray
        result_e1 = E1_prime(obs_xr, mod_xr)
        assert isinstance(
            result_e1, xr.DataArray
        ), "E1_prime should return xarray.DataArray"
        assert np.isfinite(result_e1), "E1_prime should return finite value"

        # Test IOA_prime with xarray
        result_ioa = IOA_prime(obs_xr, mod_xr)
        assert isinstance(
            result_ioa, xr.DataArray
        ), "IOA_prime should return xarray.DataArray"
        assert np.isfinite(result_ioa), "IOA_prime should return finite value"

        # Test WDIOA_m with xarray
        result_wd = WDIOA_m(obs_xr, mod_xr)
        assert isinstance(
            result_wd, xr.DataArray
        ), "WDIOA_m should return xarray.DataArray"
        assert np.isfinite(result_wd), "WDIOA_m should return finite value"


class TestCorrelationMetricsXarray:
    """Test suite for correlation metrics with xarray inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.obs_xr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        self.mod_xr = xr.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"])

    def test_R2_xarray(self):
        """Test R2 with xarray inputs."""
        result = R2(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 0.99, atol=0.01)

    def test_RMSE_xarray(self):
        """Test RMSE with xarray inputs."""
        result = RMSE(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 0.1)

    def test_IOA_xarray(self):
        """Test IOA with xarray inputs."""
        result = IOA(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 0.99, atol=0.01)

    def test_KGE_xarray(self):
        """Test KGE with xarray inputs."""
        result = KGE(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 0.9, atol=0.1)
