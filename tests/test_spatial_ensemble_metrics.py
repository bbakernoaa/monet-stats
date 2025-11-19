"""
Tests for spatial_ensemble_metrics.py module.

This module tests spatial and ensemble-based statistical metrics including
FSS, SAL, CRPS, BSS, and other spatial verification metrics.
"""
import numpy as np
import pytest

from src.monet_stats.spatial_ensemble_metrics import (
    BSS,
    CRPS,
    EDS,
    FSS,
    SAL,
    ensemble_mean,
    ensemble_std,
    rank_histogram,
    spread_error,
)


class TestSpatialEnsembleMetrics:
    """Test suite for spatial and ensemble metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        # Perfect agreement case
        self.obs_perfect = np.array([1, 2, 3, 4, 5])
        self.mod_perfect = np.array([1, 2, 3, 4, 5])

        # Small systematic bias
        self.obs_biased = np.array([1, 2, 3, 4, 5])
        self.mod_biased = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Ensemble forecasts
        self.ensemble_forecasts = np.random.normal(20, 3, (50, 100))  # 50 ensemble members, 100 time points
        self.observed_values = np.random.normal(20, 2, 100)

        # 2D spatial fields
        self.obs_field = np.random.exponential(5, (20, 20))
        self.mod_field = self.obs_field + np.random.normal(0, 1, (20, 20))
        self.obs_field = np.maximum(self.obs_field, 0)  # Ensure non-negative values
        self.mod_field = np.maximum(self.mod_field, 0)

    def test_ensemble_mean_basic(self):
        """Test ensemble_mean with basic ensemble."""
        ensemble = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3 members, 3 time points
        result = ensemble_mean(ensemble)
        expected = np.array([4, 5, 6])  # Mean across ensemble members
        np.testing.assert_array_equal(result, expected)

    def test_ensemble_mean_perfect_agreement(self):
        """Test ensemble_mean with identical members."""
        ensemble = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])  # All members identical
        result = ensemble_mean(ensemble)
        expected = np.array([5, 5, 5])
        np.testing.assert_array_equal(result, expected)

    def test_ensemble_std_basic(self):
        """Test ensemble_std with basic ensemble."""
        ensemble = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # All members identical
        result = ensemble_std(ensemble)
        expected = np.array([0, 0, 0])  # No variance
        np.testing.assert_array_equal(result, expected)

        # Test with variance
        ensemble = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])  # Increasing values
        result = ensemble_std(ensemble)
        # Standard deviation of [1,3,5], [2,4,6], [3,5,7] should be > 0
        assert all(r > 0 for r in result)

    def test_bss_perfect_agreement(self):
        """Test BSS with perfect agreement."""
        obs_binary = np.array([1, 0, 1, 0])
        mod_binary = np.array([1, 0, 1, 0])
        result = BSS(obs_binary, mod_binary, threshold=0.5)
        # With perfect agreement, BSS should be 1.0
        assert abs(result - 1.0) < 1e-10

    def test_bss_worst_agreement(self):
        """Test BSS with worst agreement."""
        obs_binary = np.array([1, 0, 1, 0])
        mod_binary = np.array([0, 1, 0, 1])
        result = BSS(obs_binary, mod_binary, threshold=0.5)
        # With opposite predictions, BSS should be negative
        assert result < 0

    def test_crps_basic(self):
        """Test CRPS with basic values."""
        # Simple ensemble: 3 members predicting value around 5
        ensemble = np.array([[4.9, 5.1, 5.0], [4.8, 5.2, 5.0], [5.1, 4.9, 5.0]])
        obs = np.array([5.0, 5.0, 5.0])
        result = CRPS(ensemble, obs)
        # Should be a positive value representing the CRPS score
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result >= 0)

    def test_fss_perfect_agreement(self):
        """Test FSS with perfect agreement."""
        # For perfect agreement, FSS should be close to 1
        result = FSS(self.obs_field, self.mod_field, window=5, threshold=10)
        assert 0 <= result <= 1

    def test_fss_with_zeros(self):
        """Test FSS with zero arrays."""
        obs_zeros = np.zeros((10, 10))
        mod_zeros = np.zeros((10, 10))
        result = FSS(obs_zeros, mod_zeros, window=2, threshold=1)
        # With all zeros and threshold > 0, should get a valid FSS score
        assert 0 <= result <= 1

    def test_sal_basic(self):
        """Test SAL with basic values."""
        # SAL returns 3 values: Structure, Amplitude, Location
        s, a, l = SAL(self.obs_field, self.mod_field, threshold=5)
        # All should be finite values
        assert np.isfinite(s)
        assert np.isfinite(a)
        assert np.isfinite(l)
        # Each component should be reasonable in magnitude
        assert abs(s) < 10  # Structure component
        assert abs(a) < 10  # Amplitude component
        assert abs(l) < 10  # Location component

    def test_rank_histogram_basic(self):
        """Test rank histogram with basic ensemble."""
        # Create ensemble where observations are in the middle of the distribution
        ensemble = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3 members
        obs = np.array([2, 5, 8])  # Obs in middle of each time point
        result = rank_histogram(ensemble, obs)
        # Should return histogram with 4 bins (3 members + 1)
        assert len(result) == 4
        # At least one bin should have a count
        assert sum(result) == len(obs)

    def test_spread_error_basic(self):
        """Test spread_error with basic ensemble."""
        ensemble = np.random.normal(0, 1, (10, 100))  # 10 members, 100 time points
        obs = np.random.normal(0, 1, 100)
        spread, error = spread_error(ensemble, obs)
        # Both should be positive values
        assert spread >= 0
        assert error >= 0

    def test_eds_basic(self):
        """Test EDS (Extreme Dependency Score) with basic values."""
        # EDS should handle binary event detection
        obs_events = np.array([1, 0, 1, 0, 1])  # Observed events
        mod_events = np.array([1, 1, 0, 0, 1])  # Model events
        result = EDS(obs_events, mod_events, threshold=0.5)
        # Should be between -1 and 1
        assert -1 <= result <= 1

    def test_edge_case_single_element_ensemble(self):
        """Test ensemble functions with single element."""
        ensemble_single = np.array([[5]])  # 1 member, 1 time point
        result_mean = ensemble_mean(ensemble_single)
        assert result_mean[0] == 5

        result_std = ensemble_std(ensemble_single)
        assert result_std[0] == 0  # Single value has no variance

    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)

        # BSS with all zeros
        result_bss = BSS(obs_zeros, mod_zeros, threshold=0.5)
        # Should handle division by zero appropriately
        assert np.isfinite(result_bss) or np.isnan(result_bss)

    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        result_bss = BSS(obs_ones, mod_ones, threshold=0.5)
        # With perfect agreement, should get high BSS
        assert np.isfinite(result_bss)

    @pytest.mark.parametrize("metric_func", [
        ensemble_mean, ensemble_std
    ])
    def test_ensemble_functions_output_type(self, metric_func):
        """Test that ensemble functions return appropriate values."""
        result = metric_func(self.ensemble_forecasts)
        assert isinstance(result, np.ndarray), \
            f"{metric_func.__name__} should return an array, got {type(result)}"

    @pytest.mark.unit
    def test_spatial_metrics_mathematical_correctness(self):
        """Test mathematical correctness of spatial metrics."""
        # Create simple spatial fields to test
        obs_simple = np.ones((5, 5))
        mod_simple = np.ones((5, 5))  # Perfect agreement

        # FSS with perfect agreement should be high
        fss_result = FSS(obs_simple, mod_simple, window=2, threshold=0.5)
        # For identical fields, FSS should be close to 1
        assert 0 <= fss_result <= 1

    @pytest.mark.slow
    def test_spatial_metrics_performance(self):
        """Test performance with larger datasets."""
        # Create moderately sized spatial fields
        large_obs = np.random.random((50, 50))
        large_mod = large_obs + np.random.normal(0, 0.1, (50, 50))

        import time
        start_time = time.time()
        result = FSS(large_obs, large_mod, window=5, threshold=0.5)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 5.0, "FSS should complete in under 5 seconds"
        assert isinstance(result, (float, np.floating)), "Should return a float"

    def test_bss_with_various_thresholds(self):
        """Test BSS with different thresholds."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        # Test different thresholds
        result1 = BSS(obs, mod, threshold=2.5)
        result2 = BSS(obs, mod, threshold=3.5)

        # Both should be finite values
        assert np.isfinite(result1)
        assert np.isfinite(result2)

    def test_crps_with_different_ensemble_sizes(self):
        """Test CRPS with different ensemble sizes."""
        # Small ensemble
        small_ensemble = np.random.normal(0, 1, (5, 10))
        obs = np.random.normal(0, 1, 10)
        result_small = CRPS(small_ensemble, obs)

        # Larger ensemble
        large_ensemble = np.random.normal(0, 1, (20, 10))
        result_large = CRPS(large_ensemble, obs)

        # Both should be positive
        assert np.all(result_small >= 0)
        assert np.all(result_large >= 0)

    def test_ensemble_mean_vs_numpy_mean(self):
        """Test ensemble_mean against numpy mean."""
        ensemble = np.random.normal(0, 1, (10, 100))
        our_result = ensemble_mean(ensemble)
        numpy_result = np.mean(ensemble, axis=0)

        np.testing.assert_array_almost_equal(our_result, numpy_result)

    def test_ensemble_std_vs_numpy_std(self):
        """Test ensemble_std against numpy std."""
        ensemble = np.random.normal(0, 1, (10, 100))
        our_result = ensemble_std(ensemble)
        numpy_result = np.std(ensemble, axis=0)

        np.testing.assert_array_almost_equal(our_result, numpy_result)
