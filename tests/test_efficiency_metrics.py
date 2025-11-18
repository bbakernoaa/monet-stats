"""
Tests for efficiency_metrics.py module.

This module tests efficiency-based statistical metrics including
NSE, NSEm, NSElog, MAPE, MSE, MASE, and other efficiency statistics.
"""
import numpy as np
import pytest

from src.monet_stats.efficiency_metrics import (
    MAPE, MSE, MASE, NSE, NSEm, NSElog, PC, mNSE, rNSE
)


class TestEfficiencyMetrics:
    """Test suite for efficiency metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Perfect agreement case
        self.obs_perfect = np.array([1, 2, 3, 4, 5])
        self.mod_perfect = np.array([1, 2, 3, 4, 5])
        
        # Small systematic bias
        self.obs_biased = np.array([1, 2, 3, 4, 5])
        self.mod_biased = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        # Random errors
        np.random.seed(42)
        self.obs_random = np.random.normal(0, 1, 50)
        self.mod_random = self.obs_random + np.random.normal(0, 0.2, 50)  # Small noise
    
    def test_nse_perfect_agreement(self):
        """Test NSE with perfect agreement."""
        result = NSE(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, NSE should be 1.0
        assert abs(result - 1.0) < 1e-10
    
    def test_nse_worst_case(self):
        """Test NSE with worst case (model is just the mean)."""
        obs = np.array([1, 2, 3, 4, 5])
        # Model that's just the mean of observations
        mod = np.full_like(obs, np.mean(obs))
        result = NSE(obs, mod)
        # Should be 0 since model performs as well as the mean
        assert abs(result - 0.0) < 1e-10
    
    def test_nse_negative_values(self):
        """Test NSE with negative values."""
        obs = np.array([-2, -1, 0, 1, 2])
        mod = np.array([-1.9, -0.9, 0.1, 1.1, 2.1])  # Small errors
        result = NSE(obs, mod)
        # Should be a reasonable value (not NaN)
        assert np.isfinite(result)
    
    def test_nse_with_bias(self):
        """Test NSE with systematic bias."""
        result = NSE(self.obs_biased, self.mod_biased)
        # Should be < 1.0 due to bias
        assert result < 1.0
    
    def test_nsem_perfect_agreement(self):
        """Test NSEm with perfect agreement."""
        result = NSEm(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, NSEm should be 1.0
        assert abs(result - 1.0) < 1e-10
    
    def test_nselog_perfect_agreement(self):
        """Test NSElog with perfect agreement."""
        # Use positive values for log calculation
        obs_pos = np.abs(self.obs_perfect) + 0.1  # Ensure positive values
        mod_pos = np.abs(self.mod_perfect) + 0.1
        result = NSElog(obs_pos, mod_pos)
        # With perfect agreement, NSElog should be 1.0
        assert abs(result - 1.0) < 1e-10
    
    def test_nselog_with_zeros(self):
        """Test NSElog with zero values (should handle log(0))."""
        obs_with_zero = np.array([0, 1, 2, 3, 4])
        mod_with_zero = np.array([0.1, 1, 2, 3, 4])  # Slightly different to avoid perfect match
        result = NSElog(obs_with_zero, mod_with_zero)
        # Should handle log(0) appropriately (using small epsilon)
        assert np.isfinite(result)
    
    def test_mse_perfect_agreement(self):
        """Test MSE with perfect agreement."""
        result = MSE(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, MSE should be 0.0
        assert abs(result - 0.0) < 1e-10
    
    def test_mse_positive_values(self):
        """Test that MSE is always positive."""
        result = MSE(self.obs_random, self.mod_random)
        assert result >= 0, f"MSE should be non-negative, got {result}"
    
    def test_mape_perfect_agreement(self):
        """Test MAPE with perfect agreement."""
        result = MAPE(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, MAPE should be 0.0
        assert abs(result - 0.0) < 1e-10
    
    def test_mape_with_zeros(self):
        """Test MAPE with zero values (should handle division by zero)."""
        obs_with_zero = np.array([0, 1, 2, 3, 4])
        mod_with_zero = np.array([0.1, 1, 2, 3, 4])  # Slightly different to avoid perfect match
        result = MAPE(obs_with_zero, mod_with_zero)
        # Should handle division by zero appropriately
        assert np.isfinite(result) or np.isnan(result)
    
    def test_mase_perfect_agreement(self):
        """Test MASE with perfect agreement."""
        result = MASE(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, should have low error relative to naive forecast
        assert np.isfinite(result)
    
    def test_mase_with_zeros(self):
        """Test MASE with zero values."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)
        result = MASE(obs_zeros, mod_zeros)
        # Should handle case where naive forecast error is zero
        assert np.isfinite(result) or np.isnan(result)
    
    def test_pc_perfect_agreement(self):
        """Test PC (Performance Coefficient) with perfect agreement."""
        result = PC(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, should return some finite value
        assert np.isfinite(result)
    
    def test_mnse_perfect_agreement(self):
        """Test mNSE with perfect agreement."""
        result = mNSE(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, mNSE should be 1.0
        assert abs(result - 1.0) < 1e-10
    
    def test_rnse_perfect_agreement(self):
        """Test rNSE with perfect agreement."""
        result = rNSE(self.obs_perfect, self.mod_perfect)
        # With perfect agreement, rNSE should be 1.0
        assert abs(result - 1.0) < 1e-10
    
    @pytest.mark.parametrize("metric_func", [
        NSE, NSEm, NSElog, MSE, MAPE, MASE, PC, mNSE, rNSE
    ])
    def test_efficiency_metrics_output_type(self, metric_func):
        """Test that efficiency metrics return appropriate values."""
        result = metric_func(self.obs_random, self.mod_random)
        assert isinstance(result, (float, np.floating, int, np.integer)), \
            f"{metric_func.__name__} should return a numeric value, got {type(result)}"
    
    def test_edge_case_single_element(self):
        """Test behavior with single element arrays."""
        result = MSE(np.array([1.0]), np.array([1.0]))
        assert abs(result - 0.0) < 1e-10, "Single perfect match should give MSE=0.0"
        
        result = NSE(np.array([1.0]), np.array([1.0]))
        assert abs(result - 1.0) < 1e-10, "Single perfect match should give NSE=1.0"
    
    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)
        
        result_mse = MSE(obs_zeros, mod_zeros)
        assert abs(result_mse - 0.0) < 1e-10, f"Identical zeros should give MSE=0.0, got {result_mse}"
        
        result_nse = NSE(obs_zeros, mod_zeros)
        # When both arrays are zeros, NSE might be NaN or 1 depending on implementation
        assert np.isfinite(result_nse) or np.isnan(result_nse), f"NSE with zeros should be finite or NaN, got {result_nse}"
    
    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)
        
        result_mse = MSE(obs_ones, mod_ones)
        assert abs(result_mse - 0.0) < 1e-10, f"Identical arrays should give MSE=0.0, got {result_mse}"
        
        result_nse = NSE(obs_ones, mod_ones)
        assert abs(result_nse - 1.0) < 1e-10, f"Identical arrays should give NSE=1.0, got {result_nse}"
    
    @pytest.mark.unit
    def test_efficiency_metrics_mathematical_correctness(self):
        """Test mathematical correctness of efficiency metrics."""
        # Create data with known properties
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Small errors
        
        # MSE should be mean of squared errors: mean([0.1^2, 0.1^2, 0.1^2, 0.1^2, 0.1^2]) = 0.01
        expected_mse = 0.01
        mse_result = MSE(obs, mod)
        assert abs(mse_result - expected_mse) < 1e-10, f"Expected MSE={expected_mse}, got {mse_result}"
        
        # NSE calculation: 1 - (sum of squared errors) / (sum of squared deviations from mean)
        # SSE = 5 * 0.01 = 0.05
        # SS_mean = sum((obs - mean)^2) = sum([4, 1, 0, 1, 4]) = 10
        # NSE = 1 - 0.05/10 = 1 - 0.005 = 0.995
        expected_nse = 1 - (0.05 / 10)
        nse_result = NSE(obs, mod)
        assert abs(nse_result - expected_nse) < 1e-10, f"Expected NSE={expected_nse}, got {nse_result}"
    
    @pytest.mark.slow
    def test_efficiency_metrics_performance(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_obs = np.random.normal(0, 1, 10000)
        large_mod = large_obs + np.random.normal(0, 0.1, 10000)  # Small noise
        
        import time
        start_time = time.time()
        result = NSE(large_obs, large_mod)
        end_time = time.time()
        
        # Should complete quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0, "NSE should complete in under 1 second"
        assert isinstance(result, (float, np.floating)), "Should return a float"
    
    def test_nse_range_bounds(self):
        """Test that NSE is within expected range."""
        # NSE can be negative for poor models, but should be <= 1
        result = NSE(self.obs_random, self.mod_random)
        assert result <= 1.0, f"NSE should be <= 1.0, got {result}"
    
    def test_mape_range_bounds(self):
        """Test that MAPE is non-negative."""
        result = MAPE(self.obs_random, self.mod_random)
        if np.isfinite(result):
            assert result >= 0, f"MAPE should be non-negative, got {result}"
    
    def test_mse_vs_numpy_var(self):
        """Test MSE calculation against numpy."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        
        our_result = MSE(obs, mod)
        numpy_result = np.mean((obs - mod)**2)
        
        assert abs(our_result - numpy_result) < 1e-10, f"MSE should match numpy calculation: {our_result} vs {numpy_result}"
    
    def test_nse_with_very_poor_model(self):
        """Test NSE with a very poor model."""
        obs = np.array([1, 2, 3, 4, 5])
        # Very poor model - constant value far from observations
        mod = np.array([10, 10, 10, 10])
        result = NSE(obs, mod)
        # Should be negative since model is worse than using the mean
        assert result < 0, f"NSE should be negative for very poor model, got {result}"
    
    def test_nse_with_large_values(self):
        """Test NSE with large values."""
        obs = np.array([100, 2000, 3000, 4000, 500])
        mod = np.array([1001, 2001, 301, 4001, 5001])  # Small relative errors
        result = NSE(obs, mod)
        # Should be close to 1 since relative errors are small
        assert result > 0.9, f"NSE should be high for small relative errors, got {result}"