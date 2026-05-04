#!/usr/bin/env python3
"""
Additional tests for efficiency_metrics.py to improve coverage.
Tests focus on functions that are not well-covered by existing tests.
"""

import numpy as np

from monet_stats.efficiency_metrics import (
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


class TestNSEMetrics:
    """Test Nash-Sutcliffe Efficiency metrics."""

    def test_nse_basic(self):
        """Test NSE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 2.9, 4.1])
        result = NSE(obs, mod)
        # Calculate expected: 1 - sum((obs-mod)^2)/sum((obs-mean(obs))^2)
        obs_mean = np.mean(obs)
        numerator = np.sum((obs - mod) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        expected = 1.0 - (numerator / denominator)
        assert np.isclose(result, expected)

    def test_nsem_basic(self):
        """Test NSEm with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 2.9, 4.1])
        result = NSEm(obs, mod)
        # Calculate expected: 1 - sum((obs-mod)^2)/sum((obs-mean(obs))^2)
        obs_mean = np.mean(obs)
        numerator = np.sum((obs - mod) ** 2)
        denominator = np.sum((obs - obs_mean) ** 2)
        expected = 1.0 - (numerator / denominator)
        assert np.isclose(result, expected)


class TestLogNSEMetrics:
    """Test logarithmic NSE metrics."""

    def test_nselog_basic(self):
        """Test NSElog with basic numpy arrays."""
        obs = np.array([1.1, 2.1, 3.1, 4.1])  # Use positive values > 1
        mod = np.array([1.2, 2.2, 3.2, 4.2])
        result = NSElog(obs, mod)
        # NSElog = 1 - sum((log(obs)-log(mod))^2)/sum((log(obs)-mean(log(obs)))^2)
        epsilon = 1e-6
        log_obs = np.log(obs + epsilon)
        log_mod = np.log(mod + epsilon)
        log_obs_mean = np.mean(log_obs)
        numerator = np.sum((log_obs - log_mod) ** 2)
        denominator = np.sum((log_obs - log_obs_mean) ** 2)
        expected = 1.0 - (numerator / denominator)
        assert np.isclose(result, expected)


class TestRelativeNSEMetrics:
    """Test relative NSE metrics."""

    def test_rnse_basic(self):
        """Test rNSE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 2.9, 4.1])
        result = rNSE(obs, mod)
        # rNSE should return a reasonable efficiency value
        # Typical range for NSE-like metrics is -∞ to 1, with 1 being perfect
        assert isinstance(result, (float, np.floating))
        assert result <= 1.0  # Should not exceed 1 (perfect efficiency)
        assert result > 0.9  # Should be high for this good prediction


class TestModifiedNSEMetrics:
    """Test modified NSE metrics."""

    def test_mnse_basic(self):
        """Test mNSE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 2.9, 4.1])
        result = mNSE(obs, mod)
        # mNSE = 1 - sum(|obs-mod|)/sum(|obs-mean(obs)|)
        obs_mean = np.mean(obs)
        numerator = np.sum(np.abs(obs - mod))
        denominator = np.sum(np.abs(obs - obs_mean))
        expected = 1.0 - (numerator / denominator)
        assert np.isclose(result, expected)


class TestPercentCorrectMetrics:
    """Test percent correct metrics."""

    def test_pc_basic(self):
        """Test PC with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.05, 2.1, 2.95, 4.05])
        result = PC(obs, mod)
        # PC = 100 * (number of predictions within tolerance) / total predictions
        # Default tolerance: 10% of observed value
        tolerance = 0.1 * np.abs(obs)
        correct = np.abs(obs - mod) <= tolerance
        expected = (np.sum(correct) / len(correct)) * 100.0
        assert np.isclose(result, expected)


class TestAbsoluteErrorMetrics:
    """Test absolute error metrics."""

    def test_mae_basic(self):
        """Test MAE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MAE(obs, mod)
        expected = np.ma.abs(mod - obs).mean()
        assert np.isclose(result, expected)


class TestSquaredErrorMetrics:
    """Test squared error metrics."""

    def test_mse_basic(self):
        """Test MSE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MSE(obs, mod)
        expected = np.ma.mean((mod - obs) ** 2)
        assert np.isclose(result, expected)


class TestPercentageErrorMetrics:
    """Test percentage error metrics."""

    def test_mape_basic(self):
        """Test MAPE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0])
        mod = np.array([2.0, 2.0, 4.0])
        result = MAPE(obs, mod)
        expected = (100 * np.ma.abs(mod - obs) / np.ma.abs(obs)).mean()
        assert np.isclose(result, expected)


class TestScaledErrorMetrics:
    """Test scaled error metrics."""

    def test_mase_basic(self):
        """Test MASE with basic numpy arrays."""
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mod = np.array([1.1, 2.1, 3.1, 4.1])
        result = MASE(obs, mod)
        # Calculate naive forecast error (using previous observation)
        naive_diff = np.diff(obs)
        naive_error = np.mean(np.abs(naive_diff))
        model_error = np.mean(np.abs(mod - obs))
        expected = model_error / naive_error
        assert np.isclose(result, expected)
