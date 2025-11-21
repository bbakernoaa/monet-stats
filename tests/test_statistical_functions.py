"""
Comprehensive unit tests for all statistical functions in monet_stats.
"""

import numpy as np
import pytest
from scipy.stats import pearsonr

from monet_stats.contingency_metrics import FAR, POD
from monet_stats.correlation_metrics import AC, R2
from monet_stats.correlation_metrics import pearsonr as stats_pearsonr
from monet_stats.efficiency_metrics import NSE as eff_NSE
from monet_stats.error_metrics import LOG_ERROR, MAE, MAPE, MASE, MB, NMSE, NRMSE, RMSE, MedAE, sMAPE
from monet_stats.relative_metrics import NMB
from monet_stats.spatial_ensemble_metrics import CRPS, FSS
from monet_stats.utils_stats import mae, rmse


class TestErrorMetrics:
    """Test error metrics functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
        return obs, mod

    def test_MAE(self, sample_data):
        """Test Mean Absolute Error."""
        obs, mod = sample_data
        result = MAE(obs, mod)
        expected = np.mean(np.abs(obs - mod))
        assert np.isclose(result, expected)

    def test_RMSE(self, sample_data):
        """Test Root Mean Square Error."""
        obs, mod = sample_data
        result = RMSE(obs, mod)
        expected = np.sqrt(np.mean((obs - mod) ** 2))
        assert np.isclose(result, expected)

    def test_MB(self, sample_data):
        """Test Mean Bias."""
        obs, mod = sample_data
        result = MB(obs, mod)
        expected = np.mean(obs - mod)
        assert np.isclose(result, expected)

    def test_NRMSE(self, sample_data):
        """Test Normalized Root Mean Square Error."""
        obs, mod = sample_data
        result = NRMSE(obs, mod)
        rmse_val = np.sqrt(np.mean((obs - mod) ** 2))
        obs_range = np.max(obs) - np.min(obs)
        expected = rmse_val / obs_range
        assert np.isclose(result, expected)

    def test_MAPE(self, sample_data):
        """Test Mean Absolute Percentage Error."""
        obs, mod = sample_data
        result = MAPE(obs, mod)
        expected = np.mean(np.abs((mod - obs) / obs)) * 100
        assert np.isclose(result, expected)

    def test_MASE(self, sample_data):
        """Test Mean Absolute Scaled Error."""
        obs, mod = sample_data
        result = MASE(obs, mod)
        # This is a simplified test - full implementation would involve naive forecast
        model_error = np.mean(np.abs(mod - obs))
        naive_diff = np.diff(obs)
        naive_error = np.mean(np.abs(naive_diff))
        expected = model_error / naive_error
        assert np.isclose(result, expected)

    def test_MedAE(self, sample_data):
        """Test Median Absolute Error."""
        obs, mod = sample_data
        result = MedAE(obs, mod)
        expected = np.median(np.abs(obs - mod))
        assert np.isclose(result, expected)

    def test_sMAPE(self, sample_data):
        """Test Symmetric Mean Absolute Percentage Error."""
        obs, mod = sample_data
        result = sMAPE(obs, mod)
        expected = np.mean(200 * np.abs(mod - obs) / (np.abs(mod) + np.abs(obs)))
        assert np.isclose(result, expected)

    def test_NMSE(self, sample_data):
        """Test Normalized Mean Square Error."""
        obs, mod = sample_data
        result = NMSE(obs, mod)
        mse = np.mean((mod - obs) ** 2)
        obs_var = np.var(obs)
        expected = mse / obs_var
        assert np.isclose(result, expected)

    def test_LOG_ERROR(self, sample_data):
        """Test Logarithmic Error Metric."""
        obs, mod = sample_data
        # Add small offset to ensure positive values
        obs_pos = obs + 0.1
        mod_pos = mod + 0.1
        result = LOG_ERROR(obs_pos, mod_pos)
        obs_log = np.log(obs_pos)
        mod_log = np.log(mod_pos)
        expected = np.sqrt(np.mean((mod_log - obs_log) ** 2))
        assert np.isclose(result, expected)


class TestCorrelationMetrics:
    """Test correlation metrics functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
        return obs, mod

    def test_R2(self, sample_data):
        """Test Coefficient of Determination."""
        obs, mod = sample_data
        result = R2(obs, mod)
        from scipy.stats import pearsonr

        r_val, _ = pearsonr(obs, mod)
        # Ensure r_val is a scalar
        if isinstance(r_val, tuple):
            r_val = r_val[0]
        elif hasattr(r_val, "__iter__"):
            r_val = list(r_val)[0]
        expected = float(r_val) ** 2
        assert np.isclose(result, expected, rtol=1e-10)

    def test_pearsonr(self, sample_data):
        """Test Pearson correlation coefficient."""
        obs, mod = sample_data
        result = stats_pearsonr(obs, mod)
        expected, _ = pearsonr(obs, mod)
        assert np.isclose(result, expected)

    def test_AC(self, sample_data):
        """Test Anomaly Correlation."""
        obs, mod = sample_data
        result = AC(obs, mod)
        # Calculate anomaly correlation manually
        obs_bar = np.mean(obs)
        mod_bar = np.mean(mod)
        obs_anom = obs - obs_bar
        mod_anom = mod - mod_bar
        numerator = np.sum(obs_anom * mod_anom)
        denom = np.sqrt(np.sum(obs_anom**2) * np.sum(mod_anom**2))
        expected = numerator / denom
        assert np.isclose(result, expected)


class TestContingencyMetrics:
    """Test contingency table metrics."""

    @pytest.fixture
    def contingency_data(self):
        """Create contingency table data for testing."""
        # Perfect detection case
        obs = np.array([1, 1, 0, 0])  # 2 events, 2 non-events
        mod = np.array([1, 1, 0, 0])  # Perfect forecast
        return obs, mod

    def test_POD(self, contingency_data):
        """Test Probability of Detection."""
        obs, mod = contingency_data
        result = POD(obs, mod, 0.5)  # Threshold
        # With our test data, this should be 1.0 (perfect detection)
        assert np.isclose(result, 1.0)

    def test_FAR(self, contingency_data):
        """Test False Alarm Rate."""
        obs, mod = contingency_data
        result = FAR(obs, mod, 0.5)  # Threshold
        # With perfect forecast, FAR should be 0
        assert np.isclose(result, 0.0)


class TestEfficiencyMetrics:
    """Test efficiency metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
        return obs, mod

    def test_NSE(self, sample_data):
        """Test Nash-Sutcliffe Efficiency."""
        obs, mod = sample_data
        result = eff_NSE(obs, mod)
        numerator = np.sum((obs - mod) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)
        expected = 1 - (numerator / denominator)
        assert np.isclose(result, expected)


class TestRelativeMetrics:
    """Test relative metrics."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
        return obs, mod

    def test_NMB(self, sample_data):
        """Test Normalized Mean Bias."""
        obs, mod = sample_data
        result = NMB(obs, mod)
        expected = (np.sum(mod - obs) / np.sum(obs)) * 100.0
        assert np.isclose(result, expected)


class TestUtilsStats:
    """Test utility functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod = np.array([1.1, 1.9, 3.1, 3.9, 5.2])
        return obs, mod

    def test_rmse(self, sample_data):
        """Test RMSE utility function."""
        obs, mod = sample_data
        result = rmse(obs, mod)
        expected = np.sqrt(np.mean((obs - mod) ** 2))
        assert np.isclose(result, expected)

    def test_mae(self, sample_data):
        """Test MAE utility function."""
        obs, mod = sample_data
        result = mae(obs, mod)
        expected = np.mean(np.abs(obs - mod))
        assert np.isclose(result, expected)


class TestSpatialEnsembleMetrics:
    """Test spatial and ensemble metrics."""

    def test_FSS(self):
        """Test Fractions Skill Score."""
        # Simple 2D arrays for testing
        obs = np.array([[1, 0], [0, 1]], dtype=float)
        mod = np.array([[1, 0], [0, 1]], dtype=float)
        result = FSS(obs, mod, window=1, threshold=0.5)
        # Perfect match should give high FSS
        assert 0 <= result <= 1

    def test_CRPS(self):
        """Test Continuous Ranked Probability Score."""
        # Ensemble with 3 members
        ensemble = np.array([[1, 2], [2, 3], [3, 4]], dtype=float)
        obs = np.array([2, 3], dtype=float)
        result = CRPS(ensemble, obs)
        # CRPS should be non-negative
        assert np.all(result >= 0)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__])
