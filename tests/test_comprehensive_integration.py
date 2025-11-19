"""
Comprehensive integration tests for the complete Monet Stats system.
Tests module interactions, API consistency, and end-to-end workflows.
"""
from typing import Any
import numpy as np
import pytest
import xarray as xr
from pytest_benchmark.fixture import BenchmarkFixture

from monet_stats.test_aliases import (
    coefficient_of_determination,
    critical_success_index,
    false_alarm_rate,
    hit_rate,
    index_of_agreement,
    mean_absolute_error,
    mean_bias_error,
    pearson_correlation,
    root_mean_squared_error,
)
from tests.test_utils import TestDataGenerator


class TestModuleInteractions:
    """Test interactions between different modules and consistency of APIs."""

    def test_error_correlation_consistency(self) -> None:
        """Test that error and correlation metrics work together consistently."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=0.8)

        # Compute error metrics
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)

        # Compute correlation metrics
        pearson_r = pearson_correlation(obs, mod)
        r2 = coefficient_of_determination(obs, mod)

        # Consistency checks
        assert rmse >= mae, "RMSE should be >= MAE"
        assert pearson_r >= 0, "Correlation should be positive for correlated data"
        assert r2 >= 0, "R² should be non-negative"
        assert abs(pearson_r**2 - r2) < 1e-10, "R² should equal Pearson correlation squared"

    def test_spatial_error_consistency(self) -> None:
        """Test spatial data handling across modules."""
        data_gen = TestDataGenerator()
        obs_grid, mod_grid = data_gen.generate_spatial_data(shape=(100, 100))

        # Flatten for compatibility with current metrics
        obs_flat = obs_grid.flatten()
        mod_flat = mod_grid.flatten()

        # Compute metrics
        mae = mean_absolute_error(obs_flat, mod_flat)
        rmse = root_mean_squared_error(obs_flat, mod_flat)
        pearson_r = pearson_correlation(obs_flat, mod_flat)

        # Should produce reasonable results for spatial data
        assert mae > 0, "MAE should be positive for spatial data"
        assert rmse > 0, "RMSE should be positive for spatial data"
        assert -1 <= pearson_r <= 1, "Correlation should be in valid range"

    def test_xarray_integration(self) -> None:
        """Test integration with xarray DataArrays."""
        np.random.seed(42)

        # Create xarray DataArrays
        obs_da = xr.DataArray(
            np.random.normal(15, 5, (10, 20, 30)),
            coords={'time': range(10), 'lat': range(20), 'lon': range(30)},
            dims=['time', 'lat', 'lon'],
            name='temperature'
        )
        mod_da = obs_da + np.random.normal(0, 1, obs_da.shape)

        # Test that we can extract values and compute metrics
        obs_vals = obs_da.values.flatten()
        mod_vals = mod_da.values.flatten()

        mae = mean_absolute_error(obs_vals, mod_vals)
        rmse = root_mean_squared_error(obs_vals, mod_vals)

        assert mae > 0, "MAE should be positive"
        assert rmse > 0, "RMSE should be positive"


class TestDataFormatCompatibility:
    """Test compatibility with different data formats."""

    def test_numpy_array_compatibility(self) -> None:
        """Test that numpy arrays work correctly."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100)

        # Should work with numpy arrays
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)
        pearson_r = pearson_correlation(obs, mod)

        assert isinstance(mae, (float, np.number))
        assert isinstance(rmse, (float, np.number))
        assert isinstance(pearson_r, (float, np.number))

    def test_list_compatibility(self) -> None:
        """Test that Python lists work correctly."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100)

        # Convert to lists
        obs_list = obs.tolist()
        mod_list = mod.tolist()

        # Should work with lists
        mae = mean_absolute_error(obs_list, mod_list)
        rmse = root_mean_squared_error(obs_list, mod_list)
        pearson_r = pearson_correlation(obs_list, mod_list)

        assert isinstance(mae, (float, np.number))
        assert isinstance(rmse, (float, np.number))
        assert isinstance(pearson_r, (float, np.number))

    def test_mixed_array_types(self) -> None:
        """Test mixed array types (numpy array vs list)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100)

        # Mix types
        obs_list = obs.tolist()

        # Should handle mixed types gracefully
        mae = mean_absolute_error(obs_list, mod)
        rmse = root_mean_squared_error(obs_list, mod)
        pearson_r = pearson_correlation(obs_list, mod)

        assert isinstance(mae, (float, np.number))
        assert isinstance(rmse, (float, np.number))
        assert isinstance(pearson_r, (float, np.number))


class TestAPIConsistency:
    """Test API consistency across different metrics."""

    def test_parameter_consistency(self) -> None:
        """Test that similar metrics have consistent parameter signatures."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100)

        # All metrics should accept (obs, mod) as first two parameters
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)
        pearson_r = pearson_correlation(obs, mod)
        ioe = index_of_agreement(obs, mod)

        # Should all produce scalar results
        assert np.isscalar(mae), "MAE should be scalar"
        assert np.isscalar(rmse), "RMSE should be scalar"
        assert np.isscalar(pearson_r), "Pearson correlation should be scalar"
        assert np.isscalar(ioe), "IOA should be scalar"

    def test_return_type_consistency(self) -> None:
        """Test that metrics return consistent types."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100)

        metrics = [
            mean_absolute_error(obs, mod),
            root_mean_squared_error(obs, mod),
            pearson_correlation(obs, mod),
            coefficient_of_determination(obs, mod),
            index_of_agreement(obs, mod)
        ]

        # All should be numeric types
        for metric in metrics:
            assert isinstance(metric, (int, float, np.number)), f"Metric {metric} should be numeric"


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_climate_model_evaluation(self) -> None:
        """Test typical climate model evaluation workflow."""
        # Simulate monthly temperature data for 10 years at 100 grid points
        n_time = 120  # 10 years * 12 months
        n_grid = 100

        data_gen = TestDataGenerator()
        obs_data = np.random.normal(15, 10, (n_time, n_grid))  # Monthly temps
        mod_data = obs_data + np.random.normal(0, 1, (n_time, n_grid))  # Model bias

        # Flatten for current metrics
        obs_flat = obs_data.flatten()
        mod_flat = mod_data.flatten()

        # Compute comprehensive evaluation metrics
        mae = mean_absolute_error(obs_flat, mod_flat)
        rmse = root_mean_squared_error(obs_flat, mod_flat)
        bias = mean_bias_error(obs_flat, mod_flat)
        pearson_r = pearson_correlation(obs_flat, mod_flat)
        r2 = coefficient_of_determination(obs_flat, mod_flat)
        ioa = index_of_agreement(obs_flat, mod_flat)

        # Validate results
        assert mae > 0, "MAE should be positive"
        assert rmse > 0, "RMSE should be positive"
        assert -np.inf < bias < np.inf, "Bias should be finite"
        assert -1 <= pearson_r <= 1, "Correlation should be in valid range"
        assert 0 <= r2 <= 1, "R² should be in valid range"
        assert 0 <= ioa <= 1, "IOA should be in valid range"

    def test_weather_forecast_verification(self) -> None:
        """Test typical weather forecast verification workflow."""
        # Simulate 24-hour forecast for 30 days at 50 locations
        n_forecasts = 30
        n_hours = 24
        n_locations = 50

        data_gen = TestDataGenerator()
        obs_forecast = np.random.normal(15, 8, (n_forecasts, n_hours, n_locations))
        mod_forecast = obs_forecast + np.random.normal(0, 2, (n_forecasts, n_hours, n_locations))

        # Flatten for current metrics
        obs_flat = obs_forecast.flatten()
        mod_flat = mod_forecast.flatten()

        # Compute verification metrics
        mae = mean_absolute_error(obs_flat, mod_flat)
        rmse = root_mean_squared_error(obs_flat, mod_flat)

        # Test contingency metrics for threshold-based evaluation
        threshold = 20  # Temperature threshold
        obs_binary = (obs_flat > threshold).astype(int)
        mod_binary = (mod_flat > threshold).astype(int)

        hr = hit_rate(obs_binary, mod_binary)
        far = false_alarm_rate(obs_binary, mod_binary)
        csi = critical_success_index(obs_binary, mod_binary)

        # Validate results
        assert mae > 0, "MAE should be positive"
        assert rmse > 0, "RMSE should be positive"
        assert 0 <= hr <= 1, "Hit rate should be in [0, 1]"
        assert 0 <= far <= 1, "False alarm rate should be in [0, 1]"
        assert 0 <= csi <= 1, "CSI should be in [0, 1]"


class TestPerformanceCharacteristics:
    """Test performance characteristics of the system."""

    def test_small_dataset_performance(self, benchmark: BenchmarkFixture) -> None:
        """Test performance on small datasets."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=0.8)

        def run_comprehensive_metrics():
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            bias = mean_bias_error(obs, mod)
            pearson_r = pearson_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)
            ioa = index_of_agreement(obs, mod)
            return mae, rmse, bias, pearson_r, r2, ioa

        result = benchmark(run_comprehensive_metrics)
        assert result is not None

    def test_medium_dataset_performance(self, benchmark: BenchmarkFixture) -> None:
        """Test performance on medium datasets."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=10000, correlation=0.8)

        def run_comprehensive_metrics():
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            bias = mean_bias_error(obs, mod)
            pearson_r = pearson_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)
            ioa = index_of_agreement(obs, mod)
            return mae, rmse, bias, pearson_r, r2, ioa

        result = benchmark(run_comprehensive_metrics)
        assert result is not None

    def test_large_dataset_performance(self, benchmark: BenchmarkFixture) -> None:
        """Test performance on large datasets."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100000, correlation=0.8)

        def run_comprehensive_metrics():
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            bias = mean_bias_error(obs, mod)
            pearson_r = pearson_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)
            ioa = index_of_agreement(obs, mod)
            return mae, rmse, bias, pearson_r, r2, ioa

        result = benchmark(run_comprehensive_metrics)
        assert result is not None


class TestMathematicalCorrectness:
    """Test mathematical correctness of the implementation."""

    def test_perfect_correlation_properties(self) -> None:
        """Test properties with perfectly correlated data."""
        # Generate perfectly correlated data
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1  # Perfect linear relationship

        # Test correlation properties
        pearson_r = pearson_correlation(x, y)
        r2 = coefficient_of_determination(x, y)

        assert abs(pearson_r - 1.0) < 1e-10, f"Perfect linear data should have correlation 1.0, got {pearson_r}"
        assert abs(r2 - 1.0) < 1e-10, f"Perfect linear data should have R² = 1.0, got {r2}"

        # Error should be zero for perfect linear relationship
        mae = mean_absolute_error(x, y/2 - 0.5)  # y/2 - 0.5 = x
        assert abs(mae) < 1e-10, f"Perfect linear data should have zero error, got {mae}"

    def test_identity_properties(self) -> None:
        """Test properties when obs == mod."""
        data = np.random.normal(10, 2, 100)

        # When obs == mod, errors should be zero and correlations perfect
        mae = mean_absolute_error(data, data)
        rmse = root_mean_squared_error(data, data)
        bias = mean_bias_error(data, data)
        pearson_r = pearson_correlation(data, data)
        r2 = coefficient_of_determination(data, data)
        ioa = index_of_agreement(data, data)

        assert abs(mae) < 1e-10, "MAE should be zero when obs == mod"
        assert abs(rmse) < 1e-10, "RMSE should be zero when obs == mod"
        assert abs(bias) < 1e-10, "Bias should be zero when obs == mod"
        assert abs(pearson_r - 1.0) < 1e-10, "Correlation should be 1.0 when obs == mod"
        assert abs(r2 - 1.0) < 1e-10, "R² should be 1.0 when obs == mod"
        assert abs(ioa - 1.0) < 1e-10, "IOA should be 1.0 when obs == mod"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
