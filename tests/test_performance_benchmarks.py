"""
Performance benchmarking tests using pytest-benchmark.
Tests various data sizes and optimization strategies.
"""
import numpy as np
import pytest
import xarray as xr
from pytest_benchmark.fixture import BenchmarkFixture

from monet_stats.test_aliases import (
    akaike_information_criterion,
    bayesian_information_criterion,
    coefficient_of_determination,
    critical_success_index,
    equitable_threat_score,
    false_alarm_rate,
    heidke_skill_score,
    hit_rate,
    index_of_agreement,
    mean_absolute_error,
    mean_bias_error,
    modified_index_of_agreement,
    normalized_mean_absolute_error,
    normalized_mean_bias_error,
    pearson_correlation,
    peirce_skill_score,
    root_mean_squared_error,
    spearman_correlation,
)
from tests.test_utils import TestDataGenerator


class TestPerformanceBenchmarks:
    """Performance benchmarks for statistical metrics."""

    def test_error_metrics_performance_small(self, benchmark: BenchmarkFixture):
        """Benchmark error metrics on small datasets (1K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=0.8)

        def run_metrics():
            mbe = mean_bias_error(obs, mod)
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            nmbe = normalized_mean_bias_error(obs, mod)
            nmae = normalized_mean_absolute_error(obs, mod)
            return mbe, mae, rmse, nmbe, nmae

        result = benchmark(run_metrics)
        assert result is not None

    def test_error_metrics_performance_medium(self, benchmark: BenchmarkFixture):
        """Benchmark error metrics on medium datasets (10K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=10000, correlation=0.8)

        def run_metrics():
            mbe = mean_bias_error(obs, mod)
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            nmbe = normalized_mean_bias_error(obs, mod)
            nmae = normalized_mean_absolute_error(obs, mod)
            return mbe, mae, rmse, nmbe, nmae

        result = benchmark(run_metrics)
        assert result is not None

    def test_error_metrics_performance_large(self, benchmark: BenchmarkFixture):
        """Benchmark error metrics on large datasets (100K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100000, correlation=0.8)

        def run_metrics():
            mbe = mean_bias_error(obs, mod)
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            nmbe = normalized_mean_bias_error(obs, mod)
            nmae = normalized_mean_absolute_error(obs, mod)
            return mbe, mae, rmse, nmbe, nmae

        result = benchmark(run_metrics)
        assert result is not None

    def test_correlation_metrics_performance_small(self, benchmark: BenchmarkFixture):
        """Benchmark correlation metrics on small datasets (1K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=0.8)

        def run_metrics():
            pearson_r = pearson_correlation(obs, mod)
            spearman_r = spearman_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)
            return pearson_r, spearman_r, r2

        result = benchmark(run_metrics)
        assert result is not None

    def test_correlation_metrics_performance_medium(self, benchmark: BenchmarkFixture):
        """Benchmark correlation metrics on medium datasets (10K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=10000, correlation=0.8)

        def run_metrics():
            pearson_r = pearson_correlation(obs, mod)
            spearman_r = spearman_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)
            return pearson_r, spearman_r, r2

        result = benchmark(run_metrics)
        assert result is not None

    def test_correlation_metrics_performance_large(self, benchmark: BenchmarkFixture):
        """Benchmark correlation metrics on large datasets (100K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100000, correlation=0.8)

        def run_metrics():
            pearson_r = pearson_correlation(obs, mod)
            spearman_r = spearman_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)
            return pearson_r, spearman_r, r2

        result = benchmark(run_metrics)
        assert result is not None

    def test_contingency_metrics_performance_small(self, benchmark: BenchmarkFixture):
        """Benchmark contingency metrics on small datasets (1K points)."""
        data_gen = TestDataGenerator()
        obs_binary = np.random.choice([0, 1], size=1000)
        mod_binary = np.random.choice([0, 1], size=1000)

        def run_metrics():
            hr = hit_rate(obs_binary, mod_binary)
            far = false_alarm_rate(obs_binary, mod_binary)
            csi = critical_success_index(obs_binary, mod_binary)
            ets = equitable_threat_score(obs_binary, mod_binary)
            pss = peirce_skill_score(obs_binary, mod_binary)
            hss = heidke_skill_score(obs_binary, mod_binary)
            return hr, far, csi, ets, pss, hss

        result = benchmark(run_metrics)
        assert result is not None

    def test_contingency_metrics_performance_large(self, benchmark: BenchmarkFixture):
        """Benchmark contingency metrics on large datasets (100K points)."""
        data_gen = TestDataGenerator()
        obs_binary = np.random.choice([0, 1], size=100000)
        mod_binary = np.random.choice([0, 1], size=100000)

        def run_metrics():
            hr = hit_rate(obs_binary, mod_binary)
            far = false_alarm_rate(obs_binary, mod_binary)
            csi = critical_success_index(obs_binary, mod_binary)
            ets = equitable_threat_score(obs_binary, mod_binary)
            pss = peirce_skill_score(obs_binary, mod_binary)
            hss = heidke_skill_score(obs_binary, mod_binary)
            return hr, far, csi, ets, pss, hss

        result = benchmark(run_metrics)
        assert result is not None

    def test_efficiency_metrics_performance_small(self, benchmark: BenchmarkFixture):
        """Benchmark efficiency metrics on small datasets (1K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=0.8)

        def run_metrics():
            ioa = index_of_agreement(obs, mod)
            mioa = modified_index_of_agreement(obs, mod)
            aic = akaike_information_criterion(obs, mod)
            bic = bayesian_information_criterion(obs, mod)
            return ioa, mioa, aic, bic

        result = benchmark(run_metrics)
        assert result is not None

    def test_efficiency_metrics_performance_large(self, benchmark: BenchmarkFixture):
        """Benchmark efficiency metrics on large datasets (100K points)."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100000, correlation=0.8)

        def run_metrics():
            ioa = index_of_agreement(obs, mod)
            mioa = modified_index_of_agreement(obs, mod)
            aic = akaike_information_criterion(obs, mod)
            bic = bayesian_information_criterion(obs, mod)
            return ioa, mioa, aic, bic

        result = benchmark(run_metrics)
        assert result is not None


class TestSpatialPerformance:
    """Performance tests for spatial data operations."""

    def test_spatial_error_metrics_performance(self, benchmark: BenchmarkFixture):
        """Benchmark spatial error metrics on 2D grids."""
        data_gen = TestDataGenerator()
        obs_grid, mod_grid = data_gen.generate_spatial_data(shape=(100, 100))

        def run_metrics():
            mae = mean_absolute_error(obs_grid.flatten(), mod_grid.flatten())
            rmse = root_mean_squared_error(obs_grid.flatten(), mod_grid.flatten())
            return mae, rmse

        result = benchmark(run_metrics)
        assert result is not None

    def test_large_spatial_grids(self, benchmark: BenchmarkFixture):
        """Benchmark performance on large spatial grids."""
        data_gen = TestDataGenerator()
        obs_grid, mod_grid = data_gen.generate_spatial_data(shape=(500, 500))

        def run_metrics():
            mae = mean_absolute_error(obs_grid.flatten(), mod_grid.flatten())
            rmse = root_mean_squared_error(obs_grid.flatten(), mod_grid.flatten())
            return mae, rmse

        result = benchmark(run_metrics)
        assert result is not None


class TestXarrayPerformance:
    """Performance tests for xarray DataArray operations."""

    @pytest.mark.xarray
    def test_xarray_error_metrics_performance_small(self, benchmark: BenchmarkFixture):
        """Benchmark xarray error metrics on small datasets."""
        np.random.seed(42)

        # Create xarray DataArrays
        obs_da = xr.DataArray(
            np.random.normal(15, 5, (10, 20, 30)),
            coords={'time': range(10), 'lat': range(20), 'lon': range(30)},
            dims=['time', 'lat', 'lon']
        )
        mod_da = obs_da + np.random.normal(0, 1, obs_da.shape)

        def run_metrics():
            # Flatten for compatibility with current metrics
            obs_flat = obs_da.values.flatten()
            mod_flat = mod_da.values.flatten()

            mae = mean_absolute_error(obs_flat, mod_flat)
            rmse = root_mean_squared_error(obs_flat, mod_flat)
            return mae, rmse

        result = benchmark(run_metrics)
        assert result is not None

    @pytest.mark.xarray
    def test_xarray_error_metrics_performance_medium(self, benchmark: BenchmarkFixture):
        """Benchmark xarray error metrics on medium datasets."""
        np.random.seed(42)

        # Create larger xarray DataArrays
        obs_da = xr.DataArray(
            np.random.normal(15, 5, (20, 50, 50)),
            coords={'time': range(20), 'lat': range(50), 'lon': range(50)},
            dims=['time', 'lat', 'lon']
        )
        mod_da = obs_da + np.random.normal(0, 1, obs_da.shape)

        def run_metrics():
            # Flatten for compatibility with current metrics
            obs_flat = obs_da.values.flatten()
            mod_flat = mod_da.values.flatten()

            mae = mean_absolute_error(obs_flat, mod_flat)
            rmse = root_mean_squared_error(obs_flat, mod_flat)
            return mae, rmse

        result = benchmark(run_metrics)
        assert result is not None


class TestAlgorithmOptimization:
    """Test different algorithm implementations for performance."""

    def test_vectorized_performance(self, benchmark: BenchmarkFixture):
        """Benchmark the vectorized MAE implementation."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=10000, correlation=0.8)

        def vectorized_mae():
            """Vectorized implementation."""
            return np.mean(np.abs(obs - mod))

        result = benchmark(vectorized_mae)
        assert result is not None

    def test_loop_performance(self, benchmark: BenchmarkFixture):
        """Benchmark the loop-based MAE implementation."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=10000, correlation=0.8)

        def loop_mae():
            """Loop-based implementation."""
            total = 0.0
            for i in range(len(obs)):
                total += abs(obs[i] - mod[i])
            return total / len(obs)

        result = benchmark(loop_mae)
        assert result is not None


class TestMemoryUsage:
    """Test memory usage patterns for different data sizes."""

    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_memory_scaling_error_metrics(self, benchmark: BenchmarkFixture, size: int):
        """Test memory scaling of error metrics."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=size, correlation=0.8)

        def run_with_size():
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            return mae, rmse

        benchmark(run_with_size)

    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_memory_scaling_correlation_metrics(self, benchmark: BenchmarkFixture, size: int):
        """Test memory scaling of correlation metrics."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=size, correlation=0.8)

        def run_with_size():
            pearson_r = pearson_correlation(obs, mod)
            spearman_r = spearman_correlation(obs, mod)
            return pearson_r, spearman_r

        benchmark(run_with_size)


class TestRealWorldScenarios:
    """Performance tests for realistic atmospheric data scenarios."""

    def test_climate_model_evaluation(self, benchmark: BenchmarkFixture):
        """Benchmark typical climate model evaluation scenario."""
        # Simulate monthly temperature data for 10 years at 100 grid points
        n_time = 120  # 10 years * 12 months
        n_grid = 100

        data_gen = TestDataGenerator()
        obs_data = np.random.normal(15, 10, (n_time, n_grid))  # Monthly temps
        mod_data = obs_data + np.random.normal(0, 1, (n_time, n_grid))  # Model bias

        def run_evaluation():
            # Flatten for current metrics
            obs_flat = obs_data.flatten()
            mod_flat = mod_data.flatten()

            # Compute common evaluation metrics
            mae = mean_absolute_error(obs_flat, mod_flat)
            rmse = root_mean_squared_error(obs_flat, mod_flat)
            pearson_r = pearson_correlation(obs_flat, mod_flat)
            r2 = coefficient_of_determination(obs_flat, mod_flat)

            return mae, rmse, pearson_r, r2

        result = benchmark(run_evaluation)
        assert result is not None

    def test_weather_forecast_verification(self, benchmark: BenchmarkFixture):
        """Benchmark typical weather forecast verification scenario."""
        # Simulate 24-hour forecast for 30 days at 50 locations
        n_forecasts = 30
        n_hours = 24
        n_locations = 50

        data_gen = TestDataGenerator()
        obs_forecast = np.random.normal(15, 8, (n_forecasts, n_hours, n_locations))
        mod_forecast = obs_forecast + np.random.normal(0, 2, (n_forecasts, n_hours, n_locations))

        def run_verification():
            # Flatten for current metrics
            obs_flat = obs_forecast.flatten()
            mod_flat = mod_forecast.flatten()

            # Compute verification metrics
            mae = mean_absolute_error(obs_flat, mod_flat)
            rmse = root_mean_squared_error(obs_flat, mod_flat)
            hit_rate_val = hit_rate(obs_flat > 20, mod_flat > 20)  # Precip threshold

            return mae, rmse, hit_rate_val

        result = benchmark(run_verification)
        assert result is not None


# Performance regression thresholds (in milliseconds)
PERFORMANCE_THRESHOLDS = {
    'error_metrics_small': 10.0,      # 1K data points
    'error_metrics_medium': 100.0,    # 10K data points
    'error_metrics_large': 1000.0,    # 100K data points
    'correlation_metrics_small': 20.0,
    'correlation_metrics_medium': 200.0,
    'correlation_metrics_large': 2000.0,
    'contingency_metrics_small': 5.0,
    'contingency_metrics_large': 50.0,
    'efficiency_metrics_small': 15.0,
    'efficiency_metrics_large': 150.0,
}


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-compare"])
