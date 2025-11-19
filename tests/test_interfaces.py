"""
Tests for interfaces.py module.
"""
import numpy as np

from src.monet_stats.interfaces import (
    BaseStatisticalMetric,
    DataProcessor,
    PerformanceOptimizer,
)


class TestBaseStatisticalMetric(BaseStatisticalMetric):
    """Test implementation of BaseStatisticalMetric."""
    def compute(self, obs, mod, **kwargs):
        return np.mean(obs - mod)


class TestInterfaces:
    """Test suite for the interfaces module."""

    def test_base_statistical_metric(self):
        """Test the BaseStatisticalMetric class."""
        metric = TestBaseStatisticalMetric()
        obs = np.array([1, 2, 3])
        mod = np.array([2, 3, 4])
        assert metric.validate_inputs(obs, mod)
        assert np.isclose(metric.compute(obs, mod), -1.0)

    def test_data_processor(self):
        """Test the DataProcessor class."""
        # Test to_numpy
        assert isinstance(DataProcessor.to_numpy([1, 2, 3]), np.ndarray)
        # Test align_arrays
        obs_np = np.array([1, 2, 3])
        mod_np = np.array([4, 5, 6])
        obs_aligned, mod_aligned = DataProcessor.align_arrays(obs_np, mod_np)
        assert np.array_equal(obs_aligned, obs_np)
        # Test handle_missing_values
        obs_nan = np.array([1, 2, np.nan])
        mod_nan = np.array([4, np.nan, 6])
        obs_clean, mod_clean = DataProcessor.handle_missing_values(obs_nan, mod_nan)
        assert len(obs_clean) == 1

    def test_performance_optimizer(self):
        """Test the PerformanceOptimizer class."""
        # Test chunk_array
        arr = np.arange(10)
        chunks = PerformanceOptimizer.chunk_array(arr, chunk_size=3)
        assert len(chunks) == 4
        # Test vectorize_function
        def square(x):
            return x ** 2
        arr = np.array([1, 2, 3])
        result = PerformanceOptimizer.vectorize_function(square, arr)
        assert np.array_equal(result, np.array([1, 4, 9]))
