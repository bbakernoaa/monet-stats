"""
Tests for performance.py module.
"""
import numpy as np

from src.monet_stats.performance import (
    chunk_array,
    fast_mae,
    fast_rmse,
    memory_efficient_correlation,
    optimize_for_size,
    parallel_compute,
    vectorize_function,
)


class TestPerformance:
    """Test suite for performance optimization functions."""

    def test_chunk_array(self):
        """Test that chunk_array splits an array into chunks."""
        arr = np.arange(10)
        chunks = chunk_array(arr, chunk_size=3)
        assert len(chunks) == 4
        assert chunks[0].shape == (3,)
        assert chunks[-1].shape == (1,)

    def test_vectorize_function(self):
        """Test that vectorize_function applies a function to an array."""
        def square(x):
            return x ** 2

        arr = np.array([1, 2, 3])
        result = vectorize_function(square, arr)
        assert np.array_equal(result, np.array([1, 4, 9]))

    def test_parallel_compute(self):
        """Test that parallel_compute chunks and computes a function."""
        data = np.arange(10)
        result = parallel_compute(np.mean, data, chunk_size=3)
        assert np.isclose(result, np.mean(data))

    def test_optimize_for_size(self):
        """Test that optimize_for_size calls the function."""
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2, 3])
        result = optimize_for_size(lambda o, m, axis: np.mean(o - m), obs, mod)
        assert np.isclose(result, 0)

    def test_memory_efficient_correlation(self):
        """Test memory_efficient_correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        result = memory_efficient_correlation(x, y)
        assert np.isclose(result, 1.0)

    def test_fast_rmse(self):
        """Test fast_rmse."""
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2, 4])
        result = fast_rmse(obs, mod)
        assert np.isclose(result, np.sqrt(1/3))

    def test_fast_mae(self):
        """Test fast_mae."""
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2, 4])
        result = fast_mae(obs, mod)
        assert np.isclose(result, 1/3)
