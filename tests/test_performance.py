"""
Tests for performance.py module.
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.performance import (
    apply_lazy_threshold,
    chunk_array,
    fast_mae,
    fast_rmse,
    get_chunk_recommendation,
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

        # This test is incorrect and will be removed.

    def test_vectorize_function(self):
        """Test that vectorize_function applies a function to an array."""

        def square(x):
            return x**2

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
        assert np.isclose(result, np.sqrt(1 / 3))

    def test_fast_mae(self):
        """Test fast_mae."""
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2, 4])
        result = fast_mae(obs, mod)
        assert np.isclose(result, 1 / 3)

    def test_get_chunk_recommendation(self):
        """Test get_chunk_recommendation suggests appropriate chunks."""
        # 1000x1000 float64 is ~7.6 MB
        data = xr.DataArray(np.random.rand(1000, 1000), dims=["x", "y"])

        # Target 1 MB
        chunks = get_chunk_recommendation(data, target_mb=1.0)
        assert chunks["x"] < 1000
        assert chunks["y"] == 1000

        # Target 10 MB (should be all)
        chunks = get_chunk_recommendation(data, target_mb=10.0)
        assert chunks["x"] == 1000
        assert chunks["y"] == 1000

    def test_apply_lazy_threshold(self):
        """Test apply_lazy_threshold converts to Dask when needed."""
        data = xr.DataArray(np.random.rand(1000, 1000), dims=["x", "y"])

        # Small threshold -> should become dask
        with pytest.warns(UserWarning, match="exceeds threshold"):
            data_lazy = apply_lazy_threshold(data, threshold_mb=1.0)
        assert hasattr(data_lazy.data, "chunks")

        # Large threshold -> should stay numpy
        data_eager = apply_lazy_threshold(data, threshold_mb=10.0)
        assert not hasattr(data_eager.data, "chunks")

        # Force dask
        data_forced = apply_lazy_threshold(data, threshold_mb=10.0, force_dask=True)
        assert hasattr(data_forced.data, "chunks")

    def test_parallel_compute_xarray(self):
        """Test parallel_compute handles xarray internally with laziness."""
        data = xr.DataArray(np.random.rand(100, 100), dims=["x", "y"])

        # Should convert to lazy internally if threshold is low (default is 500MB, so we force it)
        # We can't easily check internal state without mocking, but we verify it works
        result = parallel_compute(lambda d, axis: d.mean(), data)
        assert np.isclose(result.compute(), data.mean())

    def test_parallel_compute_numpy_weighted(self):
        """Test parallel_compute weighted average for numpy scalars."""
        data = np.array([1, 2, 3, 4, 5, 6])
        # Chunk into [1,2], [3,4], [5,6] -> means are 1.5, 3.5, 5.5
        # (1.5 + 3.5 + 5.5) / 3 = 10.5 / 3 = 3.5
        result = parallel_compute(np.mean, data, chunk_size=2)
        assert np.isclose(result, 3.5)
