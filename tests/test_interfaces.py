"""
Tests for interfaces.py module.
"""

from typing import Any

import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

from monet_stats.interfaces import BaseStatisticalMetric, DataProcessor, PerformanceOptimizer


class TestBaseStatisticalMetric(BaseStatisticalMetric):
    """Test implementation of BaseStatisticalMetric."""

    def compute(self, obs: Any, mod: Any, **kwargs: Any) -> Any:
        if isinstance(obs, xr.DataArray):
            return self._handle_xarray(obs, mod, lambda o, m: (o - m).mean())
        return np.mean(obs - mod)


class TestInterfaces:
    """Test suite for the interfaces module."""

    def test_base_statistical_metric(self) -> None:
        """Test the BaseStatisticalMetric class."""
        metric = TestBaseStatisticalMetric()
        obs = np.array([1, 2, 3])
        mod = np.array([2, 3, 4])
        assert metric.validate_inputs(obs, mod)
        assert np.isclose(metric.compute(obs, mod), -1.0)

    def test_shape_mismatch(self) -> None:
        """Test that validate_inputs raises ValueError on shape mismatch."""
        metric = TestBaseStatisticalMetric()
        obs = np.array([1, 2, 3])
        mod = np.array([1, 2])
        with pytest.raises(ValueError, match="compatible shapes"):
            metric.validate_inputs(obs, mod)

    def test_data_processor(self) -> None:
        """Test the DataProcessor class."""
        # Test to_numpy
        assert isinstance(DataProcessor.to_numpy([1, 2, 3]), np.ndarray)
        # Test align_arrays
        obs_np = np.array([1, 2, 3])
        mod_np = np.array([1, 2, 3])
        obs_aligned, mod_aligned = DataProcessor.align_arrays(obs_np, mod_np)
        assert np.array_equal(obs_aligned, obs_np)
        # Test handle_missing_values
        obs_nan = np.array([1, 2, np.nan])
        mod_nan = np.array([4, np.nan, 6])
        obs_clean, mod_clean = DataProcessor.handle_missing_values(obs_nan, mod_nan)
        assert len(obs_clean) == 1

    def test_performance_optimizer(self) -> None:
        """Test the PerformanceOptimizer class."""
        # Test chunk_array
        arr = np.arange(10)
        chunks = PerformanceOptimizer.chunk_array(arr, chunk_size=3)
        assert len(chunks) == 4

        # Test vectorize_function
        def square(x: float) -> float:
            return x**2

        arr = np.array([1.0, 2.0, 3.0])
        result = PerformanceOptimizer.vectorize_function(square, arr)
        assert np.array_equal(result, np.array([1.0, 4.0, 9.0]))

    @pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
    def test_dask_laziness(self) -> None:
        """Test that validate_inputs does not trigger compute for Dask arrays."""
        metric = TestBaseStatisticalMetric()
        dask_arr = da.from_array(np.array([1, 2, 3]), chunks=3)
        obs = xr.DataArray(dask_arr, dims=["x"])
        mod = xr.DataArray(dask_arr, dims=["x"])

        # validate_inputs should return True without triggering compute
        assert metric.validate_inputs(obs, mod)

        # Check if it's still a dask array (lazy)
        assert hasattr(obs.data, "chunks")

    def test_history_update(self) -> None:
        """Test that _handle_xarray updates history."""
        metric = TestBaseStatisticalMetric()
        obs = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], attrs={"history": "initial"})
        mod = xr.DataArray([2.0, 3.0, 4.0], dims=["x"])

        result = metric.compute(obs, mod)
        assert "history" in result.attrs
        assert "Calculated TestBaseStatisticalMetric" in result.attrs["history"]
        assert "initial" in result.attrs["history"]
