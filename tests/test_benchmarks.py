"""
Tests for benchmarks.py module.
"""

from typing import Any

import numpy as np
import xarray as xr

from monet_stats.benchmarks import AccuracyVerification, PerformanceBenchmark


class TestPerformanceBenchmark:
    """Test suite for PerformanceBenchmark class."""

    def test_run_all_benchmarks(self) -> None:
        """Test that run_all_benchmarks runs without errors."""
        benchmark = PerformanceBenchmark()
        results = benchmark.run_all_benchmarks(sizes=[10], backends=["numpy", "xarray"])
        assert isinstance(results, dict)
        assert "numpy" in results
        assert "xarray" in results
        assert 10 in results["numpy"]
        assert "MAE" in results["numpy"][10]
        assert "avg_time" in results["numpy"][10]["MAE"]

    def test_generate_test_data_backends(self) -> None:
        """Test that generate_test_data correctly handles different backends."""
        benchmark = PerformanceBenchmark()
        obs_np, _ = benchmark.generate_test_data(10, backend="numpy")
        assert isinstance(obs_np, np.ndarray)

        obs_xr, _ = benchmark.generate_test_data(10, backend="xarray")
        assert isinstance(obs_xr, xr.DataArray)
        assert not hasattr(obs_xr.data, "chunks")

        # Test dask backend if installed
        try:
            import dask.array as da  # noqa: F401

            obs_dask, _ = benchmark.generate_test_data(10, backend="dask")
            assert isinstance(obs_dask, xr.DataArray)
            assert hasattr(obs_dask.data, "chunks")
        except ImportError:
            import pytest

            pytest.skip("Dask not installed")

    def test_plot_performance(self) -> None:
        """Test that plot_performance runs without errors."""
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks(sizes=[10], backends=["numpy"])

        # Track A
        try:
            import matplotlib.pyplot as plt  # noqa: F401

            ax = benchmark.plot_performance(track="A")
            assert ax is not None
        except ImportError:
            pass  # Expected if matplotlib is not installed

        # Track B
        try:
            import hvplot.pandas  # noqa: F401

            plot = benchmark.plot_performance(track="B")
            assert plot is not None
        except ImportError:
            pass  # Expected if hvplot is not installed


class TestAccuracyVerification:
    """Test suite for AccuracyVerification class."""

    def test_test_known_values(self) -> None:
        """Test that test_known_values runs without errors."""
        verification = AccuracyVerification()
        results = verification.test_known_values()
        assert isinstance(results, dict)
        assert "R2_perfect" in results
        assert "KGE_perfect" in results
        assert "CRMSE_perfect" in results
        assert "passed" in results["R2_perfect"]

    def test_cross_backend_verification(self) -> None:
        """Test that cross_backend_verification correctly identifies consistency."""
        verification = AccuracyVerification()
        results = verification.cross_backend_verification(size=100)
        assert isinstance(results, dict)
        assert "MAE" in results
        assert "xarray" in results["MAE"]
        assert "numpy" in results["MAE"]
        assert results["MAE"]["passed"]

    def test_print_accuracy_report(self, capsys: Any) -> None:
        """Test that print_accuracy_report runs without errors."""
        verification = AccuracyVerification()
        verification.print_accuracy_report()
        captured = capsys.readouterr()
        assert "ACCURACY VERIFICATION REPORT" in captured.out
        assert "CROSS-BACKEND CONSISTENCY" in captured.out
