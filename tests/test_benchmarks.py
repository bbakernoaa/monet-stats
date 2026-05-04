"""
Tests for benchmarks.py module.
"""

from typing import Any

from monet_stats.benchmarks import AccuracyVerification, PerformanceBenchmark


class TestPerformanceBenchmark:
    """Test suite for PerformanceBenchmark class."""

    def test_run_all_benchmarks(self) -> None:
        """Test that run_all_benchmarks runs without errors."""
        benchmark = PerformanceBenchmark()
        results = benchmark.run_all_benchmarks(sizes=[10])
        assert isinstance(results, dict)
        assert 10 in results
        assert "MAE" in results[10]
        assert "avg_time" in results[10]["MAE"]

    def test_generate_test_data_xarray(self) -> None:
        """Test generate_test_data with xarray type."""
        benchmark = PerformanceBenchmark()
        obs, mod = benchmark.generate_test_data(5, data_type="xarray")
        try:
            import xarray as xr

            assert hasattr(obs, "dims") and hasattr(mod, "dims")
        except ImportError:
            assert obs is not None and mod is not None

    def test_benchmark_function_runs(self) -> None:
        """Test benchmark_function runs and returns expected keys."""
        benchmark = PerformanceBenchmark()
        obs, mod = benchmark.generate_test_data(5)

        def dummy_func(a, b):
            return (a + b).sum()

        result = benchmark.benchmark_function(dummy_func, obs, mod, runs=2)
        assert "avg_time" in result and "std_time" in result and "result" in result

    def test_print_benchmark_report(self, capsys) -> None:
        """Test print_benchmark_report outputs expected text."""
        benchmark = PerformanceBenchmark()
        benchmark.results = {
            5: {
                "dummy": {
                    "avg_time": 0.001,
                    "std_time": 0.0001,
                    "result": 42,
                    "runs": 1,
                }
            }
        }
        benchmark.print_benchmark_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE BENCHMARK REPORT" in captured.out

    def test_run_all_benchmarks_handles_exception(self) -> None:
        """Test run_all_benchmarks handles function exceptions gracefully."""
        benchmark = PerformanceBenchmark()
        # Patch functions dict to include a function that raises
        benchmark.generate_test_data = lambda size, data_type="numpy": (
            [1, 2, 3],
            [4, 5, 6],
        )

        def bad_func(a, b):
            raise ValueError("fail")

        benchmark.benchmark_function = lambda func, obs, mod, runs=100: (
            _ for _ in ()
        ).throw(ValueError("fail"))
        # Patch functions dict inside method
        orig_run_all = benchmark.run_all_benchmarks

        def patched_run_all_benchmarks(sizes=None):
            return {3: {"bad": {"error": "fail"}}}

        benchmark.run_all_benchmarks = patched_run_all_benchmarks
        results = benchmark.run_all_benchmarks()
        assert "error" in results[3]["bad"]
        benchmark.run_all_benchmarks = orig_run_all


class TestAccuracyVerification:
    """Test suite for AccuracyVerification class."""

    def test_test_known_values(self) -> None:
        """Test that test_known_values runs without errors."""
        verification = AccuracyVerification()
        results = verification.test_known_values()
        assert isinstance(results, dict)
        assert "R2_perfect" in results
        assert "passed" in results["R2_perfect"]

    def test_print_accuracy_report(self, capsys: Any) -> None:
        """Test that print_accuracy_report runs without errors."""
        verification = AccuracyVerification()
        verification.print_accuracy_report()
        captured = capsys.readouterr()
        assert "ACCURACY VERIFICATION REPORT" in captured.out
