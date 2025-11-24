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
