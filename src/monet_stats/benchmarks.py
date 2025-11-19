"""
Performance benchmarks and accuracy verification for statistical functions.
"""

import time
from typing import Callable, Dict, List, Tuple

import numpy as np
import xarray as xr

from .correlation_metrics import R2
from .correlation_metrics import pearsonr as stats_pearsonr
from .efficiency_metrics import NSE
from .error_metrics import MAE, MAPE, MASE, MB, RMSE, MedAE, sMAPE
from .relative_metrics import FB, FE, NMB
from .utils_stats import correlation, mae, rmse


class PerformanceBenchmark:
    """
    Class to run performance benchmarks for statistical functions.
    """

    def __init__(self):
        self.results = {}

    def generate_test_data(self, size: int, data_type: str = "numpy") -> Tuple:
        """
        Generate test data of specified size.

        Parameters
        ----------
        size : int
            Number of data points to generate.
        data_type : str
            Type of data to generate ('numpy', 'xarray', 'pandas').

        Returns
        -------
        tuple
            Generated obs and mod arrays.
        """
        np.random.seed(42)  # For reproducible results
        obs = np.random.normal(10, 3, size)
        mod = obs + np.random.normal(0, 1, size)  # Add some error

        if data_type == "xarray":
            obs = xr.DataArray(obs, dims=["time"], coords={"time": range(size)})
            mod = xr.DataArray(mod, dims=["time"], coords={"time": range(size)})

        return obs, mod

    def benchmark_function(
        self, func: Callable, obs: np.ndarray, mod: np.ndarray, runs: int = 100
    ) -> Dict:
        """
        Benchmark a single function.

        Parameters
        ----------
        func : callable
            Function to benchmark.
        obs : array-like
            Observed values.
        mod : array-like
            Model values.
        runs : int
            Number of runs for averaging.

        Returns
        -------
        dict
            Benchmark results including timing and results.
        """
        times = []
        results = []

        for _ in range(runs):
            start_time = time.perf_counter()
            result = func(obs, mod)
            end_time = time.perf_counter()

            times.append(end_time - start_time)
            results.append(result)

        avg_time = np.mean(times)
        std_time = np.std(times)

        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "result": results[0],  # Use first result as representative
            "runs": runs,
        }

    def run_all_benchmarks(self, sizes: List[int] = None) -> Dict:
        """
        Run benchmarks for all functions with different data sizes.

        Parameters
        ----------
        sizes : list of int
            List of data sizes to test.

        Returns
        -------
        dict
            Complete benchmark results.
        """
        if sizes is None:
            sizes = [100, 1000, 10000]
        functions = {
            "MAE": MAE,
            "RMSE": RMSE,
            "MB": MB,
            "R2": R2,
            "NSE": NSE,
            "MAPE": MAPE,
            "MASE": MASE,
            "MedAE": MedAE,
            "sMAPE": sMAPE,
            "NMB": NMB,
            "FB": FB,
            "FE": FE,
            "stats_pearsonr": stats_pearsonr,
            "rmse_util": rmse,
            "mae_util": mae,
            "corr_util": correlation,
        }

        results = {}

        for size in sizes:
            print(f"Benchmarking with data size: {size}")
            obs, mod = self.generate_test_data(size)

            size_results = {}
            for name, func in functions.items():
                try:
                    bench_result = self.benchmark_function(func, obs, mod)
                    size_results[name] = bench_result
                except Exception as e:
                    print(f"Error benchmarking {name}: {e!s}")
                    size_results[name] = {"error": str(e)}

            results[size] = size_results

        self.results = results
        return results

    def print_benchmark_report(self):
        """
        Print a formatted benchmark report.
        """
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        for size, size_results in self.results.items():
            print(f"\nData Size: {size:,} elements")
            print("-" * 40)

            # Sort by average time
            sorted_results = sorted(
                size_results.items(), key=lambda x: x[1].get("avg_time", float("inf"))
            )

            for name, result in sorted_results:
                if "error" not in result:
                    avg_time = result["avg_time"]
                    std_time = result["std_time"]
                    print(f"{name:<20}: {avg_time*1000:>8.4f}±{std_time*1000:.4f} ms")
                else:
                    print(f"{name:<20}: ERROR - {result['error']}")


class AccuracyVerification:
    """
    Class to verify mathematical accuracy of statistical functions.
    """

    def __init__(self):
        self.tolerance = 1e-10

    def test_known_values(self) -> Dict:
        """
        Test functions with known analytical values.

        Returns
        -------
        dict
            Results of accuracy tests.
        """
        results = {}

        # Perfect correlation case
        obs_perfect = np.array([1, 2, 3, 4, 5])
        mod_perfect = np.array([1, 2, 3, 4, 5])

        # Test R2 for perfect correlation
        r2_result = R2(obs_perfect, mod_perfect)
        results["R2_perfect"] = {
            "computed": r2_result,
            "expected": 1.0,
            "passed": np.isclose(r2_result, 1.0, atol=self.tolerance),
        }

        # Test correlation for perfect correlation
        corr_result = correlation(obs_perfect, mod_perfect)
        results["correlation_perfect"] = {
            "computed": corr_result,
            "expected": 1.0,
            "passed": np.isclose(corr_result, 1.0, atol=self.tolerance),
        }

        # Test RMSE for perfect match
        rmse_result = RMSE(obs_perfect, mod_perfect)
        results["RMSE_perfect"] = {
            "computed": rmse_result,
            "expected": 0.0,
            "passed": np.isclose(rmse_result, 0.0, atol=self.tolerance),
        }

        # Test MAE for perfect match
        mae_result = MAE(obs_perfect, mod_perfect)
        results["MAE_perfect"] = {
            "computed": mae_result,
            "expected": 0.0,
            "passed": np.isclose(mae_result, 0.0, atol=self.tolerance),
        }

        # Test NSE for perfect match
        nse_result = NSE(obs_perfect, mod_perfect)
        results["NSE_perfect"] = {
            "computed": nse_result,
            "expected": 1.0,
            "passed": np.isclose(nse_result, 1.0, atol=self.tolerance),
        }

        # Test with known bias
        obs_bias = np.ones(10)
        mod_bias = np.ones(10) * 2  # 100% bias
        mb_result = MB(obs_bias, mod_bias)
        expected_mb = 1.0  # (2-1)/1 = 1
        results["MB_bias"] = {
            "computed": mb_result,
            "expected": expected_mb,
            "passed": np.isclose(mb_result, expected_mb, atol=self.tolerance),
        }

        # Test with known MAPE
        obs_mape = np.array([10, 10, 10])
        mod_mape = np.array([11, 9, 10])  # 10%, -10%, 0% errors
        mape_result = MAPE(obs_mape, mod_mape)
        expected_mape = (10 + 10 + 0) / 3  # Average absolute percentage error
        results["MAPE_known"] = {
            "computed": mape_result,
            "expected": expected_mape,
            "passed": np.isclose(
                mape_result, expected_mape, atol=0.1
            ),  # Higher tolerance for MAPE
        }

        return results

    def print_accuracy_report(self):
        """
        Print a formatted accuracy verification report.
        """
        results = self.test_known_values()

        print("\n" + "=" * 80)
        print("ACCURACY VERIFICATION REPORT")
        print("=" * 80)

        passed = 0
        total = len(results)

        for test_name, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{test_name:<20}: {status:<4} | "
                f"Computed: {result['computed']:.6f}, "
                f"Expected: {result['expected']:.6f}"
            )
            if result["passed"]:
                passed += 1

        print(f"\nAccuracy Summary: {passed}/{total} tests passed")

        if passed == total:
            print("✓ All accuracy tests PASSED!")
        else:
            print("✗ Some accuracy tests FAILED!")


def run_comprehensive_benchmarks():
    """
    Run both performance benchmarks and accuracy verification.
    """
    print("Running comprehensive benchmarks and accuracy verification...")

    # Performance benchmark
    perf_bench = PerformanceBenchmark()
    perf_bench.run_all_benchmarks(sizes=[100, 1000, 10000])
    perf_bench.print_benchmark_report()

    # Accuracy verification
    acc_verify = AccuracyVerification()
    acc_verify.print_accuracy_report()


if __name__ == "__main__":
    run_comprehensive_benchmarks()
