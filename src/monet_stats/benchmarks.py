"""
Performance benchmarks and accuracy verification for statistical functions (Aero Protocol Compliant).

This module provides tools to benchmark the execution time of various metrics
and verify their mathematical accuracy against known values.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    Performance benchmarking suite for statistical functions.

    This class enables timing analysis of metrics across different backends
    (NumPy, Xarray, Dask) and data sizes.
    """

    def __init__(self) -> None:
        """Initialize the benchmark suite with an empty results dictionary."""
        self.results: Dict[int, Dict[str, Any]] = {}

    def generate_test_data(
        self, size: int, data_type: str = "numpy", chunks: Optional[Dict[str, int]] = None
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Generate synthetic test data for benchmarking.

        Parameters
        ----------
        size : int
            Number of data points to generate.
        data_type : str, optional
            Type of data to generate ('numpy' or 'xarray'). Default is 'numpy'.
        chunks : Optional[Dict[str, int]], optional
            Dask chunk sizes if data_type is 'xarray' and lazy evaluation is desired.

        Returns
        -------
        Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]
            A tuple containing (obs, mod) arrays.
        """
        np.random.seed(42)  # For reproducible results
        obs_raw = np.random.normal(10, 3, size)
        mod_raw = obs_raw + np.random.normal(0, 1, size)  # Add some error

        if data_type == "xarray":
            obs = xr.DataArray(obs_raw, dims=["time"], coords={"time": np.arange(size)}, name="obs")
            mod = xr.DataArray(mod_raw, dims=["time"], coords={"time": np.arange(size)}, name="mod")
            if chunks:
                obs = obs.chunk(chunks)
                mod = mod.chunk(chunks)
            return obs, mod

        return obs_raw, mod_raw

    def benchmark_function(
        self,
        func: Callable,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        runs: int = 100,
    ) -> Dict[str, Any]:
        """
        Benchmark a single statistical function.

        .. note::
           For Dask-backed arrays, this function explicitly calls `.compute()`
           to measure the full execution time of the calculation.

        Parameters
        ----------
        func : Callable
            The function to benchmark.
        obs : Union[np.ndarray, xr.DataArray]
            Observed values.
        mod : Union[np.ndarray, xr.DataArray]
            Model values.
        runs : int, optional
            Number of iterations for averaging. Default is 100.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'avg_time', 'std_time', 'result', and 'runs'.
        """
        times: List[float] = []
        results: List[Any] = []

        # Warm-up run (especially for Dask/JIT if applicable)
        _ = func(obs, mod)

        for _ in range(runs):
            start_time = time.perf_counter()
            result = func(obs, mod)
            # If dask-backed, we might want to trigger compute to measure execution time
            # however, the protocol says "No Hidden Computes".
            # For benchmarking, we might explicitly compute if we want to measure wall time of the calculation.
            if hasattr(result, "compute"):
                result = result.compute()

            end_time = time.perf_counter()

            times.append(end_time - start_time)
            results.append(result)

        avg_time = float(np.mean(times))
        std_time = float(np.std(times))

        return {
            "avg_time": avg_time,
            "std_time": std_time,
            "result": results[0],
            "runs": runs,
        }

    def run_all_benchmarks(self, sizes: Optional[List[int]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Run benchmarks for a standard set of functions across multiple sizes.

        Parameters
        ----------
        sizes : Optional[List[int]], optional
            List of data sizes to test. Defaults to [100, 1000, 10000].

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Comprehensive benchmark results indexed by size.
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

        results: Dict[int, Dict[str, Any]] = {}

        for size in sizes:
            print(f"Benchmarking with data size: {size:,}")
            obs, mod = self.generate_test_data(size)

            size_results: Dict[str, Any] = {}
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

    def print_benchmark_report(self) -> None:
        """Print a formatted performance report to the console."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        for size, size_results in self.results.items():
            print(f"\nData Size: {size:,} elements")
            print("-" * 40)

            # Sort by average time
            sorted_results = sorted(
                size_results.items(),
                key=lambda x: x[1].get("avg_time", float("inf")) if isinstance(x[1], dict) else float("inf"),
            )

            for name, result in sorted_results:
                if isinstance(result, dict) and "error" not in result:
                    avg_time = result["avg_time"]
                    std_time = result["std_time"]
                    print(f"{name:<20}: {avg_time*1000:>8.4f}±{std_time*1000:.4f} ms")
                elif isinstance(result, dict):
                    print(f"{name:<20}: ERROR - {result.get('error')}")
                else:
                    print(f"{name:<20}: ERROR - Unknown result format")


class AccuracyVerification:
    """
    Suite for verifying the mathematical correctness of statistical functions.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        """
        Initialize the verification suite.

        Parameters
        ----------
        tolerance : float, optional
            Numerical tolerance for floating-point comparisons. Default is 1e-10.
        """
        self.tolerance = tolerance

    def test_known_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Run a series of tests against analytically known values.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of test results including computed vs expected values.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Perfect correlation case
        obs_perfect = np.array([1, 2, 3, 4, 5])
        mod_perfect = np.array([1, 2, 3, 4, 5])

        # Test R2 for perfect correlation
        r2_result = R2(obs_perfect, mod_perfect)
        results["R2_perfect"] = {
            "computed": float(r2_result),
            "expected": 1.0,
            "passed": bool(np.isclose(r2_result, 1.0, atol=self.tolerance)),
        }

        # Test correlation for perfect correlation
        corr_result = correlation(obs_perfect, mod_perfect)
        results["correlation_perfect"] = {
            "computed": float(corr_result),
            "expected": 1.0,
            "passed": bool(np.isclose(corr_result, 1.0, atol=self.tolerance)),
        }

        # Test RMSE for perfect match
        rmse_result = RMSE(obs_perfect, mod_perfect)
        results["RMSE_perfect"] = {
            "computed": float(rmse_result),
            "expected": 0.0,
            "passed": bool(np.isclose(rmse_result, 0.0, atol=self.tolerance)),
        }

        # Test MAE for perfect match
        mae_result = MAE(obs_perfect, mod_perfect)
        results["MAE_perfect"] = {
            "computed": float(mae_result),
            "expected": 0.0,
            "passed": bool(np.isclose(mae_result, 0.0, atol=self.tolerance)),
        }

        # Test NSE for perfect match
        nse_result = NSE(obs_perfect, mod_perfect)
        results["NSE_perfect"] = {
            "computed": float(nse_result),
            "expected": 1.0,
            "passed": bool(np.isclose(nse_result, 1.0, atol=self.tolerance)),
        }

        # Test with known bias
        obs_bias = np.ones(10)
        mod_bias = np.ones(10) * 2  # 100% bias
        mb_result = MB(obs_bias, mod_bias)
        # MB = mean(mod - obs) = (2 - 1) = 1.0
        expected_mb = 1.0
        results["MB_bias"] = {
            "computed": float(mb_result),
            "expected": expected_mb,
            "passed": bool(np.isclose(mb_result, expected_mb, atol=self.tolerance)),
        }

        # Test with known MAPE
        obs_mape = np.array([10, 10, 10])
        mod_mape = np.array([11, 9, 10])  # 10%, -10%, 0% errors
        mape_result = MAPE(obs_mape, mod_mape)
        expected_mape = (10 + 10 + 0) / 3  # Average absolute percentage error
        results["MAPE_known"] = {
            "computed": float(mape_result),
            "expected": expected_mape,
            "passed": bool(np.isclose(mape_result, expected_mape, atol=0.1)),
        }

        return results

    def print_accuracy_report(self) -> None:
        """Print a formatted accuracy report to the console."""
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


def run_comprehensive_benchmarks() -> None:
    """
    Execute both performance and accuracy suites.
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
