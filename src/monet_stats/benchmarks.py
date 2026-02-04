"""
Performance benchmarks and accuracy verification for statistical functions (Aero Protocol Compliant).

This module provides tools to benchmark the execution time of various metrics
and verify their mathematical accuracy against known values.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .correlation_metrics import KGE, R2
from .correlation_metrics import pearsonr as stats_pearsonr
from .efficiency_metrics import NSE
from .error_metrics import CRMSE, IOA, MAE, MAPE, MASE, MB, RMSE, MedAE, sMAPE
from .relative_metrics import FB, FE, NMB, NME
from .utils_stats import _update_history, correlation, mae, rmse


class PerformanceBenchmark:
    """
    Performance benchmarking suite for statistical functions.

    This class enables timing analysis of metrics across different backends
    (NumPy, Xarray, Dask) and data sizes.
    """

    def __init__(self) -> None:
        """Initialize the benchmark suite with an empty results dictionary."""
        self.results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    def generate_test_data(
        self, size: int, backend: str = "numpy", chunks: Optional[Dict[str, int]] = None
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Generate synthetic test data for benchmarking.

        Parameters
        ----------
        size : int
            Number of data points to generate.
        backend : str, optional
            Backend to use ('numpy', 'xarray', or 'dask'). Default is 'numpy'.
        chunks : Optional[Dict[str, int]], optional
            Dask chunk sizes if backend is 'dask'. If None, defaults to {'time': 12500000}
            (approx 100MB for float64).

        Returns
        -------
        Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]
            A tuple containing (obs, mod) arrays.
        """
        np.random.seed(42)  # For reproducible results
        obs_raw = np.random.normal(10, 3, size)
        mod_raw = obs_raw + np.random.normal(0, 1, size)  # Add some error

        if backend in ["xarray", "dask"]:
            obs = xr.DataArray(obs_raw, dims=["time"], coords={"time": np.arange(size)}, name="obs")
            mod = xr.DataArray(mod_raw, dims=["time"], coords={"time": np.arange(size)}, name="mod")

            _update_history(obs, "Generated benchmark data")
            _update_history(mod, "Generated benchmark data")

            if backend == "dask":
                if chunks is None:
                    # ~100MB chunk size for float64 (8 bytes per element)
                    chunks = {"time": 12500000}
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

    def run_all_benchmarks(
        self,
        sizes: Optional[List[int]] = None,
        backends: Optional[List[str]] = None,
    ) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Run benchmarks for a standard set of functions across multiple sizes and backends.

        Parameters
        ----------
        sizes : Optional[List[int]], optional
            List of data sizes to test. Defaults to [100, 1000, 10000].
        backends : Optional[List[str]], optional
            List of backends to test. Defaults to ['numpy', 'xarray', 'dask'].

        Returns
        -------
        Dict[str, Dict[int, Dict[str, Any]]]
            Comprehensive benchmark results indexed by backend, then size.
        """
        if sizes is None:
            sizes = [100, 1000, 10000]
        if backends is None:
            backends = ["numpy", "xarray", "dask"]

        functions = {
            "MAE": MAE,
            "RMSE": RMSE,
            "CRMSE": CRMSE,
            "MB": MB,
            "R2": R2,
            "NSE": NSE,
            "KGE": KGE,
            "IOA": IOA,
            "MAPE": MAPE,
            "MASE": MASE,
            "MedAE": MedAE,
            "sMAPE": sMAPE,
            "NMB": NMB,
            "NME": NME,
            "FB": FB,
            "FE": FE,
            "stats_pearsonr": stats_pearsonr,
            "rmse_util": rmse,
            "mae_util": mae,
            "corr_util": correlation,
        }

        results: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for backend in backends:
            print(f"Benchmarking backend: {backend}")
            backend_results: Dict[int, Dict[str, Any]] = {}
            for size in sizes:
                print(f"  Data size: {size:,}")
                obs, mod = self.generate_test_data(size, backend=backend)

                size_results: Dict[str, Any] = {}
                for name, func in functions.items():
                    try:
                        bench_result = self.benchmark_function(func, obs, mod)
                        size_results[name] = bench_result
                    except Exception as e:
                        print(f"    Error benchmarking {name}: {e!s}")
                        size_results[name] = {"error": str(e)}

                backend_results[size] = size_results
            results[backend] = backend_results

        self.results = results
        return results

    def print_benchmark_report(self) -> None:
        """Print a formatted performance report to the console."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)

        if not self.results:
            print("No benchmark results found. Run run_all_benchmarks() first.")
            return

        for backend, backend_results in self.results.items():
            print(f"\nBackend: {backend.upper()}")
            print("=" * 20)
            for size, size_results in backend_results.items():
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
                        print(f"{name:<20}: {avg_time * 1000:>8.4f}±{std_time * 1000:.4f} ms")
                    elif isinstance(result, dict):
                        print(f"{name:<20}: ERROR - {result.get('error')}")
                    else:
                        print(f"{name:<20}: ERROR - Unknown result format")

    def plot_performance(
        self,
        metrics: Optional[List[str]] = None,
        track: str = "A",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Visualize benchmarking results (Two-Track Rule).

        Parameters
        ----------
        metrics : Optional[List[str]], optional
            List of metrics to plot. If None, plots top metrics.
        track : str, optional
            'A' for static (Matplotlib) or 'B' for interactive (HvPlot).
            Default is 'A'.
        save_path : Optional[str], optional
            Path to save the static plot.

        Returns
        -------
        Any
            The plot object (Axes for Track A, HoloViews for Track B).
        """
        if not self.results:
            print("No results to plot. Run run_all_benchmarks() first.")
            return None

        # Prepare data for plotting
        data_list = []
        for backend, sizes_dict in self.results.items():
            for size, metrics_dict in sizes_dict.items():
                for metric, result in metrics_dict.items():
                    if isinstance(result, dict) and "avg_time" in result:
                        data_list.append(
                            {
                                "Backend": backend,
                                "Size": size,
                                "Metric": metric,
                                "Time (ms)": result["avg_time"] * 1000,
                            }
                        )

        if not data_list:
            print("No valid results found to plot.")
            return None

        import pandas as pd

        df = pd.DataFrame(data_list)

        if metrics:
            df = df[df["Metric"].isin(metrics)]
        else:
            # Default to a few common metrics if none specified
            common_metrics = ["MAE", "RMSE", "R2", "NSE", "KGE"]
            df = df[df["Metric"].isin(common_metrics)]

        if track.upper() == "A":
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("Matplotlib not found. Track A requires Matplotlib.")
                return None

            fig, ax = plt.subplots(figsize=(10, 6))
            for (metric, backend), group in df.groupby(["Metric", "Backend"]):
                label = f"{metric} ({backend})"
                ax.plot(group["Size"], group["Time (ms)"], marker="o", label=label)

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Data Size (number of elements)")
            ax.set_ylabel("Execution Time (ms)")
            ax.set_title("MONET-stats Performance Benchmark")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                print(f"Plot saved to {save_path}")

            return ax

        elif track.upper() == "B":
            try:
                import hvplot.pandas  # noqa: F401
            except ImportError:
                print("HvPlot not found. Track B requires HvPlot.")
                return None

            plot = df.hvplot.line(
                x="Size",
                y="Time (ms)",
                by=["Metric", "Backend"],
                logx=True,
                logy=True,
                title="MONET-stats Performance Benchmark",
                width=800,
                height=500,
            )
            return plot

        else:
            print(f"Unknown track: {track}. Use 'A' or 'B'.")
            return None


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

        # Test CRMSE
        crmse_result = CRMSE(obs_perfect, mod_perfect)
        results["CRMSE_perfect"] = {
            "computed": float(crmse_result),
            "expected": 0.0,
            "passed": bool(np.isclose(crmse_result, 0.0, atol=self.tolerance)),
        }

        # Test IOA
        ioa_result = IOA(obs_perfect, mod_perfect)
        results["IOA_perfect"] = {
            "computed": float(ioa_result),
            "expected": 1.0,
            "passed": bool(np.isclose(ioa_result, 1.0, atol=self.tolerance)),
        }

        # Test NMB
        nmb_result = NMB(obs_perfect, mod_perfect)
        results["NMB_perfect"] = {
            "computed": float(nmb_result),
            "expected": 0.0,
            "passed": bool(np.isclose(nmb_result, 0.0, atol=self.tolerance)),
        }

        # Test NME
        nme_result = NME(obs_perfect, mod_perfect)
        results["NME_perfect"] = {
            "computed": float(nme_result),
            "expected": 0.0,
            "passed": bool(np.isclose(nme_result, 0.0, atol=self.tolerance)),
        }

        # Test KGE
        kge_result = KGE(obs_perfect, mod_perfect)
        results["KGE_perfect"] = {
            "computed": float(kge_result),
            "expected": 1.0,
            "passed": bool(np.isclose(kge_result, 1.0, atol=self.tolerance)),
        }

        return results

    def cross_backend_verification(self, size: int = 1000) -> Dict[str, Dict[str, Any]]:
        """
        Verify consistency of metrics across different backends.

        Parameters
        ----------
        size : int, optional
            Size of test data. Default is 1000.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results of cross-backend verification.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Use the same data payload for both backends to ensure exact comparison
        obs_np, mod_np = PerformanceBenchmark().generate_test_data(size, backend="numpy")
        obs_xr = xr.DataArray(obs_np, dims=["time"], coords={"time": np.arange(size)}, name="obs")
        mod_xr = xr.DataArray(mod_np, dims=["time"], coords={"time": np.arange(size)}, name="mod")

        metrics = {
            "MAE": MAE,
            "RMSE": RMSE,
            "MB": MB,
            "R2": R2,
            "NSE": NSE,
            "KGE": KGE,
            "IOA": IOA,
            "NMB": NMB,
            "NME": NME,
        }

        for name, func in metrics.items():
            res_np = float(func(obs_np, mod_np))
            res_xr = float(func(obs_xr, mod_xr))

            passed = bool(np.isclose(res_np, res_xr, atol=self.tolerance))
            results[name] = {
                "numpy": res_np,
                "xarray": res_xr,
                "diff": abs(res_np - res_xr),
                "passed": passed,
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
                f"{test_name:<25}: {status:<4} | Computed: {result['computed']:.6f}, Expected: {result['expected']:.6f}"
            )
            if result["passed"]:
                passed += 1

        print(f"\nAccuracy Summary: {passed}/{total} tests passed")

        # Cross-backend verification
        print("\nCROSS-BACKEND CONSISTENCY (NumPy vs Xarray)")
        print("-" * 45)
        cross_results = self.cross_backend_verification()
        cross_passed = 0
        for name, res in cross_results.items():
            status = "PASS" if res["passed"] else "FAIL"
            print(f"{name:<10}: {status:<4} | Diff: {res['diff']:.2e}")
            if res["passed"]:
                cross_passed += 1
        print(f"Consistency Summary: {cross_passed}/{len(cross_results)} passed")

        if passed == total and cross_passed == len(cross_results):
            print("\n✓ All accuracy and consistency tests PASSED!")
        else:
            print("\n✗ Some tests FAILED!")


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
