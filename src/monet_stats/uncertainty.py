"""
Uncertainty quantification tools for model evaluation (Aero Protocol Compliant).
"""

from typing import Any, Callable, Dict, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history, ensure_single_chunk


def block_bootstrap(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    metric_func: Callable,
    n_boot: int = 1000,
    block_size: int = 1,
    dim: str = "time",
    axis: int = -1,
    confidence_level: float = 0.95,
    **metric_kwargs: Any,
) -> Union[Dict[str, float], xr.Dataset]:
    """
    Perform block-bootstrapping to calculate confidence intervals for a metric (Aero Protocol).

    Typical Use Cases
    -----------------
    - Calculating 95% confidence intervals for any performance metric (RMSE, Bias, etc.).
    - Accounting for temporal or spatial autocorrelation in the data.
    - Providing context for whether model improvements are statistically significant.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    metric_func : Callable
        The metric function to evaluate (e.g., MAE, RMSE).
    n_boot : int, optional
        Number of bootstrap iterations, by default 1000.
    block_size : int, optional
        Size of the blocks for bootstrapping, by default 1 (standard bootstrap).
    dim : str, optional
        Dimension along which to bootstrap (xarray only).
    axis : int, optional
        Axis along which to bootstrap (numpy only).
    confidence_level : float, optional
        Confidence level for the intervals, by default 0.95.
    **metric_kwargs : Any
        Additional keyword arguments passed to the metric function.

    Returns
    -------
    dict or xarray.Dataset
        Mean, lower bound, and upper bound of the bootstrapped metric.
    """

    def _bootstrap_numpy(o: np.ndarray, m: np.ndarray) -> np.ndarray:
        n = o.shape[axis]
        n_blocks = n // block_size
        if n_blocks == 0:
            return np.array([np.nan] * n_boot)

        # Pre-calculate block indices
        block_indices = [np.arange(i, i + block_size) for i in range(0, n - block_size + 1)]
        n_available_blocks = len(block_indices)

        boot_metrics = []
        for _ in range(n_boot):
            # Sample blocks with replacement
            idx = np.random.choice(n_available_blocks, n_blocks)
            sample_idx = np.concatenate([block_indices[i] for i in idx])

            # Slice along the bootstrap axis
            o_boot = np.take(o, sample_idx, axis=axis)
            m_boot = np.take(m, sample_idx, axis=axis)

            boot_metrics.append(metric_func(o_boot, m_boot, **metric_kwargs))

        return np.array(boot_metrics)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs = ensure_single_chunk(obs, dim)
        mod = ensure_single_chunk(mod, dim)

        res = xr.apply_ufunc(
            _bootstrap_numpy,
            obs,
            mod,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["bootstrap"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"bootstrap": n_boot}},
        )

        # Handle NaN values explicitly
        res = res.where(~xr.ufuncs.isnan(res), drop=True)

        # Calculate percentiles for confidence intervals
        alpha = (1 - confidence_level) / 2
        res_mean = res.mean(dim="bootstrap")
        res_low = res.quantile(alpha, dim="bootstrap").drop_vars("quantile", errors="ignore")
        res_high = res.quantile(1 - alpha, dim="bootstrap").drop_vars("quantile", errors="ignore")

        ds = xr.Dataset({"mean": res_mean, "lower": res_low, "upper": res_high})
        return _update_history(ds, f"Block-Bootstrap ({metric_func.__name__}, n={n_boot})")

    # NumPy path
    boot_samples = _bootstrap_numpy(np.asarray(obs), np.asarray(mod))
    alpha = (1 - confidence_level) / 2
    return {
        "mean": float(np.nanmean(boot_samples)),
        "lower": float(np.nanquantile(boot_samples, alpha)),
        "upper": float(np.nanquantile(boot_samples, 1 - alpha)),
    }
