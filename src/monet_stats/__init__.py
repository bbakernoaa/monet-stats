"""
Statistics submodule for MONET utility functions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

# Explicit imports for all public API symbols (for lint compliance)
from .analysis import (
    anomalies,
    climatology,
    detrend,
    diurnal_cycle,
    exceedance_count,
    fft_analysis,
    kz_filter,
    mda1,
    mda8,
    peak_timing,
    percentile,
    power_spectrum,
    resample_data,
    rolling_mean_8h,
    rolling_mean_24h,
    weighted_spatial_mean,
)
from .contingency_metrics import CSI, ETS, FAR, FBI, HSS, POD, TSS, scores
from .correlation_metrics import (
    AC,
    CCC,
    E1,
    KGE,
    R2,
    WDAC,
    WDIOA,
    WDRMSE,
    RMSEs,
    RMSEu,
    WDIOA_m,
    WDRMSE_m,
    d1,
    kendalltau,
    pearsonr,
    spearmanr,
    taylor_skill,
)
from .efficiency_metrics import MAPE, MASE, NSE, PC, NSElog, NSEm, mNSE, rNSE
from .error_metrics import (
    COE,
    CORR_INDEX,
    CRMSE,
    IOA,
    LOG_ERROR,
    MAE,
    MB,
    MNB,
    MNE,
    MO,
    MSE,
    NMSE,
    NOP,
    NP,
    NRMSE,
    RMSE,
    STDO,
    STDP,
    VOLUMETRIC_ERROR,
    WDMB,
    IOA_m,
    MAE_m,
    MAE_norm,
    MAPE_mod,
    MASE_mod,
    MdnB,
    MdnNB,
    MdnNE,
    MdnO,
    MdnP,
    MedAE,
    MedAE_m,
    NMdnGE,
    NSE_alpha,
    NSE_beta,
    RMdn,
    RMSE_m,
    RMSE_norm,
    WDMB_m,
    WDMdnB,
    bias_fraction,
)
from .performance import (
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
from .relative_metrics import FB, FE, MPE, NMB, NMB_ABS, NMdnB
from .spatial_ensemble_metrics import BSS, CRPS, EDS, SAL, ensemble_mean, ensemble_std, rank_histogram, spread_error
from .spatial_skill_metrics import FSS, VETS
from .utils_stats import (
    angular_difference,
    circlebias,
    circlebias_m,
    correlation,
    mae,
    matchedcompressed,
    matchmasks,
    rmse,
)
from .visualize import plot_spatial

__all__ = [
    # analysis
    "resample_data",
    "climatology",
    "anomalies",
    "detrend",
    "kz_filter",
    "diurnal_cycle",
    "rolling_mean_8h",
    "rolling_mean_24h",
    "mda1",
    "mda8",
    "exceedance_count",
    "percentile",
    "peak_timing",
    "weighted_spatial_mean",
    "fft_analysis",
    "power_spectrum",
    # contingency_metrics
    "CSI",
    "ETS",
    "FAR",
    "FBI",
    "HSS",
    "POD",
    "TSS",
    "scores",
    # correlation_metrics
    "R2",
    "WDRMSE_m",
    "WDRMSE",
    "RMSEs",
    "RMSEu",
    "d1",
    "E1",
    "WDIOA_m",
    "WDIOA",
    "AC",
    "WDAC",
    "taylor_skill",
    "KGE",
    "CCC",
    "pearsonr",
    "spearmanr",
    "kendalltau",
    # error_metrics
    "COE",
    "CORR_INDEX",
    "CRMSE",
    "LOG_ERROR",
    "NMSE",
    "STDO",
    "STDP",
    "MNB",
    "MNE",
    "MdnNB",
    "MdnNE",
    "NMdnGE",
    "NOP",
    "NP",
    "MO",
    "MdnO",
    "MdnP",
    "RMdn",
    "MB",
    "MdnB",
    "WDMB_m",
    "WDMB",
    "WDMdnB",
    "NRMSE",
    "MAE",
    "MAE_m",
    "MedAE",
    "MedAE_m",
    "RMSE",
    "RMSE_m",
    "IOA",
    "IOA_m",
    "NSE_alpha",
    "NSE_beta",
    "MAPE_mod",
    "MASE_mod",
    "RMSE_norm",
    "MAE_norm",
    "bias_fraction",
    "VOLUMETRIC_ERROR",
    # efficiency_metrics
    "NSE",
    "NSEm",
    "NSElog",
    "rNSE",
    "mNSE",
    "PC",
    "MSE",
    "MAPE",
    "MASE",
    # relative_metrics
    "NMB",
    "NMB_ABS",
    "NMdnB",
    "FB",
    "FE",
    "MPE",
    # spatial_ensemble_metrics
    "EDS",
    "CRPS",
    "spread_error",
    "SAL",
    "BSS",
    "ensemble_mean",
    "ensemble_std",
    "rank_histogram",
    # spatial_skill_metrics
    "FSS",
    "VETS",
    # utils_stats
    "matchedcompressed",
    "matchmasks",
    "circlebias_m",
    "circlebias",
    "angular_difference",
    "rmse",
    "mae",
    "correlation",
    # performance
    "get_chunk_recommendation",
    "apply_lazy_threshold",
    "chunk_array",
    "vectorize_function",
    "parallel_compute",
    "optimize_for_size",
    "memory_efficient_correlation",
    "fast_rmse",
    "fast_mae",
    # visualize
    "plot_spatial",
]

# Register xarray accessors
from . import accessor as accessor
from .plugin_system import plugin_manager


def stats(
    data: Union[pd.DataFrame, xr.Dataset],
    obs_name: str = "Obs",
    mod_name: str = "Mod",
    threshold: float = 0.0,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    plugins: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate summary statistics for observations and model results (Aero Protocol).

    Supports both pandas DataFrames and xarray Datasets. For xarray, it optimizes
    performance by bundling all computations into a single Dask graph evaluation.

    Parameters
    ----------
    data : pd.DataFrame or xr.Dataset
        Input data containing observations and model results.
    obs_name : str, optional
        Name of the observation column/variable, by default "Obs".
    mod_name : str, optional
        Name of the model column/variable, by default "Mod".
    threshold : float, optional
        Threshold for contingency scores (POD, FAR), by default 0.0.
    minval : float, optional
        Minimum value for filtering observations, by default None.
    maxval : float, optional
        Maximum value for filtering observations, by default None.
    plugins : List[str], optional
        List of registered plugin names to include in the statistics, by default None.

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated statistics:
        - N: Number of valid points
        - Obs: Mean of observations
        - Mod: Mean of model
        - MB: Mean Bias
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Square Error
        - R: Pearson Correlation
        - IOA: Index of Agreement
        - NMB: Normalized Mean Bias
        - MNB: Mean Normalized Bias
        - POD: Probability of Detection (at threshold)
        - FAR: False Alarm Rate (at threshold)
        - HSS: Heidke Skill Score (at threshold)
        - NSE: Nash-Sutcliffe Efficiency
        - CRMSE: Centered Root Mean Square Error
        - MdnB: Median Bias
        - KGE: Kling-Gupta Efficiency
        - R2: Coefficient of Determination
        - CCC: Concordance Correlation Coefficient
        - MNE: Mean Normalized Gross Error
        - NMSE: Normalized Mean Square Error

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from monet_stats import stats
    >>> df = pd.DataFrame({'Obs': [1, 2, 3], 'Mod': [1.1, 1.9, 3.2]})
    >>> results = stats(df)
    >>> print(results['MB'])
    0.06666666666666665
    """
    # Restore legacy parameters filtering logic
    if minval is not None:
        if isinstance(data, pd.DataFrame):
            data = data[data[obs_name] >= minval]
        else:
            data = data.where(data[obs_name] >= minval, drop=True)
    if maxval is not None:
        if isinstance(data, pd.DataFrame):
            data = data[data[obs_name] <= maxval]
        else:
            data = data.where(data[obs_name] <= maxval, drop=True)

    if isinstance(data, pd.DataFrame):
        obs_s = data[obs_name]
        mod_s = data[mod_name]
        obs = obs_s.values
        mod = mod_s.values

        res: Dict[str, Any] = {}
        res["N"] = obs_s.dropna().count()
        res["Obs"] = obs_s.mean()
        res["Mod"] = mod_s.mean()
        res["MB"] = MB(obs, mod)
        res["MAE"] = MAE(obs, mod)
        res["RMSE"] = RMSE(obs, mod)
        res["R"] = pearsonr(obs, mod)
        res["IOA"] = IOA(obs, mod)
        res["NMB"] = NMB(obs, mod)
        res["MNB"] = MNB(obs, mod)
        res["MNE"] = MNE(obs, mod)
        res["NSE"] = NSE(obs, mod)
        res["CRMSE"] = CRMSE(obs, mod)
        res["MdnB"] = MdnB(obs, mod)
        res["KGE"] = KGE(obs, mod)
        res["R2"] = R2(obs, mod)
        res["CCC"] = CCC(obs, mod)
        res["NMSE"] = NMSE(obs, mod)

        # Include plugins
        if plugins:
            for p_name in plugins:
                try:
                    res[p_name] = plugin_manager.compute_metric(p_name, obs, mod)
                except Exception:
                    res[p_name] = np.nan

        try:
            res["POD"] = POD(obs, mod, threshold)
            res["FAR"] = FAR(obs, mod, threshold)
            res["HSS"] = HSS(obs, mod, threshold)
        except Exception:
            res["POD"] = np.nan
            res["FAR"] = np.nan
            res["HSS"] = np.nan
        return res

    elif isinstance(data, xr.Dataset):
        obs = data[obs_name]
        mod = data[mod_name]

        # Gather all metrics that can be computed together to optimize dask graph
        metrics_lazy = {
            "N": obs.count(),
            "Obs": obs.mean(),
            "Mod": mod.mean(),
            "MB": MB(obs, mod),
            "MAE": MAE(obs, mod),
            "RMSE": RMSE(obs, mod),
            "R": pearsonr(obs, mod),
            "IOA": IOA(obs, mod),
            "NMB": NMB(obs, mod),
            "MNB": MNB(obs, mod),
            "MNE": MNE(obs, mod),
            "NSE": NSE(obs, mod),
            "CRMSE": CRMSE(obs, mod),
            "MdnB": MdnB(obs, mod),
            "KGE": KGE(obs, mod),
            "R2": R2(obs, mod),
            "CCC": CCC(obs, mod),
            "NMSE": NMSE(obs, mod),
        }

        # Include plugins (lazy evaluation)
        if plugins:
            for p_name in plugins:
                try:
                    metrics_lazy[p_name] = plugin_manager.compute_metric(p_name, obs, mod)
                except Exception:
                    metrics_lazy[p_name] = xr.DataArray(np.nan)

        # Contingency scores (optional if threshold is valid)
        try:
            metrics_lazy["POD"] = POD(obs, mod, threshold)
            metrics_lazy["FAR"] = FAR(obs, mod, threshold)
            metrics_lazy["HSS"] = HSS(obs, mod, threshold)
        except (ValueError, TypeError):
            # If thresholding fails during graph construction
            metrics_lazy["POD"] = xr.DataArray(np.nan)
            metrics_lazy["FAR"] = xr.DataArray(np.nan)
            metrics_lazy["HSS"] = xr.DataArray(np.nan)

        # Single optimized compute call using a dummy Dataset to bundle dask graph
        # This avoids a direct dependency on dask.base.compute
        ds_lazy = xr.Dataset({k: v for k, v in metrics_lazy.items() if isinstance(v, (xr.DataArray, xr.Dataset))})
        ds_computed = ds_lazy.compute()

        results = {}
        for k, v in metrics_lazy.items():
            if k in ds_computed:
                val = ds_computed[k].values
                results[k] = val.item() if hasattr(val, "item") else val
            else:
                # For non-xarray types (already computed or scalar)
                results[k] = v.item() if hasattr(v, "item") else v

        return results

    else:
        raise TypeError("data must be a pandas DataFrame or xarray Dataset")
