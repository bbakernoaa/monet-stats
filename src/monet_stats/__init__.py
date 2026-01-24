"""
Statistics submodule for MONET utility functions.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

# Explicit imports for all public API symbols (for lint compliance)
from .contingency_metrics import CSI, ETS, FAR, FBI, HSS, POD, TSS, scores
from .correlation_metrics import (
    AC,
    CCC,
    E1,
    IOA,
    KGE,
    R2,
    RMSE,
    WDAC,
    WDIOA,
    WDRMSE,
    IOA_m,
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
from .efficiency_metrics import MAPE, MASE, MSE, NSE, PC, NSElog, NSEm, mNSE, rNSE
from .error_metrics import (
    MAE,
    MB,
    MNB,
    MNE,
    MO,
    NOP,
    NP,
    NRMSE,
    STDO,
    STDP,
    WDMB,
    MAE_m,
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
    WDMB_m,
    WDMdnB,
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

__all__ = [
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
    "RMSE",
    "WDRMSE_m",
    "WDRMSE",
    "RMSEs",
    "RMSEu",
    "d1",
    "E1",
    "IOA_m",
    "IOA",
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
    "RMSE_m",
    "IOA_m",
    "NSE_alpha",
    "NSE_beta",
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
]


def stats(
    data: Union[pd.DataFrame, xr.Dataset],
    obs_name: str = "Obs",
    mod_name: str = "Mod",
    threshold: float = 0.0,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate summary statistics for observations and model results.

    Supports both pandas DataFrames and xarray Datasets.

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

    Returns
    -------
    Dict[str, Any]
        Dictionary of calculated statistics.
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
        obs = data[obs_name]
        mod = data[mod_name]
        res: Dict[str, Any] = {}
        res["N"] = obs.dropna().count()
        res["Obs"] = obs.mean()
        res["Mod"] = mod.mean()
        res["MB"] = MB(obs.values, mod.values)
        res["R"] = pearsonr(obs.values, mod.values)
        res["IOA"] = IOA(obs.values, mod.values)
        res["RMSE"] = RMSE(obs.values, mod.values)
        res["NMB"] = NMB(obs.values, mod.values)

        try:
            a, b, c, d = scores(obs.values, mod.values, threshold)
            res["POD"] = a / (a + b) if (a + b) > 0 else 0.0
            res["FAR"] = c / (a + c) if (a + c) > 0 else 0.0
        except Exception:
            res["POD"] = np.nan
            res["FAR"] = np.nan
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
            "R": pearsonr(obs, mod),
            "IOA": IOA(obs, mod),
            "RMSE": RMSE(obs, mod),
            "NMB": NMB(obs, mod),
        }

        # Contingency scores
        try:
            a_l, b_l, c_l, d_l = scores(obs, mod, threshold)
            metrics_lazy["a"] = a_l
            metrics_lazy["b"] = b_l
            metrics_lazy["c"] = c_l
        except Exception:
            pass

        # Single optimized compute call
        computed = xr.compute(*metrics_lazy.values())
        results = dict(zip(metrics_lazy.keys(), [v.item() if hasattr(v, "item") else v for v in computed]))

        # Format final output
        res = {k: results[k] for k in ["N", "Obs", "Mod", "MB", "R", "IOA", "RMSE", "NMB"]}
        if "a" in results:
            a, b, c = results["a"], results["b"], results["c"]
            res["POD"] = a / (a + b) if (a + b) > 0 else 0.0
            res["FAR"] = c / (a + c) if (a + c) > 0 else 0.0
        else:
            res["POD"] = np.nan
            res["FAR"] = np.nan
        return res

    else:
        raise TypeError("data must be a pandas DataFrame or xarray Dataset")
