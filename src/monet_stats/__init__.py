"""
Statistics submodule for MONET utility functions.
"""

# Expose all functions from all stats submodules
# Dynamically build __all__ from all submodules

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
from .efficiency_metrics import (
    MAPE,
    MASE,
    MSE,
    NSE,
    PC,
    NSElog,
    NSEm,
    mNSE,
    rNSE,
)
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
from .relative_metrics import (
    FB,
    FE,
    MPE,
    NMB,
    NMB_ABS,
    NMdnB,
)
from .spatial_ensemble_metrics import (
    BSS,
    CRPS,
    EDS,
    FSS,
    SAL,
    ensemble_mean,
    ensemble_std,
    rank_histogram,
    spread_error,
)
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
    "FSS",
    "EDS",
    "CRPS",
    "spread_error",
    "SAL",
    "BSS",
    "ensemble_mean",
    "ensemble_std",
    "rank_histogram",
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


from typing import Any, Dict

import pandas as pd


def stats(df: pd.DataFrame, minval: Any, maxval: Any) -> Dict[str, float]:
    """Short summary.

    Parameters
    ----------
    df : pd.DataFrame
        Description of parameter `df`.
    minval : Any
        Description of parameter `minval`.
    maxval : Any
        Description of parameter `maxval`.

    Returns
    -------
    Dict[str, float]
        Description of returned object.

    """
    from numpy import sqrt

    dd: Dict[str, float] = {}
    dd["N"] = df.Obs.dropna().count()
    dd["Obs"] = df.Obs.mean()
    dd["Mod"] = df.CMAQ.mean()
    dd["MB"] = MB(df.Obs.values, df.CMAQ.values)  # mean bias
    dd["R"] = sqrt(R2(df.Obs.values, df.CMAQ.values))  # pearsonr ** 2
    dd["IOA"] = IOA(df.Obs.values, df.CMAQ.values)  # Index of Agreement
    dd["RMSE"] = RMSE(df.Obs.values, df.CMAQ.values)
    dd["NMB"] = NMB(df.Obs.values, df.CMAQ.values)
    try:
        a, b, c, d = scores(df.Obs.values, df.CMAQ.values, 70, 1000)
        dd["POD"] = a / (a + b)
        dd["FAR"] = c / (a + c)
    except Exception:
        dd["POD"] = 1.0
        dd["FAR"] = 0.0
    return dd
