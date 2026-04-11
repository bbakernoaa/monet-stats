"""
Relative/Percentage Metrics for Model Evaluation (Aero Protocol Compliant)
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr

from .error_metrics import MAE, MedAE
from .utils_stats import _resolve_axis_to_dim, _update_history, circlebias, circlebias_m, ensure_single_chunk

__all__ = [
    "NMB",
    "WDNMB_m",
    "NMB_ABS",
    "NMdnB",
    "FB",
    "ME",
    "MdnE",
    "WDME_m",
    "WDME",
    "WDMdnE",
    "NME_m",
    "NME_m_ABS",
    "NME",
    "NMdnE",
    "FE",
    "USUTPB",
    "USUTPE",
    "MNPB",
    "MdnNPB",
    "MNPE",
    "MdnNPE",
    "NMPB",
    "NMdnPB",
    "NMPE",
    "NMdnPE",
    "PSUTMNPB",
    "PSUTMdnNPB",
    "PSUTMNPE",
    "PSUTMdnNPE",
    "PSUTNMPB",
    "PSUTNMPE",
    "PSUTNMdnPB",
    "PSUTNMdnPE",
    "MPE",
    "MdnPE",
    "MG",
    "VG",
]


def NMB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Bias (%)

    Typical Use Cases
    -----------------
    - Comparing model bias across variables or datasets with different units or scales.
    - Common in regulatory and operational air quality model performance reports.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply. For NMB, this weights the sums in the numerator and denominator.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([1.1, 2.2, 3.3])
    >>> NMB(obs, mod)
    10.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        diff = mod - obs
        if weights is not None:
            res = (diff * weights).sum(dim=dim) / (obs * weights).sum(dim=dim) * 100.0
            return _update_history(res, "Weighted Normalized Mean Bias (NMB)")
        res = diff.sum(dim=dim) / obs.sum(dim=dim) * 100.0
        return _update_history(res, "Normalized Mean Bias (NMB)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        if obs_arr.size == 0:
            return np.nan
        diff_arr = mod_arr - obs_arr
        if weights is not None:
            res = np.nansum(diff_arr * weights, axis=axis) / np.nansum(obs_arr * weights, axis=axis) * 100.0
        else:
            res = np.nansum(diff_arr, axis=axis) / np.nansum(obs_arr, axis=axis) * 100.0
        # If denominator was zero or all NaN, return NaN instead of 0.0 or inf
        if np.ndim(res) == 0:
            return res.item() if np.isfinite(res) else np.nan
        return np.where(np.isfinite(res), res, np.nan)


def WDNMB_m(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Wind Direction Normalized Mean Bias (%) (avoid single block error in np.ma)

    Typical Use Cases
    -----------------
    - Comparing the average wind direction bias, normalized by observed wind direction, across sites or time periods.
    - Used in wind energy and meteorological model evaluation for directionally normalized performance.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed wind direction values (degrees).
    mod : xarray.DataArray or numpy.ndarray
        Model predicted wind direction values (degrees).
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Wind direction normalized mean bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([345, 15, 25])
    >>> WDNMB_m(obs, mod)
    -5.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        diff = mod - obs
        cb = circlebias_m(diff)
        res = cb.sum(dim=dim) / obs.sum(dim=dim) * 100.0
        return _update_history(res, "Wind Direction Normalized Mean Bias (WDNMB_m)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        diff = mod_arr - obs_arr
        res = circlebias_m(diff).sum(axis=axis) / obs_arr.sum(axis=axis) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def NMB_ABS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Bias - Absolute of the denominator (%)

    Typical Use Cases
    -----------------
    - Quantifying normalized mean bias when the denominator (sum of observations) may be negative or zero.
    - Used for robust model evaluation in cases with possible sign changes in the observed data sum.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean bias with absolute denominator (percent).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (mod - obs).sum(dim=dim) / abs(obs.sum(dim=dim)) * 100.0
        return _update_history(res, "Normalized Mean Bias Absolute (NMB_ABS)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        res = (mod_arr - obs_arr).sum(axis=axis) / np.abs(obs_arr.sum(axis=axis)) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def NMdnB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Median Bias (%)

    Typical Use Cases
    -----------------
    - Assessing the central tendency of normalized bias, robust to outliers and non-normal distributions.
    - Used for robust model evaluation across variables or sites with different scales.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized median bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4, 100])  # 100 is an outlier
    >>> mod = np.array([1.1, 2.2, 3.3, 4.4, 105])
    >>> NMdnB(obs, mod)
    10.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = mod - obs

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        diff = ensure_single_chunk(diff, dim)
        obs = ensure_single_chunk(obs, dim)

        res = (
            diff.quantile(q=0.5, dim=dim).drop_vars("quantile", errors="ignore")
            / obs.quantile(q=0.5, dim=dim).drop_vars("quantile", errors="ignore")
            * 100.0
        )
        return _update_history(res, "Normalized Median Bias (NMdnB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = np.ma.median(mod_arr - obs_arr, axis=axis) / np.ma.median(obs_arr, axis=axis) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def FB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Fractional Bias (%)

    Typical Use Cases
    -----------------
    - Quantifying the average bias as a fraction of the sum of model and observed values.
    - Used in air quality and meteorological model evaluation for normalized bias assessment.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Fractional bias (percent).

    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (((mod - obs) / (mod + obs)).mean(dim=dim) * 2.0) * 100.0
        return _update_history(res, "Fractional Bias (FB)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        res = (np.ma.masked_invalid((mod_arr - obs_arr) / (mod_arr + obs_arr)).mean(axis=axis) * 2.0) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def ME(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Mean Gross Error (model and obs unit). Alias for MAE.
    """
    res = MAE(obs, mod, axis=axis)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "Mean Gross Error (ME)")
    return res


def MdnE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Median Gross Error (model and obs unit). Alias for MedAE.
    """
    res = MedAE(obs, mod, axis=axis)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "Median Gross Error (MdnE)")
    return res


def WDME_m(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Wind Direction Mean Gross Error (model and obs unit)
    (avoid single block error in np.ma)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of wind direction errors, regardless of direction.
    - Used in wind energy, meteorology, and air quality studies to assess wind direction model performance.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed wind direction values (degrees).
    mod : xarray.DataArray or numpy.ndarray
        Model predicted wind direction values (degrees).
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Mean gross error in wind direction (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([10, 20, 30])
    >>> WDME_m(obs, mod)
    20.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = abs(circlebias_m(mod - obs)).mean(dim=dim)
        return _update_history(res, "Wind Direction Mean Gross Error (WDME_m)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        res = np.abs(circlebias_m(mod_arr - obs_arr)).mean(axis=axis)
        return res.item() if np.ndim(res) == 0 else res


def WDME(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Wind Direction Mean Gross Error (model and obs unit)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of wind direction errors, regardless of direction.
    - Used in wind energy, meteorology, and air quality studies to assess wind direction model performance.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed wind direction values (degrees).
    mod : xarray.DataArray or numpy.ndarray
        Model predicted wind direction values (degrees).
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Mean gross error in wind direction (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([10, 20, 30])
    >>> WDME(obs, mod)
    20.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = abs(circlebias(mod - obs)).mean(dim=dim)
        return _update_history(res, "Wind Direction Mean Gross Error (WDME)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = np.ma.mean(np.ma.abs(circlebias(mod_arr - obs_arr)), axis=axis)
        return res.item() if np.ndim(res) == 0 else res


def WDMdnE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Wind Direction Median Gross Error (model and obs unit)

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of wind direction errors, robust to outliers.
    - Used in wind energy and meteorological applications for robust wind direction model evaluation.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed wind direction values (degrees).
    mod : xarray.DataArray or numpy.ndarray
        Model predicted wind direction values (degrees).
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Median gross error in wind direction (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([10, 20, 30])
    >>> WDMdnE(obs, mod)
    10.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        cb = abs(circlebias(mod - obs))

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        cb = ensure_single_chunk(cb, dim)

        res = cb.quantile(q=0.5, dim=dim).drop_vars("quantile", errors="ignore")
        return _update_history(res, "Wind Direction Median Gross Error (WDMdnE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        cb = circlebias(mod_arr - obs_arr)
        result = np.ma.median(np.ma.abs(cb), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NME_m(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Error (%) (avoid single block error in np.ma)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of model errors relative to observations, robust to masked arrays.
    - Used for model evaluation when data may contain masked or missing values.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NME_m(obs, mod)
    37.5
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (abs(mod - obs).sum(dim=dim) / obs.sum(dim=dim)) * 100
        return _update_history(res, "Normalized Mean Error (NME_m)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        res = (np.abs(mod_arr - obs_arr).sum(axis=axis) / obs_arr.sum(axis=axis)) * 100
        return res.item() if np.ndim(res) == 0 else res


def NME_m_ABS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Error (%) - Absolute of the denominator
    (avoid single block error in np.ma)

    Typical Use Cases
    -----------------
    - Quantifying normalized mean error when the denominator (sum of observations)
      may be negative or zero, robust to masked arrays.
    - Used for model evaluation with possible sign changes or missing values in observed data.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean error with absolute denominator (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NME_m_ABS(obs, mod)
    37.5
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (abs(mod - obs).sum(dim=dim) / abs(obs.sum(dim=dim))) * 100
        return _update_history(res, "Normalized Mean Error Absolute (NME_m_ABS)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        res = (np.abs(mod_arr - obs_arr).sum(axis=axis) / np.abs(obs_arr.sum(axis=axis))) * 100
        return res.item() if np.ndim(res) == 0 else res


def NME(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Error (%)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of model errors relative to observations.
    - Used for model evaluation and comparison across variables or datasets with different scales.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NME(obs, mod)
    37.5
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (abs(mod - obs).sum(dim=dim) / obs.sum(dim=dim)) * 100
        return _update_history(res, "Normalized Mean Error (NME)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        if obs_arr.size == 0:
            return np.nan
        res = (np.ma.abs(mod_arr - obs_arr).sum(axis=axis) / obs_arr.sum(axis=axis)) * 100
        return res.item() if np.ndim(res) == 0 else res


def NMdnE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Median Error (%)

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of model errors relative to observations, robust to outliers.
    - Used for robust model evaluation and comparison across variables or datasets with different scales.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized median error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NMdnE(obs, mod)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff_abs = abs(mod - obs)

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        diff_abs = ensure_single_chunk(diff_abs, dim)
        obs = ensure_single_chunk(obs, dim)

        res = (
            diff_abs.quantile(q=0.5, dim=dim).drop_vars("quantile", errors="ignore")
            / obs.quantile(q=0.5, dim=dim).drop_vars("quantile", errors="ignore")
            * 100
        )
        return _update_history(res, "Normalized Median Error (NMdnE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = np.ma.median(np.ma.abs(mod_arr - obs_arr), axis=axis) / np.ma.median(obs_arr, axis=axis) * 100
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def FE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Fractional Error (%)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of model errors as a fraction of the sum of model and observed values.
    - Used in air quality and meteorological model evaluation for normalized error assessment.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Fractional error (percent).

    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (abs(mod - obs) / (mod + obs)).mean(dim=dim) * 2.0 * 100.0
        return _update_history(res, "Fractional Error (FE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = (np.ma.mean(np.ma.abs(mod_arr - obs_arr) / (mod_arr + obs_arr), axis=axis)) * 2.0 * 100.0
        return res.item() if np.ndim(res) == 0 else res


def USUTPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Unpaired Space/Unpaired Time Peak Bias (%)

    Typical Use Cases
    -----------------
    - Assessing the bias in peak values between model and observations, regardless of spatial or temporal pairing.
    - Used in event-based or extreme value model evaluation, especially for air quality and meteorological extremes.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Peak bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 5])
    >>> USUTPB(obs, mod)
    25.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = ((mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)) * 100.0
        return _update_history(res, "Unpaired Space/Unpaired Time Peak Bias (USUTPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = ((np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis)) / np.ma.max(obs_arr, axis=axis)) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def USUTPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Unpaired Space/Unpaired Time Peak Error (%)

    Typical Use Cases
    -----------------
    - Quantifying the error in peak values between model and observations, regardless of spatial or temporal pairing.
    - Used in event-based or extreme value model evaluation, especially for air quality and meteorological extremes.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the statistic.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 5])
    >>> USUTPE(obs, mod)
    25.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (abs(mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)) * 100.0
        return _update_history(res, "Unpaired Space/Unpaired Time Peak Error (USUTPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = (
            np.ma.abs(np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis)) / np.ma.max(obs_arr, axis=axis)
        ) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def MNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Mean Normalized Peak Bias (%)

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the mean of normalized peak bias.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Mean normalized peak bias (percent).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _intermediate = (mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)
        mdim = _resolve_axis_to_dim(_intermediate, axis)
        res = _intermediate.mean(dim=mdim) * 100.0
        return _update_history(res, "Mean Normalized Peak Bias (MNPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = ((np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)) / np.ma.max(obs_arr, axis=paxis)).mean(
            axis=axis
        ) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def MdnNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Median Normalized Peak Bias (%)

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of normalized peak bias.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Median normalized peak bias (percent).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _intermediate = (mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)
        mdim = _resolve_axis_to_dim(_intermediate, axis)
        if mdim is None:
            mdim = list(_intermediate.dims)

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        _intermediate = ensure_single_chunk(_intermediate, mdim)

        res = _intermediate.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore") * 100.0
        return _update_history(res, "Median Normalized Peak Bias (MdnNPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = (
            np.ma.median(
                ((np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)) / np.ma.max(obs_arr, axis=paxis)),
                axis=axis,
            )
            * 100.0
        )
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MG(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Geometric Mean Bias (MG).

    Typical Use Cases
    -----------------
    - Air quality model evaluation, especially for variables with log-normal
      distributions.
    - Provides a measure of central tendency for ratios mod/obs.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values (must be positive).
    mod : xarray.DataArray or numpy.ndarray
        Model values (must be positive).
    axis : int or str or None, optional
        Axis along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply. If provided, computes a weighted mean of log-ratios.

    Returns
    -------
    MG : xarray.DataArray or numpy.ndarray or float
        Geometric Mean Bias. MG = exp(mean(log(mod) - log(obs)))
    """
    epsilon = 1e-10
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        log_ratio = np.log(xr.where(mod <= 0, epsilon, mod)) - np.log(xr.where(obs <= 0, epsilon, obs))
        if weights is not None:
            if not isinstance(weights, xr.DataArray):
                weights = xr.DataArray(weights, coords=obs.coords, dims=obs.dims)
            res = np.exp(log_ratio.weighted(weights).mean(dim=dim))
            return _update_history(res, "Weighted Geometric Mean Bias (MG)")
        res = np.exp(log_ratio.mean(dim=dim))
        return _update_history(res, "Geometric Mean Bias (MG)")
    else:
        obs = np.asarray(obs)
        mod = np.asarray(mod)
        log_ratio = np.log(np.where(mod <= 0, epsilon, mod)) - np.log(np.where(obs <= 0, epsilon, obs))
        if weights is not None:
            res = np.exp(np.ma.average(np.ma.masked_invalid(log_ratio), axis=axis, weights=weights))
        else:
            res = np.exp(np.nanmean(log_ratio, axis=axis))
        return res.item() if np.ndim(res) == 0 else res


def VG(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Geometric Variance (VG).

    Typical Use Cases
    -----------------
    - Air quality model evaluation for log-normally distributed variables.
    - Measures the spread of the ratios mod/obs.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values (must be positive).
    mod : xarray.DataArray or numpy.ndarray
        Model values (must be positive).
    axis : int or str or None, optional
        Axis along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply. If provided, computes a weighted mean of squared log-ratios.

    Returns
    -------
    VG : xarray.DataArray or numpy.ndarray or float
        Geometric Variance. VG = exp(mean((log(mod) - log(obs))^2))
    """
    epsilon = 1e-10
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        log_ratio_sq = (np.log(xr.where(mod <= 0, epsilon, mod)) - np.log(xr.where(obs <= 0, epsilon, obs))) ** 2
        if weights is not None:
            if not isinstance(weights, xr.DataArray):
                weights = xr.DataArray(weights, coords=obs.coords, dims=obs.dims)
            res = np.exp(log_ratio_sq.weighted(weights).mean(dim=dim))
            return _update_history(res, "Weighted Geometric Variance (VG)")
        res = np.exp(log_ratio_sq.mean(dim=dim))
        return _update_history(res, "Geometric Variance (VG)")
    else:
        obs = np.asarray(obs)
        mod = np.asarray(mod)
        log_ratio_sq = (np.log(np.where(mod <= 0, epsilon, mod)) - np.log(np.where(obs <= 0, epsilon, obs))) ** 2
        if weights is not None:
            res = np.exp(np.ma.average(np.ma.masked_invalid(log_ratio_sq), axis=axis, weights=weights))
        else:
            res = np.exp(np.nanmean(log_ratio_sq, axis=axis))
        return res.item() if np.ndim(res) == 0 else res


def MNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Mean Normalized Peak Error (MNPE, %)

    Typical Use Cases
    -----------------
    - Quantifying the average error in peak values between model and observations, normalized by observed peaks.
    - Used in model evaluation for extreme events, such as air quality exceedances or meteorological extremes.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the mean of normalized peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Mean normalized peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> MNPE(obs, mod, paxis=1)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _intermediate = abs(mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)
        mdim = _resolve_axis_to_dim(_intermediate, axis)
        res = _intermediate.mean(dim=mdim) * 100.0
        return _update_history(res, "Mean Normalized Peak Error (MNPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = (
            np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)) / np.ma.max(obs_arr, axis=paxis)
        ).mean(axis=axis) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def MdnNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Median Normalized Peak Error (MdnNPE, %)

    Typical Use Cases
    -----------------
    - Evaluating the typical error in peak values between model and observations,
      normalized by observed peaks, robust to outliers.
    - Used in robust model evaluation for extreme events, such as air quality exceedances
      or meteorological extremes.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of normalized peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Median normalized peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> MdnNPE(obs, mod, paxis=1)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _intermediate = abs(mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)
        mdim = _resolve_axis_to_dim(_intermediate, axis)
        if mdim is None:
            mdim = list(_intermediate.dims)

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        _intermediate = ensure_single_chunk(_intermediate, mdim)

        res = _intermediate.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore") * 100.0
        return _update_history(res, "Median Normalized Peak Error (MdnNPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = (
            np.ma.median(
                (
                    np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis))
                    / np.ma.max(obs_arr, axis=paxis)
                ),
                axis=axis,
            )
            * 100.0
        )
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NMPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Peak Bias (NMPB, %)

    Typical Use Cases
    -----------------
    - Quantifying the average bias in peak values, normalized by the mean of observed peaks.
    - Used in model evaluation for extreme events, especially when comparing across sites or time periods.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the mean of normalized peak bias.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean peak bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> NMPB(obs, mod, paxis=1)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _diff_mean = (mod.max(dim=pdim) - obs.max(dim=pdim)).mean(dim=_resolve_axis_to_dim(mod.max(dim=pdim), axis))
        _obs_mean = obs.max(dim=pdim).mean(dim=_resolve_axis_to_dim(obs.max(dim=pdim), axis))
        res = (_diff_mean / _obs_mean) * 100.0
        return _update_history(res, "Normalized Mean Peak Bias (NMPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = (
            (np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)).mean(axis=axis)
            / np.ma.max(obs_arr, axis=paxis).mean(axis=axis)
        ) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def NMdnPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Median Peak Bias (NMdnPB, %)

    Typical Use Cases
    -----------------
    - Evaluating the typical bias in peak values, normalized by the median of observed peaks, robust to outliers.
    - Used in robust model evaluation for extreme events, especially when comparing across sites or time periods.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of normalized peak bias.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized median peak bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> NMdnPB(obs, mod, paxis=1)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _diff = mod.max(dim=pdim) - obs.max(dim=pdim)
        _obs_max = obs.max(dim=pdim)
        mdim = _resolve_axis_to_dim(_diff, axis)
        if mdim is None:
            mdim = list(_diff.dims)

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        _diff = ensure_single_chunk(_diff, mdim)
        _obs_max = ensure_single_chunk(_obs_max, mdim)

        res = (
            _diff.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore")
            / _obs_max.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore")
            * 100.0
        )
        return _update_history(res, "Normalized Median Peak Bias (NMdnPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = (
            np.ma.median(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis), axis=axis)
            / np.ma.median(np.ma.max(obs_arr, axis=paxis), axis=axis)
        ) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NMPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Mean Peak Error (NMPE, %)

    Typical Use Cases
    -----------------
    - Quantifying the average error in peak values, normalized by the mean of observed peaks.
    - Used in model evaluation for extreme events, especially when comparing across sites or time periods.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the mean of normalized peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized mean peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> NMPE(obs, mod, paxis=1)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _diff_abs_mean = abs(mod.max(dim=pdim) - obs.max(dim=pdim)).mean(
            dim=_resolve_axis_to_dim(mod.max(dim=pdim), axis)
        )
        _obs_mean = obs.max(dim=pdim).mean(dim=_resolve_axis_to_dim(obs.max(dim=pdim), axis))
        res = (_diff_abs_mean / _obs_mean) * 100.0
        return _update_history(res, "Normalized Mean Peak Error (NMPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = (
            np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)).mean(axis=axis)
            / np.ma.max(obs_arr, axis=paxis).mean(axis=axis)
        ) * 100.0
        return res.item() if np.ndim(res) == 0 else res


def NMdnPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str, Iterable[Union[int, str]]],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Normalized Median Peak Error (NMdnPE, %)

    Typical Use Cases
    -----------------
    - Evaluating the typical error in peak values, normalized by the median of observed peaks, robust to outliers.
    - Used in robust model evaluation for extreme events, especially when comparing across sites or time periods.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    paxis : int or str
        Axis or dimension along which to compute the peak (e.g., time or space).
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of normalized peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized median peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> NMdnPE(obs, mod, paxis=1)
    33.33333333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        pdim = _resolve_axis_to_dim(obs, paxis)
        _diff_abs = abs(mod.max(dim=pdim) - obs.max(dim=pdim))
        _obs_max = obs.max(dim=pdim)
        mdim = _resolve_axis_to_dim(_diff_abs, axis)
        if mdim is None:
            mdim = list(_diff_abs.dims)

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        _diff_abs = ensure_single_chunk(_diff_abs, mdim)
        _obs_max = ensure_single_chunk(_obs_max, mdim)

        res = (
            _diff_abs.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore")
            / _obs_max.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore")
            * 100.0
        )
        return _update_history(res, "Normalized Median Peak Error (NMdnPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = (
            np.ma.median(
                np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)),
                axis=axis,
            )
            / np.ma.median(np.ma.max(obs_arr, axis=paxis), axis=axis)
        ) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def PSUTMNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Mean Normalized Peak Bias (PSUTMNPB, %)

    Wrapper for MNPB with paxis=0, axis=None.
    """
    return MNPB(obs, mod, paxis=0, axis=None)


def PSUTMdnNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Median Normalized Peak Bias (PSUTMdnNPB, %)

    Wrapper for MdnNPB with paxis=0, axis=None.
    """
    return MdnNPB(obs, mod, paxis=0, axis=None)


def PSUTMNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Mean Normalized Peak Error (PSUTMNPE, %)

    Wrapper for MNPE with paxis=0, axis=None.
    """
    return MNPE(obs, mod, paxis=0, axis=None)


def PSUTMdnNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Median Normalized Peak Error (PSUTMdnNPE, %)

    Wrapper for MdnNPE with paxis=0, axis=None.
    """
    return MdnNPE(obs, mod, paxis=0, axis=None)


def PSUTNMPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Normalized Mean Peak Bias (PSUTNMPB, %)

    Wrapper for NMPB with paxis=0, axis=None.
    """
    return NMPB(obs, mod, paxis=0, axis=None)


def PSUTNMPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Normalized Mean Peak Error (PSUTNMPE, %)

    Wrapper for NMPE with paxis=0, axis=None.
    """
    return NMPE(obs, mod, paxis=0, axis=None)


def PSUTNMdnPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Normalized Median Peak Bias (PSUTNMdnPB, %)

    Typical Use Cases
    -----------------
    - Evaluating the normalized median peak bias for spatially paired, temporally unpaired datasets, robust to outliers.
    - Used in robust model evaluation for spatial ensemble or multi-time analysis.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of normalized peak bias.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized median peak bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> PSUTNMdnPB(obs, mod)
    33.33333333333333
    """
    return NMdnPB(obs, mod, paxis=0, axis=None)


def PSUTNMdnPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Normalized Median Peak Error (PSUTNMdnPE, %)

    Typical Use Cases
    -----------------
    - Evaluating the normalized median peak error for spatially paired, temporally unpaired
      datasets, robust to outliers.
    - Used in robust model evaluation for spatial ensemble or multi-time analysis.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of normalized peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Normalized median peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> PSUTNMdnPE(obs, mod)
    33.33333333333333
    """
    return NMdnPE(obs, mod, paxis=0, axis=None)


def MPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Mean Peak Error (%)

    Typical Use Cases
    -----------------
    - Quantifying the average error in peak values between model and observations.
    - Used in model evaluation for extreme events, such as air quality exceedances
      or meteorological extremes.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the mean of peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Mean peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> MPE(obs, mod)
    33.33333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        res = (abs(mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)).mean() * 100.0
        return _update_history(res, "Mean Peak Error (MPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        res = (
            np.ma.abs(np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis)) / np.ma.max(obs_arr, axis=axis)
        ).mean() * 100.0
        return res.item() if np.ndim(res) == 0 else res


def MdnPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Median Peak Error (%)

    Typical Use Cases
    -----------------
    - Evaluating the typical error in peak values between model and observations,
      robust to outliers.
    - Used in robust model evaluation for extreme events, such as air quality
      exceedances or meteorological extremes.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model predicted values.
    axis : int or str or None, optional
        Axis or dimension along which to compute the median of peak error.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Median peak error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([[1, 2, 3], [2, 3, 4]])
    >>> mod = np.array([[2, 2, 2], [2, 2, 5]])
    >>> MdnPE(obs, mod)
    33.333333333
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        _intermediate = abs(mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)
        mdim = list(_intermediate.dims)

        # Aero Protocol: Ensure reduction dimensions are single-chunked for quantile
        _intermediate = ensure_single_chunk(_intermediate, mdim)

        res = _intermediate.quantile(q=0.5, dim=mdim).drop_vars("quantile", errors="ignore") * 100.0
        return _update_history(res, "Median Peak Error (MdnPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        result = (
            np.ma.median(
                (
                    np.ma.abs(np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis))
                    / np.ma.max(obs_arr, axis=axis)
                ),
                axis=axis,
            )
            * 100.0
        )
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result
