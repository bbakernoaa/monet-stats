"""
Efficiency Metrics for Model Evaluation (Aero Protocol Compliant).
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr

from .correlation_metrics import KGE  # noqa: F401
from .error_metrics import MAE, MAPE, MASE, MSE  # noqa: F401
from .utils_stats import _resolve_axis_to_dim, _update_history

__all__ = ["NSE", "NSEm", "NSElog", "rNSE", "mNSE", "PC", "KGE", "MAE", "MAPE", "MASE", "MSE"]


def NSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Nash-Sutcliffe Efficiency (NSE).

    Typical Use Cases
    -----------------
    - Quantifying the predictive power of hydrological models relative to the
      mean of observations.
    - Used in hydrology, meteorology, and environmental model evaluation.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply for area-weighted statistics.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.efficiency_metrics import NSE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> NSE(obs, mod)
    0.992
    """
    # Standardize to DataArray if either input is a DataArray (Aero Protocol)
    if isinstance(obs, xr.DataArray) or isinstance(mod, xr.DataArray):
        if not isinstance(obs, xr.DataArray):
            obs = xr.DataArray(obs, dims=mod.dims, coords=mod.coords)
        if not isinstance(mod, xr.DataArray):
            mod = xr.DataArray(mod, dims=obs.dims, coords=obs.coords)

        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)

        if weights is not None:
            # If weights is numpy, wrap it as DataArray to align
            if not isinstance(weights, xr.DataArray):
                weights = xr.DataArray(weights, dims=obs.dims, coords=obs.coords)
            obs_mean = obs.weighted(weights).mean(dim=dim)
            numerator = ((obs - mod) ** 2).weighted(weights).sum(dim=dim)
            denominator = ((obs - obs_mean) ** 2).weighted(weights).sum(dim=dim)
        else:
            obs_mean = obs.mean(dim=dim)
            numerator = ((obs - mod) ** 2).sum(dim=dim)
            denominator = ((obs - obs_mean) ** 2).sum(dim=dim)

        # Handle division by zero
        result = 1.0 - (numerator / denominator)
        result = xr.where((numerator == 0) & (denominator == 0), 1.0, result)
        result = xr.where((numerator != 0) & (denominator == 0), -np.inf, result)

        return _update_history(result, "NSE" if weights is None else "Weighted NSE")
    else:
        obs = np.asanyarray(obs)
        mod = np.asanyarray(mod)
        if obs.size == 0:
            return np.nan
        if weights is not None:
            obs_mean = np.ma.average(np.ma.masked_invalid(obs), axis=axis, weights=weights, keepdims=True)
            numerator = np.nansum(((obs - mod) ** 2) * weights, axis=axis)
            denominator = np.nansum(((obs - obs_mean) ** 2) * weights, axis=axis)
        else:
            obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
            numerator = np.nansum((obs - mod) ** 2, axis=axis)
            denominator = np.nansum((obs - obs_mean) ** 2, axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (numerator / denominator)
            result = np.where((numerator == 0) & (denominator == 0), 1.0, result)
            result = np.where((numerator != 0) & (denominator == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def NSEm(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Nash-Sutcliffe Efficiency (NSE) - robust to masked arrays.

    This function is a wrapper for NSE that explicitly handles masked data and NaNs.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply for area-weighted statistics.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.efficiency_metrics import NSEm
    >>> obs = np.array([1, 2, np.nan, 4])
    >>> mod = np.array([1.1, 2.1, 3.0, 4.1])
    >>> NSEm(obs, mod)
    0.995
    """
    # Standard NSE implementation already handles NaNs and mixed types
    res = NSE(obs, mod, axis=axis, weights=weights)
    if isinstance(res, xr.DataArray):
        return _update_history(res, "NSEm")
    return res


def NSElog(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Log Nash-Sutcliffe Efficiency (NSElog).

    Calculates NSE on logarithmic-transformed data to focus on lower values.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values (positive values only).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values (positive values only).
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply for area-weighted statistics.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Log Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.efficiency_metrics import NSElog
    >>> obs = np.array([1, 10, 100])
    >>> mod = np.array([1.1, 9.0, 110])
    >>> NSElog(obs, mod)
    0.988
    """
    epsilon = 1e-6
    # Avoid double history update by using .data or being careful
    # We apply log first, then call NSE. NSE will handle history if it's a DataArray.
    obs_log = np.log(obs + epsilon)
    mod_log = np.log(mod + epsilon)
    result = NSE(obs_log, mod_log, axis=axis, weights=weights)

    # If it's a DataArray, NSE already updated history to "NSE".
    # We might want to "fix" it to "NSElog".
    if isinstance(result, xr.DataArray):
        # Update history specifically for NSElog
        return _update_history(result, "NSElog")
    return result


def rNSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Relative Nash-Sutcliffe Efficiency (rNSE).

    Normalizes errors by the magnitude of observed values.
    Formula: 1 - [ sum( ((obs - mod)/obs)^2 ) / sum( ((obs - mean)/obs)^2 ) ]

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values (should be non-zero for normalization).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply for area-weighted statistics.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Relative Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.efficiency_metrics import rNSE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> rNSE(obs, mod)
    0.994261721483555
    """
    epsilon = 1e-8
    if isinstance(obs, xr.DataArray) or isinstance(mod, xr.DataArray):
        if not isinstance(obs, xr.DataArray):
            obs = xr.DataArray(obs, dims=mod.dims, coords=mod.coords)
        if not isinstance(mod, xr.DataArray):
            mod = xr.DataArray(mod, dims=obs.dims, coords=obs.coords)

        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)

        if weights is not None:
            if not isinstance(weights, xr.DataArray):
                weights = xr.DataArray(weights, dims=obs.dims, coords=obs.coords)
            obs_mean = obs.weighted(weights).mean(dim=dim)
            obs_safe = xr.where(abs(obs) < epsilon, epsilon, obs)
            numerator = (((obs - mod) / obs_safe) ** 2).weighted(weights).sum(dim=dim)
            denominator = (((obs - obs_mean) / obs_safe) ** 2).weighted(weights).sum(dim=dim)
        else:
            obs_mean = obs.mean(dim=dim)
            obs_safe = xr.where(abs(obs) < epsilon, epsilon, obs)
            numerator = (((obs - mod) / obs_safe) ** 2).sum(dim=dim)
            denominator = (((obs - obs_mean) / obs_safe) ** 2).sum(dim=dim)

        result = 1.0 - (numerator / denominator)
        result = xr.where((numerator == 0) & (denominator == 0), 1.0, result)
        result = xr.where((numerator != 0) & (denominator == 0), -np.inf, result)

        return _update_history(result, "rNSE")
    else:
        obs = np.asanyarray(obs)
        mod = np.asanyarray(mod)
        if obs.size == 0:
            return np.nan
        if weights is not None:
            obs_mean = np.ma.average(np.ma.masked_invalid(obs), axis=axis, weights=weights, keepdims=True)
            obs_safe = np.where(np.abs(obs) < epsilon, epsilon, obs)
            numerator = np.nansum((((obs - mod) / obs_safe) ** 2) * weights, axis=axis)
            denominator = np.nansum((((obs - obs_mean) / obs_safe) ** 2) * weights, axis=axis)
        else:
            obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
            obs_safe = np.where(np.abs(obs) < epsilon, epsilon, obs)
            numerator = np.nansum(((obs - mod) / obs_safe) ** 2, axis=axis)
            denominator = np.nansum(((obs - obs_mean) / obs_safe) ** 2, axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (numerator / denominator)
            result = np.where((numerator == 0) & (denominator == 0), 1.0, result)
            result = np.where((numerator != 0) & (denominator == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def mNSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Modified Nash-Sutcliffe Efficiency (mNSE).

    Uses absolute differences instead of squared differences.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.
    weights : numpy.ndarray or xarray.DataArray, optional
        Weights to apply for area-weighted statistics.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Modified Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.efficiency_metrics import mNSE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> mNSE(obs, mod)
    0.92
    """
    if isinstance(obs, xr.DataArray) or isinstance(mod, xr.DataArray):
        if not isinstance(obs, xr.DataArray):
            obs = xr.DataArray(obs, dims=mod.dims, coords=mod.coords)
        if not isinstance(mod, xr.DataArray):
            mod = xr.DataArray(mod, dims=obs.dims, coords=obs.coords)

        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)

        if weights is not None:
            if not isinstance(weights, xr.DataArray):
                weights = xr.DataArray(weights, dims=obs.dims, coords=obs.coords)
            obs_mean = obs.weighted(weights).mean(dim=dim)
            numerator = abs(obs - mod).weighted(weights).sum(dim=dim)
            denominator = abs(obs - obs_mean).weighted(weights).sum(dim=dim)
        else:
            obs_mean = obs.mean(dim=dim)
            numerator = abs(obs - mod).sum(dim=dim)
            denominator = abs(obs - obs_mean).sum(dim=dim)

        result = 1.0 - (numerator / denominator)
        result = xr.where((numerator == 0) & (denominator == 0), 1.0, result)
        result = xr.where((numerator != 0) & (denominator == 0), -np.inf, result)

        return _update_history(result, "mNSE")
    else:
        obs = np.asanyarray(obs)
        mod = np.asanyarray(mod)
        if obs.size == 0:
            return np.nan
        if weights is not None:
            obs_mean = np.ma.average(np.ma.masked_invalid(obs), axis=axis, weights=weights, keepdims=True)
            numerator = np.nansum(np.abs(obs - mod) * weights, axis=axis)
            denominator = np.nansum(np.abs(obs - obs_mean) * weights, axis=axis)
        else:
            obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
            numerator = np.nansum(np.abs(obs - mod), axis=axis)
            denominator = np.nansum(np.abs(obs - obs_mean), axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (numerator / denominator)
            result = np.where((numerator == 0) & (denominator == 0), 1.0, result)
            result = np.where((numerator != 0) & (denominator == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def PC(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
    tolerance: float = 0.1,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Percent of Correct (PC).

    Calculates the percentage of model predictions that are within a specified
    tolerance of the observations.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.
    tolerance : float, optional
        Fraction of observed value used as tolerance (default 0.1).

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Percent of correct predictions (0-100%).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.efficiency_metrics import PC
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.05, 2.5, 2.95, 4.05])
    >>> PC(obs, mod)
    75.0
    """
    if isinstance(obs, xr.DataArray) or isinstance(mod, xr.DataArray):
        if not isinstance(obs, xr.DataArray):
            obs = xr.DataArray(obs, dims=mod.dims, coords=mod.coords)
        if not isinstance(mod, xr.DataArray):
            mod = xr.DataArray(mod, dims=obs.dims, coords=obs.coords)

        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)

        tol = tolerance * abs(obs)
        correct = abs(obs - mod) <= tol
        # Only consider where both obs and mod are not NaN
        mask = obs.notnull() & mod.notnull()
        result = (correct.where(mask).sum(dim=dim) / mask.sum(dim=dim)) * 100.0

        return _update_history(result, "PC")
    else:
        obs = np.asanyarray(obs)
        mod = np.asanyarray(mod)
        mask = np.isnan(obs) | np.isnan(mod)

        tol = tolerance * np.abs(obs)
        correct = np.abs(obs - mod) <= tol

        total = np.sum(~mask, axis=axis)
        correct_sum = np.sum(correct & ~mask, axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = (correct_sum / total) * 100.0
            result = np.where(total == 0, np.nan, result)

        return result.item() if np.ndim(result) == 0 else result
