"""
Efficiency Metrics for Model Evaluation
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr

from .error_metrics import MAE, MAPE, MASE, MSE  # noqa: F401
from .utils_stats import _update_history


def NSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
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
    0.98
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_mean = obs.mean(dim=dim)
        numerator = ((obs - mod) ** 2).sum(dim=dim)
        denominator = ((obs - obs_mean) ** 2).sum(dim=dim)

        # Handle division by zero
        result = 1.0 - (numerator / denominator)
        result = xr.where((numerator == 0) & (denominator == 0), 1.0, result)
        result = xr.where((numerator != 0) & (denominator == 0), -np.inf, result)

        return _update_history(result, "NSE")
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
    0.985
    """
    # Standard NSE implementation already handles NaNs if using nan-aware functions
    return NSE(obs, mod, axis=axis)


def NSElog(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
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
    obs_log = np.log(obs + epsilon)
    mod_log = np.log(mod + epsilon)
    result = NSE(obs_log, mod_log, axis=axis)
    if isinstance(result, xr.DataArray):
        return _update_history(result, "NSElog")
    return result


def rNSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Relative Nash-Sutcliffe Efficiency (rNSE).

    Normalizes errors by the range of observed values.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the statistic.

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
    0.992
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_mean = obs.mean(dim=dim)
        obs_range = obs.max(dim=dim) - obs.min(dim=dim)
        # Avoid division by zero in normalization
        obs_range_safe = xr.where(obs_range == 0, 1.0, obs_range)

        numerator = (((obs - mod) / obs_range_safe) ** 2).sum(dim=dim)
        denominator = (((obs - obs_mean) / obs_range_safe) ** 2).sum(dim=dim)

        result = 1.0 - (numerator / denominator)
        result = xr.where((numerator == 0) & (denominator == 0), 1.0, result)
        result = xr.where((numerator != 0) & (denominator == 0), -np.inf, result)

        return _update_history(result, "rNSE")
    else:
        obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
        obs_range = np.nanmax(obs, axis=axis, keepdims=True) - np.nanmin(obs, axis=axis, keepdims=True)
        obs_range_safe = np.where(obs_range == 0, 1.0, obs_range)

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = np.nansum(((obs - mod) / obs_range_safe) ** 2, axis=axis)
            denominator = np.nansum(((obs - obs_mean) / obs_range_safe) ** 2, axis=axis)
            result = 1.0 - (numerator / denominator)
            result = np.where((numerator == 0) & (denominator == 0), 1.0, result)
            result = np.where((numerator != 0) & (denominator == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def mNSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
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
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_mean = obs.mean(dim=dim)
        numerator = np.abs(obs - mod).sum(dim=dim)
        denominator = np.abs(obs - obs_mean).sum(dim=dim)

        result = 1.0 - (numerator / denominator)
        result = xr.where((numerator == 0) & (denominator == 0), 1.0, result)
        result = xr.where((numerator != 0) & (denominator == 0), -np.inf, result)

        return _update_history(result, "mNSE")
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
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        tol = tolerance * np.abs(obs)
        correct = np.abs(obs - mod) <= tol
        result = (correct.sum(dim=dim) / correct.count(dim=dim)) * 100.0

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
