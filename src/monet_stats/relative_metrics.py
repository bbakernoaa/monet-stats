"""
Relative/Percentage Metrics for Model Evaluation (Aero Protocol Compliant)
"""

from typing import Any, Optional, Union

import numpy as np
import xarray as xr

from .utils_stats import circlebias, circlebias_m


def _update_history(obj: Any, metric_name: str) -> Any:
    """Update the history attribute of an xarray object."""
    if isinstance(obj, (xr.DataArray, xr.Dataset)):
        history = obj.attrs.get("history", "")
        new_entry = f"Calculated {metric_name} using monet-stats."
        obj.attrs["history"] = f"{history}\n{new_entry}".strip()
    return obj


def NMB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        # Ensure we use dimension name if axis is int
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (mod - obs).sum(dim=dim) / obs.sum(dim=dim) * 100.0
        return _update_history(res, "Normalized Mean Bias (NMB)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return (mod_arr - obs_arr).sum(axis=axis) / obs_arr.sum(axis=axis) * 100.0


def WDNMB_m(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        diff = mod - obs
        cb = circlebias_m(diff)
        res = cb.sum(dim=dim) / obs.sum(dim=dim) * 100.0
        return _update_history(res, "Wind Direction Normalized Mean Bias (WDNMB_m)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        diff = mod_arr - obs_arr
        return circlebias_m(diff).sum(axis=axis) / obs_arr.sum(axis=axis) * 100.0


def NMB_ABS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (mod - obs).sum(dim=dim) / abs(obs.sum(dim=dim)) * 100.0
        return _update_history(res, "Normalized Mean Bias Absolute (NMB_ABS)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return (mod_arr - obs_arr).sum(axis=axis) / np.abs(obs_arr.sum(axis=axis)) * 100.0


def NMdnB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (mod - obs).median(dim=dim) / obs.median(dim=dim) * 100.0
        return _update_history(res, "Normalized Median Bias (NMdnB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return np.ma.median(mod_arr - obs_arr, axis=axis) / np.ma.median(obs_arr, axis=axis) * 100.0


def FB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (((mod - obs) / (mod + obs)).mean(dim=dim) * 2.0) * 100.0
        return _update_history(res, "Fractional Bias (FB)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return (np.ma.masked_invalid((mod_arr - obs_arr) / (mod_arr + obs_arr)).mean(axis=axis) * 2.0) * 100.0


def ME(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Mean Gross Error (model and obs unit)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of model errors, regardless of direction.
    - Used in model evaluation to summarize overall error size.

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
        Mean gross error value(s).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> ME(obs, mod)
    1.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = abs(mod - obs).mean(dim=dim)
        return _update_history(res, "Mean Gross Error (ME)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return np.mean(np.abs(mod_arr - obs_arr), axis=axis)


def MdnE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Median Gross Error (model and obs unit)

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of model errors, robust to outliers.
    - Used in model evaluation when error distributions are skewed or non-normal.

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
        Median gross error value(s).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> MdnE(obs, mod)
    1.0
    """

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = abs(mod - obs).median(dim=dim)
        return _update_history(res, "Median Gross Error (MdnE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return np.ma.median(np.ma.abs(mod_arr - obs_arr), axis=axis)


def WDME_m(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = abs(circlebias_m(mod - obs)).mean(dim=dim)
        return _update_history(res, "Wind Direction Mean Gross Error (WDME_m)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return np.abs(circlebias_m(mod_arr - obs_arr)).mean(axis=axis)


def WDME(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = abs(circlebias(mod - obs)).mean(dim=dim)
        return _update_history(res, "Wind Direction Mean Gross Error (WDME)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return np.ma.mean(np.ma.abs(circlebias(mod_arr - obs_arr)), axis=axis)


def WDMdnE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        cb = circlebias(mod - obs)
        res = abs(cb).median(dim=dim)
        return _update_history(res, "Wind Direction Median Gross Error (WDMdnE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        cb = circlebias(mod_arr - obs_arr)
        return np.ma.median(np.ma.abs(cb), axis=axis)


def NME_m(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod - obs).sum(dim=dim) / obs.sum(dim=dim)) * 100
        return _update_history(res, "Normalized Mean Error (NME_m)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return (np.abs(mod_arr - obs_arr).sum(axis=axis) / obs_arr.sum(axis=axis)) * 100


def NME_m_ABS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod - obs).sum(dim=dim) / abs(obs.sum(dim=dim))) * 100
        return _update_history(res, "Normalized Mean Error Absolute (NME_m_ABS)")
    else:
        obs_arr = np.asanyarray(obs)
        mod_arr = np.asanyarray(mod)
        return (np.abs(mod_arr - obs_arr).sum(axis=axis) / np.abs(obs_arr.sum(axis=axis))) * 100


def NME(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod - obs).sum(dim=dim) / obs.sum(dim=dim)) * 100
        return _update_history(res, "Normalized Mean Error (NME)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (np.ma.abs(mod_arr - obs_arr).sum(axis=axis) / obs_arr.sum(axis=axis)) * 100


def NMdnE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = abs(mod - obs).median(dim=dim) / obs.median(dim=dim) * 100
        return _update_history(res, "Normalized Median Error (NMdnE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return np.ma.median(np.ma.abs(mod_arr - obs_arr), axis=axis) / np.ma.median(obs_arr, axis=axis) * 100


def FE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod - obs) / (mod + obs)).mean(dim=dim) * 2.0 * 100.0
        return _update_history(res, "Fractional Error (FE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (np.ma.mean(np.ma.abs(mod_arr - obs_arr) / (mod_arr + obs_arr), axis=axis)) * 2.0 * 100.0


def USUTPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = ((mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)) * 100.0
        return _update_history(res, "Unpaired Space/Unpaired Time Peak Bias (USUTPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return ((np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis)) / np.ma.max(obs_arr, axis=axis)) * 100.0


def USUTPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)) * 100.0
        return _update_history(res, "Unpaired Space/Unpaired Time Peak Error (USUTPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.abs(np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis)) / np.ma.max(obs_arr, axis=axis)
        ) * 100.0


def MNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = (((mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)).mean(dim=mdim)) * 100.0
        return _update_history(res, "Mean Normalized Peak Bias (MNPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            (np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)) / np.ma.max(obs_arr, axis=paxis)
        ).mean(axis=axis) * 100.0


def MdnNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = ((mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)).median(dim=mdim) * 100.0
        return _update_history(res, "Median Normalized Peak Bias (MdnNPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.median(
                ((np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)) / np.ma.max(obs_arr, axis=paxis)),
                axis=axis,
            )
            * 100.0
        )


def MNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = (abs(mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)).mean(dim=mdim) * 100.0
        return _update_history(res, "Mean Normalized Peak Error (MNPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)) / np.ma.max(obs_arr, axis=paxis)
        ).mean(axis=axis) * 100.0


def MdnNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = (abs(mod.max(dim=pdim) - obs.max(dim=pdim)) / obs.max(dim=pdim)).median(dim=mdim) * 100.0
        return _update_history(res, "Median Normalized Peak Error (MdnNPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.median(
                (
                    np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis))
                    / np.ma.max(obs_arr, axis=paxis)
                ),
                axis=axis,
            )
            * 100.0
        )


def NMPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = ((mod.max(dim=pdim) - obs.max(dim=pdim)).mean(dim=mdim) / obs.max(dim=pdim).mean(dim=mdim)) * 100.0
        return _update_history(res, "Normalized Mean Peak Bias (NMPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            (np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)).mean(axis=axis)
            / np.ma.max(obs_arr, axis=paxis).mean(axis=axis)
        ) * 100.0


def NMdnPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = (mod.max(dim=pdim) - obs.max(dim=pdim)).median(dim=mdim) / obs.max(dim=pdim).median(dim=mdim) * 100.0
        return _update_history(res, "Normalized Median Peak Bias (NMdnPB)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.median(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis), axis=axis)
            / np.ma.median(np.ma.max(obs_arr, axis=paxis), axis=axis)
        ) * 100.0


def NMPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = (abs(mod.max(dim=pdim) - obs.max(dim=pdim)).mean(dim=mdim) / obs.max(dim=pdim).mean(dim=mdim)) * 100.0
        return _update_history(res, "Normalized Mean Peak Error (NMPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)).mean(axis=axis)
            / np.ma.max(obs_arr, axis=paxis).mean(axis=axis)
        ) * 100.0


def NMdnPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    paxis: Union[int, str],
    axis: Optional[Union[int, str]] = None,
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
        pdim = paxis
        if isinstance(paxis, int):
            pdim = obs.dims[paxis]
        mdim = axis
        if isinstance(axis, int):
            mdim = obs.dims[axis]
        res = (abs(mod.max(dim=pdim) - obs.max(dim=pdim))).median(dim=mdim) / obs.max(dim=pdim).median(dim=mdim) * 100.0
        return _update_history(res, "Normalized Median Peak Error (NMdnPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.median(
                np.ma.abs(np.ma.max(mod_arr, axis=paxis) - np.ma.max(obs_arr, axis=paxis)),
                axis=axis,
            )
            / np.ma.median(np.ma.max(obs_arr, axis=paxis), axis=axis)
        ) * 100.0


def PSUTMNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Mean Normalized Peak Bias (PSUTMNPB, %)

    Wrapper for MNPB with paxis=0, axis=None.
    """

    return MNPB(obs, mod, paxis=0, axis=None)


def PSUTMdnNPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Median Normalized Peak Bias (PSUTMdnNPB, %)

    Wrapper for MdnNPB with paxis=0, axis=None.
    """

    return MdnNPB(obs, mod, paxis=0, axis=None)


def PSUTMNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Mean Normalized Peak Error (PSUTMNPE, %)

    Wrapper for MNPE with paxis=0, axis=None.
    """

    return MNPE(obs, mod, paxis=0, axis=None)


def PSUTMdnNPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Median Normalized Peak Error (PSUTMdnNPE, %)

    Wrapper for MdnNPE with paxis=0, axis=None.
    """

    return MdnNPE(obs, mod, paxis=0, axis=None)


def PSUTNMPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Normalized Mean Peak Bias (PSUTNMPB, %)

    Wrapper for NMPB with paxis=0, axis=None.
    """
    return NMPB(obs, mod, paxis=0, axis=None)


def PSUTNMPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Paired Space/Unpaired Time Normalized Mean Peak Error (PSUTNMPE, %)

    Wrapper for NMPE with paxis=0, axis=None.
    """
    return NMPE(obs, mod, paxis=0, axis=None)


def PSUTNMdnPB(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
    axis: Optional[Union[int, str]] = None,
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
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)).mean() * 100.0
        return _update_history(res, "Mean Peak Error (MPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.abs(np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis)) / np.ma.max(obs_arr, axis=axis)
        ).mean() * 100.0


def MdnPE(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
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
        dim = axis
        if isinstance(axis, int):
            dim = obs.dims[axis]
        res = (abs(mod.max(dim=dim) - obs.max(dim=dim)) / obs.max(dim=dim)).median() * 100.0
        return _update_history(res, "Median Peak Error (MdnPE)")
    else:
        obs_arr = np.ma.asanyarray(obs)
        mod_arr = np.ma.asanyarray(mod)
        return (
            np.ma.median(
                (
                    np.ma.abs(np.ma.max(mod_arr, axis=axis) - np.ma.max(obs_arr, axis=axis))
                    / np.ma.max(obs_arr, axis=axis)
                ),
                axis=axis,
            )
            * 100.0
        )
