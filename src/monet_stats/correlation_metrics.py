"""
Correlation and Agreement Metrics for Model Evaluation
"""

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

from .utils_stats import circlebias, circlebias_m, matchedcompressed


def R2(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Coefficient of Determination (R^2, unitless).

    Typical Use Cases
    -----------------
    - Quantifying how well model predictions explain the variance in observations.
    - Used in regression analysis, model skill assessment, and forecast
      verification.

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
        Coefficient of determination (R^2).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import R2
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 1.9, 3.2, 3.8])
    >>> R2(obs, mod)
    0.9846153846153847
    """
    from scipy.stats import pearsonr

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is None:
            # Default to all dimensions if None
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        def _pearsonr2(a, b):
            if np.var(a) == 0 or np.var(b) == 0:
                return 0.0
            r_val, _ = pearsonr(a, b)
            if np.isnan(r_val):
                return 0.0
            return r_val**2

        result = xr.apply_ufunc(
            _pearsonr2,
            obs,
            mod,
            input_core_dims=[[dim] if isinstance(dim, str) else list(dim)] * 2,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Update history
        history = f"R2 computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        if axis is None:
            obsc, modc = matchedcompressed(obs, mod)
            if np.var(obsc) == 0 or np.var(modc) == 0:
                return 0.0
            r_val, _ = pearsonr(obsc, modc)
            if np.isnan(r_val):
                return 0.0
            return r_val**2
        else:
            # Manual vectorized R2
            obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
            mod_mean = np.nanmean(mod, axis=axis, keepdims=True)
            obs_std = obs - obs_mean
            mod_std = mod - mod_mean
            num = np.nansum(obs_std * mod_std, axis=axis)
            den = np.sqrt(np.nansum(obs_std**2, axis=axis) * np.nansum(mod_std**2, axis=axis))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = num / den
                result = np.where(np.isnan(r), 0.0, r**2)
                return result.item() if np.ndim(result) == 0 else result


def RMSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Root Mean Square Error (RMSE, model unit).

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations.
    - Used in model evaluation, forecast verification, and regression analysis.

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
        Root mean square error value(s).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import RMSE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> RMSE(obs, mod)
    1.118033988749895
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        result = ((mod - obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        # Update history
        history = f"RMSE computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        return np.ma.sqrt(np.ma.mean((np.subtract(mod, obs)) ** 2, axis=axis))


def WDRMSE_m(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Root Mean Square Error (WDRMSE, model unit).

    Robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of wind direction errors, accounting for
      circularity, robust to masked arrays.
    - Used in wind energy, meteorology, and air quality studies to assess wind
      direction model performance.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Wind direction root mean square error (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import WDRMSE_m
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([10, 20, 30])
    >>> WDRMSE_m(obs, mod)
    20.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        result = (circlebias_m(mod - obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        # Update history
        history = f"WDRMSE_m computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        return np.ma.sqrt(np.ma.mean((circlebias_m(np.subtract(mod, obs))) ** 2, axis=axis))


def WDRMSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Root Mean Square Error (WDRMSE, model unit).

    Standard version.

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of wind direction errors, accounting for
      circularity.
    - Used in wind energy, meteorology, and air quality studies to assess wind
      direction model performance.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Wind direction root mean square error (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import WDRMSE
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([10, 20, 30])
    >>> WDRMSE(obs, mod)
    20.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        result = (circlebias(mod - obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        # Update history
        history = f"WDRMSE computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        return np.ma.sqrt(np.ma.mean((circlebias(np.subtract(mod, obs))) ** 2, axis=axis))


def RMSEs(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray, None]:
    """
    Root Mean Squared Error between observations and regression fit.

    (RMSEs, model unit)

    Typical Use Cases
    -----------------
    - Quantifying the error between observations and a regression fit to the
      model predictions.
    - Used in model evaluation to assess how well a regression fit to the model
      matches the observations.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray, optional
        Root mean squared error value(s), or None if regression fails.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import RMSEs
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> RMSEs(obs, mod)
    0.7071067811865476
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is None:
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        def _rmses(a, b):
            from scipy.stats import linregress

            mask = ~np.isnan(a) & ~np.isnan(b)
            if not np.any(mask):
                return np.nan
            m, c, _, _, _ = linregress(a[mask], b[mask])
            mod_hat = c + m * a
            return np.sqrt(np.mean((mod_hat - a) ** 2))

        result = xr.apply_ufunc(
            _rmses,
            obs,
            mod,
            input_core_dims=[[dim] if isinstance(dim, str) else list(dim)] * 2,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Update history
        history = f"RMSEs computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        if axis is None:
            try:
                from scipy.stats import linregress

                obsc, modc = matchedcompressed(obs, mod)
                m, b, _, _, _ = linregress(obsc, modc)
                mod_hat = b + m * obs
                return RMSE(obs, mod_hat, axis=axis)
            except (ValueError, ZeroDivisionError):
                return None
        else:
            # Manual vectorized regression for numpy with axis
            obs = np.asarray(obs)
            mod = np.asarray(mod)
            if axis < 0:
                axis = obs.ndim + axis

            obs_moved = np.moveaxis(obs, axis, -1)
            mod_moved = np.moveaxis(mod, axis, -1)
            other_shape = obs_moved.shape[:-1]
            obs_flat = obs_moved.reshape(-1, obs_moved.shape[-1])
            mod_flat = mod_moved.reshape(-1, mod_moved.shape[-1])

            results = []
            from scipy.stats import linregress

            for i in range(len(obs_flat)):
                mask = ~np.isnan(obs_flat[i]) & ~np.isnan(mod_flat[i])
                if np.sum(mask) < 2:
                    results.append(np.nan)
                else:
                    m, b, _, _, _ = linregress(obs_flat[i][mask], mod_flat[i][mask])
                    mod_hat = b + m * obs_flat[i]
                    results.append(np.sqrt(np.nanmean((mod_hat - obs_flat[i]) ** 2)))
            return np.array(results).reshape(other_shape)


def matchmasks(a1: ArrayLike, a2: ArrayLike) -> Tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """
    Match and combine masks from two masked arrays.

    Typical Use Cases
    -----------------
    - Ensuring that two arrays have the same mask for paired statistical calculations.
    - Used in metrics that require both arrays to have valid data at the same locations (e.g., correlation, regression).

    Parameters
    ----------
    a1 : array-like or numpy.ma.MaskedArray
        First input array.
    a2 : array-like or numpy.ma.MaskedArray
        Second input array.

    Returns
    -------
    tuple of numpy.ma.MaskedArray
        Tuple of (a1_masked, a2_masked) with combined mask.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> a1 = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    >>> a2 = np.ma.array([4, 5, 6], mask=[0, 0, 1])
    >>> stats.matchmasks(a1, a2)
    (masked_array(data=[1, --, 3], mask=[False,  True, False]),
     masked_array(data=[4, --, --], mask=[False, False,  True]))
    """
    mask = np.ma.getmaskarray(a1) | np.ma.getmaskarray(a2)
    return np.ma.masked_where(mask, a1), np.ma.masked_where(mask, a2)


def RMSEu(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray, None]:
    """
    Root Mean Squared Error between regression fit and model predictions.

    (RMSEu, model unit)

    Typical Use Cases
    -----------------
    - Quantifying the error between a linear regression fit to observations and
      the model predictions.
    - Used in model evaluation to assess how well a regression fit to obs
      matches the model output.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray, optional
        Root mean squared error value(s), or None if regression fails.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import RMSEu
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> RMSEu(obs, mod)
    0.7071067811865476
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is None:
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        def _rmseu(a, b):
            from scipy.stats import linregress

            mask = ~np.isnan(a) & ~np.isnan(b)
            if not np.any(mask):
                return np.nan
            m, c, _, _, _ = linregress(a[mask], b[mask])
            mod_hat = c + m * a
            return np.sqrt(np.mean((mod_hat - b) ** 2))

        result = xr.apply_ufunc(
            _rmseu,
            obs,
            mod,
            input_core_dims=[[dim] if isinstance(dim, str) else list(dim)] * 2,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Update history
        history = f"RMSEu computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        if axis is None:
            try:
                from scipy.stats import linregress

                obsc, modc = matchedcompressed(obs, mod)
                m, b, _, _, _ = linregress(obsc, modc)
                mod_hat = b + m * obs
                return RMSE(mod_hat, mod, axis=axis)
            except (ValueError, ZeroDivisionError):
                return None
        else:
            obs = np.asarray(obs)
            mod = np.asarray(mod)
            if axis < 0:
                axis = obs.ndim + axis

            obs_moved = np.moveaxis(obs, axis, -1)
            mod_moved = np.moveaxis(mod, axis, -1)
            other_shape = obs_moved.shape[:-1]
            obs_flat = obs_moved.reshape(-1, obs_moved.shape[-1])
            mod_flat = mod_moved.reshape(-1, mod_moved.shape[-1])

            results = []
            from scipy.stats import linregress

            for i in range(len(obs_flat)):
                mask = ~np.isnan(obs_flat[i]) & ~np.isnan(mod_flat[i])
                if np.sum(mask) < 2:
                    results.append(np.nan)
                else:
                    m, b, _, _, _ = linregress(obs_flat[i][mask], mod_flat[i][mask])
                    mod_hat = b + m * obs_flat[i]
                    results.append(np.sqrt(np.nanmean((mod_hat - mod_flat[i]) ** 2)))
            return np.array(results).reshape(other_shape)


def d1(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Modified Index of Agreement (d1).

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, less sensitive
      to outliers than IOA.
    - Used in model evaluation for robust skill assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Modified index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import d1
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> d1(obs, mod)
    0.5
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        num = abs(obs - mod).sum(dim=dim)
        mean_obs = obs.mean(dim=dim)
        denom = (abs(mod - mean_obs) + abs(obs - mean_obs)).sum(dim=dim)
        result = 1.0 - (num / denom)
        result = xr.where((num == 0) & (denom == 0), 1.0, result)
        result = xr.where((num != 0) & (denom == 0), -np.inf, result)

        # Update history
        history = f"d1 computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        num = np.ma.abs(np.subtract(obs, mod)).sum(axis=axis)
        mean_obs = np.ma.mean(obs, axis=axis)
        denom = (np.ma.abs(np.subtract(mod, mean_obs)) + np.ma.abs(np.subtract(obs, mean_obs))).sum(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
            result = np.where((num == 0) & (denom == 0), 1.0, result)
            result = np.where((num != 0) & (denom == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def E1(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Modified Coefficient of Efficiency (E1).

    Typical Use Cases
    -----------------
    - Quantifying the efficiency of model predictions relative to observed mean,
      robust to outliers.
    - Used in hydrology, meteorology, and model skill assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Modified coefficient of efficiency (unitless, -inf to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import E1
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> E1(obs, mod)
    0.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        num = abs(obs - mod).sum(dim=dim)
        denom = abs(obs - obs.mean(dim=dim)).sum(dim=dim)
        result = 1.0 - (num / denom)
        result = xr.where((num == 0) & (denom == 0), 1.0, result)
        result = xr.where((num != 0) & (denom == 0), -np.inf, result)

        # Update history
        history = f"E1 computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        num = np.ma.abs(np.subtract(obs, mod)).sum(axis=axis)
        mean_obs = np.ma.mean(obs, axis=axis)
        denom = np.ma.abs(np.subtract(obs, mean_obs)).sum(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
            result = np.where((num == 0) & (denom == 0), 1.0, result)
            result = np.where((num != 0) & (denom == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def IOA_m(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Index of Agreement (IOA), robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, normalized by
      total deviation.
    - Used in model evaluation for skill assessment, robust to masked arrays.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import IOA_m
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> IOA_m(obs, mod)
    0.8
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obsmean = obs.mean(dim=dim)
        num = ((obs - mod) ** 2).sum(dim=dim)
        denom = ((abs(mod - obsmean) + abs(obs - obsmean)) ** 2).sum(dim=dim)
        result = 1.0 - (num / denom)
        result = xr.where((num == 0) & (denom == 0), 1.0, result)
        result = xr.where((num != 0) & (denom == 0), -np.inf, result)

        # Update history
        history = f"IOA_m computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obsmean = np.ma.mean(obs, axis=axis)
        num = (np.ma.abs(np.subtract(obs, mod)) ** 2).sum(axis=axis)
        denom = ((np.ma.abs(np.subtract(mod, obsmean)) + np.ma.abs(np.subtract(obs, obsmean))) ** 2).sum(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
            result = np.where((num == 0) & (denom == 0), 1.0, result)
            result = np.where((num != 0) & (denom == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def IOA(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Index of Agreement (IOA).

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, normalized by
      total deviation.
    - Used in model evaluation for skill assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import IOA
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> IOA(obs, mod)
    0.8
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obsmean = obs.mean(dim=dim)
        num = ((obs - mod) ** 2).sum(dim=dim)
        denom = ((abs(mod - obsmean) + abs(obs - obsmean)) ** 2).sum(dim=dim)
        result = 1.0 - (num / denom)
        result = xr.where((num == 0) & (denom == 0), 1.0, result)
        result = xr.where((num != 0) & (denom == 0), -np.inf, result)

        # Update history
        history = f"IOA computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obsmean = np.ma.mean(obs, axis=axis)
        num = (np.ma.abs(np.subtract(obs, mod)) ** 2).sum(axis=axis)
        denom = ((np.ma.abs(np.subtract(mod, obsmean)) + np.ma.abs(np.subtract(obs, obsmean))) ** 2).sum(axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
            result = np.where((num == 0) & (denom == 0), 1.0, result)
            result = np.where((num != 0) & (denom == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def WDIOA_m(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Index of Agreement (WDIOA_m).

    Robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the agreement between observed and modeled wind directions,
      accounting for circularity.
    - Used in wind energy, meteorology, and air quality studies to assess wind
      direction model performance.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 1: Perfect agreement between observed and modeled wind directions
    - 0: No agreement (as bad as using the mean of observations)

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Modeled wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Wind direction index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import WDIOA_m
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([345, 15, 25])
    >>> WDIOA_m(obs, mod)
    0.8
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obsmean = obs.mean(dim=dim)
        num = (abs(circlebias_m(obs - mod))).sum(dim=dim)
        denom = (abs(circlebias_m(mod - obsmean)) + abs(circlebias_m(obs - obsmean))).sum(dim=dim)

        result = 1.0 - (num / denom)
        result = xr.where(denom == 0, 1.0, result)

        # Update history
        history = f"WDIOA_m computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obsmean = np.ma.mean(obs, axis=axis)
        num = np.ma.sum(np.ma.abs(circlebias_m(np.subtract(obs, mod))), axis=axis)
        denom = np.ma.sum(
            np.ma.abs(circlebias_m(np.subtract(mod, obsmean))) + np.ma.abs(circlebias_m(np.subtract(obs, obsmean))),
            axis=axis,
        )
        result = np.where(denom == 0, 1.0, 1.0 - (num / denom))
        return result.item() if np.ndim(result) == 0 else result


def WDIOA(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Index of Agreement (WDIOA).

    Standard version.

    Typical Use Cases
    -----------------
    - Quantifying the agreement between observed and modeled wind directions,
      accounting for circularity.
    - Used in wind energy, meteorology, and air quality studies to assess wind
      direction model performance.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 1: Perfect agreement between observed and modeled wind directions
    - 0: No agreement (as bad as using the mean of observations)

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Modeled wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Wind direction index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import WDIOA
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([345, 15, 25])
    >>> WDIOA(obs, mod)
    0.8
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        num = abs(circlebias(obs - mod)).sum(dim=dim)
        mean_obs = obs.mean(dim=dim)
        denom = (abs(circlebias(mod - mean_obs)) + abs(circlebias(obs - mean_obs))).sum(dim=dim)

        result = 1.0 - (num / denom)
        result = xr.where(denom == 0, 1.0, result)

        # Update history
        history = f"WDIOA computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        num = np.ma.sum(np.ma.abs(circlebias(np.subtract(obs, mod))), axis=axis)
        mean_obs = np.ma.mean(obs, axis=axis)
        denom = np.ma.sum(
            np.ma.abs(circlebias(np.subtract(mod, mean_obs))) + np.ma.abs(circlebias(np.subtract(obs, mean_obs))),
            axis=axis,
        )
        result = np.where(denom == 0, 1.0, 1.0 - (num / denom))
        return result.item() if np.ndim(result) == 0 else result


def AC(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Anomaly Correlation (AC).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Anomaly correlation coefficient (unitless, -1 to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import AC
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> AC(obs, mod)
    0.9922778767136677
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_bar = obs.mean(dim=dim)
        mod_bar = mod.mean(dim=dim)
        obs_anom = obs - obs_bar
        mod_anom = mod - mod_bar
        p1 = (mod_anom * obs_anom).sum(dim=dim)
        p2 = ((mod_anom**2).sum(dim=dim) * (obs_anom**2).sum(dim=dim)) ** 0.5
        result = p1 / p2
        # Update history
        history = f"AC computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obs_bar = np.ma.mean(obs, axis=axis)
        mod_bar = np.ma.mean(mod, axis=axis)
        if axis is not None:
            # Need to keep dims for subtraction if axis is not None
            obs_bar_kd = np.ma.mean(obs, axis=axis, keepdims=True)
            mod_bar_kd = np.ma.mean(mod, axis=axis, keepdims=True)
        else:
            obs_bar_kd = obs_bar
            mod_bar_kd = mod_bar
        obs_anom = np.subtract(obs, obs_bar_kd)
        mod_anom = np.subtract(mod, mod_bar_kd)
        p1 = np.ma.sum(np.ma.multiply(mod_anom, obs_anom), axis=axis)
        p2 = np.ma.sqrt(np.ma.multiply(np.ma.sum(obs_anom**2, axis=axis), np.ma.sum(mod_anom**2, axis=axis)))
        return p1 / p2


def WDAC(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Anomaly Correlation (WDAC).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Modeled wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        WDAC value(s).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import WDAC
    >>> obs = np.array([350, 10, 20])
    >>> mod = np.array([345, 15, 25])
    >>> WDAC(obs, mod)
    0.9992386127814763
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_rad = obs * np.pi / 180.0
        mod_rad = mod * np.pi / 180.0
        obs_anom = obs_rad - obs_rad.mean(dim=dim)
        mod_anom = mod_rad - mod_rad.mean(dim=dim)
        numerator = (np.sin(obs_anom) * np.sin(mod_anom)).sum(dim=dim)
        denominator = np.sqrt((np.sin(obs_anom) ** 2).sum(dim=dim) * (np.sin(mod_anom) ** 2).sum(dim=dim))
        result = numerator / denominator
        # Update history
        history = f"WDAC computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obs_rad = np.deg2rad(obs)
        mod_rad = np.deg2rad(mod)
        if axis is not None:
            obs_bar_rad = np.ma.mean(obs_rad, axis=axis, keepdims=True)
            mod_bar_rad = np.ma.mean(mod_rad, axis=axis, keepdims=True)
        else:
            obs_bar_rad = np.ma.mean(obs_rad)
            mod_bar_rad = np.ma.mean(mod_rad)

        obs_anom = obs_rad - obs_bar_rad
        mod_anom = mod_rad - mod_bar_rad
        numerator = np.ma.sum(np.sin(obs_anom) * np.sin(mod_anom), axis=axis)
        denominator = np.ma.sqrt(
            np.ma.sum(np.sin(obs_anom) ** 2, axis=axis) * np.ma.sum(np.sin(mod_anom) ** 2, axis=axis)
        )
        return numerator / denominator


def taylor_skill(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Taylor Skill Score (TSS).

    Typical Use Cases
    -----------------
    - Summarizing model performance in a single skill score for use in Taylor
      diagrams.
    - Used in climate, weather, and environmental model evaluation.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 1: Perfect agreement between model and observations
    - 0: No skill

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the skill score.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Taylor skill score (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import taylor_skill
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([1.1, 1.9, 3.2])
    >>> taylor_skill(obs, mod)
    0.9995574044955781
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        std_obs = obs.std(dim=dim)
        std_mod = mod.std(dim=dim)
        corr = xr.corr(obs, mod, dim=dim)

        # Calculate Taylor Skill Score using the common formula
        # S = 4 * (1 + R) / ( (sigma_p/sigma_o + sigma_o/sigma_p)^2 * (1 + R_max) )
        # Assuming R_max = 1.0
        norm_std = std_mod / std_obs
        result = (4.0 * (corr + 1.0)) / ((norm_std + 1.0 / norm_std) ** 2 * 2.0)
        # Update history
        history = f"taylor_skill computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        std_obs = np.ma.std(obs, axis=axis)
        std_mod = np.ma.std(mod, axis=axis)
        from scipy.stats import pearsonr

        if axis is None:
            if np.ma.is_masked(obs):
                corr = pearsonr(obs.compressed(), mod.compressed())[0]
            else:
                corr = pearsonr(obs, mod)[0]
        else:
            # Vectorized correlation over axis for numpy
            obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
            mod_mean = np.nanmean(mod, axis=axis, keepdims=True)
            obs_anom = obs - obs_mean
            mod_anom = mod - mod_mean
            num_corr = np.nansum(obs_anom * mod_anom, axis=axis)
            den_corr = np.sqrt(np.nansum(obs_anom**2, axis=axis) * np.nansum(mod_anom**2, axis=axis))
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = num_corr / den_corr

        norm_std = std_mod / std_obs
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (4.0 * (corr + 1.0)) / ((norm_std + 1.0 / norm_std) ** 2 * 2.0)
            result = np.where(np.isnan(result) | np.isinf(result), 1.0, result)
        return result.item() if np.ndim(result) == 0 else result


def KGE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Kling-Gupta Efficiency (KGE).

    Typical Use Cases
    -----------------
    - Quantifying the overall agreement between model and observations,
      combining correlation, bias, and variability.
    - Used in hydrology, meteorology, and environmental model evaluation.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect agreement between model and observations
    - 0: Moderate skill
    - Negative values: Poor skill

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute KGE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Kling-Gupta efficiency (unitless, -∞ to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import KGE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([1.1, 1.9, 3.2])
    >>> KGE(obs, mod)
    0.8988771192996924
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        r = xr.corr(obs, mod, dim=dim)
        alpha = mod.std(dim=dim) / obs.std(dim=dim)
        beta = mod.mean(dim=dim) / obs.mean(dim=dim)
        result = 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5
        # Update history
        history = f"KGE computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        from scipy.stats import pearsonr

        if np.ma.is_masked(obs):
            r = pearsonr(obs.compressed(), mod.compressed())[0]
        else:
            r = pearsonr(obs, mod)[0]
        alpha = np.ma.std(mod, axis=axis) / np.ma.std(obs, axis=axis)
        beta = np.ma.mean(mod, axis=axis) / np.ma.mean(obs, axis=axis)
        return 1.0 - ((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) ** 0.5


def pearsonr(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Pearson correlation coefficient.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension name along which to compute the coefficient.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Pearson correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import pearsonr
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 4, 6])
    >>> pearsonr(obs, mod)
    1.0
    """
    from scipy.stats import pearsonr as _pearsonr

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is None:
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        def _pearsonr_onlyr(a, b):
            mask = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(mask) < 2 or np.var(a[mask]) == 0 or np.var(b[mask]) == 0:
                return np.nan
            return _pearsonr(a[mask], b[mask])[0]

        result = xr.apply_ufunc(
            _pearsonr_onlyr,
            obs,
            mod,
            input_core_dims=[[dim] if isinstance(dim, str) else list(dim)] * 2,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Update history
        history = f"pearsonr computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        if axis is None:
            obsc, modc = matchedcompressed(obs, mod)
            if len(obsc) < 2 or np.var(obsc) == 0 or np.var(modc) == 0:
                return 0.0
            r_val, _ = _pearsonr(obsc, modc)
            return r_val if not np.isnan(r_val) else 0.0
        else:
            # For numpy with axis, use manual vectorized correlation
            obs_mean = np.nanmean(obs, axis=axis, keepdims=True)
            mod_mean = np.nanmean(mod, axis=axis, keepdims=True)
            obs_std = obs - obs_mean
            mod_std = mod - mod_mean
            num = np.nansum(obs_std * mod_std, axis=axis)
            den = np.sqrt(np.nansum(obs_std**2, axis=axis) * np.nansum(mod_std**2, axis=axis))
            with np.errstate(divide="ignore", invalid="ignore"):
                result = num / den
                return result.item() if np.ndim(result) == 0 else result


def spearmanr(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Spearman rank correlation coefficient.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the coefficient.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Spearman rank correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import spearmanr
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> spearmanr(obs, mod)
    0.8660254037844387
    """
    from scipy.stats import spearmanr as _spearmanr

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is None:
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        def _spearmanr_onlyrho(a, b):
            mask = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(mask) < 2:
                return np.nan
            return _spearmanr(a[mask], b[mask])[0]

        result = xr.apply_ufunc(
            _spearmanr_onlyrho,
            obs,
            mod,
            input_core_dims=[[dim] if isinstance(dim, str) else list(dim)] * 2,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Update history
        history = f"spearmanr computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        if axis is None:
            obsc, modc = matchedcompressed(obs, mod)
            if len(obsc) < 2:
                return np.nan
            return _spearmanr(obsc, modc)[0]
        else:
            # Fallback for numpy with axis: manual loop over other axes
            obs = np.asarray(obs)
            mod = np.asarray(mod)
            if axis < 0:
                axis = obs.ndim + axis

            # Move axis to last position
            obs_moved = np.moveaxis(obs, axis, -1)
            mod_moved = np.moveaxis(mod, axis, -1)

            # Reshape all other axes into one
            other_shape = obs_moved.shape[:-1]
            obs_flat = obs_moved.reshape(-1, obs_moved.shape[-1])
            mod_flat = mod_moved.reshape(-1, mod_moved.shape[-1])

            results = []
            for i in range(len(obs_flat)):
                mask = ~np.isnan(obs_flat[i]) & ~np.isnan(mod_flat[i])
                if np.sum(mask) < 2:
                    results.append(np.nan)
                else:
                    results.append(_spearmanr(obs_flat[i][mask], mod_flat[i][mask])[0])

            return np.array(results).reshape(other_shape)


def kendalltau(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Kendall rank correlation coefficient.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension name along which to compute the coefficient.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Kendall rank correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import kendalltau
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> kendalltau(obs, mod)
    1.0
    """
    from scipy.stats import kendalltau as _kendalltau

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is None:
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        def _kendalltau_onlytau(a, b):
            mask = ~np.isnan(a) & ~np.isnan(b)
            if np.sum(mask) < 2:
                return np.nan
            return _kendalltau(a[mask], b[mask])[0]

        result = xr.apply_ufunc(
            _kendalltau_onlytau,
            obs,
            mod,
            input_core_dims=[[dim] if isinstance(dim, str) else list(dim)] * 2,
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # Update history
        history = f"kendalltau computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        if axis is None:
            obsc, modc = matchedcompressed(obs, mod)
            if len(obsc) < 2:
                return np.nan
            return _kendalltau(obsc, modc)[0]
        else:
            # Fallback for numpy with axis: manual loop over other axes
            obs = np.asarray(obs)
            mod = np.asarray(mod)
            if axis < 0:
                axis = obs.ndim + axis

            obs_moved = np.moveaxis(obs, axis, -1)
            mod_moved = np.moveaxis(mod, axis, -1)

            other_shape = obs_moved.shape[:-1]
            obs_flat = obs_moved.reshape(-1, obs_moved.shape[-1])
            mod_flat = mod_moved.reshape(-1, mod_moved.shape[-1])

            results = []
            for i in range(len(obs_flat)):
                mask = ~np.isnan(obs_flat[i]) & ~np.isnan(mod_flat[i])
                if np.sum(mask) < 2:
                    results.append(np.nan)
                else:
                    results.append(_kendalltau(obs_flat[i][mask], mod_flat[i][mask])[0])

            return np.array(results).reshape(other_shape)


def CCC(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Concordance Correlation Coefficient (CCC).

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, accounting for
      precision and accuracy.
    - Used in model evaluation to assess how well model predictions agree with
      observations.
    - Measures how far the values deviate from the line of perfect concordance
      (slope=1, intercept=0).

    Typical Values and Range
    ------------------------
    - Range: -1 to 1
    - 1: Perfect agreement between model and observations
    - 0: No agreement
    - -1: Perfect negative agreement

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the coefficient.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Concordance correlation coefficient (unitless, -1 to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import CCC
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> CCC(obs, mod)
    0.9984779299847792
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        # Calculate means
        obs_mean = obs.mean(dim=dim)
        mod_mean = mod.mean(dim=dim)

        # Calculate variances and covariance
        obs_var = obs.var(dim=dim)
        mod_var = mod.var(dim=dim)
        covar = ((obs - obs_mean) * (mod - mod_mean)).mean(dim=dim)

        # Calculate CCC
        numerator = 2 * covar
        denominator = obs_var + mod_var + (obs_mean - mod_mean) ** 2
        result = numerator / denominator
        # Update history
        history = f"CCC computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        # Calculate means
        obs_mean = np.nanmean(obs, axis=axis)
        mod_mean = np.nanmean(mod, axis=axis)

        # Calculate variances and covariance
        obs_var = np.nanvar(obs, axis=axis)
        mod_var = np.nanvar(mod, axis=axis)
        if axis is not None:
            obs_mean_kd = np.nanmean(obs, axis=axis, keepdims=True)
            mod_mean_kd = np.nanmean(mod, axis=axis, keepdims=True)
        else:
            obs_mean_kd = obs_mean
            mod_mean_kd = mod_mean
        covar = np.nanmean((obs - obs_mean_kd) * (mod - mod_mean_kd), axis=axis)

        # Calculate CCC
        numerator = 2 * covar
        denominator = obs_var + mod_var + (obs_mean - mod_mean) ** 2
        return numerator / denominator


def E1_prime(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Modified Coefficient of Efficiency (E1') - Alternative formulation.

    Typical Use Cases
    -----------------
    - Quantifying the efficiency of model predictions relative to observed mean,
      robust to outliers.
    - Used in hydrology, meteorology, and model skill assessment as an
      alternative to E1.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Modified coefficient of efficiency (unitless, -inf to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import E1_prime
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> E1_prime(obs, mod)
    0.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_mean = obs.mean(dim=dim)
        num = abs(obs - mod).sum(dim=dim)
        denom = abs(obs - obs_mean).sum(dim=dim)
        # Handle case where denominator is 0
        result = 1.0 - (num / denom)
        result = xr.where((num == 0) & (denom == 0), 1.0, result)
        result = xr.where((num != 0) & (denom == 0), -np.inf, result)

        # Update history
        history = f"E1_prime computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obs_c, mod_c = matchedcompressed(obs, mod)
        obs_mean = np.nanmean(obs_c, axis=axis)
        if axis is not None:
            obs_mean_kd = np.nanmean(obs_c, axis=axis, keepdims=True)
        else:
            obs_mean_kd = obs_mean
        num = np.nansum(np.abs(obs_c - mod_c), axis=axis)
        denom = np.nansum(np.abs(obs_c - obs_mean_kd), axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
            if np.ndim(result) == 0:
                if num == 0 and denom == 0:
                    result = np.array(1.0)
                elif denom == 0:
                    result = np.array(-np.inf)
            else:
                result = np.where((num == 0) & (denom == 0), 1.0, result)
                result = np.where((num != 0) & (denom == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result


def IOA_prime(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Index of Agreement (IOA') - Alternative formulation.

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, normalized by
      total deviation.
    - Used in model evaluation for skill assessment as an alternative to IOA.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the statistic.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.correlation_metrics import IOA_prime
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> IOA_prime(obs, mod)
    0.8
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obsmean = obs.mean(dim=dim)
        num = ((obs - mod) ** 2).sum(dim=dim)
        denom = ((abs(mod - obsmean) + abs(obs - obsmean)) ** 2).sum(dim=dim)
        # Handle case where denominator is 0
        result = 1.0 - (num / denom)
        result = xr.where((num == 0) & (denom == 0), 1.0, result)
        result = xr.where((num != 0) & (denom == 0), -np.inf, result)

        # Update history
        history = f"IOA_prime computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
    else:
        obs_c, mod_c = matchedcompressed(obs, mod)
        obsmean = np.nanmean(obs_c, axis=axis)
        if axis is not None:
            obsmean_kd = np.nanmean(obs_c, axis=axis, keepdims=True)
        else:
            obsmean_kd = obsmean
        num = np.nansum((obs_c - mod_c) ** 2, axis=axis)
        denom = np.nansum((np.abs(mod_c - obsmean_kd) + np.abs(obs_c - obsmean_kd)) ** 2, axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
            if np.ndim(result) == 0:
                if num == 0 and denom == 0:
                    result = np.array(1.0)
                elif denom == 0:
                    result = np.array(-np.inf)
            else:
                result = np.where((num == 0) & (denom == 0), 1.0, result)
                result = np.where((num != 0) & (denom == 0), -np.inf, result)
        return result.item() if np.ndim(result) == 0 else result
