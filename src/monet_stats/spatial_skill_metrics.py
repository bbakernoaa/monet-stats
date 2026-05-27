"""
Spatial Skill Metrics for Model Evaluation (Aero Protocol Compliant)
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from .utils_stats import _resolve_axis_to_dim, _update_history


def _uniform_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a uniform filter to the data.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    window_size : int
        The size of the filter window.

    Returns
    -------
    np.ndarray
        The filtered data.
    """
    return uniform_filter(data.astype(float), size=window_size)


def FSS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    window_size: int = 3,
    threshold: Optional[float] = None,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Fractions Skill Score (FSS) (Aero Protocol).

    Typical Use Cases
    -----------------
    - Evaluating the spatial skill of a forecast, particularly for high-resolution models.
    - Assessing forecast performance at different spatial scales by varying the window size.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    window_size : int, optional
        The size of the square window for calculating fractions, by default 3.
    threshold : float, optional
        The threshold to define an event. If None, uses mean of observations (Lazy-friendly).
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the fractions (xarray only).
        If None, uses all dimensions.
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the fractions.

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Fractions Skill Score, ranging from 0 (no skill) to 1 (perfect skill).

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> obs = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> mod = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> score = FSS(obs, mod, window_size=3, threshold=0.5)
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")

        if threshold is None:
            # Use xarray's native mean to maintain laziness
            threshold = obs.mean()

        obs_binary = (obs >= threshold).astype(float)
        mod_binary = (mod >= threshold).astype(float)

        # Rolling mean for fractions
        # Resolve dimensions using Aero Protocol utility
        roll_dims = _resolve_axis_to_dim(obs, dim if dim is not None else axis)

        if isinstance(roll_dims, str):
            roll_dims = [roll_dims]

        roll_kwargs = {d: window_size for d in roll_dims}
        obs_frac = obs_binary.rolling(dim=roll_kwargs, center=True).mean()
        mod_frac = mod_binary.rolling(dim=roll_kwargs, center=True).mean()

        mse = ((obs_frac - mod_frac) ** 2).mean(dim=roll_dims)
        mse_ref = (obs_frac**2).mean(dim=roll_dims) + (mod_frac**2).mean(dim=roll_dims)

        res = xr.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))
        return _update_history(res, "Fractions Skill Score (FSS)")

    # NumPy path
    obs_arr = np.asarray(obs)
    mod_arr = np.asarray(mod)

    if threshold is None:
        threshold = np.nanmean(obs_arr)

    obs_binary = (obs_arr >= threshold).astype(float)
    mod_binary = (mod_arr >= threshold).astype(float)

    obs_frac = _uniform_filter(obs_binary, window_size)
    mod_frac = _uniform_filter(mod_binary, window_size)

    # Use specified axis for mean calculation if provided
    mse = np.nanmean((obs_frac - mod_frac) ** 2, axis=axis)
    mse_ref = np.nanmean(obs_frac**2, axis=axis) + np.nanmean(mod_frac**2, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))
        return res.item() if np.ndim(res) == 0 else res


def VETS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Volumetric Equitable Threat Score (VETS) (Aero Protocol).

    Typical Use Cases
    -----------------
    - Evaluating the skill of volumetric forecasts, such as precipitation accumulation.
    - Provides an equitable score that accounts for random chance.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the score (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the score.

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Volumetric Equitable Threat Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")

        # Resolve reduction dimensions
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)

        # Use xr.where for minimum/maximum to be safer with dask
        hits = xr.where(obs < mod, obs, mod).sum(dim=reduction_dim)
        sum_obs = obs.sum(dim=reduction_dim)
        sum_mod = mod.sum(dim=reduction_dim)
        misses = sum_obs - hits
        false_alarms = sum_mod - hits
        total_union = xr.where(obs > mod, obs, mod).sum(dim=reduction_dim)

        hits_random = (sum_obs * sum_mod) / total_union
        denominator = hits + misses + false_alarms - hits_random

        res = xr.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
        return _update_history(res, "Volumetric Equitable Threat Score (VETS)")

    # NumPy path
    from .utils_stats import _nanmask_inputs

    o_, m_ = _nanmask_inputs(obs, mod)
    obs_arr = o_.filled(np.nan)
    mod_arr = m_.filled(np.nan)

    hits = np.nansum(np.minimum(obs_arr, mod_arr), axis=axis)
    sum_obs = np.nansum(obs_arr, axis=axis)
    sum_mod = np.nansum(mod_arr, axis=axis)
    misses = sum_obs - hits
    false_alarms = sum_mod - hits
    total_union = obs_sum + mod_sum - hits

    hits_random = (sum_obs * sum_mod) / total_union
    denominator = hits + misses + false_alarms - hits_random

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
        return res.item() if np.ndim(res) == 0 else res
