"""
Spatial Skill Metrics for Model Evaluation (Aero Protocol Compliant)
"""

from typing import Optional, Union

import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter

from .utils_stats import _update_history


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
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Fractions Skill Score (FSS).

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
        The threshold to define an event. If None, uses mean of observations.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        The Fractions Skill Score, ranging from 0 (no skill) to 1 (perfect skill).
    """
    if threshold is None:
        threshold = np.nanmean(obs)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_binary = (obs >= threshold).astype(float)
        mod_binary = (mod >= threshold).astype(float)

        # Rolling mean for fractions
        obs_frac = obs_binary.rolling(dim={d: window_size for d in obs.dims}, center=True).mean()
        mod_frac = mod_binary.rolling(dim={d: window_size for d in mod.dims}, center=True).mean()

        mse = ((obs_frac - mod_frac) ** 2).mean()
        mse_ref = (obs_frac**2).mean() + (mod_frac**2).mean()

        res = xr.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))
        return _update_history(res, "Fractions Skill Score (FSS)")

    obs_binary = (np.asarray(obs) >= threshold).astype(float)
    mod_binary = (np.asarray(mod) >= threshold).astype(float)

    obs_frac = _uniform_filter(obs_binary, window_size)
    mod_frac = _uniform_filter(mod_binary, window_size)

    mse = np.nanmean((obs_frac - mod_frac) ** 2)
    mse_ref = np.nanmean(obs_frac**2) + np.nanmean(mod_frac**2)

    if mse_ref == 0:
        return 1.0
    return 1 - (mse / mse_ref)


def VETS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    axis: Optional[Union[int, str]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Volumetric Equitable Threat Score (VETS).

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
    axis : int or str, optional
        Axis along which to compute the score.

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        The Volumetric Equitable Threat Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Use xr.where for minimum/maximum to be safer with dask
        hits = xr.where(obs < mod, obs, mod).sum(dim=axis)
        sum_obs = obs.sum(dim=axis)
        sum_mod = mod.sum(dim=axis)
        misses = sum_obs - hits
        false_alarms = sum_mod - hits
        total_union = xr.where(obs > mod, obs, mod).sum(dim=axis)
        hits_random = (sum_obs * sum_mod) / total_union
        denominator = hits + misses + false_alarms - hits_random

        res = xr.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
        return _update_history(res, "Volumetric Equitable Threat Score (VETS)")

    obs_arr = np.asarray(obs)
    mod_arr = np.asarray(mod)
    hits = np.sum(np.minimum(obs_arr, mod_arr), axis=axis)
    sum_obs = np.sum(obs_arr, axis=axis)
    sum_mod = np.sum(mod_arr, axis=axis)
    misses = sum_obs - hits
    false_alarms = sum_mod - hits
    total_union = np.sum(np.maximum(obs_arr, mod_arr), axis=axis)
    hits_random = (sum_obs * sum_mod) / total_union
    denominator = hits + misses + false_alarms - hits_random
    if denominator == 0:
        return 1.0
    return (hits - hits_random) / denominator
