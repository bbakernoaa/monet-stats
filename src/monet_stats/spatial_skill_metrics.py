"""
Spatial Skill Metrics for Model Evaluation following the Aero Protocol.
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
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    threshold: float,
    window_size: int,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Fractions Skill Score (FSS).

    Typical Use Cases
    -----------------
    - Evaluating the spatial skill of a forecast, particularly for high-resolution models.
    - Assessing forecast performance at different spatial scales by varying the window size.
    - Useful for precipitation, convection, and other spatially-defined events.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    threshold : float
        The threshold to define an event.
    window_size : int
        The size of the square window for calculating fractions.

    Returns
    -------
    fss : float, numpy.ndarray, or xarray.DataArray
        The Fractions Skill Score, ranging from 0 (no skill) to 1 (perfect skill).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_binary = (obs >= threshold).astype(float)
        mod_binary = (mod >= threshold).astype(float)

        # rolling handles Dask arrays
        obs_frac = obs_binary.rolling(dim={d: window_size for d in obs.dims}, center=True).mean()
        mod_frac = mod_binary.rolling(dim={d: window_size for d in mod.dims}, center=True).mean()

        mse = ((obs_frac - mod_frac) ** 2).mean()
        mse_ref = (obs_frac**2).mean() + (mod_frac**2).mean()

        result = xr.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))
        return _update_history(result, "FSS")
    else:
        obs_binary = (np.asarray(obs) >= threshold).astype(float)
        mod_binary = (np.asarray(mod) >= threshold).astype(float)

        obs_frac = _uniform_filter(obs_binary, window_size)
        mod_frac = _uniform_filter(mod_binary, window_size)

        mse = np.mean((obs_frac - mod_frac) ** 2)
        mse_ref = np.mean(obs_frac**2) + np.mean(mod_frac**2)

        if mse_ref == 0:
            return 1.0
        else:
            return 1 - (mse / mse_ref)


def VETS(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Volumetric Equitable Threat Score (VETS).

    Typical Use Cases
    -----------------
    - Evaluating the skill of volumetric forecasts, such as precipitation accumulation.
    - Provides an equitable score that accounts for random chance, making it suitable for rare events.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int or str, optional
        Axis along which to compute the score.

    Returns
    -------
    vets : float, numpy.ndarray, or xarray.DataArray
        The Volumetric Equitable Threat Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        hits = xr.where(obs < mod, obs, mod).sum(dim=dim)
        sum_obs = obs.sum(dim=dim)
        sum_mod = mod.sum(dim=dim)
        misses = sum_obs - hits
        false_alarms = sum_mod - hits
        total_union = xr.where(obs > mod, obs, mod).sum(dim=dim)
        hits_random = (sum_obs * sum_mod) / total_union
        denominator = hits + misses + false_alarms - hits_random

        result = xr.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
        return _update_history(result, "VETS")
    else:
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
