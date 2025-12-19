"""
Spatial Skill Metrics for Model Evaluation
"""

from typing import Any, Optional

import numpy as np
import xarray as xr


def FSS(
    obs: xr.DataArray,
    mod: xr.DataArray,
    threshold: float,
    window_size: int,
) -> Any:
    """
    Fractions Skill Score (FSS).

    Typical Use Cases
    -----------------
    - Evaluating the spatial skill of a forecast, particularly for high-resolution models.
    - Assessing forecast performance at different spatial scales by varying the window size.
    - Useful for precipitation, convection, and other spatially-defined events.

    Parameters
    ----------
    obs : xr.DataArray
        Observed values.
    mod : xr.DataArray
        Model or predicted values.
    threshold : float
        The threshold to define an event.
    window_size : int
        The size of the square window for calculating fractions.

    Returns
    -------
    fss : float or ndarray
        The Fractions Skill Score, ranging from 0 (no skill) to 1 (perfect skill).
    """
    obs, mod = xr.align(obs, mod, join="inner")
    obs_binary = (obs >= threshold).astype(float)
    mod_binary = (mod >= threshold).astype(float)

    # Use xarray's rolling window operation, which is Dask-compatible
    obs_frac = obs_binary.rolling(dim={d: window_size for d in obs.dims}, center=True).mean()
    mod_frac = mod_binary.rolling(dim={d: window_size for d in mod.dims}, center=True).mean()

    mse = ((obs_frac - mod_frac) ** 2).mean()
    mse_ref = (obs_frac**2).mean() + (mod_frac**2).mean()

    # Using xr.where to handle potential division by zero in a Dask-friendly way
    return xr.where(mse_ref == 0, 1.0, 1 - (mse / mse_ref))


def VETS(obs: xr.DataArray, mod: xr.DataArray, axis: Optional[int] = None) -> Any:
    """
    Volumetric Equitable Threat Score (VETS).

    Typical Use Cases
    -----------------
    - Evaluating the skill of volumetric forecasts, such as precipitation accumulation.
    - Provides an equitable score that accounts for random chance, making it suitable for rare events.

    Parameters
    ----------
    obs : xr.DataArray
        Observed values.
    mod : xr.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute the score.

    Returns
    -------
    vets : float or ndarray
        The Volumetric Equitable Threat Score.
    """
    obs, mod = xr.align(obs, mod, join="inner")

    # Use xr.ufuncs for element-wise operations to stay within xarray/dask ecosystem
    hits = xr.ufuncs.minimum(obs, mod).sum(dim=axis)
    sum_obs = obs.sum(dim=axis)
    sum_mod = mod.sum(dim=axis)
    misses = sum_obs - hits
    false_alarms = sum_mod - hits
    total_union = xr.ufuncs.maximum(obs, mod).sum(dim=axis)
    hits_random = (sum_obs * sum_mod) / total_union
    denominator = hits + misses + false_alarms - hits_random

    # Use xr.where for safe division
    return xr.where(denominator == 0, 1.0, (hits - hits_random) / denominator)
