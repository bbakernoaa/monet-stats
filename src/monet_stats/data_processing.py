"""
Data processing utilities for statistical computations.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


def to_numpy(
    data: Union[np.ndarray, xr.DataArray, pd.Series, pd.DataFrame, list],
) -> np.ndarray:
    """
    Convert data to numpy array.

    Parameters
    ----------
    data : array-like or xarray.DataArray or pandas Series/DataFrame or list
        Input data to convert.

    Returns
    -------
    numpy.ndarray
        Converted numpy array.
    """
    if isinstance(data, xr.DataArray) or isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        return np.asarray(data.values)
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.asarray(data)


def align_arrays(obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray]) -> Tuple:
    """
    Align two arrays for comparison.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model/predicted values.

    Returns
    -------
    tuple
        Aligned obs and mod arrays.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        return obs, mod
    else:
        # For numpy arrays, ensure they have the same shape
        obs = to_numpy(obs)
        mod = to_numpy(mod)

        if obs.shape != mod.shape:
            raise ValueError(f"Arrays must have the same shape, got {obs.shape} and {mod.shape}")

        return obs, mod


def handle_missing_values(
    obs: np.ndarray, mod: np.ndarray, strategy: str = "pairwise"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values in arrays.

    Parameters
    ----------
    obs : numpy.ndarray
        Observed values.
    mod : numpy.ndarray
        Model/predicted values.
    strategy : str, optional
        Strategy for handling missing values ('pairwise', 'listwise').

    Returns
    -------
    tuple of numpy.ndarray
        Arrays with missing values handled according to strategy.
    """
    if strategy == "pairwise":
        # Remove pairs where either value is NaN
        mask = ~(np.isnan(obs) | np.isnan(mod))
        return obs[mask], mod[mask]
    elif strategy == "listwise":
        # Remove all pairs if any value is NaN
        mask = ~(np.isnan(obs) | np.isnan(mod))
        return obs[mask], mod[mask]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def normalize_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "zscore",
) -> Tuple:
    """
    Normalize data using various methods.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model/predicted values.
    method : str, optional
        Normalization method ('zscore', 'minmax', 'robust').

    Returns
    -------
    tuple
        Normalized obs and mod arrays.
    """
    obs, mod = align_arrays(obs, mod)

    if method == "zscore":
        # Z-score normalization (mean=0, std=1)
        if isinstance(obs, xr.DataArray):
            obs_mean = float(obs.mean().values)
            obs_std = float(obs.std().values)
            mod_mean = float(mod.mean().values)
            mod_std = float(mod.std().values)

            obs_norm = (obs - obs_mean) / obs_std
            mod_norm = (mod - mod_mean) / mod_std
        else:
            obs_mean = float(np.mean(obs))
            obs_std = float(np.std(obs))
            mod_mean = float(np.mean(mod))
            mod_std = float(np.std(mod))

            obs_norm = (obs - obs_mean) / obs_std
            mod_norm = (mod - mod_mean) / mod_std
    elif method == "minmax":
        # Min-max normalization (range [0, 1])
        if isinstance(obs, xr.DataArray):
            obs_min = float(obs.min().values)
            obs_max = float(obs.max().values)
            mod_min = float(mod.min().values)
            mod_max = float(mod.max().values)

            obs_norm = (obs - obs_min) / (obs_max - obs_min)
            mod_norm = (mod - mod_min) / (mod_max - mod_min)
        else:
            obs_min = float(np.min(obs))
            obs_max = float(np.max(obs))
            mod_min = float(np.min(mod))
            mod_max = float(np.max(mod))

            obs_norm = (obs - obs_min) / (obs_max - obs_min)
            mod_norm = (mod - mod_min) / (mod_max - mod_min)
    elif method == "robust":
        # Robust normalization using median and MAD
        if isinstance(obs, xr.DataArray):
            obs_median = float(obs.median().values)
            obs_mad = float(np.median(np.abs(obs.values - obs_median)))
            mod_median = float(mod.median().values)
            mod_mad = float(np.median(np.abs(mod.values - mod_median)))

            obs_norm = (obs - obs_median) / obs_mad
            mod_norm = (mod - mod_median) / mod_mad
        else:
            obs_median = float(np.median(obs))
            obs_mad = float(np.median(np.abs(obs - obs_median)))
            mod_median = float(np.median(mod))
            mod_mad = float(np.median(np.abs(mod - mod_median)))

            obs_norm = (obs - obs_median) / obs_mad
            mod_norm = (mod - mod_median) / mod_mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return obs_norm, mod_norm


def detrend_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "linear",
) -> Tuple:
    """
    Remove trend from data.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model/predicted values.
    method : str, optional
        Detrending method ('linear', 'constant').

    Returns
    -------
    tuple
        Detrended obs and mod arrays.
    """
    obs, mod = align_arrays(obs, mod)

    if method == "linear":
        if isinstance(obs, xr.DataArray):
            # For xarray, use scipy's detrend function
            from scipy.signal import detrend

            # Simple approach: convert to numpy, detrend, then back to xarray
            obs_np = np.asarray(obs.values)
            mod_np = np.asarray(mod.values)
            obs_detrended = detrend(obs_np, axis=-1)
            mod_detrended = detrend(mod_np, axis=-1)
            # Convert back to xarray if original was xarray
            obs_detrended = xr.DataArray(obs_detrended, coords=obs.coords, dims=obs.dims)
            mod_detrended = xr.DataArray(mod_detrended, coords=mod.coords, dims=mod.dims)
        else:
            from scipy.signal import detrend

            obs_detrended = detrend(obs, axis=-1)
            mod_detrended = detrend(mod, axis=-1)
    elif method == "constant":
        # Remove mean
        if isinstance(obs, xr.DataArray):
            obs_mean = float(obs.mean().values)
            mod_mean = float(mod.mean().values)
            obs_detrended = obs - obs_mean
            mod_detrended = mod - mod_mean
        else:
            obs_mean = float(np.mean(obs))
            mod_mean = float(np.mean(mod))
            obs_detrended = obs - obs_mean
            mod_detrended = mod - mod_mean
    else:
        raise ValueError(f"Unknown detrending method: {method}")

    return obs_detrended, mod_detrended


def compute_anomalies(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    reference_period: Optional[Tuple[int, int]] = None,
) -> Tuple:
    """
    Compute anomalies relative to a reference period.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model/predicted values.
    reference_period : tuple of int, optional
        Reference period as (start_year, end_year) for computing climatology.

    Returns
    -------
    tuple
        Anomalies for obs and mod arrays.
    """
    obs, mod = align_arrays(obs, mod)

    if isinstance(obs, xr.DataArray):
        # For xarray, we can potentially use time dimensions
        if reference_period is not None:
            # In a real implementation, we would filter for reference period
            # For now, we'll just use overall mean as climatology
            obs_clim = float(obs.mean().values)
            mod_clim = float(mod.mean().values)
        else:
            # Use overall mean as climatology
            obs_clim = float(obs.mean().values)
            mod_clim = float(mod.mean().values)

        # Compute anomalies
        obs_anom = obs - obs_clim
        mod_anom = mod - mod_clim
    else:
        # For numpy arrays, use overall mean as climatology
        obs_clim = float(np.mean(obs))
        mod_clim = float(np.mean(mod))

        # Compute anomalies
        obs_anom = obs - obs_clim
        mod_anom = mod - mod_clim

    return obs_anom, mod_anom
