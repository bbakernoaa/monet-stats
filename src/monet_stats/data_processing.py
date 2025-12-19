"""
Data processing utilities for statistical computations.
"""

from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr


def to_numpy(
    data: Union[np.ndarray, xr.DataArray, list],
) -> np.ndarray:
    """
    Convert data to numpy array.

    Parameters
    ----------
    data : array-like or xarray.DataArray or list
        Input data to convert.

    Returns
    -------
    numpy.ndarray
        Converted numpy array.
    """
    if isinstance(data, xr.DataArray):
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
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    strategy: str = "pairwise",
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Handle missing values in arrays.

    Parameters
    ----------
    obs : np.ndarray or xr.DataArray
        Observed values.
    mod : np.ndarray or xr.DataArray
        Model/predicted values.
    strategy : str, optional
        Strategy for handling missing values ('pairwise', 'listwise').

    Returns
    -------
    tuple of np.ndarray or xr.DataArray
        Arrays with missing values handled.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        if strategy in ("pairwise", "listwise"):
            mask = (obs.notnull() & mod.notnull()).compute()
            return obs.where(mask, drop=True), mod.where(mask, drop=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    else:
        obs_np, mod_np = to_numpy(obs), to_numpy(mod)
        if strategy in ("pairwise", "listwise"):
            mask = ~(np.isnan(obs_np) | np.isnan(mod_np))
            return obs_np[mask], mod_np[mask]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def normalize_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "zscore",
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
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
    dim: str = "time",
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
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
    dim : str, optional
        The dimension along which to detrend for xarray objects, by default "time".

    Returns
    -------
    tuple
        Detrended obs and mod arrays.
    """
    obs, mod = align_arrays(obs, mod)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        if method == "linear":
            from scipy.signal import detrend

            obs_detrended = xr.apply_ufunc(
                detrend,
                obs,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                keep_attrs=True,
            )
            mod_detrended = xr.apply_ufunc(
                detrend,
                mod,
                input_core_dims=[[dim]],
                output_core_dims=[[dim]],
                dask="parallelized",
                keep_attrs=True,
            )
        elif method == "constant":
            obs_detrended = obs - obs.mean(dim=dim)
            mod_detrended = mod - mod.mean(dim=dim)
        else:
            raise ValueError(f"Unknown detrending method: {method}")
        return obs_detrended, mod_detrended

    else:
        obs_np, mod_np = to_numpy(obs), to_numpy(mod)
        if method == "linear":
            from scipy.signal import detrend

            obs_detrended = detrend(obs_np, axis=-1)
            mod_detrended = detrend(mod_np, axis=-1)
        elif method == "constant":
            obs_detrended = obs_np - obs_np.mean()
            mod_detrended = mod_np - mod_np.mean()
        else:
            raise ValueError(f"Unknown detrending method: {method}")
        return obs_detrended, mod_detrended


def compute_anomalies(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    reference_period: Optional[Tuple[int, int]] = None,
    groupby: str = "time.month",
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Compute anomalies relative to a reference period.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model/predicted values.
    reference_period : tuple of int, optional
        Reference period for climatology calculation (for xarray objects only).
    groupby : str, optional
        The dimension to group by for climatology calculation for xarray objects, e.g., 'time.month'.

    Returns
    -------
    tuple
        Anomalies for obs and mod arrays.
    """
    obs, mod = align_arrays(obs, mod)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        if reference_period:
            obs_clim = obs.sel(time=slice(*reference_period)).groupby(groupby).mean(dim="time")
            mod_clim = mod.sel(time=slice(*reference_period)).groupby(groupby).mean(dim="time")
        else:
            obs_clim = obs.groupby(groupby).mean(dim="time")
            mod_clim = mod.groupby(groupby).mean(dim="time")
        obs_anom = obs.groupby(groupby) - obs_clim
        mod_anom = mod.groupby(groupby) - mod_clim
        return obs_anom, mod_anom
    else:
        obs_np, mod_np = to_numpy(obs), to_numpy(mod)
        # For numpy arrays, use overall mean as climatology
        obs_clim = obs_np.mean()
        mod_clim = mod_np.mean()
        return obs_np - obs_clim, mod_np - mod_clim
