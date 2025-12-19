"""
Data processing utilities for statistical computations.
"""

from typing import Optional, Tuple

import numpy as np
import xarray as xr


def align_arrays(obs: xr.DataArray, mod: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Align two xarray DataArrays for comparison.

    Parameters
    ----------
    obs : xr.DataArray
        Observed values.
    mod : xr.DataArray
        Model/predicted values.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Aligned obs and mod arrays.
    """
    obs, mod = xr.align(obs, mod, join="inner")
    return obs, mod


def handle_missing_values(
    obs: xr.DataArray, mod: xr.DataArray, strategy: str = "pairwise"
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Handle missing values in arrays.

    Note
    ----
    When using Dask arrays, the boolean mask for missing values is
    computed and loaded into memory. This is a known limitation of Xarray's
    indexing capabilities with Dask.

    Parameters
    ----------
    obs : xr.DataArray
        Observed values.
    mod : xr.DataArray
        Model/predicted values.
    strategy : str, optional
        Strategy for handling missing values ('pairwise', 'listwise').

    Returns
    -------
    tuple of xr.DataArray
        Arrays with missing values handled according to strategy.
    """
    if strategy in ("pairwise", "listwise"):
        # When using Dask, the boolean mask must be computed before indexing
        # to allow Xarray to determine the output shape.
        mask = (obs.notnull() & mod.notnull()).compute()
        return obs.where(mask, drop=True), mod.where(mask, drop=True)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def normalize_data(
    obs: xr.DataArray,
    mod: xr.DataArray,
    method: str = "zscore",
) -> tuple[xr.DataArray, xr.DataArray]:
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
        obs_norm = (obs - obs.mean()) / obs.std()
        mod_norm = (mod - mod.mean()) / mod.std()
    elif method == "minmax":
        obs_norm = (obs - obs.min()) / (obs.max() - obs.min())
        mod_norm = (mod - mod.min()) / (mod.max() - mod.min())
    elif method == "robust":
        obs_median = obs.median()
        mod_median = mod.median()
        obs_mad = abs(obs - obs_median).median()
        mod_mad = abs(mod - mod_median).median()
        obs_norm = (obs - obs_median) / obs_mad
        mod_norm = (mod - mod_median) / mod_mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return obs_norm, mod_norm


def detrend_data(
    obs: xr.DataArray,
    mod: xr.DataArray,
    method: str = "linear",
    dim: str = "time",
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Remove trend from data.

    Parameters
    ----------
    obs : xr.DataArray
        Observed values.
    mod : xr.DataArray
        Model/predicted values.
    method : str, optional
        Detrending method ('linear', 'constant').
    dim : str, optional
        The dimension along which to detrend, by default "time".

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Detrended obs and mod arrays.
    """
    obs, mod = align_arrays(obs, mod)

    if method == "linear":
        from scipy.signal import detrend

        # Use xr.apply_ufunc to apply scipy's detrend along the specified dimension
        # This makes it compatible with Dask arrays.
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


def compute_anomalies(
    data: xr.DataArray,
    climatology: Optional[xr.DataArray] = None,
    reference_period: Optional[tuple[str, str]] = None,
    groupby: str = "time.month",
) -> xr.DataArray:
    """
    Compute anomalies relative to a climatology.

    Parameters
    ----------
    data : xr.DataArray
        The input data (e.g., observations or model output).
    climatology : xr.DataArray, optional
        A pre-computed climatology. If None, it will be computed from the data.
    reference_period : tuple of str, optional
        The reference period for computing the climatology, e.g., ('1990-01-01', '2020-12-31').
    groupby : str, optional
        The dimension to group by for climatology calculation, e.g., 'time.month'.

    Returns
    -------
    xr.DataArray
        Anomalies computed from the data.
    """
    if climatology is None:
        if reference_period:
            climatology = data.sel(time=slice(*reference_period)).groupby(groupby).mean(dim="time")
        else:
            climatology = data.groupby(groupby).mean(dim="time")

    anomalies = data.groupby(groupby) - climatology
    return anomalies
