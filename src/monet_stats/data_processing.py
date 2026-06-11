"""
Data processing utilities for statistical computations (Aero Protocol Compliant).
"""

from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .utils_stats import _update_history


def to_numpy(
    data: Any,
) -> np.ndarray:
    """
    Convert data to numpy array (Eager operation).

    .. warning::
        This operation triggers immediate computation if the input is a Dask-backed
        xarray object. Use with caution in lazy pipelines.

    Parameters
    ----------
    data : Any
        Input data to convert (xarray.DataArray, xarray.Dataset, pandas.Series/DataFrame, list, etc.).

    Returns
    -------
    numpy.ndarray
        Converted numpy array.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([1, 2, 3])
    >>> to_numpy(da)
    array([1, 2, 3])
    """
    if isinstance(data, xr.DataArray):
        return data.values
    elif isinstance(data, xr.Dataset):
        return data.to_array().values
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        return np.asarray(data)


def align_arrays(
    obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray]
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Align two arrays for comparison.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model/predicted values.

    Returns
    -------
    tuple of (numpy.ndarray or xarray.DataArray)
        Aligned (obs, mod) arrays.

    Examples
    --------
    >>> import xarray as xr
    >>> obs = xr.DataArray([1, 2], coords={'x': [0, 1]}, dims='x')
    >>> mod = xr.DataArray([2, 3], coords={'x': [1, 2]}, dims='x')
    >>> obs_a, mod_a = align_arrays(obs, mod)
    >>> obs_a.x.values
    array([1])
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        return xr.align(obs, mod, join="inner")

    # Fallback for numpy or mixed types
    obs_arr = np.asanyarray(obs)
    mod_arr = np.asanyarray(mod)

    if obs_arr.shape != mod_arr.shape:
        try:
            obs_arr, mod_arr = np.broadcast_arrays(obs_arr, mod_arr)
        except ValueError:
            raise ValueError(f"Arrays must have compatible shapes, got {obs_arr.shape} and {mod_arr.shape}")

    return obs_arr, mod_arr


def handle_missing_values(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    strategy: str = "pairwise",
    preserve_shape: bool = False,
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Handle missing values in arrays (Aero Protocol: Lazy-friendly).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model/predicted values.
    strategy : str, optional
        Strategy for handling missing values ('pairwise', 'listwise').
        For xarray, ensures NaNs are matched across both arrays without dropping coordinates.
    preserve_shape : bool, optional
        If True, returns masked arrays (for NumPy) instead of flattened arrays.
        Default is False.

    Returns
    -------
    tuple of (numpy.ndarray or xarray.DataArray)
        (obs, mod) with missing values handled.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, np.nan, 3])
    >>> mod = np.array([1, 2, np.nan])
    >>> handle_missing_values(obs, mod)
    (array([1.]), array([1.]))
    >>> o, m = handle_missing_values(obs, mod, preserve_shape=True)
    >>> o
    masked_array(data=[1.0, --, --], mask=[False,  True,  True], fill_value=1e+20)
    """
    obs, mod = align_arrays(obs, mod)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        mask = obs.isnull() | mod.isnull()
        res_obs = obs.where(~mask)
        res_mod = mod.where(~mask)
        return _update_history(res_obs, "Missing value handling"), _update_history(res_mod, "Missing value handling")
    else:
        mask = np.isnan(obs) | np.isnan(mod)
        if strategy in ["pairwise", "listwise"]:
            if preserve_shape:
                return np.ma.masked_where(mask, obs), np.ma.masked_where(mask, mod)
            return obs[~mask], mod[~mask]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def normalize_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "zscore",
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, Iterable[int]]] = None,
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Normalize data using various methods (Aero Protocol: Lazy-friendly).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model/predicted values.
    method : str, optional
        Normalization method ('zscore', 'minmax', 'robust').
        - 'zscore': (x - mean) / std
        - 'minmax': (x - min) / (max - min)
        - 'robust': (x - median) / MAD (Median Absolute Deviation)
        Default is 'zscore'.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the normalization (xarray only).
    axis : int or iterable of int, optional
        Axis or axes along which to compute the normalization (numpy only).

    Returns
    -------
    tuple of (numpy.ndarray or xarray.DataArray)
        Normalized (obs, mod) arrays.

    Notes
    -----
    Aero Protocol: Supports vectorized normalization along specified dimensions
    and handles Dask-backed arrays lazily.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> obs = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> mod = xr.DataArray(np.random.rand(10, 10), dims=['lat', 'lon'])
    >>> # Normalize over all dimensions (default)
    >>> obs_norm, mod_norm = normalize_data(obs, mod, method='zscore')
    >>> # Normalize along the 'lat' dimension (per-longitude)
    >>> obs_norm, mod_norm = normalize_data(obs, mod, method='zscore', dim='lat')
    """
    from .utils_stats import _resolve_axis_to_dim, ensure_single_chunk

    obs, mod = align_arrays(obs, mod)

    # Resolve dimensions for Xarray or axes for NumPy
    if isinstance(obs, xr.DataArray):
        reduction_dim = dim if dim is not None else _resolve_axis_to_dim(obs, axis)
    else:
        reduction_dim = axis

    if method == "zscore":
        if isinstance(obs, xr.DataArray):
            obs_norm = (obs - obs.mean(dim=reduction_dim)) / obs.std(dim=reduction_dim)
            mod_norm = (mod - mod.mean(dim=reduction_dim)) / mod.std(dim=reduction_dim)
        else:
            obs_norm = (obs - np.nanmean(obs, axis=reduction_dim, keepdims=True)) / np.nanstd(
                obs, axis=reduction_dim, keepdims=True
            )
            mod_norm = (mod - np.nanmean(mod, axis=reduction_dim, keepdims=True)) / np.nanstd(
                mod, axis=reduction_dim, keepdims=True
            )
    elif method == "minmax":
        if isinstance(obs, xr.DataArray):
            obs_min = obs.min(dim=reduction_dim)
            obs_max = obs.max(dim=reduction_dim)
            mod_min = mod.min(dim=reduction_dim)
            mod_max = mod.max(dim=reduction_dim)
        else:
            obs_min = np.nanmin(obs, axis=reduction_dim, keepdims=True)
            obs_max = np.nanmax(obs, axis=reduction_dim, keepdims=True)
            mod_min = np.nanmin(mod, axis=reduction_dim, keepdims=True)
            mod_max = np.nanmax(mod, axis=reduction_dim, keepdims=True)
        obs_norm = (obs - obs_min) / (obs_max - obs_min)
        mod_norm = (mod - mod_min) / (mod_max - mod_min)
    elif method == "robust":
        if isinstance(obs, xr.DataArray):
            # Ensure reduction dimensions are single-chunked for Dask
            # If reduction_dim is None, it means global reduction over all dims
            rechunk_dims = reduction_dim if reduction_dim is not None else list(obs.dims)
            obs_c = ensure_single_chunk(obs, rechunk_dims)
            mod_c = ensure_single_chunk(mod, rechunk_dims)

            obs_median = obs_c.quantile(0.5, dim=reduction_dim).drop_vars("quantile", errors="ignore")
            obs_mad = abs(obs_c - obs_median).quantile(0.5, dim=reduction_dim).drop_vars("quantile", errors="ignore")

            mod_median = mod_c.quantile(0.5, dim=reduction_dim).drop_vars("quantile", errors="ignore")
            mod_mad = abs(mod_c - mod_median).quantile(0.5, dim=reduction_dim).drop_vars("quantile", errors="ignore")
        else:
            # Use nanmedian for NumPy to handle NaNs if strategy was pairwise
            obs_median = np.nanmedian(obs, axis=reduction_dim, keepdims=True)
            obs_mad = np.nanmedian(np.abs(obs - obs_median), axis=reduction_dim, keepdims=True)

            mod_median = np.nanmedian(mod, axis=reduction_dim, keepdims=True)
            mod_mad = np.nanmedian(np.abs(mod - mod_median), axis=reduction_dim, keepdims=True)

        obs_norm = (obs - obs_median) / obs_mad
        mod_norm = (mod - mod_median) / mod_mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    hist_msg = f"Normalized ({method}) along {reduction_dim}"
    return _update_history(obs_norm, hist_msg), _update_history(mod_norm, hist_msg)


def detrend_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "linear",
    dim: Optional[str] = None,
    axis: int = -1,
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Remove trend from data (Aero Protocol: Lazy-friendly).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model/predicted values.
    method : str, optional
        Detrending method ('linear', 'constant').
        - 'linear': least-squares linear detrend.
        - 'constant': subtract mean.
    dim : str, optional
        Dimension along which to detrend (xarray only).
    axis : int, optional
        Axis along which to detrend (numpy only, or if dim is None). Default is -1.

    Returns
    -------
    tuple of (numpy.ndarray or xarray.DataArray)
        Detrended (obs, mod) arrays.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([1, 2, 3])
    >>> obs_d, mod_d = detrend_data(obs, mod, method='linear')
    >>> np.allclose(obs_d, 0)
    True
    """
    from .analysis import detrend

    obs, mod = align_arrays(obs, mod)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        # Delegate to canonical implementation in analysis module
        if dim is None:
            dim = obs.dims[axis]
        return detrend(obs, method=method, dim=dim), detrend(mod, method=method, dim=dim)

    # NumPy path
    if method == "linear":
        from scipy.signal import detrend as scipy_detrend

        obs_detrended = scipy_detrend(obs, axis=axis)
        mod_detrended = scipy_detrend(mod, axis=axis)
    elif method == "constant":
        obs_detrended = obs - np.mean(obs, axis=axis, keepdims=True)
        mod_detrended = mod - np.mean(mod, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown detrending method: {method}")

    return _update_history(obs_detrended, f"Detrended ({method})"), _update_history(
        mod_detrended, f"Detrended ({method})"
    )


def compute_anomalies(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    climatology: Optional[Union[np.ndarray, xr.DataArray]] = None,
    freq: Optional[str] = None,
    dim: Optional[str] = None,
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Compute anomalies relative to climatology (Aero Protocol: Lazy-friendly).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model/predicted values.
    climatology : numpy.ndarray or xarray.DataArray, optional
        Climatology to subtract. If None, the mean of each array is used.
    freq : str, optional
        Frequency for climatology ('month', 'season', etc.). Only used if climatology is None
        and inputs are Xarray objects with time dimensions.
    dim : str, optional
        Dimension along which to compute anomalies.

    Returns
    -------
    tuple of (numpy.ndarray or xarray.DataArray)
        (obs_anom, mod_anom)

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1, 2, 3, 4, 5])
    >>> obs_anom, _ = compute_anomalies(obs, mod)
    >>> np.isclose(np.mean(obs_anom), 0)
    True
    """
    from .analysis import anomalies

    obs, mod = align_arrays(obs, mod)

    # If freq or dim is specified, use the canonical climatology-based anomalies
    if (freq is not None or dim is not None) and isinstance(obs, xr.DataArray) and climatology is None:
        return anomalies(obs, freq=freq or "month", dim=dim or "time"), anomalies(
            mod, freq=freq or "month", dim=dim or "time"
        )

    if climatology is not None:
        obs_anom = obs - climatology
        mod_anom = mod - climatology
    else:
        obs_anom = obs - obs.mean()
        mod_anom = mod - mod.mean()

    return _update_history(obs_anom, "Anomaly computation"), _update_history(mod_anom, "Anomaly computation")
