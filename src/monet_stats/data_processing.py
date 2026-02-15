"""
Data processing utilities for statistical computations (Aero Protocol Compliant).
"""

from typing import Any, Optional, Tuple, Union

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
    obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray], strategy: str = "pairwise"
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
        For numpy, returns flattened arrays with NaNs removed.

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
            return obs[~mask], mod[~mask]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def normalize_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "zscore",
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Normalize data using various methods (Lazy-friendly).

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

    Returns
    -------
    tuple of (numpy.ndarray or xarray.DataArray)
        Normalized (obs, mod) arrays.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> obs = xr.DataArray(np.random.rand(10, 10))
    >>> mod = xr.DataArray(np.random.rand(10, 10))
    >>> obs_norm, mod_norm = normalize_data(obs, mod, method='zscore')
    """
    obs, mod = align_arrays(obs, mod)

    if method == "zscore":
        obs_norm = (obs - obs.mean()) / obs.std()
        mod_norm = (mod - mod.mean()) / mod.std()
    elif method == "minmax":
        obs_norm = (obs - obs.min()) / (obs.max() - obs.min())
        mod_norm = (mod - mod.min()) / (mod.max() - mod.min())
    elif method == "robust":
        if isinstance(obs, xr.DataArray):
            obs_median = obs.median()
            obs_mad = abs(obs - obs_median).median()
        else:
            obs_median = np.median(obs)
            obs_mad = np.median(np.abs(obs - obs_median))

        if isinstance(mod, xr.DataArray):
            mod_median = mod.median()
            mod_mad = abs(mod - mod_median).median()
        else:
            mod_median = np.median(mod)
            mod_mad = np.median(np.abs(mod - mod_median))

        obs_norm = (obs - obs_median) / obs_mad
        mod_norm = (mod - mod_median) / mod_mad
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return _update_history(obs_norm, f"Normalization ({method})"), _update_history(
        mod_norm, f"Normalization ({method})"
    )


def detrend_data(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    method: str = "linear",
    dim: Optional[str] = None,
    axis: int = -1,
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Remove trend from data (Lazy-friendly).

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
    obs, mod = align_arrays(obs, mod)

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        from .analysis import detrend as analysis_detrend

        if dim is None:
            dim = obs.dims[axis]
        obs_detrended = analysis_detrend(obs, method=method, dim=dim)
        mod_detrended = analysis_detrend(mod, method=method, dim=dim)
        return obs_detrended, mod_detrended

    if method == "linear":
        from scipy.signal import detrend

        obs_detrended = detrend(obs, axis=axis)
        mod_detrended = detrend(mod, axis=axis)
    elif method == "constant":
        obs_detrended = obs - np.nanmean(obs, axis=axis, keepdims=True)
        mod_detrended = mod - np.nanmean(mod, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unknown detrending method: {method}")

    return _update_history(obs_detrended, f"Detrending ({method})"), _update_history(
        mod_detrended, f"Detrending ({method})"
    )


def compute_anomalies(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    climatology: Optional[Union[np.ndarray, xr.DataArray]] = None,
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Compute anomalies relative to climatology (Lazy-friendly).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model/predicted values.
    climatology : numpy.ndarray or xarray.DataArray, optional
        Climatology to subtract. If None, the mean of each array is used.

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
    obs, mod = align_arrays(obs, mod)

    if climatology is not None:
        obs_anom = obs - climatology
        mod_anom = mod - climatology
    else:
        obs_anom = obs - obs.mean()
        mod_anom = mod - mod.mean()

    return _update_history(obs_anom, "Anomaly computation"), _update_history(mod_anom, "Anomaly computation")
