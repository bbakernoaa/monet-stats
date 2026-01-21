"""
Utility Functions for Statistics
"""

from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike


def matchedcompressed(a1: ArrayLike, a2: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return compressed (non-masked) values from two masked arrays with matched masks.

    Note: For Xarray DataArrays, this function will trigger a computation if
    the data is Dask-backed, as it returns NumPy ndarrays. For lazy operations,
    prefer using Xarray-native methods with `skipna=True`.

    Parameters
    ----------
    a1 : array-like
        First input array.
    a2 : array-like
        Second input array.

    Returns
    -------
    tuple of ndarray
        Tuple of (a1_compressed, a2_compressed), both 1D arrays of valid values.
    """
    # Handle Xarray objects by extracting values (explicitly mentioned as computation-triggering)
    if hasattr(a1, "values") and hasattr(a1, "coords"):
        a1 = a1.values
    if hasattr(a2, "values") and hasattr(a2, "coords"):
        a2 = a2.values

    # Convert to masked arrays to handle existing masks and NaNs
    a1_m = np.ma.masked_invalid(a1)
    a2_m = np.ma.masked_invalid(a2)

    # Handle mismatched shapes for numpy arrays by truncating
    if a1_m.shape != a2_m.shape:
        min_size = min(a1_m.size, a2_m.size)
        a1_m = a1_m.flat[:min_size]
        a2_m = a2_m.flat[:min_size]

    mask = np.ma.getmaskarray(a1_m) | np.ma.getmaskarray(a2_m)
    a1_matched = np.ma.masked_where(mask, a1_m)
    a2_matched = np.ma.masked_where(mask, a2_m)
    return a1_matched.compressed(), a2_matched.compressed()


def _update_history(obj: Any, metric_name: str) -> Any:
    """
    Update the scientific history attribute of an xarray object.

    Parameters
    ----------
    obj : Any
        The object to update.
    metric_name : str
        The name of the metric or operation performed.

    Returns
    -------
    Any
        The original object with an updated history attribute if applicable.
    """
    if hasattr(obj, "attrs"):
        try:
            timestamp = pd.Timestamp.now().isoformat()
            new_entry = f"[{timestamp}] Calculated {metric_name} using monet-stats."
            if obj.attrs is not None:
                history = obj.attrs.get("history", "")
                obj.attrs["history"] = f"{history}\n{new_entry}".strip()
        except Exception:
            pass
    return obj


def matchmasks(a1: ArrayLike, a2: ArrayLike) -> Tuple[Any, Any]:
    """
    Match and combine masks from two masked arrays or align Xarray objects.

    Parameters
    ----------
    a1 : array-like
        First input array.
    a2 : array-like
        Second input array.

    Returns
    -------
    tuple
        Tuple of (a1_matched, a2_matched).
    """
    if isinstance(a1, xr.DataArray) and isinstance(a2, xr.DataArray):
        # Align xarray objects (works for dask-backed as well)
        return xr.align(a1, a2, join="inner")
    else:
        a1_arr = np.asanyarray(a1)
        a2_arr = np.asanyarray(a2)
        mask = np.ma.getmaskarray(a1_arr) | np.ma.getmaskarray(a2_arr)
        return np.ma.masked_where(mask, a1_arr), np.ma.masked_where(mask, a2_arr)


def circlebias_m(b: ArrayLike) -> Any:
    """
    Circular bias for wind direction (robust to masked arrays).

    Parameters
    ----------
    b : array-like
        Difference between two wind directions (degrees).

    Returns
    -------
    array-like
        Circularly wrapped difference (degrees).
    """
    if hasattr(b, "attrs") and hasattr(b, "coords"):
        res = (b + 180) % 360 - 180
        return _update_history(res, "circlebias_m")

    b_masked = np.ma.masked_invalid(b)
    return (b_masked + 180) % 360 - 180


def circlebias(b: ArrayLike) -> Any:
    """
    Circular bias (wind direction difference, wrapped to [-180, 180] degrees).

    Parameters
    ----------
    b : array-like
        Difference between two wind directions (degrees).

    Returns
    -------
    array-like
        Circularly wrapped difference (degrees).
    """
    if hasattr(b, "attrs") and hasattr(b, "coords"):
        res = (b + 180) % 360 - 180
        return _update_history(res, "circlebias")

    return (np.asarray(b) + 180) % 360 - 180


def angular_difference(angle1: ArrayLike, angle2: ArrayLike, units: str = "degrees") -> Any:
    """
    Calculate the smallest angular difference between two angles.

    Backend-agnostic (supports NumPy and Xarray/Dask).

    Parameters
    ----------
    angle1 : array-like
        First angle(s).
    angle2 : array-like
        Second angle(s).
    units : str, optional
        Units of angles ('degrees' or 'radians'). Default is 'degrees'.

    Returns
    -------
    array-like
        Smallest angular difference between the two angles.
    """
    if units == "degrees":
        max_val = 360.0
    elif units == "radians":
        max_val = 2 * np.pi
    else:
        raise ValueError("units must be 'degrees' or 'radians'")

    if (hasattr(angle1, "attrs") and hasattr(angle1, "coords")) or (
        hasattr(angle2, "attrs") and hasattr(angle2, "coords")
    ):
        if (hasattr(angle1, "attrs") and hasattr(angle1, "coords")) and (
            hasattr(angle2, "attrs") and hasattr(angle2, "coords")
        ):
            angle1, angle2 = xr.align(angle1, angle2, join="inner")
        diff = abs(angle1 - angle2)
        result = xr.where(diff > max_val / 2, max_val - diff, diff)
        return _update_history(result, "angular_difference")

    angle1_arr = np.asarray(angle1)
    angle2_arr = np.asarray(angle2)
    diff = np.abs(angle1_arr - angle2_arr)
    return np.minimum(diff, max_val - diff)


def rmse(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Calculate Root Mean Square Error between observations and model.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable, optional
        Axis or dimension along which to compute RMSE.

    Returns
    -------
    Union[np.number, np.ndarray, xr.DataArray]
        Root mean square error.
    """
    from .error_metrics import RMSE

    res = RMSE(obs, mod, axis=axis)
    if hasattr(res, "attrs") and hasattr(res, "coords"):
        return _update_history(res, "rmse")
    return res


def mae(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Calculate Mean Absolute Error between observations and model.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable, optional
        Axis along which to compute MAE.

    Returns
    -------
    Union[np.number, np.ndarray, xr.DataArray]
        Mean absolute error.
    """
    from .error_metrics import MAE

    res = MAE(obs, mod, axis=axis)
    if hasattr(res, "attrs") and hasattr(res, "coords"):
        return _update_history(res, "mae")
    return res


def correlation(
    x: Union[np.ndarray, xr.DataArray],
    y: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Calculate Pearson correlation coefficient between x and y.

    Parameters
    ----------
    x : numpy.ndarray or xarray.DataArray
        First variable.
    y : numpy.ndarray or xarray.DataArray
        Second variable.
    axis : int, str, or iterable, optional
        Axis along which to compute correlation.

    Returns
    -------
    Union[np.number, np.ndarray, xr.DataArray]
        Pearson correlation coefficient.

    Raises
    ------
    ValueError
        If input arrays are empty.
    """
    if hasattr(x, "size") and x.size == 0:
        raise ValueError("Input arrays cannot be empty")
    if hasattr(y, "size") and y.size == 0:
        raise ValueError("Input arrays cannot be empty")

    from .correlation_metrics import pearsonr

    res = pearsonr(x, y, axis=axis)
    if hasattr(res, "attrs") and hasattr(res, "coords"):
        return _update_history(res, "correlation")
    return res
