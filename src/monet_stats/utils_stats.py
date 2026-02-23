"""
Utility Functions for Statistics (Aero Protocol Compliant).
"""

import warnings
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike


def _resolve_axis_to_dim(
    obj: Any, axis: Optional[Union[int, str, Iterable[Union[int, str]]]]
) -> Optional[Union[str, Iterable[str]]]:
    """
    Resolve axis index or name(s) to Xarray dimension name(s).

    Parameters
    ----------
    obj : Any
        Xarray DataArray or Dataset.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute.

    Returns
    -------
    str or iterable of str, optional
        Dimension name(s) corresponding to the axis.
    """
    # Centralized resolution for Aero Protocol compliance
    if not hasattr(obj, "dims"):
        return axis

    if axis is None:
        return obj.dims

    if isinstance(axis, int):
        return obj.dims[axis]

    if isinstance(axis, str):
        return axis

    if isinstance(axis, Iterable):
        return [obj.dims[a] if isinstance(a, int) else a for a in axis]

    return axis


def matchedcompressed(a1: ArrayLike, a2: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return compressed (non-masked) values from two matched arrays.

    Note: For Xarray DataArrays, this function will trigger a computation if
    the data is Dask-backed, as it returns NumPy ndarrays. For lazy operations,
    prefer using Xarray-native methods with `skipna=True`.

    Parameters
    ----------
    a1 : ArrayLike
        First input array.
    a2 : ArrayLike
        Second input array.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of (a1_compressed, a2_compressed), both 1D arrays of valid values.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, np.nan, 3])
    >>> b = np.array([4, 5, 6])
    >>> matchedcompressed(a, b)
    (array([1., 3.]), array([4., 6.]))
    """
    # Handle Xarray objects by extracting values (triggers computation for Dask)
    if isinstance(a1, (xr.DataArray, xr.Dataset)):
        if hasattr(a1, "data") and hasattr(a1.data, "chunks"):
            warnings.warn(
                "matchedcompressed triggered computation on Dask-backed array. "
                "Consider using backend-agnostic xarray operations instead.",
                UserWarning,
                stacklevel=2,
            )
        a1 = a1.values
    if isinstance(a2, (xr.DataArray, xr.Dataset)):
        if hasattr(a2, "data") and hasattr(a2.data, "chunks"):
            warnings.warn(
                "matchedcompressed triggered computation on Dask-backed array. "
                "Consider using backend-agnostic xarray operations instead.",
                UserWarning,
                stacklevel=2,
            )
        a2 = a2.values

    # Convert to masked arrays to handle existing masks and NaNs
    a1_m = np.ma.masked_invalid(a1)
    a2_m = np.ma.masked_invalid(a2)

    # Handle mismatched shapes by truncating both to the minimum size
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
            else:
                obj.attrs = {"history": new_entry}
        except Exception:
            pass
    return obj


def matchmasks(a1: Any, a2: Any) -> Tuple[Any, Any]:
    """
    Match and combine masks from two arrays or align Xarray objects (Aero Protocol).

    Parameters
    ----------
    a1 : Any
        First input array or DataArray.
    a2 : Any
        Second input array or DataArray.

    Returns
    -------
    Tuple[Any, Any]
        Tuple of (a1_matched, a2_matched).
    """
    if isinstance(a1, (xr.DataArray, xr.Dataset)) and isinstance(a2, (xr.DataArray, xr.Dataset)):
        return xr.align(a1, a2, join="inner")

    a1_arr = np.asanyarray(a1)
    a2_arr = np.asanyarray(a2)

    # Unify mask handling for numpy
    a1_m = np.ma.masked_invalid(a1_arr)
    a2_m = np.ma.masked_invalid(a2_arr)

    common_mask = np.ma.getmaskarray(a1_m) | np.ma.getmaskarray(a2_m)
    return np.ma.masked_where(common_mask, a1_m), np.ma.masked_where(common_mask, a2_m)


def circlebias(b: ArrayLike) -> Any:
    """
    Circular bias (wrapped to [-180, 180] degrees) (Aero Protocol).

    Handles both dense and masked arrays, as well as Xarray/Dask objects.

    Parameters
    ----------
    b : ArrayLike
        Difference between two wind directions (degrees).

    Returns
    -------
    Any
        Circularly wrapped difference (degrees).

    Examples
    --------
    >>> circlebias(190)
    -170
    >>> circlebias(-190)
    170
    """
    if isinstance(b, (xr.DataArray, xr.Dataset)):
        res = (b + 180) % 360 - 180
        return _update_history(res, "circlebias")

    # Handle both dense and masked numpy arrays
    b_arr = np.ma.masked_invalid(b)
    res_arr = (b_arr + 180) % 360 - 180

    if not np.ma.is_masked(res_arr) and not isinstance(b, np.ma.MaskedArray):
        if np.issubdtype(res_arr.dtype, np.floating):
            return res_arr.filled(np.nan)
        return np.ma.getdata(res_arr)

    return res_arr


def circlebias_m(b: ArrayLike) -> Any:
    """
    Robust circular bias for wind direction (Alias for circlebias).

    Parameters
    ----------
    b : ArrayLike
        Difference between two wind directions (degrees).

    Returns
    -------
    Any
        Circularly wrapped difference.
    """
    return circlebias(b)


def angular_difference(angle1: ArrayLike, angle2: ArrayLike, units: str = "degrees") -> Any:
    """
    Calculate the smallest angular difference between two angles (Aero Protocol).

    Backend-agnostic (supports NumPy and Xarray/Dask).

    Parameters
    ----------
    angle1 : ArrayLike
        First angle(s).
    angle2 : ArrayLike
        Second angle(s).
    units : str, optional
        Units of angles ('degrees' or 'radians'). Default is 'degrees'.

    Returns
    -------
    Any
        Smallest angular difference between the two angles.
    """
    if units == "degrees":
        max_val = 360.0
    elif units == "radians":
        max_val = 2 * np.pi
    else:
        raise ValueError("units must be 'degrees' or 'radians'")

    if isinstance(angle1, (xr.DataArray, xr.Dataset)) or isinstance(angle2, (xr.DataArray, xr.Dataset)):
        if isinstance(angle1, (xr.DataArray, xr.Dataset)) and isinstance(angle2, (xr.DataArray, xr.Dataset)):
            angle1, angle2 = xr.align(angle1, angle2, join="inner")

        diff = abs(angle1 - angle2)
        result = xr.where(diff > max_val / 2, max_val - diff, diff)
        return _update_history(result, "angular_difference")

    angle1_arr = np.asanyarray(angle1)
    angle2_arr = np.asanyarray(angle2)
    diff = np.abs(angle1_arr - angle2_arr)
    return np.minimum(diff, max_val - diff)


def rmse(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Calculate Root Mean Square Error (Alias for error_metrics.RMSE).

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model or predicted values.
    axis : Union[int, str, Iterable], optional
        Axis along which to compute RMSE.

    Returns
    -------
    Union[np.number, np.ndarray, xr.DataArray]
        Root mean square error.
    """
    from .error_metrics import RMSE

    res = RMSE(obs, mod, axis=axis)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "rmse")
    return res


def mae(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Calculate Mean Absolute Error (Alias for error_metrics.MAE).

    Parameters
    ----------
    obs : Union[np.ndarray, xr.DataArray]
        Observed values.
    mod : Union[np.ndarray, xr.DataArray]
        Model or predicted values.
    axis : Union[int, str, Iterable], optional
        Axis along which to compute MAE.

    Returns
    -------
    Union[np.number, np.ndarray, xr.DataArray]
        Mean absolute error.
    """
    from .error_metrics import MAE

    res = MAE(obs, mod, axis=axis)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "mae")
    return res


def correlation(
    x: Union[np.ndarray, xr.DataArray],
    y: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Calculate Pearson correlation coefficient (Alias for correlation_metrics.pearsonr).

    Parameters
    ----------
    x : Union[np.ndarray, xr.DataArray]
        First variable.
    y : Union[np.ndarray, xr.DataArray]
        Second variable.
    axis : Union[int, str, Iterable], optional
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
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "correlation")
    return res
