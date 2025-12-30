"""
Utility Functions for Statistics
"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


def matchedcompressed(a1: ArrayLike, a2: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return compressed (non-masked) values from two masked arrays with matched masks.

    Typical Use Cases
    -----------------
    - Ensuring paired, valid (non-masked) values for statistical calculations (e.g., correlation, regression).
    - Used in metrics that require both arrays to have valid data at the same locations.

    Parameters
    ----------
    a1 : array-like or numpy.ma.MaskedArray
        First input array.
    a2 : array-like or numpy.ma.MaskedArray
        Second input array.

    Returns
    -------
    tuple of ndarray
        Tuple of (a1_compressed, a2_compressed), both 1D arrays of valid values.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import utils_stats as stats
    >>> a1 = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    >>> a2 = np.ma.array([4, 5, 6], mask=[0, 0, 1])
    >>> stats.matchedcompressed(a1, a2)
    (array([1]), array([4]))
    """
    a1, a2 = matchmasks(a1, a2)
    return a1.compressed(), a2.compressed()


def matchmasks(a1: ArrayLike, a2: ArrayLike) -> Tuple[Any, Any]:
    """
    Match and combine masks from two masked arrays, supporting numpy, masked, and xarray arrays.

    This function is designed to handle mixed-type inputs, including dask-backed xarray.DataArray objects.
    It ensures that if a dask array is involved, its boolean mask is explicitly computed before being applied.

    Typical Use Cases
    -----------------
    - Ensuring that two arrays have the same mask for paired statistical calculations.
    - Used in metrics that require both arrays to have valid data at the same locations (e.g., correlation, regression).

    Parameters
    ----------
    a1 : array-like, numpy.ma.MaskedArray, or xarray.DataArray
        First input array.
    a2 : array-like, numpy.ma.MaskedArray, or xarray.DataArray
        Second input array.

    Returns
    -------
    tuple of numpy.ma.MaskedArray or xarray.DataArray
        Tuple of (a1_masked, a2_masked) with a combined, synchronized mask.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.utils_stats import matchmasks
    >>> a1 = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    >>> a2 = np.ma.array([4, 5, 6], mask=[0, 0, 1])
    >>> a1_m, a2_m = matchmasks(a1, a2)
    >>> a1_m.compressed()
    array([1])
    >>> a2_m.compressed()
    array([4])
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    is_a1_xr = xr is not None and isinstance(a1, xr.DataArray)
    is_a2_xr = xr is not None and isinstance(a2, xr.DataArray)

    if is_a1_xr and is_a2_xr:
        # If both are xarray, align them, which also aligns their masks
        a1, a2 = xr.align(a1, a2, join="inner")
        # Ensure that NaN values are also treated as masked
        # The returned arrays from align will have NaNs where there was no overlap
        # isnull() will create a boolean mask for these
        mask = a1.isnull() | a2.isnull()
        return a1.where(~mask), a2.where(~mask)

    # For mixed types (e.g., xarray and numpy) or just numpy arrays.
    # We will work with numpy masks.

    # Extract masks. np.ma.getmaskarray returns `False` for unmasked arrays.
    mask1 = np.ma.getmaskarray(a1)
    mask2 = np.ma.getmaskarray(a2)

    # Also consider NaNs as masked values.
    # Need to handle non-numeric data that would raise TypeError with np.isnan
    # Also, np.isnan on a masked array can be problematic, so get the data.
    data1 = a1.data if hasattr(a1, "data") else a1
    data2 = a2.data if hasattr(a2, "data") else a2
    try:
        nan_mask1 = np.isnan(data1)
        mask1 = mask1 | nan_mask1
    except TypeError:
        pass  # Non-numeric data

    try:
        nan_mask2 = np.isnan(data2)
        mask2 = mask2 | nan_mask2
    except TypeError:
        pass  # Non-numeric data

    # Combine the masks
    combined_mask = mask1 | mask2

    # If the combined_mask is a dask array, compute it to get a numpy array.
    if hasattr(combined_mask, "dask"):
        combined_mask = combined_mask.compute()

    # Apply the combined mask. masked_where works for numpy, masked, and xarray inputs.
    a1_masked = np.ma.masked_where(combined_mask, a1)
    a2_masked = np.ma.masked_where(combined_mask, a2)

    # If inputs were xarray, we should try to return xarray objects.
    # However, np.ma.masked_where returns a MaskedArray.
    # For now, this is acceptable, as downstream functions can handle it.
    # The key is that the data and mask are correct.

    return a1_masked, a2_masked


def circlebias_m(b: ArrayLike) -> Any:
    """
    Circular bias for wind direction (avoid single block error in np.ma).

    Typical Use Cases
    -----------------
    - Calculating the signed difference between two wind directions, accounting for circularity,
      robust to masked arrays.
    - Used in wind direction bias and error metrics for masked or missing data.

    Parameters
    ----------
    b : array-like or numpy.ma.MaskedArray
        Difference between two wind directions (degrees).

    Returns
    -------
    array-like or numpy.ma.MaskedArray
        Circularly wrapped difference (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import efficiency_metrics as stats
    >>> stats.circlebias_m(np.array([190, -190, 10, -10]))
    array([-170, 170,  10, -10])
    """
    b = np.ma.masked_invalid(b)
    out = (b + 180) % 360 - 180
    return out


def circlebias(b: ArrayLike) -> Any:
    """
    Circular bias (wind direction difference, wrapped to [-180, 180] degrees).

    Typical Use Cases
    -----------------
    - Calculating the signed difference between two wind directions, accounting for circularity.
    - Used in wind direction bias and error metrics to avoid artificial large errors across 0/360 boundaries.

    Parameters
    ----------
    b : array-like
        Difference between two wind directions (degrees).

    Returns
    -------
    array-like
        Circularly wrapped difference (degrees).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import efficiency_metrics as stats
    >>> stats.circlebias(np.array([190, -190, 10, -10]))
    array([-170, 170,  10, -10])
    """
    b = np.asarray(b)
    return (b + 180) % 360 - 180


def angular_difference(
    angle1: ArrayLike, angle2: ArrayLike, units: str = "degrees"
) -> Any:
    """
    Calculate the smallest angular difference between two angles.

    Typical Use Cases
    -----------------
    - Computing the difference between wind directions, headings, or other angular measurements.
    - Used in meteorology, navigation, and circular statistics.

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

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import efficiency_metrics as stats
    >>> stats.angular_difference(10, 350, units='degrees')
    20.0
    """
    angle1 = np.asarray(angle1)
    angle2 = np.asarray(angle2)

    if units == "degrees":
        max_val = 360.0
    elif units == "radians":
        max_val = 2 * np.pi
    else:
        raise ValueError("units must be 'degrees' or 'radians'")

    diff = np.abs(angle1 - angle2)
    return np.minimum(diff, max_val - diff)


def rmse(predictions: ArrayLike, targets: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Calculate Root Mean Square Error between predictions and targets.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    targets : array-like
        Target (true) values.
    axis : int, optional
        Axis along which to compute RMSE.

    Returns
    -------
    float or array
        Root mean square error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import efficiency_metrics as stats
    >>> stats.rmse([1, 2, 3], [1.1, 2.1, 2.9])
    0.1
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return np.sqrt(np.mean((predictions - targets) ** 2, axis=axis))


def mae(predictions: ArrayLike, targets: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Calculate Mean Absolute Error between predictions and targets.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    targets : array-like
        Target (true) values.
    axis : int, optional
        Axis along which to compute MAE.

    Returns
    -------
    float or array
        Mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import efficiency_metrics as stats
    >>> stats.mae([1, 2, 3], [1.1, 2.1, 2.9])
    0.1
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    return np.mean(np.abs(predictions - targets), axis=axis)


def correlation(x: ArrayLike, y: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Calculate Pearson correlation coefficient between x and y.

    Parameters
    ----------
    x : array-like
        First variable.
    y : array-like
        Second variable.
    axis : int, optional
        Axis along which to compute correlation.

    Returns
    -------
    float
        Pearson correlation coefficient.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats import efficiency_metrics as stats
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 6, 8, 10]
    >>> stats.correlation(x, y)
    1.0
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays cannot be empty")

    if axis is None:
        # Flatten arrays for 1D correlation
        x = x.flatten()
        y = y.flatten()

    # Calculate means
    mean_x = np.mean(x, axis=axis, keepdims=True)
    mean_y = np.mean(y, axis=axis, keepdims=True)

    # Calculate numerator and denominators
    numerator = np.mean((x - mean_x) * (y - mean_y), axis=axis)
    var_x = np.mean((x - mean_x) ** 2, axis=axis)
    var_y = np.mean((y - mean_y) ** 2, axis=axis)

    # Calculate correlation
    correlation = numerator / np.sqrt(var_x * var_y)

    return correlation
