"""
Performance optimization utilities for statistical computations.
"""

from typing import Any, Callable, Union

import numpy as np
import xarray as xr


def chunk_array(arr: np.ndarray, chunk_size: int = 1000000) -> list:
    """
    Split array into chunks for memory-efficient processing.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array to chunk.
    chunk_size : int, optional
        Size of each chunk (number of elements).

    Returns
    -------
    list
        List of array chunks.
    """
    if arr.size == 0:
        return []

    num_elements = arr.size
    chunks = []
    for i in range(0, num_elements, chunk_size):
        chunks.append(arr[i : i + chunk_size])  # noqa: E203
    return chunks


def vectorize_function(func: Callable, *args, **kwargs) -> Any:
    """
    Apply function in a vectorized manner.

    Parameters
    ----------
    func : callable
        Function to vectorize.
    *args : tuple
        Arguments to pass to function.
    **kwargs : dict
        Keyword arguments to pass to function.

    Returns
    -------
    result
        Result of vectorized function application.
    """
    return np.vectorize(func)(*args, **kwargs)


def parallel_compute(
    func: Callable,
    data: Union[np.ndarray, xr.DataArray],
    chunk_size: int = 1000000,
    axis=None,
) -> Any:
    """
    Compute function in parallel using chunking strategy.

    Parameters
    ----------
    func : callable
        Function to apply to data chunks.
    data : numpy.ndarray or xarray.DataArray
        Input data to process.
    chunk_size : int, optional
        Size of data chunks for processing.
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    result
        Result of parallel computation.
    """
    if isinstance(data, xr.DataArray):
        # Handle xarray DataArray
        result = func(data, axis=axis)
        return result
    else:
        # For numpy arrays, chunk if large
        if data.size > chunk_size:
            chunks = chunk_array(data, chunk_size)
            results = [func(chunk, axis=axis) for chunk in chunks]
            # Combine results based on function type
            if isinstance(results[0], np.ndarray):
                return np.concatenate(results)
            else:
                # For scalar results, average or combine as appropriate
                weights = [len(chunk) for chunk in chunks]
                return np.average(results, weights=weights)
        else:
            return func(data, axis=axis)


def optimize_for_size(
    func: Callable,
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis=None,
) -> Any:
    """
    Optimize function computation based on data size.

    Parameters
    ----------
    func : callable
        Function to optimize.
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model values.
    axis : int, optional
        Axis along which to compute.

    Returns
    -------
    result
        Optimized computation result.
    """
    # Check if data is large enough to warrant optimization
    if hasattr(obs, "size") and hasattr(mod, "size"):
        max_size = max(obs.size, mod.size)
        if max_size > 100000:  # 100K elements
            # Use chunked processing for large arrays
            # Instead of calling parallel_compute with multiple arguments, just use the function directly
            pass  # Skip optimization for now to avoid complexity

    # Use standard computation for smaller arrays
    return func(obs, mod, axis=axis)


def memory_efficient_correlation(x: np.ndarray, y: np.ndarray, axis=None) -> float:
    """
    Memory-efficient computation of Pearson correlation coefficient.

    Parameters
    ----------
    x : numpy.ndarray
        First variable.
    y : numpy.ndarray
        Second variable.
    axis : int, optional
        Axis along which to compute correlation.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    if axis is None:
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


def fast_rmse(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis=None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Fast computation of Root Mean Square Error.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute RMSE.

    Returns
    -------
    rmse : float or ndarray or DataArray
        Root mean square error.
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if (
        xr is not None
        and isinstance(obs, xr.DataArray)
        and isinstance(mod, xr.DataArray)
    ):
        obs, mod = xr.align(obs, mod, join="inner")
        return ((mod - obs) ** 2).mean(dim=axis) ** 0.5
    else:
        return np.sqrt(np.mean((mod - obs) ** 2, axis=axis))


def fast_mae(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis=None,
) -> Any:
    """
    Fast computation of Mean Absolute Error.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MAE.

    Returns
    -------
    mae : float or ndarray or DataArray
        Mean absolute error.
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if (
        xr is not None
        and isinstance(obs, xr.DataArray)
        and isinstance(mod, xr.DataArray)
    ):
        obs, mod = xr.align(obs, mod, join="inner")
        return abs(mod - obs).mean(dim=axis)
    else:
        return np.mean(np.abs(mod - obs), axis=axis)
