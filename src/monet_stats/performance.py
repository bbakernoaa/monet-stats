"""
Performance optimization utilities for statistical computations.
"""

from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


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
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Any:
    """
    Compute function in parallel using chunking strategy.

    For Xarray DataArrays, this relies on the underlying Dask support.
    For NumPy arrays, it uses manual chunking.

    Parameters
    ----------
    func : callable
        Function to apply to data chunks.
    data : numpy.ndarray or xarray.DataArray
        Input data to process.
    chunk_size : int, optional
        Size of data chunks for processing (NumPy only).
    axis : int, str, or iterable, optional
        Axis along which to compute.

    Returns
    -------
    result
        Result of parallel computation.
    """
    if hasattr(data, "attrs") and hasattr(data, "coords"):
        # Handle xarray DataArray natively
        res = func(data, axis=axis)
        return _update_history(res, "parallel_compute")
    else:
        # For numpy arrays, chunk if large
        data_arr = np.asanyarray(data)
        if data_arr.size > chunk_size:
            chunks = chunk_array(data_arr, chunk_size)
            results = [func(chunk, axis=axis) for chunk in chunks]
            # Combine results based on function type
            if isinstance(results[0], np.ndarray):
                return np.concatenate(results)
            else:
                # For scalar results, average or combine as appropriate
                weights = [len(chunk) for chunk in chunks]
                return np.average(results, weights=weights)
        else:
            return func(data_arr, axis=axis)


def optimize_for_size(
    func: Callable,
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
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
    axis : int, str, or iterable, optional
        Axis along which to compute.

    Returns
    -------
    result
        Optimized computation result.
    """
    return func(obs, mod, axis=axis)


def memory_efficient_correlation(
    x: Union[np.ndarray, xr.DataArray],
    y: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Any:
    """
    Memory-efficient computation of Pearson correlation coefficient.

    Alias for correlation_metrics.pearsonr.

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
    Any
        Pearson correlation coefficient.
    """
    from .correlation_metrics import pearsonr

    res = pearsonr(x, y, axis=axis)
    if hasattr(res, "attrs") and hasattr(res, "coords"):
        return _update_history(res, "memory_efficient_correlation")
    return res


def fast_rmse(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Any:
    """
    Fast computation of Root Mean Square Error.

    Alias for error_metrics.RMSE.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable, optional
        Axis along which to compute RMSE.

    Returns
    -------
    Any
        Root mean square error.
    """
    from .error_metrics import RMSE

    res = RMSE(obs, mod, axis=axis)
    if hasattr(res, "attrs") and hasattr(res, "coords"):
        return _update_history(res, "fast_rmse")
    return res


def fast_mae(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Any:
    """
    Fast computation of Mean Absolute Error.

    Alias for error_metrics.MAE.

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
    Any
        Mean absolute error.
    """
    from .error_metrics import MAE

    res = MAE(obs, mod, axis=axis)
    if hasattr(res, "attrs") and hasattr(res, "coords"):
        return _update_history(res, "fast_mae")
    return res
