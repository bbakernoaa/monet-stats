"""
Performance optimization utilities for statistical computations.
"""

import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from .utils_stats import _update_history


def get_chunk_recommendation(
    data: Union[xr.DataArray, xr.Dataset],
    target_mb: float = 100.0,
    chunk_dim: Optional[str] = None,
) -> Dict[str, int]:
    """
    Recommend chunk sizes for an xarray object to target a specific chunk size (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data to analyze.
    target_mb : float, optional
        Target size for each chunk in Megabytes. Default is 100.0.
    chunk_dim : str, optional
        Specific dimension to chunk along. If None, it tries to chunk the largest dimension.

    Returns
    -------
    Dict[str, int]
        Recommended chunk sizes dictionary.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> data_array = xr.DataArray(np.random.rand(1000, 1000), dims=['x', 'y'])
    >>> get_chunk_recommendation(data_array, target_mb=1.0)
    {'x': 125, 'y': 1000}
    """
    if isinstance(data, xr.Dataset):
        if not data.data_vars:
            return {}
        # Use the first variable to estimate
        var_name = list(data.data_vars)[0]
        data_arr = data[var_name]
    else:
        data_arr = data

    itemsize = data_arr.dtype.itemsize
    total_elements = data_arr.size

    # Target elements per chunk
    target_elements = (target_mb * 1024 * 1024) / itemsize

    if target_elements >= total_elements:
        return {str(d): s for d, s in data_arr.sizes.items()}

    chunks = {str(d): s for d, s in data_arr.sizes.items()}

    if chunk_dim is not None:
        if chunk_dim not in data_arr.dims:
            raise ValueError(f"Dimension {chunk_dim} not found in data.")

        # Calculate size of other dimensions
        other_dims_size = total_elements / data_arr.sizes[chunk_dim]
        recommended_size = max(1, int(target_elements / other_dims_size))
        chunks[chunk_dim] = min(recommended_size, data_arr.sizes[chunk_dim])
    else:
        # Multi-dimensional chunking: iteratively reduce largest dimensions
        current_elements = float(total_elements)
        # Sort dimensions by size descending
        sorted_dims = sorted(data_arr.sizes.items(), key=lambda x: x[1], reverse=True)

        for d, s in sorted_dims:
            if current_elements <= target_elements:
                break

            # Calculate ideal size for this dimension to reach target elements
            other_dims_size = current_elements / s
            ideal_s = max(1, int(target_elements / other_dims_size))

            chunks[str(d)] = min(ideal_s, s)
            current_elements = (current_elements / s) * chunks[str(d)]

    return chunks


def apply_lazy_threshold(
    data: Union[xr.DataArray, xr.Dataset],
    threshold_mb: float = 500.0,
    force_dask: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Ensure data is lazy (Dask-backed) if it exceeds a certain size (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data to check.
    threshold_mb : float, optional
        Size threshold in Megabytes. Default is 500.0.
    force_dask : bool, optional
        If True, always convert to Dask regardless of size. Default is False.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        DataArray or Dataset, potentially converted to Dask.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> da = xr.DataArray(np.random.rand(1000, 1000), dims=['x', 'y']) # ~7.6 MB
    >>> da_lazy = apply_lazy_threshold(da, threshold_mb=1.0) # Will convert to Dask
    """
    # Check if already lazy
    is_lazy = False
    if isinstance(data, xr.DataArray):
        if hasattr(data.data, "chunks") and data.data.chunks is not None:
            is_lazy = True
    elif isinstance(data, xr.Dataset):
        # Check if any variable is lazy
        if any(hasattr(data[v].data, "chunks") and data[v].data.chunks is not None for v in data.data_vars):
            is_lazy = True

    if is_lazy and not force_dask:
        return data

    size_mb = data.nbytes / (1024 * 1024)

    if force_dask or size_mb > threshold_mb:
        recommendation = get_chunk_recommendation(data)
        if not is_lazy:
            warnings.warn(
                f"Data size ({size_mb:.2f} MB) exceeds threshold ({threshold_mb:.2f} MB). "
                f"Converting to Dask with recommended chunks: {recommendation}",
                UserWarning,
                stacklevel=2,
            )
        res = data.chunk(recommendation)
        return _update_history(res, f"Converted to Dask (threshold={threshold_mb}MB)")

    return data


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


def vectorize_function(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Apply function in a vectorized manner (Aero Protocol).

    .. note::
        This uses `np.vectorize`, which is a convenience wrapper and not
        necessarily high-performance as it often involves an internal loop.
        Prefer true NumPy/Xarray vectorization where possible.

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
    Any
        Result of vectorized function application.
    """
    return np.vectorize(func)(*args, **kwargs)


def parallel_compute(
    func: Callable,
    data: Union[np.ndarray, xr.DataArray, xr.Dataset],
    chunk_size: int = 1000000,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Any:
    """
    Compute function in parallel using chunking strategy (Aero Protocol).

    For Xarray objects, this automatically ensures laziness via `apply_lazy_threshold`
    and leverages Dask if the data is already chunked.

    Parameters
    ----------
    func : callable
        Function to apply to data chunks.
    data : numpy.ndarray, xarray.DataArray, or xarray.Dataset
        Input data to process.
    chunk_size : int, optional
        Size of data chunks for processing (NumPy only). Default is 1,000,000.
    axis : int, str, or iterable, optional
        Axis along which to compute.

    Returns
    -------
    Any
        Result of parallel computation.
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        # Ensure it's lazy if large
        data_lazy = apply_lazy_threshold(data)
        res = func(data_lazy, axis=axis)
        return _update_history(res, "parallel_compute")
    else:
        # For numpy arrays, eliminate explicit loops by converting to Dask
        data_arr = np.asanyarray(data)
        if data_arr.size > chunk_size:
            # Automatically chunk and parallelize via Dask
            dask_arr = da.from_array(data_arr, chunks=chunk_size)
            res = func(dask_arr, axis=axis)
            # If result is dask-backed, compute it for return
            if hasattr(res, "compute"):
                return res.compute()
            return res
        else:
            return func(data_arr, axis=axis)


def optimize_for_size(
    func: Callable,
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Any:
    """
    Optimize function computation based on data size (Aero Protocol).

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
    Any
        Optimized computation result.
    """
    if isinstance(obs, (xr.DataArray, xr.Dataset)):
        obs = apply_lazy_threshold(obs)
    if isinstance(mod, (xr.DataArray, xr.Dataset)):
        mod = apply_lazy_threshold(mod)

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
