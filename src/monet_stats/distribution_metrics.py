"""
Distributional metrics for model evaluation (Aero Protocol Compliant).
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr
from scipy.stats import entropy, wasserstein_distance

from .utils_stats import _resolve_axis_to_dim, _update_history, ensure_single_chunk


def WassersteinDistance(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute the Wasserstein distance (Earth Mover's Distance) (Aero Protocol).

    Typical Use Cases
    -----------------
    - Measuring the "work" required to transform the model's distribution into the observed distribution.
    - Evaluating how well the model captures the overall probability density function.
    - Highly robust against outliers and excellent for evaluating climatological distributions.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the distance (xarray only).
        If None, reduces over all dimensions.
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the distance (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Wasserstein distance.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.normal(0, 1, 100)
    >>> mod = np.random.normal(0.5, 1, 100)
    >>> WassersteinDistance(obs, mod)
    0.5
    """

    def _wasserstein_numpy(o: np.ndarray, m: np.ndarray) -> float:
        # Filter NaNs for valid comparison
        o_flat = o.flatten()
        m_flat = m.flatten()
        o_valid = o_flat[~np.isnan(o_flat)]
        m_valid = m_flat[~np.isnan(m_flat)]
        if o_valid.size == 0 or m_valid.size == 0:
            return np.nan
        return float(wasserstein_distance(o_valid, m_valid))

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)

        if isinstance(reduction_dim, str):
            core_dims = [reduction_dim]
        else:
            core_dims = list(reduction_dim)

        # Ensure core dimensions are single-chunked for apply_ufunc
        obs = ensure_single_chunk(obs, core_dims)
        mod = ensure_single_chunk(mod, core_dims)

        res = xr.apply_ufunc(
            _wasserstein_numpy,
            obs,
            mod,
            input_core_dims=[core_dims, core_dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "Wasserstein Distance")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _wasserstein_numpy(o_arr.flatten(), m_arr.flatten())

    # For multi-dimensional numpy with axis, use apply_along_axis
    res = np.apply_along_axis(lambda x, y: _wasserstein_numpy(x, y), axis, o_arr, m_arr)
    return res.item() if np.ndim(res) == 0 else res


def KLDivergence(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    bins: int = 100,
    range: Optional[tuple] = None,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute the Kullback-Leibler (KL) Divergence (Aero Protocol).

    Typical Use Cases
    -----------------
    - Measuring how much information is lost when the model's distribution is used
      to approximate the observed distribution.
    - Quantifying the difference between two probability distributions.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values (Reference distribution).
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values (Approximating distribution).
    bins : int, optional
        Number of bins for estimating the PDF, by default 100.
    range : tuple, optional
        The lower and upper range of the bins. If None, uses the min/max of the data.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the divergence (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the divergence (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The KL divergence.

    Notes
    -----
    A small constant (epsilon) is added to the PDFs to avoid division by zero or log(0).
    """

    def _kl_numpy(o: np.ndarray, m: np.ndarray, bins: int, range_val: Optional[tuple]) -> float:
        o_flat = o.flatten()
        m_flat = m.flatten()
        o_valid = o_flat[~np.isnan(o_flat)]
        m_valid = m_flat[~np.isnan(m_flat)]
        if o_valid.size == 0 or m_valid.size == 0:
            return np.nan

        if range_val is None:
            range_val = (min(o_valid.min(), m_valid.min()), max(o_valid.max(), m_valid.max()))

        # Estimate PDFs
        p_o, _ = np.histogram(o_valid, bins=bins, range=range_val, density=True)
        p_m, _ = np.histogram(m_valid, bins=bins, range=range_val, density=True)

        # Add epsilon to avoid zeros
        eps = 1e-10
        p_o = p_o + eps
        p_m = p_m + eps

        # Normalize
        p_o /= p_o.sum()
        p_m /= p_m.sum()

        return float(entropy(p_o, p_m))

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)

        if isinstance(reduction_dim, str):
            core_dims = [reduction_dim]
        else:
            core_dims = list(reduction_dim)

        obs = ensure_single_chunk(obs, core_dims)
        mod = ensure_single_chunk(mod, core_dims)

        res = xr.apply_ufunc(
            _kl_numpy,
            obs,
            mod,
            input_core_dims=[core_dims, core_dims],
            output_core_dims=[[]],
            kwargs={"bins": bins, "range_val": range},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "KL Divergence")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _kl_numpy(o_arr.flatten(), m_arr.flatten(), bins, range)

    # For multi-dimensional numpy with axis
    res = np.apply_along_axis(lambda x, y: _kl_numpy(x, y, bins, range), axis, o_arr, m_arr)
    return res.item() if np.ndim(res) == 0 else res
