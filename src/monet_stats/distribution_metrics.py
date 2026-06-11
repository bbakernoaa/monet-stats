"""
Distributional metrics for model evaluation (Aero Protocol Compliant).
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr
from scipy.stats import energy_distance, entropy, wasserstein_distance

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
    bin_range: Optional[tuple] = None,
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
    bin_range : tuple, optional
        The lower and upper range of the bins. If None, uses the min/max of the data.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the divergence (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the divergence (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The KL divergence.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.normal(0, 1, 1000)
    >>> mod = np.random.normal(0.1, 1.1, 1000)
    >>> KLDivergence(obs, mod)
    0.01

    Notes
    -----
    A small constant (epsilon) is added to the PDFs to avoid division by zero or log(0).
    """

    def _kl_numpy(o: np.ndarray, m: np.ndarray, bins: int, r_val: Optional[tuple]) -> float:
        o_flat = o.flatten()
        m_flat = m.flatten()
        o_valid = o_flat[~np.isnan(o_flat)]
        m_valid = m_flat[~np.isnan(m_flat)]
        if o_valid.size == 0 or m_valid.size == 0:
            return np.nan

        if r_val is None:
            r_val = (min(o_valid.min(), m_valid.min()), max(o_valid.max(), m_valid.max()))

        # Estimate PDFs
        p_o, _ = np.histogram(o_valid, bins=bins, range=r_val, density=True)
        p_m, _ = np.histogram(m_valid, bins=bins, range=r_val, density=True)

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
            kwargs={"bins": bins, "r_val": bin_range},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "KL Divergence")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _kl_numpy(o_arr, m_arr, bins, bin_range)

    # For multi-dimensional numpy with axis, use manual iteration for correct pairing
    # Standard library pattern for 2-array axis reduction:
    def _wrapper(o_slice, m_slice):
        return _kl_numpy(o_slice, m_slice, bins, bin_range)

    o_rolled = np.rollaxis(o_arr, axis, -1)
    m_rolled = np.rollaxis(m_arr, axis, -1)
    shape_other = o_rolled.shape[:-1]
    o_flat = o_rolled.reshape(-1, o_rolled.shape[-1])
    m_flat = m_rolled.reshape(-1, m_rolled.shape[-1])
    results = np.array([_wrapper(o, m) for o, m in zip(o_flat, m_flat)])
    return results.reshape(shape_other) if shape_other else results.item()


def JensenShannonDivergence(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    bins: int = 100,
    bin_range: Optional[tuple] = None,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute the Jensen-Shannon (JS) Divergence (Aero Protocol).

    The JS divergence is a symmetric and smoothed version of the KL divergence.
    It is always finite and bounded between 0 and 1 (when using base 2).

    Typical Use Cases
    -----------------
    - Measuring similarity between two probability distributions.
    - Symmetric alternative to KL Divergence.
    - Comparing model and observation PDFs in a stable way.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    bins : int, optional
        Number of bins for estimating the PDF, by default 100.
    bin_range : tuple, optional
        The lower and upper range of the bins. If None, uses the min/max of the data.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the divergence (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the divergence (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The JS divergence.
    """

    def _js_numpy(o: np.ndarray, m: np.ndarray, bins: int, r_val: Optional[tuple]) -> float:
        o_flat = o.flatten()
        m_flat = m.flatten()
        o_valid = o_flat[~np.isnan(o_flat)]
        m_valid = m_flat[~np.isnan(m_flat)]
        if o_valid.size == 0 or m_valid.size == 0:
            return np.nan

        if r_val is None:
            r_val = (min(o_valid.min(), m_valid.min()), max(o_valid.max(), m_valid.max()))

        # Estimate PDFs
        p_o, _ = np.histogram(o_valid, bins=bins, range=r_val, density=True)
        p_m, _ = np.histogram(m_valid, bins=bins, range=r_val, density=True)

        # Normalize
        p_o /= p_o.sum() + 1e-10
        p_m /= p_m.sum() + 1e-10

        # M = 0.5 * (P + Q)
        m_pdf = 0.5 * (p_o + p_m)

        # JSD(P||Q) = 0.5 * KLD(P||M) + 0.5 * KLD(Q||M)
        jsd = 0.5 * entropy(p_o, m_pdf) + 0.5 * entropy(p_m, m_pdf)
        return float(jsd)

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
            _js_numpy,
            obs,
            mod,
            input_core_dims=[core_dims, core_dims],
            output_core_dims=[[]],
            kwargs={"bins": bins, "r_val": bin_range},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "Jensen-Shannon Divergence")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _js_numpy(o_arr, m_arr, bins, bin_range)

    def _wrapper(o_slice, m_slice):
        return _js_numpy(o_slice, m_slice, bins, bin_range)

    o_rolled = np.rollaxis(o_arr, axis, -1)
    m_rolled = np.rollaxis(m_arr, axis, -1)
    shape_other = o_rolled.shape[:-1]
    o_flat = o_rolled.reshape(-1, o_rolled.shape[-1])
    m_flat = m_rolled.reshape(-1, m_rolled.shape[-1])
    results = np.array([_wrapper(o, m) for o, m in zip(o_flat, m_flat)])
    return results.reshape(shape_other) if shape_other else results.item()


def EnergyDistance(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute the Energy distance between two distributions (Aero Protocol).

    Energy distance is a metric between probability distributions that characterizes
    equality of distributions.

    Typical Use Cases
    -----------------
    - Measuring distance between two multi-dimensional distributions.
    - Robust alternative to Wasserstein distance.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the distance (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the distance (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Energy distance.
    """

    def _energy_numpy(o: np.ndarray, m: np.ndarray) -> float:
        o_flat = o.flatten()
        m_flat = m.flatten()
        o_valid = o_flat[~np.isnan(o_flat)]
        m_valid = m_flat[~np.isnan(m_flat)]
        if o_valid.size == 0 or m_valid.size == 0:
            return np.nan
        return float(energy_distance(o_valid, m_valid))

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
            _energy_numpy,
            obs,
            mod,
            input_core_dims=[core_dims, core_dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "Energy Distance")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _energy_numpy(o_arr, m_arr)

    def _wrapper(o_slice, m_slice):
        return _energy_numpy(o_slice, m_slice)

    o_rolled = np.rollaxis(o_arr, axis, -1)
    m_rolled = np.rollaxis(m_arr, axis, -1)
    shape_other = o_rolled.shape[:-1]
    o_flat = o_rolled.reshape(-1, o_rolled.shape[-1])
    m_flat = m_rolled.reshape(-1, m_rolled.shape[-1])
    results = np.array([_wrapper(o, m) for o, m in zip(o_flat, m_flat)])
    return results.reshape(shape_other) if shape_other else results.item()


def SinkhornDistance(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    epsilon: float = 0.1,
    max_iter: int = 100,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute a differentiable approximation of the Earth Mover's Distance (Sinkhorn Distance).

    Uses the Sinkhorn-Knopp algorithm to solve the entropic regularized optimal
    transport problem.

    Typical Use Cases
    -----------------
    - Differentiable alternative to Wasserstein distance for machine learning.
    - Efficient approximation of EMD for large datasets.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    epsilon : float, optional
        Entropy regularization parameter, by default 0.1.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations, by default 100.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the distance (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the distance (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Sinkhorn distance.
    """

    def _sinkhorn_numpy(o: np.ndarray, m: np.ndarray, eps: float, n_iter: int) -> float:
        o_flat = o.flatten()
        m_flat = m.flatten()
        o_valid = o_flat[~np.isnan(o_flat)]
        m_valid = m_flat[~np.isnan(m_flat)]
        if o_valid.size == 0 or m_valid.size == 0:
            return np.nan

        # Normalize to probability distributions
        p = np.ones(o_valid.size) / o_valid.size
        q = np.ones(m_valid.size) / m_valid.size

        # Cost matrix (Squared Euclidean distance)
        # For 1D data, we use the values themselves
        C = (o_valid[:, None] - m_valid[None, :]) ** 2

        # Gibbs kernel
        K = np.exp(-C / eps)

        # Sinkhorn iterations
        u = np.ones(o_valid.size) / o_valid.size
        for _ in range(n_iter):
            v = q / (K.T @ u + 1e-10)
            u = p / (K @ v + 1e-10)

        # Transport plan
        P = u[:, None] * K * v[None, :]
        return float(np.sum(P * C))

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
            _sinkhorn_numpy,
            obs,
            mod,
            input_core_dims=[core_dims, core_dims],
            output_core_dims=[[]],
            kwargs={"eps": epsilon, "n_iter": max_iter},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "Sinkhorn Distance")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _sinkhorn_numpy(o_arr, m_arr, epsilon, max_iter)

    def _wrapper(o_slice, m_slice):
        return _sinkhorn_numpy(o_slice, m_slice, epsilon, max_iter)

    o_rolled = np.rollaxis(o_arr, axis, -1)
    m_rolled = np.rollaxis(m_arr, axis, -1)
    shape_other = o_rolled.shape[:-1]
    o_flat = o_rolled.reshape(-1, o_rolled.shape[-1])
    m_flat = m_rolled.reshape(-1, m_rolled.shape[-1])
    results = np.array([_wrapper(o, m) for o, m in zip(o_flat, m_flat)])
    return results.reshape(shape_other) if shape_other else results.item()


def MutualInformation(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    bins: int = 30,
    bin_range: Optional[tuple] = None,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute the Mutual Information (MI) (Aero Protocol).

    Mutual Information quantifies the amount of information obtained about one
    variable through observing the other. Unlike Pearson correlation, it captures
    non-linear dependencies.

    Typical Use Cases
    -----------------
    - Quantifying non-linear relationships in Earth System Science (e.g., aerosol-cloud).
    - Evaluating how much information the model shares with observations.
    - Robust alternative to correlation for non-Gaussian distributions.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted values.
    bins : int, optional
        Number of bins for estimating the joint PDF, by default 30.
    bin_range : tuple, optional
        The lower and upper range of the bins for both variables.
        If None, uses the min/max of the data.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the MI (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the MI (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The Mutual Information (in nats).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.normal(0, 1, 1000)
    >>> mod = obs + np.random.normal(0, 0.1, 1000)
    >>> MutualInformation(obs, mod)
    1.5

    Notes
    -----
    The calculation uses a binned joint-histogram approach.
    """

    def _mi_numpy(o: np.ndarray, m: np.ndarray, bins: int, r_val: Optional[tuple]) -> float:
        o_flat = o.flatten()
        m_flat = m.flatten()
        mask = ~np.isnan(o_flat) & ~np.isnan(m_flat)
        o_valid = o_flat[mask]
        m_valid = m_flat[mask]
        if o_valid.size == 0:
            return np.nan

        if r_val is None:
            # Common range for both to maintain symmetry
            vmin = min(o_valid.min(), m_valid.min())
            vmax = max(o_valid.max(), m_valid.max())
            r_val = [[vmin, vmax], [vmin, vmax]]
        else:
            # If a single tuple is provided, use it for both
            r_val = [r_val, r_val]

        c_xy, _, _ = np.histogram2d(o_valid, m_valid, bins=bins, range=r_val)
        p_xy = c_xy / np.sum(c_xy)

        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # MI = sum(p(x,y) * log(p(x,y) / (p(x)p(y))))
        # Avoid log(0) using a mask
        px_py = p_x[:, None] * p_y[None, :]
        nonzero = (p_xy > 0) & (px_py > 0)

        mi = np.sum(p_xy[nonzero] * np.log(p_xy[nonzero] / px_py[nonzero]))
        return float(mi)

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
            _mi_numpy,
            obs,
            mod,
            input_core_dims=[core_dims, core_dims],
            output_core_dims=[[]],
            kwargs={"bins": bins, "r_val": bin_range},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, "Mutual Information")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _mi_numpy(o_arr, m_arr, bins, bin_range)

    # For multi-dimensional numpy with axis, use manual iteration for correct pairing
    def _wrapper(o_slice, m_slice):
        return _mi_numpy(o_slice, m_slice, bins, bin_range)

    o_rolled = np.rollaxis(o_arr, axis, -1)
    m_rolled = np.rollaxis(m_arr, axis, -1)
    shape_other = o_rolled.shape[:-1]
    o_flat = o_rolled.reshape(-1, o_rolled.shape[-1])
    m_flat = m_rolled.reshape(-1, m_rolled.shape[-1])
    results = np.array([_wrapper(o, m) for o, m in zip(o_flat, m_flat)])
    return results.reshape(shape_other) if shape_other else results.item()
