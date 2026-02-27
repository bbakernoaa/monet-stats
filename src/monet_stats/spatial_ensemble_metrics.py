"""
Spatial and Ensemble Metrics for Atmospheric Sciences (Aero Protocol Compliant)
"""

from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def EDS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: float,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, Iterable[int]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Extreme Dependency Score (EDS) for rare event detection (Aero Protocol).

    Typical Use Cases
    -----------------
    - Assessing model performance for rare extreme events (e.g., heavy precipitation).
    - Used when traditional scores like CSI or ETS go to zero as the event becomes rarer.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed field.
    mod : xarray.DataArray or numpy.ndarray
        Model field.
    threshold : float
        Event threshold to define the extreme event.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the score (xarray only).
        If None, reduces over all dimensions.
    axis : int or iterable of int, optional
        Axis or axes along which to compute the score (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        Extreme Dependency Score.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.zeros((10, 10)); obs[5, 5] = 1
    >>> mod = np.zeros((10, 10)); mod[5, 5] = 1
    >>> EDS(obs, mod, threshold=0.5)
    1.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_bin = obs >= threshold
        mod_bin = mod >= threshold

        mask = obs.notnull() & mod.notnull()
        obs_bin = obs_bin & mask
        mod_bin = mod_bin & mask

        hits = (obs_bin & mod_bin).sum(dim=dim)
        n_obs = obs_bin.sum(dim=dim)
        n_mod = mod_bin.sum(dim=dim)
        n = mask.sum(dim=dim)

        # EDS = log(hits/n) / log(n_obs*n_mod / n^2)
        # Avoid log(0) and division by zero
        # p = n_obs / n, q = n_mod / n
        with np.errstate(divide="ignore", invalid="ignore"):
            eds = xr.where(
                (hits > 0) & (n_obs > 0) & (n_mod > 0), np.log(hits / n) / np.log((n_obs / n) * (n_mod / n)), np.nan
            )

        return _update_history(eds, "Extreme Dependency Score (EDS)")

    # NumPy path
    obs_arr = np.asarray(obs)
    mod_arr = np.asarray(mod)
    mask = ~np.isnan(obs_arr) & ~np.isnan(mod_arr)

    obs_bin = (obs_arr >= threshold) & mask
    mod_bin = (mod_arr >= threshold) & mask

    hits = np.sum(obs_bin & mod_bin, axis=axis)
    n_obs = np.sum(obs_bin, axis=axis)
    n_mod = np.sum(mod_bin, axis=axis)
    n = np.sum(mask, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(
            (hits > 0) & (n_obs > 0) & (n_mod > 0), np.log(hits / n) / np.log((n_obs / n) * (n_mod / n)), np.nan
        )
        return res.item() if np.ndim(res) == 0 else res


def CRPS(
    ensemble: Union[xr.DataArray, np.ndarray],
    obs: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.

    Supports lazy evaluation via Xarray/Dask.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts. If DataArray, should have an ensemble dimension.
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        CRPS values.

    Examples
    --------
    >>> import numpy as np
    >>> ens = np.array([[1, 2], [2, 3], [3, 4]])
    >>> obs = np.array([2, 3])
    >>> CRPS(ens, obs, axis=0)
    array([0.22222222, 0.22222222])
    """

    def _crps_numpy(ens: np.ndarray, observation: np.ndarray, ens_axis: int = 0) -> np.ndarray:
        """
        Core NumPy implementation of CRPS using the energy form.
        CRPS = E|X-y| - 0.5 * E|X-X'|
        """
        n = ens.shape[ens_axis]
        if n == 0:
            return np.nan

        # 1. Mean absolute error part: E|X-y|
        obs_broadcast = np.expand_dims(observation, axis=ens_axis)
        mae = np.mean(np.abs(ens - obs_broadcast), axis=ens_axis)

        # 2. Ensemble spread part: 0.5 * E|X-X'|
        # Optimized O(N log N) spread calculation using sorted weights
        ens_sorted = np.sort(ens, axis=ens_axis)
        i = np.arange(1, n + 1)
        weights = (2 * i - n - 1) / (n * n)

        # Reshape weights for broadcasting
        w_shape = [1] * ens.ndim
        w_shape[ens_axis] = n
        weights = weights.reshape(w_shape)

        spread = np.sum(weights * ens_sorted, axis=ens_axis)

        return mae - spread

    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        # Determine core dimension
        if isinstance(axis, int):
            ens_dim = ensemble.dims[axis]
        else:
            ens_dim = axis

        res = xr.apply_ufunc(
            _crps_numpy,
            ensemble,
            obs,
            input_core_dims=[[ens_dim], []],
            output_core_dims=[[]],
            kwargs={"ens_axis": -1},
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        return _update_history(res, "Continuous Ranked Probability Score (CRPS)")

    return _crps_numpy(np.asarray(ensemble), np.asarray(obs), ens_axis=axis)


def spread_error(
    ensemble: Union[xr.DataArray, np.ndarray],
    obs: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
    dim: Optional[Union[str, Iterable[str]]] = None,
    reduce_axis: Optional[Union[int, Iterable[int]]] = None,
) -> Tuple[Any, Any]:
    """
    Spread-Error Relationship for ensemble forecasts (Aero Protocol).

    Typical Use Cases
    -----------------
    - Assessing if the ensemble spread is a good proxy for the forecast error.
    - Ideally, mean spread should equal RMSE of the ensemble mean.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the mean spread and error (xarray only).
        If None, reduces over all dimensions.
    reduce_axis : int or iterable of int, optional
        Axis or axes along which to compute the mean spread and error (numpy only).

    Returns
    -------
    mean_spread : float, numpy.ndarray, or xarray.DataArray
        Mean ensemble spread.
    mean_error : float, numpy.ndarray, or xarray.DataArray
        Mean absolute error of ensemble mean vs. obs.
    """
    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        # Resolve ensemble dimension
        e_dim = axis
        if isinstance(axis, int):
            e_dim = ensemble.dims[axis]

        spread_field = ensemble.std(dim=e_dim)
        ens_mean_field = ensemble.mean(dim=e_dim)
        error_field = abs(ens_mean_field - obs)

        # Average over specified dimensions (or all remaining)
        m_spread = spread_field.mean(dim=dim)
        m_error = error_field.mean(dim=dim)

        return _update_history(m_spread, "Mean Ensemble Spread"), _update_history(m_error, "Mean Ensemble Error")

    # NumPy path
    ens = np.asarray(ensemble)
    observation = np.asarray(obs)

    # Calculate spread and ensemble mean
    spread_f = np.std(ens, axis=axis)
    ens_m_f = np.mean(ens, axis=axis)
    error_f = np.abs(ens_m_f - observation)

    # Average over specified axes
    m_spread = np.nanmean(spread_f, axis=reduce_axis)
    m_error = np.nanmean(error_f, axis=reduce_axis)

    if np.ndim(m_spread) == 0:
        return float(m_spread), float(m_error)
    return m_spread, m_error


def BSS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: float,
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, Iterable[int]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Brier Skill Score (BSS) for probabilistic forecasts (Aero Protocol).

    Typical Use Cases
    -----------------
    - Evaluating the accuracy of probabilistic binary forecasts relative to climatology.
    - Common in meteorological verification for event occurrence.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed binary outcomes (0 or 1) or continuous values (will be binarized).
    mod : xarray.DataArray or numpy.ndarray
        Forecast probabilities (0 to 1) or continuous values (will be binarized).
    threshold : float
        Threshold for converting values to binary events.
    dim : str or iterable of str, optional
        Dimension(s) along which to compute the score (xarray only).
    axis : int or iterable of int, optional
        Axis or axes along which to compute the score (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        Brier Skill Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Binarize if not already
        o_bin = (obs >= threshold).astype(float)
        m_prob = (mod >= threshold).astype(float)

        bs = ((m_prob - o_bin) ** 2).mean(dim=dim)
        obs_clim = o_bin.mean(dim=dim)
        bs_ref = ((obs_clim - o_bin) ** 2).mean(dim=dim)

        res = xr.where(bs_ref != 0, 1.0 - (bs / bs_ref), 0.0)
        return _update_history(res, "Brier Skill Score (BSS)")

    o = np.asarray(obs)
    m = np.asarray(mod)
    o_bin = (o >= threshold).astype(float)
    m_prob = (m >= threshold).astype(float)

    bs = np.nanmean((m_prob - o_bin) ** 2, axis=axis)
    obs_clim = np.nanmean(o_bin, axis=axis)
    if axis is not None:
        # Keep dimensions for subtraction
        obs_clim_kd = np.nanmean(o_bin, axis=axis, keepdims=True)
    else:
        obs_clim_kd = obs_clim

    bs_ref = np.nanmean((obs_clim_kd - o_bin) ** 2, axis=axis)

    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(bs_ref != 0, 1.0 - (bs / bs_ref), 0.0)
        return res.item() if np.ndim(res) == 0 else res


def _sal_numpy(
    obs_2d: np.ndarray,
    mod_2d: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Core NumPy implementation of SAL for a 2D field.

    Parameters
    ----------
    obs_2d : numpy.ndarray
        Observed 2D field.
    mod_2d : numpy.ndarray
        Model 2D field.
    threshold : float, optional
        Threshold for object identification.

    Returns
    -------
    S, A, L : float
        Structure, Amplitude, and Location components.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.random.rand(10, 10)
    >>> mod = np.random.rand(10, 10)
    >>> _sal_numpy(obs, mod)
    """
    import scipy.ndimage as ndi

    if threshold is None:
        threshold = np.nanmean(obs_2d)

    # Amplitude
    m_mod = np.nanmean(mod_2d)
    m_obs = np.nanmean(obs_2d)
    denom_a = m_mod + m_obs
    A = 2 * (m_mod - m_obs) / denom_a if denom_a != 0 else 0.0

    # Structure
    def structure(X):
        mask = threshold <= X
        if not np.any(mask):
            return 0.0, 0.0
        labeled, n = ndi.label(mask)
        masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
        max_mass = np.max(masses)
        total_mass = np.sum(masses)
        return max_mass, total_mass

    max_mod, sum_mod = structure(mod_2d)
    max_obs, sum_obs = structure(obs_2d)
    denom_s = (max_mod / sum_mod + max_obs / sum_obs) if sum_mod > 0 and sum_obs > 0 else 0
    S = 2 * (max_mod / sum_mod - max_obs / sum_obs) / denom_s if denom_s != 0 else 0.0

    # Location
    def centroid_and_spread(X):
        mask = threshold <= X
        if not np.any(mask):
            return np.array([0.0, 0.0]), 0.0, 0.0
        labeled, n = ndi.label(mask)
        centers = np.array(ndi.center_of_mass(X, labeled, index=np.arange(1, n + 1)))
        masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
        # Total mass-weighted center of the field's objects
        c = np.average(centers, axis=0, weights=masses)
        # Spread
        r = np.average(np.linalg.norm(centers - c, axis=1), weights=masses)
        return c, masses, r

    c_mod, _, r_mod = centroid_and_spread(mod_2d)
    c_obs, _, r_obs = centroid_and_spread(obs_2d)

    dist = np.linalg.norm(c_mod - c_obs)
    max_dist = np.sqrt(obs_2d.shape[0] ** 2 + obs_2d.shape[1] ** 2)
    L1 = dist / max_dist if max_dist != 0 else 0.0
    L2 = abs(r_mod - r_obs) / max_dist if max_dist != 0 else 0.0
    L = L1 + L2
    return float(S), float(A), float(L)


def SAL(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: Optional[float] = None,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> Union[
    Tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[float, float, float],
]:
    """
    Structure-Amplitude-Location (SAL) score for spatial verification (Aero Protocol).

    This implementation is vectorized over non-spatial dimensions using `xarray.apply_ufunc`
    and supports lazy evaluation for dimensions other than the spatial ones.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed field. Should be 2D (lat, lon) or multi-dimensional.
    mod : xarray.DataArray or numpy.ndarray
        Model field.
    threshold : float, optional
        Threshold for object identification. If None, uses mean of observations
        per slice (Lazy-friendly).
    lat_dim : str, optional
        Name of the latitude dimension. Default is 'lat'.
    lon_dim : str, optional
        Name of the longitude dimension. Default is 'lon'.

    Returns
    -------
    S, A, L : xarray.DataArray, numpy.ndarray, or float
        Structure, Amplitude, and Location components.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> obs = xr.DataArray(np.random.rand(10, 10, 10), dims=['time', 'lat', 'lon'])
    >>> mod = xr.DataArray(np.random.rand(10, 10, 10), dims=['time', 'lat', 'lon'])
    >>> S, A, L = SAL(obs, mod)
    """
    is_xr = isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray)

    if is_xr:
        obs, mod = xr.align(obs, mod, join="inner")
        # Determine spatial dimensions
        if lat_dim not in obs.dims or lon_dim not in obs.dims:
            # Fallback to last two dimensions if lat/lon not found
            spatial_dims = [obs.dims[-2], obs.dims[-1]]
        else:
            spatial_dims = [lat_dim, lon_dim]
    else:
        # For numpy, assume last two dimensions are spatial
        spatial_dims = [-2, -1]

    def _sal_wrapper(o: np.ndarray, m: np.ndarray, t: Optional[float]) -> Tuple[float, float, float]:
        """
        Wrapper for _sal_numpy to be used with xarray.apply_ufunc.

        Parameters
        ----------
        o : numpy.ndarray
            Observed field slice.
        m : numpy.ndarray
            Model field slice.
        t : float, optional
            Threshold.

        Returns
        -------
        Tuple[float, float, float]
            S, A, L components.

        Examples
        --------
        >>> import numpy as np
        >>> o = np.random.rand(10, 10)
        >>> m = np.random.rand(10, 10)
        >>> _sal_wrapper(o, m, 0.5)
        """
        return _sal_numpy(o, m, t)

    res_s, res_a, res_l = xr.apply_ufunc(
        _sal_wrapper,
        obs,
        mod,
        threshold,
        input_core_dims=[spatial_dims, spatial_dims, []],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )

    if is_xr:
        res_s = _update_history(res_s, "SAL: Structure component")
        res_a = _update_history(res_a, "SAL: Amplitude component")
        res_l = _update_history(res_l, "SAL: Location component")

    return res_s, res_a, res_l


def ensemble_mean(
    ensemble: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the ensemble mean.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Ensemble mean.
    """
    if isinstance(ensemble, xr.DataArray):
        dim = axis
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        res = ensemble.mean(dim=dim)
        return _update_history(res, "Ensemble Mean")
    return np.mean(ensemble, axis=axis)


def ensemble_std(
    ensemble: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the ensemble standard deviation.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Ensemble standard deviation.
    """
    if isinstance(ensemble, xr.DataArray):
        dim = axis
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        res = ensemble.std(dim=dim)
        return _update_history(res, "Ensemble Standard Deviation")
    return np.std(ensemble, axis=axis)


def rank_histogram(
    ensemble: Union[xr.DataArray, np.ndarray],
    obs: Union[xr.DataArray, np.ndarray],
    axis: Union[int, str] = 0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the rank histogram counts.

    Parameters
    ----------
    ensemble : xarray.DataArray or numpy.ndarray
        Ensemble forecasts.
    obs : xarray.DataArray or numpy.ndarray
        Observed values.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Rank histogram counts.

    Examples
    --------
    >>> import numpy as np
    >>> ens = np.array([[1, 2], [2, 3], [3, 4]])
    >>> obs = np.array([2, 3])
    >>> rank_histogram(ens, obs, axis=0)
    array([0., 0., 2., 0.])
    """

    def _rank_numpy(ens, observation, ens_axis=0):
        o_exp = np.expand_dims(observation, ens_axis)
        full_ensemble = np.concatenate([ens, o_exp], axis=ens_axis)
        ranks = np.argsort(full_ensemble, axis=ens_axis)
        obs_rank = np.argmax(ranks == ens.shape[ens_axis], axis=ens_axis)
        n_ens = ens.shape[ens_axis]
        hist, _ = np.histogram(obs_rank, bins=np.arange(n_ens + 2))
        return hist.astype(float)

    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        if isinstance(axis, int):
            ens_dim = ensemble.dims[axis]
        else:
            ens_dim = axis

        def _rank_ufunc(ens, observation):
            o_exp = np.expand_dims(observation, -1)
            full_ensemble = np.concatenate([ens, o_exp], axis=-1)
            ranks = np.argsort(full_ensemble, axis=-1)
            return np.argmax(ranks == ens.shape[-1], axis=-1)

        obs_rank = xr.apply_ufunc(
            _rank_ufunc,
            ensemble,
            obs,
            input_core_dims=[[ens_dim], []],
            output_core_dims=[[]],
            dask="parallelized",
            output_dtypes=[int],
        )

        n_ens = ensemble.sizes[ens_dim]
        bins = np.arange(n_ens + 2)
        if hasattr(obs_rank.data, "dask"):
            import dask.array as da

            hist, _ = da.histogram(obs_rank.data, bins=bins)
            res = xr.DataArray(hist, dims="rank", coords={"rank": np.arange(n_ens + 1)})
        else:
            hist, _ = np.histogram(obs_rank.values, bins=bins)
            res = xr.DataArray(hist, dims="rank", coords={"rank": np.arange(n_ens + 1)})
        return _update_history(res, "Rank Histogram")

    return _rank_numpy(np.asarray(ensemble), np.asarray(obs), ens_axis=axis)
