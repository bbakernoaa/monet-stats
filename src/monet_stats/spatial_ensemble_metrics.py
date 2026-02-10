"""
Spatial and Ensemble Metrics for Atmospheric Sciences (Aero Protocol Compliant)
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def EDS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: float,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Extreme Dependency Score (EDS) for rare event detection.

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

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
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
        hits = (obs_bin & mod_bin).sum()
        n_obs = obs_bin.sum()
        n_mod = mod_bin.sum()
        n = obs.size

        # Use xr.where for lazy evaluation
        p = n_obs / n
        q = n_mod / n

        # We need to handle the log carefully for dask
        eds = np.log(hits / n) / np.log(p * q)
        # Handle cases where hits=0 or n_obs/n_mod=0 which would result in inf/nan
        # EDS is undefined if p=0 or q=0 or hits=0
        res = xr.where((hits > 0) & (p > 0) & (q > 0), eds, np.nan)
        return _update_history(res, "Extreme Dependency Score (EDS)")

    obs_bin = np.asarray(obs) >= threshold
    mod_bin = np.asarray(mod) >= threshold
    hits = np.logical_and(obs_bin, mod_bin).sum()
    n_obs = obs_bin.sum()
    n_mod = mod_bin.sum()
    n = np.size(obs)
    if hits == 0 or n_obs == 0 or n_mod == 0:
        return np.nan
    p = n_obs / n
    q = n_mod / n
    return np.log(hits / n) / np.log(p * q)


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

    def _crps_numpy(ens, observation, ens_axis=0):
        ens_sorted = np.sort(ens, axis=ens_axis)
        n = ens.shape[ens_axis]
        # Compute empirical CDFs
        cdf_ens = np.arange(1, n + 1) / n
        shape = [1] * ens.ndim
        shape[ens_axis] = n
        cdf_ens = np.reshape(cdf_ens, shape)
        # Broadcast obs for comparison
        obs_broadcast = np.expand_dims(observation, ens_axis)
        cdf_obs = (ens_sorted >= obs_broadcast).astype(float)
        return np.sum((cdf_ens - cdf_obs) ** 2, axis=ens_axis)

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
) -> Tuple[Any, Any]:
    """
    Spread-Error Relationship for ensemble forecasts.

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

    Returns
    -------
    mean_spread : float or xarray.DataArray
        Mean ensemble spread.
    mean_error : float or xarray.DataArray
        Mean absolute error of ensemble mean vs. obs.
    """
    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        else:
            dim = axis

        spread = ensemble.std(dim=dim)
        ens_mean = ensemble.mean(dim=dim)
        error = abs(ens_mean - obs)

        # We return means over all remaining dimensions as well?
        # The original implementation returned np.mean(spread), np.mean(error)
        # which are scalars.
        m_spread = spread.mean()
        m_error = error.mean()

        return _update_history(m_spread, "Mean Ensemble Spread"), _update_history(m_error, "Mean Ensemble Error")

    ens = np.asarray(ensemble)
    observation = np.asarray(obs)
    spread = np.std(ens, axis=axis)
    ens_mean = np.mean(ens, axis=axis)
    error = np.abs(ens_mean - observation)
    return np.mean(spread), np.mean(error)


def BSS(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    threshold: float,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Brier Skill Score (BSS) for probabilistic forecasts.

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

    Returns
    -------
    xarray.DataArray or numpy.ndarray or float
        Brier Skill Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Binarize if not already
        o_bin = (obs >= threshold).astype(float)
        m_prob = (mod >= threshold).astype(float)

        bs = ((m_prob - o_bin) ** 2).mean()
        obs_clim = o_bin.mean()
        bs_ref = ((obs_clim - o_bin) ** 2).mean()

        res = xr.where(bs_ref != 0, 1 - (bs / bs_ref), 0.0)
        return _update_history(res, "Brier Skill Score (BSS)")

    o = np.asarray(obs)
    m = np.asarray(mod)
    o_bin = (o >= threshold).astype(float)
    m_prob = (m >= threshold).astype(float)

    bs = np.mean((m_prob - o_bin) ** 2)
    obs_clim = np.mean(o_bin)
    bs_ref = np.mean((obs_clim - o_bin) ** 2)

    if bs_ref == 0:
        return 0.0
    return 1 - (bs / bs_ref)


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
