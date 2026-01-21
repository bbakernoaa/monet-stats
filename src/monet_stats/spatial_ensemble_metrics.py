"""
Spatial and Ensemble Metrics for Atmospheric Sciences following the Aero Protocol.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def EDS(
    obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray], threshold: float
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Extreme Dependency Score (EDS) for rare event detection.

    Typical Use Cases
    -----------------
    - Evaluating the skill of model in predicting extreme/rare events.
    - Used in meteorology for rare weather phenomena.

    Typical Values and Range
    ------------------------
    - Range: -1 to 1
    - 1: Perfect skill
    - 0: No skill (random)
    - Negative: Worse than random

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed field.
    mod : numpy.ndarray or xarray.DataArray
        Model field.
    threshold : float
        Event threshold.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Extreme Dependency Score.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.spatial_ensemble_metrics import EDS
    >>> obs = np.zeros((5, 5)); obs[2, 2] = 1
    >>> mod = np.zeros((5, 5)); mod[2, 3] = 1
    >>> EDS(obs, mod, threshold=0.5)
    0.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_bin = obs >= threshold
        mod_bin = mod >= threshold

        hits = (obs_bin & mod_bin).sum().astype(float)
        n_obs = obs_bin.sum().astype(float)
        n_mod = mod_bin.sum().astype(float)
        n = obs.size

        p = n_obs / n
        q = n_mod / n

        # EDS = log(hits/n) / log(p*q)
        # Using xr.where to maintain laziness
        result = xr.where((hits > 0) & (p > 0) & (q > 0), np.log(hits / n) / np.log(p * q), np.nan)
        return _update_history(result, "EDS")
    else:
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
        return np.log(hits / n) / np.log(p * q) if p > 0 and q > 0 else np.nan


def CRPS(
    ensemble: Union[np.ndarray, xr.DataArray],
    obs: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str]] = 0,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Continuous Ranked Probability Score (CRPS) for ensemble forecasts.
    (Existing discrete approximation: Sum of squared CDF differences).

    Parameters
    ----------
    ensemble : numpy.ndarray or xarray.DataArray
        Ensemble forecasts, shape (n_ensemble, ...).
    obs : numpy.ndarray or xarray.DataArray
        Observed values, shape (...).
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        CRPS values, shape (...).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.spatial_ensemble_metrics import CRPS
    >>> ens = np.array([[1, 2], [2, 3], [3, 4]])
    >>> obs = np.array([2, 3])
    >>> CRPS(ens, obs)
    array([0.22222222, 0.22222222])
    """
    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        if isinstance(axis, int):
            ens_dim = ensemble.dims[axis]
        else:
            ens_dim = axis or "ensemble"

        # Sort along ensemble dimension
        ens_sorted = xr.apply_ufunc(
            np.sort,
            ensemble,
            input_core_dims=[[ens_dim]],
            output_core_dims=[[ens_dim]],
            kwargs={"axis": -1},
            dask="parallelized",
            output_dtypes=[ensemble.dtype],
        )
        n = ensemble.sizes[ens_dim]

        # Compute empirical CDFs
        cdf_ens = xr.DataArray(np.arange(1, n + 1) / n, dims=[ens_dim])

        cdf_obs = (ens_sorted >= obs).astype(float)
        crps = ((cdf_ens - cdf_obs) ** 2).sum(dim=ens_dim)
        return _update_history(crps, "CRPS")
    else:
        ens = np.asarray(ensemble)
        obs_arr = np.asarray(obs)
        ax = axis if isinstance(axis, int) else 0
        ens_sorted = np.sort(ens, axis=ax)
        n = ens.shape[ax]
        # Compute empirical CDFs
        cdf_ens = np.arange(1, n + 1) / n
        shape = [1] * ens.ndim
        shape[ax] = n
        cdf_ens = np.reshape(cdf_ens, shape)
        # Broadcast obs for comparison
        obs_broadcast = np.expand_dims(obs_arr, ax)
        cdf_obs = (ens_sorted >= obs_broadcast).astype(float)
        crps = np.sum((cdf_ens - cdf_obs) ** 2, axis=ax)
        return crps


def spread_error(
    ensemble: Union[np.ndarray, xr.DataArray],
    obs: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str]] = 0,
) -> Tuple[Any, Any]:
    """
    Spread-Error Relationship for ensemble forecasts.

    Parameters
    ----------
    ensemble : numpy.ndarray or xarray.DataArray
        Ensemble forecasts, shape (n_ensemble, ...).
    obs : numpy.ndarray or xarray.DataArray
        Observed values, shape (...).
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    mean_spread : float or xarray.DataArray
        Mean ensemble spread.
    mean_error : float or xarray.DataArray
        Mean absolute error of ensemble mean vs. obs.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.spatial_ensemble_metrics import spread_error
    >>> ens = np.array([[1, 2], [2, 3], [3, 4]])
    >>> obs = np.array([2, 3])
    >>> spread_error(ens, obs)
    (0.816496580927726, 0.3333333333333333)
    """
    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        if isinstance(axis, int):
            ens_dim = ensemble.dims[axis]
        else:
            ens_dim = axis or "ensemble"

        spread = ensemble.std(dim=ens_dim)
        ens_mean = ensemble.mean(dim=ens_dim)
        error = abs(ens_mean - obs)

        mean_spread = spread.mean()
        mean_error = error.mean()

        return _update_history(mean_spread, "spread_error_spread"), _update_history(mean_error, "spread_error_error")
    else:
        ens = np.asarray(ensemble)
        obs_arr = np.asarray(obs)
        ax = axis if isinstance(axis, int) else 0
        spread = np.std(ens, axis=ax)
        ens_mean = np.mean(ens, axis=ax)
        error = np.abs(ens_mean - obs_arr)
        return np.mean(spread), np.mean(error)


def BSS(
    obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray], threshold: float
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Brier Skill Score (BSS) for probabilistic forecasts.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed binary outcomes (0 or 1) or values.
    mod : numpy.ndarray or xarray.DataArray
        Forecast probabilities (0 to 1).
    threshold : float
        Probability threshold for converting forecast to binary.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Brier Skill Score.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        mod_binary = (mod >= threshold).astype(float)
        bs = ((mod_binary - obs) ** 2).mean()
        obs_clim = obs.mean()
        bs_ref = ((obs_clim - obs) ** 2).mean()
        result = xr.where(bs_ref != 0, 1 - (bs / bs_ref), 0.0)
        return _update_history(result, "BSS")
    else:
        obs_arr = np.asarray(obs)
        mod_arr = np.asarray(mod)
        mod_binary = (mod_arr >= threshold).astype(float)
        bs = np.mean((mod_binary - obs_arr) ** 2)
        obs_clim = np.mean(obs_arr)
        bs_ref = np.mean((obs_clim - obs_arr) ** 2)
        return 1 - (bs / bs_ref) if bs_ref != 0 else 0.0


def ensemble_mean(
    ensemble: Union[np.ndarray, xr.DataArray], axis: Optional[Union[int, str]] = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the ensemble mean across ensemble members.

    Parameters
    ----------
    ensemble : numpy.ndarray or xarray.DataArray
        Ensemble forecasts.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Ensemble mean.
    """
    if isinstance(ensemble, xr.DataArray):
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        else:
            dim = axis
        result = ensemble.mean(dim=dim, keep_attrs=True)
        return _update_history(result, "ensemble_mean")
    return np.mean(ensemble, axis=axis)


def ensemble_std(
    ensemble: Union[np.ndarray, xr.DataArray], axis: Optional[Union[int, str]] = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the ensemble standard deviation across ensemble members.

    Parameters
    ----------
    ensemble : numpy.ndarray or xarray.DataArray
        Ensemble forecasts.
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Ensemble standard deviation.
    """
    if isinstance(ensemble, xr.DataArray):
        if isinstance(axis, int):
            dim = ensemble.dims[axis]
        else:
            dim = axis
        result = ensemble.std(dim=dim, keep_attrs=True)
        return _update_history(result, "ensemble_std")
    return np.std(ensemble, axis=axis)


def rank_histogram(
    ensemble: Union[np.ndarray, xr.DataArray], obs: Union[np.ndarray, xr.DataArray], axis: Optional[Union[int, str]] = 0
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the rank histogram (Talagrand diagram) for ensemble forecasts.

    Parameters
    ----------
    ensemble : numpy.ndarray or xarray.DataArray
        Ensemble forecasts, shape (n_ensemble, ...).
    obs : numpy.ndarray or xarray.DataArray
        Observed values, shape (...).
    axis : int or str, optional
        Axis or dimension corresponding to ensemble members. Default is 0.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Rank histogram counts.
    """
    if isinstance(ensemble, xr.DataArray) and isinstance(obs, xr.DataArray):
        if isinstance(axis, int):
            ens_dim = ensemble.dims[axis]
        else:
            ens_dim = axis or "ensemble"

        n = ensemble.sizes[ens_dim]
        # Rank is number of ensemble members less than observation
        rank = (ensemble < obs).sum(dim=ens_dim)

        # To get histogram, we count occurrences of each rank 0..n
        counts = []
        for i in range(n + 1):
            counts.append((rank == i).sum())

        result = xr.concat(counts, dim="rank")
        result.coords["rank"] = np.arange(n + 1)
        return _update_history(result, "rank_histogram")
    else:
        ens = np.asarray(ensemble)
        obs_arr = np.asarray(obs)
        ax = axis if isinstance(axis, int) else 0

        # Add observed values to ensemble for ranking
        full_ensemble = np.concatenate([ens, obs_arr[np.newaxis, ...]], axis=ax)

        # Sort along ensemble axis
        sorted_indices = np.argsort(full_ensemble, axis=ax)

        # Find rank of observation (which was appended as the last element)
        ranks = np.where(sorted_indices == ens.shape[ax], 1, 0)

        # Sum along other dimensions to get histogram
        sum_axes = tuple(i for i in range(ranks.ndim) if i != ax)
        if sum_axes:
            rank_hist = np.sum(ranks, axis=sum_axes)
        else:
            rank_hist = ranks

        return rank_hist


def SAL(
    obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray], threshold: Optional[float] = None
) -> Tuple[Any, Any, Any]:
    """
    Structure-Amplitude-Location (SAL) score for spatial verification.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed field (2D).
    mod : numpy.ndarray or xarray.DataArray
        Model field (2D).
    threshold : float, optional
        Threshold for object identification. If None, uses mean of obs.

    Returns
    -------
    S : float or xarray.DataArray
        Structure component.
    A : float or xarray.DataArray
        Amplitude component.
    L : float or xarray.DataArray
        Location component.
    """
    import scipy.ndimage as ndi

    def _sal_numpy(o, m, thr):
        if thr is None:
            thr = np.mean(o)
        # Amplitude
        if (np.mean(m) + np.mean(o)) == 0:
            a_score = 0.0
        else:
            a_score = 2 * (np.mean(m) - np.mean(o)) / (np.mean(m) + np.mean(o))

        # Structure
        def structure(X):
            labeled, n = ndi.label(thr <= X)
            if n == 0:
                return 0.0, 0.0
            masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
            return np.max(masses), np.sum(masses)

        max_mod, sum_mod = structure(m)
        max_obs, sum_obs = structure(o)
        if sum_mod > 0 and sum_obs > 0:
            s_score = 2 * (max_mod / sum_mod - max_obs / sum_obs) / (max_mod / sum_mod + max_obs / sum_obs)
        else:
            s_score = np.nan

        # Location
        def centroid(X):
            labeled, n = ndi.label(thr <= X)
            if n == 0:
                return np.array([np.nan, np.nan])
            centers = np.array(ndi.center_of_mass(X, labeled, index=np.arange(1, n + 1)))
            masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
            return np.average(centers, axis=0, weights=masses)

        c_mod = centroid(m)
        c_obs = centroid(o)
        if np.any(np.isnan(c_mod)) or np.any(np.isnan(c_obs)):
            l1 = np.nan
        else:
            l1 = np.linalg.norm(c_mod - c_obs) / np.sqrt(o.shape[0] ** 2 + o.shape[1] ** 2)

        def spread(X):
            labeled, n = ndi.label(thr <= X)
            if n == 0:
                return 0.0
            centers = np.array(ndi.center_of_mass(X, labeled, index=np.arange(1, n + 1)))
            masses = ndi.sum(X, labeled, index=np.arange(1, n + 1))
            c = np.average(centers, axis=0, weights=masses)
            return np.average(np.linalg.norm(centers - c, axis=1), weights=masses)

        r_mod = spread(m)
        r_obs = spread(o)
        l2 = abs(r_mod - r_obs) / np.sqrt(o.shape[0] ** 2 + o.shape[1] ** 2)
        return s_score, a_score, l1 + l2

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Use apply_ufunc to wrap the numpy logic.
        # Note: This will force computation if input is Dask.
        res = xr.apply_ufunc(
            _sal_numpy,
            obs,
            mod,
            input_core_dims=[obs.dims, mod.dims],
            output_core_dims=[[], [], []],
            kwargs={"thr": threshold},
            dask="forbidden",
        )
        return _update_history(res[0], "SAL_S"), _update_history(res[1], "SAL_A"), _update_history(res[2], "SAL_L")
    else:
        return _sal_numpy(np.asarray(obs), np.asarray(mod), threshold)
