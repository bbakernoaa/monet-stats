"""
Temporal and phase error metrics for model evaluation (Aero Protocol Compliant).
"""

from typing import Iterable, Optional, Union

import numpy as np
import xarray as xr
from scipy.signal import find_peaks_cwt

from .utils_stats import _resolve_axis_to_dim, _update_history, ensure_single_chunk


def DynamicTimeWarping(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    dim: Optional[Union[str, Iterable[str]]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[xr.DataArray, np.ndarray, float]:
    """
    Compute the Dynamic Time Warping (DTW) distance (Aero Protocol).

    Typical Use Cases
    -----------------
    - Measuring similarity between two temporal sequences by finding the optimal alignment.
    - Quantifying phase errors in diurnal cycles.
    - Robust against slight temporal shifts or timing errors.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed temporal sequence.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted temporal sequence.
    dim : str or iterable of str, optional
        Dimension along which to compute the DTW distance (xarray only).
    axis : int, str, or iterable of int or str, optional
        Axis or axes along which to compute the DTW distance (numpy only).

    Returns
    -------
    xarray.DataArray, numpy.ndarray, or float
        The DTW distance.
    """

    def _dtw_numpy(o: np.ndarray, m: np.ndarray) -> float:
        o_valid = o[~np.isnan(o)]
        m_valid = m[~np.isnan(m)]
        n, mm = len(o_valid), len(m_valid)
        if n == 0 or mm == 0:
            return np.nan

        dtw_matrix = np.full((n + 1, mm + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, mm + 1):
                cost = abs(o_valid[i - 1] - m_valid[j - 1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

        return float(dtw_matrix[n, mm])

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        # Determine core dimensions first
        reduction_dim = _resolve_axis_to_dim(obs, dim if dim is not None else axis)
        if isinstance(reduction_dim, str):
            core_dims_obs = [reduction_dim]
        else:
            core_dims_obs = list(reduction_dim)

        reduction_dim_mod = _resolve_axis_to_dim(mod, dim if dim is not None else axis)
        if isinstance(reduction_dim_mod, str):
            core_dims_mod = [reduction_dim_mod]
        else:
            core_dims_mod = list(reduction_dim_mod)

        obs = ensure_single_chunk(obs, core_dims_obs)
        mod = ensure_single_chunk(mod, core_dims_mod)

        res = xr.apply_ufunc(
            _dtw_numpy,
            obs,
            mod,
            input_core_dims=[core_dims_obs, core_dims_mod],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            exclude_dims=set(core_dims_obs) | set(core_dims_mod),
        )
        return _update_history(res, "Dynamic Time Warping (DTW)")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)

    if axis is None:
        return _dtw_numpy(o_arr.flatten(), m_arr.flatten())

    res = np.apply_along_axis(lambda x, y: _dtw_numpy(x, y), axis, o_arr, m_arr)
    return res.item() if np.ndim(res) == 0 else res


def CrossWaveletTransform(
    obs: Union[xr.DataArray, np.ndarray],
    mod: Union[xr.DataArray, np.ndarray],
    widths: Optional[np.ndarray] = None,
    dim: str = "time",
    axis: int = -1,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Compute the Cross-Wavelet Transform (XWT) (Aero Protocol).

    Typical Use Cases
    -----------------
    - Decomposing time series to see at which temporal frequencies the model
      and observations are correlated or out of phase.
    - Identifying common time-frequency power and relative phase.

    Parameters
    ----------
    obs : xarray.DataArray or numpy.ndarray
        Observed time series.
    mod : xarray.DataArray or numpy.ndarray
        Model or predicted time series.
    widths : numpy.ndarray, optional
        Widths of the Morlet wavelet to use for the transform.
    dim : str, optional
        Dimension along which to compute the transform (xarray only).
    axis : int, optional
        Axis along which to compute the transform (numpy only).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The Cross-Wavelet Transform (complex array).
    """
    if widths is None:
        widths = np.arange(1, 31)

    def _xwt_numpy(o: np.ndarray, m: np.ndarray, w: np.ndarray) -> np.ndarray:
        # Since cwt and morlet2 are not directly exposed in some scipy versions,
        # we can use the private _cwt and _ricker or implement a basic one.
        # However, it seems _cwt is available in _peak_finding.
        try:
            from scipy.signal._peak_finding import _cwt
            from scipy.signal._wavelets import _ricker as ricker
        except ImportError:
            # Fallback or manual implementation if needed
            raise ImportError("Could not import required wavelet functions from scipy.signal")

        # Compute Continuous Wavelet Transform (CWT) for both
        cwt_obs = _cwt(o, ricker, w)
        cwt_mod = _cwt(m, ricker, w)
        # Cross-Wavelet Transform is W_xy = W_x * W_y*
        # Ensure complex multiplication
        return cwt_obs.astype(complex) * np.conj(cwt_mod.astype(complex))

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs = ensure_single_chunk(obs, dim)
        mod = ensure_single_chunk(mod, dim)

        res = xr.apply_ufunc(
            _xwt_numpy,
            obs,
            mod,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[["scale", dim]],
            kwargs={"w": widths},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[complex],
            dask_gufunc_kwargs={"output_sizes": {"scale": len(widths), dim: obs.sizes[dim]}},
            exclude_dims={dim},
        )
        res = res.assign_coords(scale=widths)
        return _update_history(res, "Cross-Wavelet Transform (XWT)")

    # NumPy path
    o_arr = np.asarray(obs)
    m_arr = np.asarray(mod)
    return _xwt_numpy(o_arr, m_arr, widths)
