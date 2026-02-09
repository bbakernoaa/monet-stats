from typing import Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def HSS(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Heidke Skill Score (HSS).

    Typical Use Cases
    -----------------
    - Evaluating categorical forecast skill (e.g., precipitation, air quality
      events).
    - Used in meteorology and environmental modeling to assess binary event
      prediction accuracy.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect forecast
    - 0: No skill (random forecast)
    - Negative values: Worse than random

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Modeled values.
    minval : float
        Minimum threshold value for event detection.
    maxval : float, optional
        Maximum threshold value for event detection.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        HSS value for the given threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import HSS
    >>> obs = np.array([1, 0, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> HSS(obs, mod, minval=0.5)
    -0.3333333333333333
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    if isinstance(denom, xr.DataArray):
        result = xr.where(denom > 0, 2 * (a * d - b * c) / denom, np.nan)
        return _update_history(result, "Heidke Skill Score (HSS)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, 2 * (a * d - b * c) / denom, np.nan)
            return result.item() if np.ndim(result) == 0 else result


def ETS(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Equitable Threat Score (ETS).

    Typical Use Cases
    -----------------
    - Evaluating forecast skill for rare events (e.g., precipitation, air quality
      exceedances).
    - Used in meteorology and environmental modeling to assess binary event
      prediction accuracy.

    Typical Values and Range
    ------------------------
    - Range: -1/3 to 1
    - 1: Perfect forecast
    - 0: No skill (random forecast)
    - Negative values: Worse than random

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Modeled values.
    minval : float
        Minimum threshold value for event detection.
    maxval : float, optional
        Maximum threshold value for event detection.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        ETS value for the given threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import ETS
    >>> obs = np.array([1, 0, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> ETS(obs, mod, minval=0.5)
    -0.2
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    total = a + b + c + d
    random_hits = ((a + b) * (a + c)) / total
    denom = a + b + c - random_hits
    if isinstance(denom, xr.DataArray):
        result = xr.where(denom > 0, (a - random_hits) / denom, np.nan)
        return _update_history(result, "Equitable Threat Score (ETS)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, (a - random_hits) / denom, np.nan)
            return result.item() if np.ndim(result) == 0 else result


def CSI(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Critical Success Index (CSI).

    Typical Use Cases
    -----------------
    - Evaluating forecast skill for rare or binary events (e.g., precipitation,
      air quality exceedances).
    - Used in meteorology and environmental modeling to assess event prediction
      accuracy.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 1: Perfect forecast
    - 0: No skill (no correct predictions)

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Modeled values.
    minval : float
        Minimum threshold value for event detection.
    maxval : float, optional
        Maximum threshold value for event detection.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        CSI value for the given threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import CSI
    >>> obs = np.array([1, 0, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> CSI(obs, mod, minval=0.5)
    0.3333333333333333
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    denom = a + b + c
    if isinstance(denom, xr.DataArray):
        result = xr.where(denom > 0, a / denom, np.nan)
        return _update_history(result, "Critical Success Index (CSI)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, a / denom, np.nan)
            return result.item() if np.ndim(result) == 0 else result


def scores(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Tuple[
    Union[np.number, np.ndarray, xr.DataArray],
    Union[np.number, np.ndarray, xr.DataArray],
    Union[np.number, np.ndarray, xr.DataArray],
    Union[np.number, np.ndarray, xr.DataArray],
]:
    """
    Calculate the 2x2 contingency table (Aero Protocol).

    Typical Use Cases
    -----------------
    - Obtaining the raw counts of hits, misses, false alarms, and correct negatives
      to compute custom categorical scores.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observation values ("truth").
    mod : numpy.ndarray or xarray.DataArray
        Model values ("prediction").
    minval : float
        Minimum threshold for event detection.
    maxval : float, optional
        Maximum threshold for event detection.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the scores.

    Returns
    -------
    Tuple[Union[np.number, np.ndarray, xr.DataArray], ...]
        A tuple of (hits, misses, false alarms, correct negatives).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.5, 1.8, 3.2, 3.8])
    >>> a, b, c, d = scores(obs, mod, minval=2.5)
    >>> print(f"Hits: {a}")
    Hits: 2
    """
    return _contingency_table(obs, mod, minval, maxval, axis=axis)


def POD(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Probability of Detection (POD) for a given event threshold.

    Typical Use Cases
    -----------------
    - Evaluating how well a model detects events above a critical threshold
      (e.g., pollution exceedances, precipitation events).
    - Used in contingency table analysis for categorical forecast verification.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval : float
        Minimum event threshold.
    maxval : float, optional
        Maximum event threshold.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    pod : numpy.number, numpy.ndarray, or xarray.DataArray
        Probability of detection.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import POD
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> POD(obs, mod, minval=0.5)
    0.5
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    denom = a + b
    if isinstance(denom, xr.DataArray):
        result = xr.where(denom > 0, a / denom, np.nan)
        return _update_history(result, "Probability of Detection (POD)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, a / denom, np.nan)
            return result.item() if np.ndim(result) == 0 else result


def FAR(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    False Alarm Rate (FAR) for a given event threshold.

    Typical Use Cases
    -----------------
    - Evaluating the frequency of false alarms in categorical forecasts (e.g.,
      precipitation, air quality events).
    - Used in meteorology and environmental modeling to assess forecast
      reliability.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 0: No false alarms (perfect reliability)
    - 1: All alarms are false (no reliability)

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval : float
        Minimum event threshold.
    maxval : float, optional
        Maximum event threshold.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        False alarm rate.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import FAR
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> FAR(obs, mod, minval=0.5)
    0.5
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    denom = a + c
    if isinstance(denom, xr.DataArray):
        result = xr.where(denom > 0, c / denom, np.nan)
        return _update_history(result, "False Alarm Rate (FAR)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, c / denom, np.nan)
            return result.item() if np.ndim(result) == 0 else result


def FBI(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Frequency Bias Index (FBI) for a given event threshold.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval : float
        Minimum event threshold.
    maxval : float, optional
        Maximum event threshold.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    fbi : numpy.number, numpy.ndarray, or xarray.DataArray
        Frequency bias index.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import FBI
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> FBI(obs, mod, minval=0.5)
    1.0
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    denom = a + b
    if isinstance(denom, xr.DataArray):
        result = xr.where(denom > 0, (a + c) / denom, np.nan)
        return _update_history(result, "Frequency Bias Index (FBI)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom > 0, (a + c) / denom, np.nan)
            return result.item() if np.ndim(result) == 0 else result


def TSS(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: float,
    maxval: Optional[float] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Hanssen-Kuipers Discriminant (True Skill Statistic, TSS).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval : float
        Minimum event threshold.
    maxval : float, optional
        Maximum event threshold.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the metric.

    Returns
    -------
    tss : numpy.number, numpy.ndarray, or xarray.DataArray
        True skill statistic.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import TSS
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> TSS(obs, mod, minval=0.5)
    0.0
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval, axis=axis)
    pod_denom = a + b
    pofd_denom = c + d

    if isinstance(pod_denom, xr.DataArray):
        pod = xr.where(pod_denom > 0, a / pod_denom, np.nan)
        pofd = xr.where(pofd_denom > 0, c / pofd_denom, np.nan)
        result = pod - pofd
        return _update_history(result, "True Skill Statistic (TSS)")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            pod = np.where(pod_denom > 0, a / pod_denom, np.nan)
            pofd = np.where(pofd_denom > 0, c / pofd_denom, np.nan)
            result = pod - pofd
            return result.item() if np.ndim(result) == 0 else result


def BSS_binary(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    threshold: float,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Binary Brier Skill Score for deterministic forecasts.

    Typical Use Cases
    -----------------
    - Evaluating the accuracy of deterministic binary forecasts (e.g.,
      precipitation yes/no).
    - Used in meteorology and environmental modeling to assess forecast skill
      relative to a reference.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect forecast
    - 0: Same skill as reference forecast
    - Negative: Worse than reference forecast

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed binary outcomes or continuous values.
    mod : numpy.ndarray or xarray.DataArray
        Forecast binary outcomes or continuous values.
    threshold : float
        Threshold value to convert continuous forecasts to binary.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the score.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Binary Brier Skill Score.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.contingency_metrics import BSS_binary
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([0, 1, 0, 0])
    >>> BSS_binary(obs, mod, threshold=0.5)
    0.5
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Handle axis vs dim
        if axis is not None and isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        obs_binary = (obs >= threshold).astype(float)
        mod_binary = (mod >= threshold).astype(float)

        bs = ((mod_binary - obs_binary) ** 2).mean(dim=dim)
        obs_clim = obs_binary.mean(dim=dim)
        bs_ref = ((obs_clim - obs_binary) ** 2).mean(dim=dim)

        result = xr.where(bs_ref > 0, 1.0 - (bs / bs_ref), 0.0)
        return _update_history(result, "Binary Brier Skill Score (BSS_binary)")
    else:
        obs_binary = (np.asarray(obs) >= threshold).astype(float)
        mod_binary = (np.asarray(mod) >= threshold).astype(float)

        bs = np.nanmean((mod_binary - obs_binary) ** 2, axis=axis)
        obs_clim = np.nanmean(obs_binary, axis=axis)
        if axis is not None:
            # Need to keep dims for subtraction
            obs_clim_kd = np.nanmean(obs_binary, axis=axis, keepdims=True)
        else:
            obs_clim_kd = obs_clim
        bs_ref = np.nanmean((obs_clim_kd - obs_binary) ** 2, axis=axis)

        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(bs_ref > 0, 1.0 - (bs / bs_ref), 0.0)
            return result.item() if np.ndim(result) == 0 else result


def _contingency_table(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval: Union[float, np.ndarray, xr.DataArray],
    maxval: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Tuple[
    Union[np.number, np.ndarray, xr.DataArray],
    Union[np.number, np.ndarray, xr.DataArray],
    Union[np.number, np.ndarray, xr.DataArray],
    Union[np.number, np.ndarray, xr.DataArray],
]:
    """
    Compute the 2x2 contingency table for event-based metrics (Aero Protocol).

    Supports vectorized thresholding. If `minval` is an array, it returns
    counts with a 'threshold' dimension.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval : float or array-like
        Minimum threshold value(s) for event detection.
    maxval : float or array-like, optional
        Maximum threshold value(s) for event detection (for range-based events).
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the table counts.

    Returns
    -------
    a : hits
    b : misses
    c : false alarms
    d : correct negatives

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.5, 1.8, 3.2, 3.8])
    >>> # Single threshold
    >>> a, b, c, d = _contingency_table(obs, mod, minval=2.5)
    >>> # Multiple thresholds (Vectorized)
    >>> thresholds = [1.0, 2.0, 3.0]
    >>> a, b, c, d = _contingency_table(obs, mod, minval=thresholds)
    >>> print(a.shape)
    (3,)
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")

        # Threshold vectorization for xarray
        if isinstance(minval, (list, np.ndarray, xr.DataArray)) and not isinstance(minval, float):
            if not isinstance(minval, xr.DataArray):
                minval = xr.DataArray(minval, dims="threshold", coords={"threshold": minval})
            if maxval is not None and not isinstance(maxval, xr.DataArray):
                maxval = xr.DataArray(maxval, dims="threshold", coords={"threshold": maxval})

        if axis is None:
            dim = obs.dims
        elif isinstance(axis, int):
            dim = obs.dims[axis]
        else:
            dim = axis

        mask = obs.notnull() & mod.notnull()

        if maxval is not None:
            obs_event = (obs >= minval) & (obs < maxval)
            mod_event = (mod >= minval) & (mod < maxval)
        else:
            obs_event = obs >= minval
            mod_event = mod >= minval

        # Important: only count where both have valid data
        obs_event = obs_event & mask
        mod_event = mod_event & mask

        a = (obs_event & mod_event).sum(dim=dim)
        b = (obs_event & ~mod_event).sum(dim=dim)
        c = (~obs_event & mod_event).sum(dim=dim)
        d = (~obs_event & ~mod_event & mask).sum(dim=dim)

        return a, b, c, d
    else:
        obs = np.asarray(obs)
        mod = np.asarray(mod)
        mask = ~np.isnan(obs) & ~np.isnan(mod)

        # Threshold vectorization for numpy
        if isinstance(minval, (list, np.ndarray)) and not isinstance(minval, (float, int)):
            minval = np.asarray(minval)
            # Add new axes to minval for broadcasting: (T, 1, 1, ...)
            minval_b = minval.reshape((len(minval),) + (1,) * obs.ndim)
            if maxval is not None:
                maxval_b = np.asarray(maxval).reshape((len(maxval),) + (1,) * obs.ndim)
            else:
                maxval_b = None
            # We will sum over the original axes, which are now shifted by 1
            if axis is None:
                calc_axis = tuple(range(1, obs.ndim + 1))
            elif isinstance(axis, int):
                calc_axis = axis + 1 if axis >= 0 else axis
            else:
                calc_axis = tuple(ax + 1 if ax >= 0 else ax for ax in axis)
        else:
            minval_b = minval
            maxval_b = maxval
            calc_axis = axis

        if maxval_b is not None:
            obs_event = (obs >= minval_b) & (obs < maxval_b)
            mod_event = (mod >= minval_b) & (mod < maxval_b)
        else:
            obs_event = obs >= minval_b
            mod_event = mod >= minval_b

        obs_event = obs_event & mask
        mod_event = mod_event & mask

        a = np.sum(obs_event & mod_event, axis=calc_axis)
        b = np.sum(obs_event & ~mod_event, axis=calc_axis)
        c = np.sum(~obs_event & mod_event, axis=calc_axis)
        d = np.sum(~obs_event & ~mod_event & mask, axis=calc_axis)

        return a, b, c, d


def HSS_max_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the Heidke Skill Score (HSS) over a range.

    Vectorized implementation (Aero Protocol).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval_range : float
        Minimum value of threshold range to test.
    maxval_range : float
        Maximum value of threshold range to test.
    step_size : float, optional
        Step size for testing thresholds. Default is 1.0.

    Returns
    -------
    optimal_threshold : float
        Threshold value that maximizes HSS.
    max_hss : float
        Maximum HSS value achieved.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> HSS_max_threshold(obs, mod, 1, 5, 0.5)
    (2.5, 1.0)
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    a, b, c, d = _contingency_table(obs, mod, minval=thresholds)

    denom = (a + c) * (c + d) + (a + b) * (b + d)
    with np.errstate(divide="ignore", invalid="ignore"):
        hss_values = np.where(denom > 0, 2 * (a * d - b * c) / denom, np.nan)

    if isinstance(hss_values, xr.DataArray):
        max_idx = hss_values.argmax(dim="threshold").values.item()
        max_hss = hss_values.isel(threshold=max_idx).values.item()
    else:
        max_idx = np.nanargmax(hss_values)
        max_hss = hss_values[max_idx]

    return float(thresholds[max_idx]), float(max_hss)


def ETS_max_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the Equitable Threat Score (ETS) over a range.

    Vectorized implementation (Aero Protocol).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval_range : float
        Minimum value of threshold range to test.
    maxval_range : float
        Maximum value of threshold range to test.
    step_size : float, optional
        Step size for testing thresholds. Default is 1.0.

    Returns
    -------
    optimal_threshold : float
        Threshold value that maximizes ETS.
    max_ets : float
        Maximum ETS value achieved.
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    a, b, c, d = _contingency_table(obs, mod, minval=thresholds)

    total = a + b + c + d
    random_hits = ((a + b) * (a + c)) / total
    denom = a + b + c - random_hits
    with np.errstate(divide="ignore", invalid="ignore"):
        ets_values = np.where(denom > 0, (a - random_hits) / denom, np.nan)

    if isinstance(ets_values, xr.DataArray):
        max_idx = ets_values.argmax(dim="threshold").values.item()
        max_ets = ets_values.isel(threshold=max_idx).values.item()
    else:
        max_idx = np.nanargmax(ets_values)
        max_ets = ets_values[max_idx]

    return float(thresholds[max_idx]), float(max_ets)


def POD_max_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the Probability of Detection (POD) over a range.

    Vectorized implementation (Aero Protocol).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval_range : float
        Minimum value of threshold range to test.
    maxval_range : float
        Maximum value of threshold range to test.
    step_size : float, optional
        Step size for testing thresholds. Default is 1.0.

    Returns
    -------
    optimal_threshold : float
        Threshold value that maximizes POD.
    max_pod : float
        Maximum POD value achieved.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> POD_max_threshold(obs, mod, 1, 5, 0.5)
    (1.0, 1.0)
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    a, b, c, d = _contingency_table(obs, mod, minval=thresholds)

    denom = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        pod_values = np.where(denom > 0, a / denom, np.nan)

    if isinstance(pod_values, xr.DataArray):
        max_idx = pod_values.argmax(dim="threshold").values.item()
        max_val = pod_values.isel(threshold=max_idx).values.item()
    else:
        max_idx = np.nanargmax(pod_values)
        max_val = pod_values[max_idx]

    return float(thresholds[max_idx]), float(max_val)


def FAR_min_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that minimizes the False Alarm Rate (FAR) over a range.

    Vectorized implementation (Aero Protocol).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval_range : float
        Minimum value of threshold range to test.
    maxval_range : float
        Maximum value of threshold range to test.
    step_size : float, optional
        Step size for testing thresholds. Default is 1.0.

    Returns
    -------
    optimal_threshold : float
        Threshold value that minimizes FAR.
    min_far : float
        Minimum FAR value achieved.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> FAR_min_threshold(obs, mod, 1, 5, 0.5)
    (1.5, 0.0)
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    a, b, c, d = _contingency_table(obs, mod, minval=thresholds)

    denom = a + c
    with np.errstate(divide="ignore", invalid="ignore"):
        far_values = np.where(denom > 0, c / denom, np.nan)

    if isinstance(far_values, xr.DataArray):
        min_idx = far_values.argmin(dim="threshold").values.item()
        min_val = far_values.isel(threshold=min_idx).values.item()
    else:
        min_idx = np.nanargmin(far_values)
        min_val = far_values[min_idx]

    return float(thresholds[min_idx]), float(min_val)
