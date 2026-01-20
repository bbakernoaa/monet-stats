from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


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
        history = f"HSS computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
        history = f"ETS computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
        history = f"CSI computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
    Calculate scores using the _contingency_table.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observation values ("truth").
    mod : numpy.ndarray or xarray.DataArray
        Model values ("prediction").
    minval : float
        Minimum threshold for event.
    maxval : float, optional
        Maximum threshold for event.
    axis : int, str, or iterable of such, optional
        Axis along which to compute the scores.

    Returns
    -------
    a, b, c, d
        Counts of hits, misses, false alarms, and correct negatives.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.5, 1.8, 3.2, 3.8])
    >>> a, b, c, d = scores(obs, mod, minval=2.5)
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
        history = f"POD computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
        history = f"FAR computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
        history = f"FBI computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
        history = f"TSS computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
        history = f"BSS_binary computed at {pd.Timestamp.now().isoformat()}"
        result.attrs["history"] = f"{result.attrs.get('history', '')}\n{history}".strip()
        return result
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
    Compute the 2x2 contingency table for event-based metrics.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    minval : float
        Minimum threshold value for event detection.
    maxval : float, optional
        Maximum threshold value for event detection (for range-based events).
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
    >>> a, b, c, d = _contingency_table(obs, mod, minval=2.5)
    >>> print(f"Hits: {a}, Misses: {b}, False Alarms: {c}, Correct Negatives: {d}")
    Hits: 2, Misses: 0, False Alarms: 0, Correct Negatives: 2
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
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

        if maxval is not None:
            obs_event = (obs >= minval) & (obs < maxval)
            mod_event = (mod >= minval) & (mod < maxval)
        else:
            obs_event = obs >= minval
            mod_event = mod >= minval

        obs_event = obs_event & mask
        mod_event = mod_event & mask

        a = np.sum(obs_event & mod_event, axis=axis)
        b = np.sum(obs_event & ~mod_event, axis=axis)
        c = np.sum(~obs_event & mod_event, axis=axis)
        d = np.sum(~obs_event & ~mod_event & mask, axis=axis)

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

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for binary classification in meteorological
      or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

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
    hss_values = []

    for threshold in thresholds:
        hss_val = HSS(obs, mod, threshold)
        if isinstance(hss_val, xr.DataArray):
            hss_val = hss_val.values.item()
        hss_values.append(hss_val)

    # Find the threshold that gives the maximum HSS
    max_idx = np.nanargmax(hss_values)
    optimal_threshold = thresholds[max_idx]
    max_hss = hss_values[max_idx]

    return float(optimal_threshold), float(max_hss)


def ETS_max_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the Equitable Threat Score (ETS) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for binary classification in meteorological
      or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

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

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> ETS_max_threshold(obs, mod, 1, 5, 0.5)
    (2.5, 1.0)
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    ets_values = []

    for threshold in thresholds:
        ets_val = ETS(obs, mod, threshold)
        if isinstance(ets_val, xr.DataArray):
            ets_val = ets_val.values.item()
        ets_values.append(ets_val)

    # Find the threshold that gives the maximum ETS
    max_idx = np.nanargmax(ets_values)
    optimal_threshold = thresholds[max_idx]
    max_ets = ets_values[max_idx]

    return float(optimal_threshold), float(max_ets)


def POD_max_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes the Probability of Detection (POD) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for maximizing detection rates in
      meteorological or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

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
    (2.5, 1.0)
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    pod_values = []

    for threshold in thresholds:
        pod_val = POD(obs, mod, threshold)
        if isinstance(pod_val, xr.DataArray):
            pod_val = pod_val.values.item()
        pod_values.append(pod_val)

    # Find the threshold that gives the maximum POD
    max_idx = np.nanargmax(pod_values)
    optimal_threshold = thresholds[max_idx]
    max_pod = pod_values[max_idx]

    return float(optimal_threshold), float(max_pod)


def FAR_min_threshold(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    minval_range: float,
    maxval_range: float,
    step_size: float = 1.0,
) -> Tuple[float, float]:
    """
    Find the threshold that minimizes the False Alarm Rate (FAR) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for minimizing false alarms in meteorological
      or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

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
    (2.5, 0.0)
    """
    thresholds = np.arange(minval_range, maxval_range, step_size)
    far_values = []

    for threshold in thresholds:
        far_val = FAR(obs, mod, threshold)
        if isinstance(far_val, xr.DataArray):
            far_val = far_val.values.item()
        far_values.append(far_val)

    # Find the threshold that gives the minimum FAR
    min_idx = np.nanargmin(far_values)
    optimal_threshold = thresholds[min_idx]
    min_far = far_values[min_idx]

    return float(optimal_threshold), float(min_far)
