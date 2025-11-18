import numpy as np


def HSS(obs, mod, minval, maxval=None):
    """
    Heidke Skill Score (HSS)

    Typical Use Cases
    -----------------
    - Evaluating categorical forecast skill (e.g., precipitation, air quality events).
    - Used in meteorology and environmental modeling to assess binary event prediction accuracy.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect forecast
    - 0: No skill (random forecast)
    - Negative values: Worse than random

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Modeled values.
    minval : float
        Threshold value for contingency table.

    Returns
    -------
    float
        HSS value for the given threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 0, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> stats.HSS(obs, mod, minval=0.5)
    # Output: HSS value between -∞ and 1
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    denom = (a + c) * (c + d) + (a + b) * (b + d)
    if denom > 0:
        return 2 * (a * d - b * c) / denom
    else:
        return np.nan


def ETS(obs, mod, minval, maxval=None):
    """
    Equitable Threat Score (ETS)

    Typical Use Cases
    -----------------
    - Evaluating forecast skill for rare events (e.g., precipitation, air quality exceedances).
    - Used in meteorology and environmental modeling to assess binary event prediction accuracy.

    Typical Values and Range
    ------------------------
    - Range: -1/3 to 1
    - 1: Perfect forecast
    - 0: No skill (random forecast)
    - Negative values: Worse than random

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Modeled values.
    minval : float
        Threshold value for contingency table.

    Returns
    -------
    float
        ETS value for the given threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 0, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> stats.ETS(obs, mod, minval=0.5, maxval=None)
    # Output: ETS value between -1/3 and 1
    """
    # Compute contingency table
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    total = a + b + c + d
    random_hits = ((a + b) * (a + c)) / total if total > 0 else 0
    denom = a + b + c - random_hits
    if denom > 0:
        return (a - random_hits) / denom
    else:
        return np.nan


def CSI(obs, mod, minval, maxval=None):
    """
    Critical Success Index (CSI)

    Typical Use Cases
    -----------------
    - Evaluating forecast skill for rare or binary events (e.g., precipitation, air quality exceedances).
    - Used in meteorology and environmental modeling to assess event prediction accuracy.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 1: Perfect forecast
    - 0: No skill (no correct predictions)

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Modeled values.
    minval : float
        Threshold value for contingency table.
    maxval : float
        Maximum threshold value (not used in calculation).

    Returns
    -------
    float
        CSI value for the given threshold.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 0, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> stats.CSI(obs, mod, minval=0.5, maxval=None)
    # Output: CSI value between 0 and 1
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    denom = a + b + c
    csi = a / denom if denom > 0 else np.nan
    return csi


def scores(obs, mod, minval, maxval=None):
    """Calculate scores using the new _contingency_table.

    Parameters
    ----------
    obs : array-like
        Observation values ("truth").
    mod : array-like
        Model values ("prediction").
        Should be the same size as `obs`.
    minval : float
        Threshold for event (used as threshold for _contingency_table).
    maxval : float, optional
        Unused, kept for compatibility.

    Returns
    -------
    a, b, c, d : float
        Counts of hits, misses, false alarms, and correct negatives.
    """
    return _contingency_table(obs, mod, minval, maxval)


def POD(obs, mod, minval, maxval=None):
    """
    Probability of Detection (POD) for a given event threshold.

    Typical Use Cases
    -----------------
    - Evaluating how well a model detects events above a critical threshold
      (e.g., pollution exceedances, precipitation events).
    - Used in contingency table analysis for categorical forecast verification.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
        Model or predicted values.
    threshold : float
        Event threshold.

    Returns
    -------
    pod : float
        Probability of detection.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> POD(obs, mod, threshold=0.5)
    0.5
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    return a / (a + b) if (a + b) > 0 else np.nan


def FAR(obs, mod, minval, maxval=None):
    """
    False Alarm Rate (FAR) for a given event threshold.

    Typical Use Cases
    -----------------
    - Evaluating the frequency of false alarms in categorical forecasts (e.g., precipitation, air quality events).
    - Used in meteorology and environmental modeling to assess forecast reliability.

    Typical Values and Range
    ------------------------
    - Range: 0 to 1
    - 0: No false alarms (perfect reliability)
    - 1: All alarms are false (no reliability)

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    minval : float
        Threshold value for contingency table.
    maxval : float, optional
        Maximum threshold value (not used in calculation).

    Returns
    -------
    float
        False alarm rate.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> stats.FAR(obs, mod, minval=0.5)
    0.5
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    return c / (a + c) if (a + c) > 0 else np.nan


def FBI(obs, mod, minval, maxval=None):
    """
    Frequency Bias Index (FBI) for a given event threshold.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
        Model or predicted values.
    threshold : float
        Event threshold.

    Returns
    -------
    fbi : float
        Frequency bias index.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> FBI(obs, mod, threshold=0.5)
    1.0
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    return (a + c) / (a + b) if (a + b) > 0 else np.nan


def TSS(obs, mod, minval, maxval=None):
    """
    Hanssen-Kuipers Discriminant (True Skill Statistic, TSS).

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
        Model or predicted values.
    threshold : float
        Event threshold.

    Returns
    -------
    tss : float
        True skill statistic.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([1, 1, 0, 0])
    >>> TSS(obs, mod, threshold=0.5)
    0.0
    """
    a, b, c, d = _contingency_table(obs, mod, minval, maxval)
    pod = a / (a + b) if (a + b) > 0 else np.nan
    pofd = c / (c + d) if (c + d) > 0 else np.nan
    return pod - pofd


def BSS_binary(obs, mod, threshold):
    """
    Binary Brier Skill Score for deterministic forecasts.

    Typical Use Cases
    -----------------
    - Evaluating the accuracy of deterministic binary forecasts (e.g., precipitation yes/no).
    - Used in meteorology and environmental modeling to assess forecast skill relative to a reference.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect forecast
    - 0: Same skill as reference forecast
    - Negative: Worse than reference forecast

    Parameters
    ----------
    obs : array_like
        Observed binary outcomes (0 or 1).
    mod : array_like
        Forecast binary outcomes (0 or 1).
    threshold : float
        Threshold value to convert continuous forecasts to binary.

    Returns
    -------
    float
        Binary Brier Skill Score.

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([0, 1, 1, 0])
    >>> mod = np.array([0, 1, 0, 0])
    >>> BSS_binary(obs, mod, threshold=0.5)
    0.5
    """
    obs = np.asarray(obs)
    mod = np.asarray(mod)
    
    # Convert to binary based on threshold
    obs_binary = (obs >= threshold).astype(int)
    mod_binary = (mod >= threshold).astype(int)
    
    # Calculate Brier Score
    bs = np.mean((mod_binary - obs_binary) ** 2)
    
    # Calculate reference Brier Score (climatology)
    obs_clim = np.mean(obs_binary)
    bs_ref = np.mean((obs_clim - obs_binary) ** 2)
    
    # Calculate Brier Skill Score
    bss = 1 - (bs / bs_ref) if bs_ref != 0 else 0
    
    return bss


def _contingency_table(obs, mod, minval, maxval=None):
    """
    Compute the 2x2 contingency table for event-based metrics.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
        Model or predicted values.
    minval : float
        Minimum threshold value for event detection.
    maxval : float, optional
        Maximum threshold value for event detection (for range-based events).

    Returns
    -------
    a : int
        Hits (obs >= minval and mod >= minval) or (minval <= obs < maxval and minval <= mod < maxval)
    b : int
        Misses (obs >= minval and mod < minval) or (minval <= obs < maxval and not minval <= mod < maxval)
    c : int
        False alarms (obs < minval and mod >= minval) or (not minval <= obs < maxval and minval <= mod < maxval)
    d : int
        Correct negatives (obs < minval and mod < minval) or (not minval <= obs < maxval and not minval <= mod < maxval)

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.5, 1.8, 3.2, 3.8])
    >>> a, b, c, d = _contingency_table(obs, mod, minval=2.5)
    >>> print(f"Hits: {a}, Misses: {b}, False Alarms: {c}, Correct Negatives: {d}")
    Hits: 2, Misses: 0, False Alarms: 0, Correct Negatives: 2
    """
    import numpy as np

    try:
        import xarray as xr
    except ImportError:
        xr = None
    # Drop NaNs and align for xarray
    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        mask = (~xr.ufuncs.isnan(obs)) & (~xr.ufuncs.isnan(mod))
        obs = obs.where(mask, drop=True)
        mod = mod.where(mask, drop=True)
        obs_vals = obs.values
        mod_vals = mod.values
    else:
        obs_vals = np.asarray(obs)
        mod_vals = np.asarray(mod)
        mask = ~np.isnan(obs_vals) & ~np.isnan(mod_vals)
        obs_vals = obs_vals[mask]
        mod_vals = mod_vals[mask]
    if maxval is not None:
        obs_event = (obs_vals >= minval) & (obs_vals < maxval)
        mod_event = (mod_vals >= minval) & (mod_vals < maxval)
    else:
        obs_event = obs_vals >= minval
        mod_event = mod_vals >= minval
    hits = int(np.logical_and(obs_event, mod_event).sum())
    misses = int(np.logical_and(obs_event, ~mod_event).sum())
    false_alarms = int(np.logical_and(~obs_event, mod_event).sum())
    correct_negatives = int(np.logical_and(~obs_event, ~mod_event).sum())
    return hits, misses, false_alarms, correct_negatives


def HSS_max_threshold(obs, mod, minval_range, maxval_range, step_size=1.0):
    """
    Find the threshold that maximizes the Heidke Skill Score (HSS) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for binary classification in meteorological or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
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
        hss_values.append(hss_val)
    
    # Find the threshold that gives the maximum HSS
    max_idx = np.argmax(hss_values)
    optimal_threshold = thresholds[max_idx]
    max_hss = hss_values[max_idx]
    
    return optimal_threshold, max_hss


def ETS_max_threshold(obs, mod, minval_range, maxval_range, step_size=1.0):
    """
    Find the threshold that maximizes the Equitable Threat Score (ETS) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for binary classification in meteorological or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
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
        ets_values.append(ets_val)
    
    # Find the threshold that gives the maximum ETS
    max_idx = np.argmax(ets_values)
    optimal_threshold = thresholds[max_idx]
    max_ets = ets_values[max_idx]
    
    return optimal_threshold, max_ets


def POD_max_threshold(obs, mod, minval_range, maxval_range, step_size=1.0):
    """
    Find the threshold that maximizes the Probability of Detection (POD) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for maximizing detection rates in meteorological or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
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
        pod_values.append(pod_val)
    
    # Find the threshold that gives the maximum POD
    max_idx = np.argmax(pod_values)
    optimal_threshold = thresholds[max_idx]
    max_pod = pod_values[max_idx]
    
    return optimal_threshold, max_pod


def FAR_min_threshold(obs, mod, minval_range, maxval_range, step_size=1.0):
    """
    Find the threshold that minimizes the False Alarm Rate (FAR) over a range.

    Typical Use Cases
    -----------------
    - Finding the optimal threshold for minimizing false alarms in meteorological or environmental modeling.
    - Used to optimize event detection thresholds in forecast systems.

    Parameters
    ----------
    obs : array_like
        Observed values.
    mod : array_like
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
        far_values.append(far_val)
    
    # Find the threshold that gives the minimum FAR
    min_idx = np.argmin(far_values)
    optimal_threshold = thresholds[min_idx]
    min_far = far_values[min_idx]
    
    return optimal_threshold, min_far
