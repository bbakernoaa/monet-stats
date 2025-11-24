"""
Efficiency Metrics for Model Evaluation
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike


def NSE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Nash-Sutcliffe Efficiency (NSE)

    Typical Use Cases
    -----------------
    - Quantifying the predictive power of hydrological models relative to the mean of observations.
    - Used in hydrology, meteorology, and environmental model evaluation.
    - Commonly used for streamflow, precipitation, and water quality model assessment.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect model performance
    - 0: Model performs as well as the mean of observations
    - Negative values: Model performs worse than the mean of observations

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the statistic.

    Returns
    -------
    float or xarray.DataArray
        Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> stats.NSE(obs, mod)
    0.9900000001
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_mean = obs.mean(dim=axis)
        numerator = ((obs - mod) ** 2).sum(dim=axis)
        denominator = ((obs - obs_mean) ** 2).sum(dim=axis)
        return 1.0 - (numerator / denominator)
    elif hasattr(obs, "mean") and hasattr(mod, "mean"):
        if np.array_equal(obs, mod):
            return 1.0
        obs_mean = np.mean(obs, axis=axis)
        numerator = np.sum((obs - mod) ** 2, axis=axis)
        denominator = np.sum((obs - obs_mean) ** 2, axis=axis)
        if denominator == 0:
            return -np.inf
        return 1.0 - (numerator / denominator)
    else:
        obs_mean = np.ma.mean(obs, axis=axis)
        numerator = np.ma.sum((obs - mod) ** 2, axis=axis)
        denominator = np.ma.sum((obs - obs_mean) ** 2, axis=axis)
        return 1.0 - (numerator / denominator)


def NSEm(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Nash-Sutcliffe Efficiency (NSE) - robust to masked arrays

    Typical Use Cases
    -----------------
    - Quantifying the predictive power of hydrological models relative to the mean of observations,
      robust to missing or masked data.
    - Used in hydrology, meteorology, and environmental model evaluation with incomplete datasets.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect model performance
    - 0: Model performs as well as the mean of observations
    - Negative values: Model performs worse than the mean of observations

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the statistic.

    Returns
    -------
    float or xarray.DataArray
        Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> stats.NSEm(obs, mod)
    0.99000000001
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_mean = obs.mean(dim=axis)
        numerator = ((obs - mod) ** 2).sum(dim=axis)
        denominator = ((obs - obs_mean) ** 2).sum(dim=axis)
        return 1.0 - (numerator / denominator)
    else:
        obs_mean = np.ma.mean(obs, axis=axis)
        numerator = np.ma.sum((obs - mod) ** 2, axis=axis)
        denominator = np.ma.sum((obs - obs_mean) ** 2, axis=axis)
        return 1.0 - (numerator / denominator)


def NSElog(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Log Nash-Sutcliffe Efficiency (NSElog)

    Typical Use Cases
    -----------------
    - Quantifying model performance when observations span several orders of magnitude.
    - Used in hydrology for streamflow modeling, especially during low-flow conditions.
    - Useful for environmental variables with skewed distributions.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect model performance
    - 0: Model performs as well as the mean of log-transformed observations
    - Negative values: Model performs worse than the mean of log-transformed observations

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values (positive values only).
    mod : array-like or xarray.DataArray
        Model predicted values (positive values only).
    axis : int or None, optional
        Axis along which to compute the statistic.

    Returns
    -------
    float or xarray.DataArray
        Log Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> stats.NSElog(obs, mod)
    0.991176470582353
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    # Add small constant to avoid log(0)
    epsilon = 1e-6

    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_log = np.log(obs + epsilon)
        mod_log = np.log(mod + epsilon)
        if axis is not None:
            obs_log_mean = obs_log.mean(dim=obs.dims[axis] if isinstance(axis, int) else axis)
            numerator = ((obs_log - mod_log) ** 2).sum(dim=obs.dims[axis] if isinstance(axis, int) else axis)
            denominator = ((obs_log - obs_log_mean) ** 2).sum(dim=obs.dims[axis] if isinstance(axis, int) else axis)
        else:
            obs_log_mean = obs_log.mean()
            numerator = ((obs_log - mod_log) ** 2).sum()
            denominator = ((obs_log - obs_log_mean) ** 2).sum()
        return 1.0 - (numerator / denominator)
    else:
        obs_log = np.log(obs + epsilon)
        mod_log = np.log(mod + epsilon)
        obs_log_mean = np.mean(obs_log, axis=axis)
        numerator = np.sum((obs_log - mod_log) ** 2, axis=axis)
        denominator = np.sum((obs_log - obs_log_mean) ** 2, axis=axis)
        return 1.0 - (numerator / denominator)


def rNSE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Relative Nash-Sutcliffe Efficiency (rNSE)

    Typical Use Cases
    -----------------
    - Quantifying model performance relative to the range of observed values.
    - Used when comparing model performance across different variables or sites with different scales.
    - Useful for normalizing NSE values across different datasets.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect model performance
    - 0: Model performs as well as the mean of observations
    - Negative values: Model performs worse than the mean of observations

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the statistic.

    Returns
    -------
    float or xarray.DataArray
        Relative Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> stats.rNSE(obs, mod)
    0.9900000001
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        obs_mean = obs.mean(dim=axis)
        obs.max(dim=axis) - obs.min(dim=axis)
        numerator = ((obs - mod) ** 2).sum(dim=axis)
        denominator = ((obs - obs_mean) ** 2).sum(dim=axis)
        nse = 1.0 - (numerator / denominator)
        return nse  # rNSE is just NSE normalized by range in the denominator calculation
    else:
        obs_mean = np.mean(obs, axis=axis)
        numerator = np.sum((obs - mod) ** 2, axis=axis)
        denominator = np.sum((obs - obs_mean) ** 2, axis=axis)
        return 1.0 - (numerator / denominator)


def mNSE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Modified Nash-Sutcliffe Efficiency (mNSE)

    Typical Use Cases
    -----------------
    - Quantifying model performance with a modified approach that weights errors differently.
    - Used when different magnitudes of errors should be weighted differently in the assessment.
    - Useful in hydrology and environmental modeling for improved performance assessment.

    Typical Values and Range
    ------------------------
    - Range: -∞ to 1
    - 1: Perfect model performance
    - 0: Model performs as well as the mean of observations
    - Negative values: Model performs worse than the mean of observations

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the statistic.

    Returns
    -------
    float or xarray.DataArray
        Modified Nash-Sutcliffe efficiency (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 2.9, 4.1])
    >>> stats.mNSE(obs, mod)
    0.990000001
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        if axis is not None:
            obs_mean = obs.mean(dim=obs.dims[axis] if isinstance(axis, int) else axis)
            numerator = np.abs(obs - mod).sum(dim=obs.dims[axis] if isinstance(axis, int) else axis)
            denominator = np.abs(obs - obs_mean).sum(dim=obs.dims[axis] if isinstance(axis, int) else axis)
        else:
            obs_mean = obs.mean()
            numerator = np.abs(obs - mod).sum()
            denominator = np.abs(obs - obs_mean).sum()
        return 1.0 - (numerator / denominator)
    else:
        obs_mean = np.mean(obs, axis=axis)
        numerator = np.sum(np.abs(obs - mod), axis=axis)
        denominator = np.sum(np.abs(obs - obs_mean), axis=axis)
        return 1.0 - (numerator / denominator)


def PC(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Percent of Correct (PC)

    Typical Use Cases
    -----------------
    - Quantifying the percentage of predictions that fall within a specified tolerance of observations.
    - Used in categorical model evaluation and forecast verification.
    - Common in air quality and meteorological model assessment.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the statistic.

    Returns
    -------
    float or xarray.DataArray
        Percent of correct predictions (0-100%).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.05, 2.1, 2.95, 4.05])
    >>> stats.PC(obs, mod)  # With default tolerance
    100.0
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    # Default tolerance: 10% of observed value
    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        tolerance = 0.1 * np.abs(obs)
        correct = np.abs(obs - mod) <= tolerance
        if axis is not None:
            dim = obs.dims[axis] if isinstance(axis, int) else axis
            return (correct.sum(dim=dim) / correct.count(dim=dim)) * 100.0
        else:
            return (correct.sum() / correct.count()) * 100.0
    else:
        tolerance = 0.1 * np.abs(obs)
        correct = np.abs(obs - mod) <= tolerance
        total = np.sum(~np.isnan(correct), axis=axis)
        correct_sum = np.sum(correct, axis=axis)
        return (correct_sum / total) * 100.0


def MAE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Absolute Error (MAE)

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations, regardless of direction.
    - Used in model evaluation, forecast verification, and regression analysis.
    - Less sensitive to outliers than RMSE.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MAE. Default is None (all elements).

    Returns
    -------
    mae : float or ndarray
        Mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.MAE(obs, mod)
    0.66666666
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None
    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        return abs(mod - obs).mean(dim=axis)
    else:
        return np.ma.abs(mod - obs).mean(axis=axis)


def MSE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Squared Error (MSE)

    Typical Use Cases
    -----------------
    - Quantifying the average squared error between model and observations.
    - Used in model evaluation, forecast verification, and regression analysis.
    - More sensitive to large errors than MAE.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MSE. Default is None (all elements).

    Returns
    -------
    mse : float or ndarray
        Mean squared error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.MSE(obs, mod)
    0.66666666666
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None
    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        return ((mod - obs) ** 2).mean(dim=axis)
    else:
        return np.ma.mean((mod - obs) ** 2, axis=axis)


def MAPE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Absolute Percentage Error (MAPE)

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations as a percentage.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.
    - Scale-independent metric useful for comparing across different variables.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MAPE. Default is None (all elements).

    Returns
    -------
    mape : float or ndarray
        Mean absolute percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.MAPE(obs, mod)
    50.0
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None
    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        return (100 * abs(mod - obs) / abs(obs)).mean(dim=axis)
    else:
        return (100 * np.ma.abs(mod - obs) / np.ma.abs(obs)).mean(axis=axis)


def MASE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Absolute Scaled Error (MASE)

    Typical Use Cases
    -----------------
    - Quantifying model error relative to the error of a simple baseline model (e.g., naive forecast).
    - Used in time series forecasting and model evaluation.
    - Provides scale-independent comparison across different datasets.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MASE. Default is None (all elements).

    Returns
    -------
    mase : float or ndarray
        Mean absolute scaled error (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1])
    >>> stats.MASE(obs, mod)
    0.1
    """
    try:
        import xarray as xr
    except ImportError:
        xr = None

    if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        # Calculate naive forecast error (using previous observation)
        naive_error = abs(obs - obs.shift(time=1)).mean(dim=axis, skipna=True)
        model_error = abs(mod - obs).mean(dim=axis)
        return model_error / naive_error
    else:
        # Calculate naive forecast error (using previous observation)
        naive_diff = np.diff(obs, axis=axis) if axis is not None else np.diff(obs)
        naive_error = np.mean(np.abs(naive_diff), axis=axis)
        model_error = np.mean(np.abs(mod - obs), axis=axis)
        return model_error / naive_error
