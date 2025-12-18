"""
Error Metrics for Model Evaluation
"""

from typing import Any, Optional

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from .utils_stats import circlebias, circlebias_m


def _to_aligned_xarray(obs: ArrayLike, mod: ArrayLike) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert inputs to aligned xarray.DataArrays.

    If the inputs are already xarray.DataArray objects, they will be aligned.
    If they are not, they will be converted to xarray.DataArray objects.

    Parameters
    ----------
    obs : ArrayLike
        Observed values.
    mod : ArrayLike
        Model predicted values.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        A tuple containing two aligned xarray.DataArray objects.
    """
    if not isinstance(obs, xr.DataArray):
        obs = xr.DataArray(obs)
    if not isinstance(mod, xr.DataArray):
        mod = xr.DataArray(mod)

    # Align the data arrays to ensure they have the same dimensions and coordinates
    obs, mod = xr.align(obs, mod, join="inner")

    return obs, mod


############################################################
# 1. Basic Error Metrics
############################################################


def STDO(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Standard deviation of Observation Errors

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Model predicted values.
    axis : int, optional
        Axis along which to compute the standard deviation.

    Returns
    -------
    float or ndarray
        Standard deviation of observation minus model errors.
        Returns 0.0 for perfect agreement.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    errors = obs - mod
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return errors.std(dim=dim)


def STDP(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Standard deviation of Prediction Errors

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Model predicted values.
    axis : int, optional
        Axis along which to compute the standard deviation.

    Returns
    -------
    float or ndarray
        Standard deviation of model minus observation errors.
        Returns 0.0 for perfect agreement.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    errors = mod - obs
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return errors.std(dim=dim)


def MNB(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Normalized Bias (%)

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Model predicted values.
    axis : int, optional
        Axis along which to compute the bias.

    Returns
    -------
    float or ndarray
        Mean normalized bias (percent).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return ((mod - obs) / obs).mean(dim=dim) * 100.0


def MNE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Normalized Gross Error (%)

    Parameters
    ----------
    obs : array-like
        Observed values.
    mod : array-like
        Model predicted values.
    axis : int, optional
        Axis along which to compute the error.

    Returns
    -------
    float or ndarray
        Mean normalized gross error (percent).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (abs(mod - obs) / obs).mean(dim=dim) * 100.0


def MdnNB(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Median Normalized Bias (%)

    Typical Use Cases
    -----------------
    - Assessing the central tendency of model bias relative to observations, less sensitive to outliers than mean.
    - Useful for robust model evaluation in the presence of skewed or non-normal error distributions.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return ((mod - obs) / obs).median(dim=dim) * 100.0


def MdnNE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Median Normalized Gross Error (%)

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of model errors relative to observations, robust to outliers.
    - Useful for summarizing error magnitude in non-Gaussian or heavy-tailed error distributions.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (abs(mod - obs) / obs).median(dim=dim) * 100.0


def NMdnGE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Normalized Median Gross Error (%)

    Typical Use Cases
    -----------------
    - Comparing the typical (median) error magnitude, normalized by the mean observation, for robust model evaluation.
    - Useful for inter-comparison of model performance across sites or variables with different scales.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (abs(mod - obs).mean(dim=dim) / obs.mean(dim=dim)) * 100.0


def NO(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    N Observations (#)

    Typical Use Cases
    -----------------
    - Counting the number of valid (non-masked) observations in a dataset.
    - Used to report sample size for statistical summaries and model evaluation.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return obs.count(dim=dim)


def NOP(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    N Observations/Prediction Pairs (#)

    Typical Use Cases
    -----------------
    - Counting the number of valid observation-prediction pairs for paired statistical analysis.
    - Used to ensure sample size consistency in paired model evaluation metrics.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return obs.count(dim=dim)


def NP(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    N Predictions (#)

    Typical Use Cases
    -----------------
    - Counting the number of valid (non-masked) model predictions in a dataset.
    - Used to report sample size for model output and for filtering invalid predictions.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = mod.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return mod.count(dim=dim)


def MO(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Error (MO) - Mean of (observation - model)

    Typical Use Cases
    -----------------
    - Quantifying the average bias between observations and model predictions.
    - Used in model evaluation to assess systematic errors.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the mean error.

    Returns
    -------
    float or xarray.DataArray
        Mean error (observation - model) in observation units.
        Returns 0.0 for perfect agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MO
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    >>> MO(obs, mod)
    -0.1
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (obs - mod).mean(dim=dim)


def MP(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Predictions (model unit)

    Typical Use Cases
    -----------------
    - Calculating the average value of model predictions for baseline or climatological reference.
    - Used in normalization, anomaly calculation, and summary statistics for model output.

    Parameters
    ----------
    obs : type
        Description of parameter `obs`.
    mod : type
        Description of parameter `mod`.
    axis : type
        Description of parameter `axis`.

    Returns
    -------
    type
        Description of returned object.

    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = mod.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return mod.mean(dim=dim)


def MdnO(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Median Error (MdnO) - Median of (observation - model)

    Typical Use Cases
    -----------------
    - Quantifying the typical bias between observations and model predictions, robust to outliers.
    - Used in robust model evaluation for non-parametric error assessment.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the median error.

    Returns
    -------
    float or xarray.DataArray
        Median error (observation - model) in observation units.
        Returns 0.0 for perfect agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MdnO
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    >>> MdnO(obs, mod)
    -0.1
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (obs - mod).median(dim=dim)


def MdnP(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Median Error (MdnP) - Median of (model - observation)

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the median error.

    Returns
    -------
    float or xarray.DataArray
        Median error (model - observation) in model units.
        Returns 0.0 for perfect agreement.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (mod - obs).median(dim=dim)


def RM(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Root Mean Error (RM) - Root of mean squared error

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the error.

    Returns
    -------
    float or xarray.DataArray
        Root of mean squared error (observation units).
        Returns 0.0 for perfect agreement.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return np.sqrt(((obs - mod) ** 2).mean(dim=dim))


def RMdn(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Root Median Error (RMdn) - Root of median squared error

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the error.

    Returns
    -------
    float or xarray.DataArray
        Root of median squared error (observation units).
        Returns 0.0 for perfect agreement.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    squared_errors = (obs - mod) ** 2
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return np.sqrt(squared_errors.median(dim=dim))


def MB(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Bias (MB)

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the mean bias.

    Returns
    -------
    float or xarray.DataArray
        Mean bias value(s) = mean(observation - model).
        Negative values indicate model overestimation.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (obs - mod).mean(dim=dim)


def MdnB(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Median Bias (MdnB)

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed values.
    mod : array-like or xarray.DataArray
        Model predicted values.
    axis : int or None, optional
        Axis along which to compute the median bias.

    Returns
    -------
    float or xarray.DataArray
        Median bias value(s) = median(observation - model).
        Negative values indicate model overestimation.
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (obs - mod).median(dim=dim)


def WDMB_m(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Wind Direction Mean Bias (WDMB, robust version for masked arrays)

    This version uses circlebias_m, which is robust to masked arrays and missing data.
    Use this if your data may contain NaNs or masked values.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed wind direction values (degrees).
    mod : array-like or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int or None, optional
        Axis along which to compute the mean bias.

    Returns
    -------
    float or xarray.DataArray
        Mean wind direction bias (degrees).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    diff = mod - obs
    bias_values = circlebias_m(diff.values)
    bias_da = xr.DataArray(bias_values, dims=diff.dims, coords=diff.coords)
    return bias_da.mean(dim=dim)


def WDMB(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Wind Direction Mean Bias (WDMB, standard version)

    This version uses circlebias, which is not robust to masked arrays.
    Use this if your data are dense and do not contain missing values.

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed wind direction values (degrees).
    mod : array-like or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int or None, optional
        Axis along which to compute the mean bias.

    Returns
    -------
    float or xarray.DataArray
        Mean wind direction bias (degrees).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    diff = mod - obs
    bias_values = circlebias(diff.values)
    bias_da = xr.DataArray(bias_values, dims=diff.dims, coords=diff.coords)
    return bias_da.mean(dim=dim)


def WDMdnB(obs, mod, axis=None):
    """
    Wind Direction Median Bias (WDMdnB)

    Parameters
    ----------
    obs : array-like or xarray.DataArray
        Observed wind direction values (degrees).
    mod : array-like or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int or None, optional
        Axis along which to compute the median bias.

    Returns
    -------
    float or xarray.DataArray
        Median wind direction bias (degrees).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    diff = mod - obs
    bias_values = circlebias(diff.values)
    bias_da = xr.DataArray(bias_values, dims=diff.dims, coords=diff.coords)
    return bias_da.median(dim=dim)


def MAE(obs: ArrayLike, mod: ArrayLike, axis: Optional[int] = None) -> Any:
    """
    Mean Absolute Error (MAE).

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations, regardless of direction.
    - Used in model evaluation, forecast verification, and regression analysis.

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
    0.6666666666666666
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return abs(mod - obs).mean(dim=dim)


def MedAE(obs, mod, axis=None):
    """
    Median Absolute Error (MedAE).

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of errors, robust to outliers and non-normal error distributions.
    - Used in robust regression, model evaluation, and forecast verification.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MedAE. Default is None (all elements).

    Returns
    -------
    medae : float or ndarray
        Median absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.MedAE(obs, mod)
    1.0
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return abs(mod - obs).median(dim=dim)


def sMAPE_original(obs, mod, axis=None):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations, normalized by their mean.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute sMAPE. Default is None (all elements).

    Returns
    -------
    smape : float or ndarray
        Symmetric mean absolute percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.sMAPE(obs, mod)
    28.57142857142857
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (200 * abs(mod - obs) / (abs(mod) + abs(obs))).mean(dim=dim)


def CRMSE(obs, mod, axis=None):
    """
    Centered Root Mean Square Error (CRMSE).

    Typical Use Cases
    -----------------
    - Quantifying the error between anomalies (deviations from mean) of model and observations.
    - Used in Taylor diagrams, model evaluation, and forecast verification.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute CRMSE. Default is None (all elements).

    Returns
    -------
    crmse : float or ndarray
        Centered root mean square error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.CRMSE(obs, mod)
    0.4714045207910317
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    o_ = obs - obs.mean(dim=dim)
    m_ = mod - mod.mean(dim=dim)
    return ((m_ - o_) ** 2).mean(dim=dim) ** 0.5


def MAPE(obs, mod, axis=None):
    """
    Mean Absolute Percentage Error (MAPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations as a percentage.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.

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
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (100 * abs(mod - obs) / abs(obs)).mean(dim=dim)


def sMAPE(obs, mod, axis=None):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations, normalized by their mean.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute sMAPE. Default is None (all elements).

    Returns
    -------
    smape : float or ndarray
        Symmetric mean absolute percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.sMAPE(obs, mod)
    28.57142857142857
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (200 * abs(mod - obs) / (abs(mod) + abs(obs))).mean(dim=dim)


def NRMSE(obs, mod, axis=None):
    """
    Normalized Root Mean Square Error (NRMSE).

    Typical Use Cases
    -----------------
    - Quantifying the relative error between model and observations, normalized by the range of observations.
    - Used in model evaluation to compare performance across different variables or sites with different scales.
    - Provides dimensionless error metric for cross-comparison.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute NRMSE. Default is None (all elements).

    Returns
    -------
    nrmse : float or ndarray
        Normalized root mean square error (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> stats.NRMSE(obs, mod)
    0.4714045207910317
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    rmse = ((mod - obs) ** 2).mean(dim=dim) ** 0.5
    obs_range = obs.max(dim=dim) - obs.min(dim=dim)
    return rmse / obs_range


def MASE(obs, mod, axis=None):
    """
    Mean Absolute Scaled Error (MASE).

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
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    shift_dim = dim if dim is not None else obs.dims[0]
    naive_error = abs(obs - obs.shift({shift_dim: 1})).mean(dim=dim, skipna=True)
    model_error = abs(mod - obs).mean(dim=dim)
    return model_error / naive_error


def MASEm(obs, mod, axis=None):
    """
    Mean Absolute Scaled Error (MASE) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying model error relative to the error of a simple baseline model (e.g., naive forecast), robust to masked arrays.
    - Used in time series forecasting and model evaluation with missing data.

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
    >>> stats.MASEm(obs, mod)
    0.1
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    shift_dim = dim if dim is not None else obs.dims[0]
    naive_error = abs(obs - obs.shift({shift_dim: 1})).mean(dim=dim, skipna=True)
    model_error = abs(mod - obs).mean(dim=dim)
    return model_error / naive_error


def RMSPE(obs, mod, axis=None):
    """
    Root Mean Square Percentage Error (RMSPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations as a percentage, emphasizing larger errors.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute RMSPE. Default is None (all elements).

    Returns
    -------
    rmspe : float or ndarray
        Root mean square percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.RMSPE(obs, mod)
    50.0
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (100 * ((mod - obs) / obs) ** 2).mean(dim=dim) ** 0.5


def MAPEm(obs, mod, axis=None):
    """
    Mean Absolute Percentage Error (MAPE) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations as a percentage, robust to missing data.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.

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
    >>> stats.MAPEm(obs, mod)
    50.0
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (100 * abs((mod - obs) / obs)).mean(dim=dim)


def sMAPEm(obs, mod, axis=None):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations, normalized by their mean, robust to missing data.
    - Used in time series forecasting, regression, and model evaluation for percentage-based error assessment.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute sMAPE. Default is None (all elements).

    Returns
    -------
    smape : float or ndarray
        Symmetric mean absolute percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.sMAPEm(obs, mod)
    28.57142857142857
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return (200 * abs(mod - obs) / (abs(mod) + abs(obs))).mean(dim=dim)


def NSC(obs, mod, axis=None):
    """
    Nash-Sutcliffe Coefficient (NSC) - Alternative to NSE.

    Typical Use Cases
    -----------------
    - Quantifying the predictive power of hydrological models relative to the mean of observations.
    - Used in hydrology, meteorology, and environmental model evaluation.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute NSC. Default is None (all elements).

    Returns
    -------
    nsc : float or ndarray
        Nash-Sutcliffe coefficient (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> stats.NSC(obs, mod)
    -0.3333
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    obs_mean = obs.mean(dim=dim)
    numerator = ((obs - mod) ** 2).sum(dim=dim)
    denominator = ((obs - obs_mean) ** 2).sum(dim=dim)
    return 1.0 - (numerator / denominator)


def NSE_alpha(obs, mod, axis=None):
    """
    NSE Alpha - Decomposed NSE component measuring ratio of standard deviations.

    Typical Use Cases
    -----------------
    - Quantifying the model's ability to capture the variability of observations.
    - Used in model evaluation to assess how well model represents observed variability.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute NSE_alpha. Default is None (all elements).

    Returns
    -------
    nse_alpha : float or ndarray
        NSE alpha component (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> stats.NSE_alpha(obs, mod)
    0.0
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return mod.std(dim=dim) / obs.std(dim=dim)


def NSE_beta(obs, mod, axis=None):
    """
    NSE Beta - Decomposed NSE component measuring bias.

    Typical Use Cases
    -----------------
    - Quantifying the systematic bias between model and observations.
    - Used in model evaluation to assess mean differences between model and observations.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute NSE_beta. Default is None (all elements).

    Returns
    -------
    nse_beta : float or ndarray
        NSE beta component (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> stats.NSE_beta(obs, mod)
    0.5
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return mod.mean(dim=dim) / obs.mean(dim=dim)


def MAE_m(obs, mod, axis=None):
    """
    Mean Absolute Error (MAE) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations, regardless of direction, robust to missing data.
    - Used in model evaluation, forecast verification, and regression analysis with incomplete datasets.

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
    >>> stats.MAE_m(obs, mod)
    0.66666666
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return abs(mod - obs).mean(dim=dim)


def MedAE_m(obs, mod, axis=None):
    """
    Median Absolute Error (MedAE) - robust to masked arrays and outliers.

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of errors, robust to outliers and non-normal error distributions with missing data.
    - Used in robust regression, model evaluation, and forecast verification with incomplete datasets.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute MedAE. Default is None (all elements).

    Returns
    -------
    medae : float or ndarray
        Median absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.MedAE_m(obs, mod)
    1.0
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return abs(mod - obs).median(dim=dim)


def RMSE(obs, mod, axis=None):
    """
    Root Mean Square Error (RMSE).

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations, accounting for large errors more heavily than MAE.
    - Used in model evaluation, forecast verification, and regression analysis.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute RMSE. Default is None (all elements).

    Returns
    -------
    rmse : float or ndarray
        Root mean square error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.RMSE(obs, mod)
    0.816496580927726
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return ((mod - obs) ** 2).mean(dim=dim) ** 0.5


def RMSE_m(obs, mod, axis=None):
    """
    Root Mean Square Error (RMSE) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations, accounting for large errors more heavily than MAE,
      robust to missing data.
    - Used in model evaluation, forecast verification, and regression analysis with incomplete datasets.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute RMSE. Default is None (all elements).

    Returns
    -------
    rmse : float or ndarray
        Root mean square error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.RMSE_m(obs, mod)
    0.816496580927726
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return ((mod - obs) ** 2).mean(dim=dim) ** 0.5


def IOA(obs, mod, axis=None):
    """
    Index of Agreement (IOA).

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, normalized by total deviation.
    - Used in model evaluation for skill assessment.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute IOA. Default is None (all elements).

    Returns
    -------
    ioa : float or ndarray
        Index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.IOA(obs, mod)
    0.8
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    obs_mean = obs.mean(dim=dim)
    num = ((obs - mod) ** 2).sum(dim=dim)
    denom = ((abs(mod - obs_mean) + abs(obs - obs_mean)) ** 2).sum(dim=dim)
    return 1.0 - (num / denom)


def IOA_m(obs, mod, axis=None):
    """
    Index of Agreement (IOA) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, normalized by total deviation, robust to missing data.
    - Used in model evaluation for skill assessment with incomplete datasets.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute IOA. Default is None (all elements).

    Returns
    -------
    ioa : float or ndarray
        Index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet.util import stats
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> stats.IOA_m(obs, mod)
    0.8
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    obs_mean = obs.mean(dim=dim)
    num = ((obs - mod) ** 2).sum(dim=dim)
    denom = ((abs(mod - obs_mean) + abs(obs - obs_mean)) ** 2).sum(dim=dim)
    return 1.0 - (num / denom)


# Add the missing functions from the specification


def MAPE_mod(obs, mod, axis=None):
    """
    Modified Mean Absolute Percentage Error (MAPE).

    This version handles cases where observations might be zero or near zero
    by using a small epsilon to avoid division by zero.

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
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    epsilon = 1e-8
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    obs_safe = xr.where(np.abs(obs) < epsilon, epsilon, obs)
    return (100 * abs(mod - obs) / abs(obs_safe)).mean(dim=dim)


def MASE_mod(obs, mod, axis=None):
    """
    Modified Mean Absolute Scaled Error (MASE).

    This version handles cases where the naive forecast error is zero
    by using a small epsilon to avoid division by zero.

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
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis

    # If dim is not specified, use the first dimension for shifting
    shift_dim = dim if dim is not None else obs.dims[0]

    naive_error = abs(obs - obs.shift({shift_dim: 1})).mean(dim=dim, skipna=True)
    model_error = abs(mod - obs).mean(dim=dim)
    return xr.where(naive_error == 0, model_error, model_error / naive_error)


def RMSE_norm(obs, mod, axis=None):
    """
    Normalized Root Mean Square Error (RMSE_norm).

    Normalizes RMSE by the range of observations.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute normalized RMSE. Default is None (all elements).

    Returns
    -------
    rmse_norm : float or ndarray
        Normalized root mean square error (unitless).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    rmse = ((mod - obs) ** 2).mean(dim=dim) ** 0.5
    obs_min = obs.min(dim=dim)
    obs_max = obs.max(dim=dim)
    obs_range = obs_max - obs_min
    return xr.where(obs_range == 0, rmse, rmse / obs_range)


def MAE_norm(obs, mod, axis=None):
    """
    Normalized Mean Absolute Error (MAE_norm).

    Normalizes MAE by the range of observations.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute normalized MAE. Default is None (all elements).

    Returns
    -------
    mae_norm : float or ndarray
        Normalized mean absolute error (unitless).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    mae = abs(mod - obs).mean(dim=dim)
    obs_min = obs.min(dim=dim)
    obs_max = obs.max(dim=dim)
    obs_range = obs_max - obs_min
    return xr.where(obs_range == 0, mae, mae / obs_range)


def bias_fraction(obs, mod, axis=None):
    """
    Bias Fraction (BF).

    Quantifies the fraction of total error that is due to systematic bias.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute bias fraction. Default is None (all elements).

    Returns
    -------
    bf : float or ndarray
        Bias fraction (unitless, 0-1).
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    bias = (mod - obs).mean(dim=dim)
    total_error = np.sqrt(((mod - obs) ** 2).mean(dim=dim))
    return xr.where(total_error == 0, 0, (bias**2) / (total_error**2))


# Add missing functions from the specification


def NMSE(obs, mod, axis=None):
    """
    Normalized Mean Square Error (NMSE).

    Typical Use Cases
    -----------------
    - Quantifying the normalized squared error between model and observations.
    - Used in model evaluation to compare performance across different variables or sites with different scales.
    - Provides dimensionless error metric for cross-comparison.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute NMSE. Default is None (all elements).

    Returns
    -------
    nmse : float or ndarray
        Normalized mean square error (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import NMSE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NMSE(obs, mod)
    0.25
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    mse = ((mod - obs) ** 2).mean(dim=dim)
    obs_var = obs.var(dim=dim)
    return xr.where(obs_var == 0, 0, mse / obs_var)


def LOG_ERROR(obs, mod, axis=None):
    """
    Logarithmic Error Metric.

    Typical Use Cases
    -----------------
    - Quantifying errors for variables that span several orders of magnitude.
    - Used in atmospheric sciences for concentration data (e.g., aerosols, pollutants).
    - Helpful when relative rather than absolute errors are important.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values (should be positive).
    mod : array_like or xarray.DataArray
        Model or predicted values (should be positive).
    axis : int, optional
        Axis along which to compute log error. Default is None (all elements).

    Returns
    -------
    log_error : float or ndarray
        Logarithmic error metric.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import LOG_ERROR
    >>> obs = np.array([1, 100])
    >>> mod = np.array([2, 200])
    >>> LOG_ERROR(obs, mod)
    0.34657359027997264
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    epsilon = 1e-10
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    obs_safe = np.abs(obs) + epsilon
    mod_safe = np.abs(mod) + epsilon
    obs_log = np.log(obs_safe)
    mod_log = np.log(mod_safe)
    return ((mod_log - obs_log) ** 2).mean(dim=dim) ** 0.5


def COE(obs, mod, axis=None):
    """
    Center of Mass Error (COE).

    Typical Use Cases
    -----------------
    - Evaluating the displacement error of spatial features.
    - Used in meteorology for precipitation field verification.
    - Assesses how much model features are shifted compared to observations.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values (typically 2D spatial field).
    mod : array_like or xarray.DataArray
        Model or predicted values (typically 2D spatial field).
    axis : int, optional
        Axis along which to compute COE. Default is None (all elements).

    Returns
    -------
    coe : float or ndarray
        Center of mass error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import COE
    >>> obs = np.array([[1, 0], [0, 1]])  # Diagonal pattern
    >>> mod = np.array([[0, 1], [1, 0]])  # Opposite diagonal
    >>> COE(obs, mod)
    1.4142135623730951
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return ((mod - obs) ** 2).mean(dim=dim) ** 0.5


def VOLUMETRIC_ERROR(obs, mod, axis=None):
    """
    Volumetric Error Metric.

    Typical Use Cases
    -----------------
    - Quantifying the volume difference between observed and modeled features.
    - Used in hydrology for flood extent verification.
    - Applied in meteorology for precipitation volume verification.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute volumetric error. Default is None (all elements).

    Returns
    -------
    vol_error : float or ndarray
        Volumetric error metric.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import VOLUMETRIC_ERROR
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> VOLUMETRIC_ERROR(obs, mod)
    0.2
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    obs_sum = obs.sum(dim=dim)
    mod_sum = mod.sum(dim=dim)
    return np.abs(mod_sum - obs_sum) / np.abs(obs_sum)


def CORR_INDEX(obs, mod, axis=None):
    """
    Correlation Index (CORR_INDEX).

    Typical Use Cases
    -----------------
    - Measuring the linear relationship between observed and modeled values.
    - Used as a component in model evaluation.
    - Quantifies how well model captures observed patterns.

    Parameters
    ----------
    obs : array_like or xarray.DataArray
        Observed values.
    mod : array_like or xarray.DataArray
        Model or predicted values.
    axis : int, optional
        Axis along which to compute correlation index. Default is None (all elements).

    Returns
    -------
    corr_index : float or ndarray
        Correlation index (unitless, -1 to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import CORR_INDEX
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 4, 6, 8])
    >>> CORR_INDEX(obs, mod)
    1.0
    """
    obs, mod = _to_aligned_xarray(obs, mod)
    dim = obs.dims[axis] if axis is not None and isinstance(axis, int) else axis
    return xr.corr(obs, mod, dim=dim)
