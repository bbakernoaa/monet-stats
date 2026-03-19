"""
Error Metrics for Model Evaluation
"""

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _resolve_axis_to_dim, _update_history, circlebias, ensure_single_chunk, matchmasks

############################################################
# 1. Basic Error Metrics
############################################################


def STDO(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Standard deviation of Observation Errors (obs - mod).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the standard deviation.
        If None, computes over all axes.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Standard deviation of (observation - model) errors.
        Returns 0.0 for perfect agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import STDO
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> mod = np.array([1.1, 1.9, 3.2])
    >>> STDO(obs, mod)
    0.1247219128924647
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        errors = obs - mod
        dim = _resolve_axis_to_dim(obs, axis)
        result = errors.std(dim=dim, keep_attrs=True)
        return _update_history(result, "STDO")

    # Fallback to numpy-compatible logic
    errors = np.subtract(obs, mod)
    result = np.ma.std(np.ma.masked_invalid(errors), axis=axis)
    return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def STDP(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Standard deviation of Prediction Errors (mod - obs).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the standard deviation.
        If None, computes over all axes.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Standard deviation of (model - observation) errors.
        Returns 0.0 for perfect agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import STDP
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> mod = np.array([1.1, 1.9, 3.2])
    >>> STDP(obs, mod)
    0.1247219128924647
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        errors = mod - obs
        dim = _resolve_axis_to_dim(obs, axis)
        result = errors.std(dim=dim, keep_attrs=True)
        return _update_history(result, "STDP")

    # Fallback to numpy-compatible logic
    errors = np.subtract(mod, obs)
    result = np.ma.std(np.ma.masked_invalid(errors), axis=axis)
    return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MNB(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Normalized Bias (%).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the bias.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean normalized bias (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([1.1, 2.2, 3.3])
    >>> MNB(obs, mod)
    10.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = ((mod - obs) / obs).mean(dim=dim, keep_attrs=True) * 100.0
        return _update_history(result, "MNB")
    else:
        result = np.ma.masked_invalid((mod - obs) / obs).mean(axis=axis) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MNE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Normalized Gross Error (%).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean normalized gross error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([1.1, 1.8, 3.3])
    >>> MNE(obs, mod)
    10.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = (abs(mod - obs) / obs).mean(dim=dim, keep_attrs=True) * 100.0
        return _update_history(result, "MNE")
    else:
        result = np.ma.masked_invalid(np.ma.abs(mod - obs) / obs).mean(axis=axis) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MdnNB(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Median Normalized Bias (%).

    Typical Use Cases
    -----------------
    - Assessing the central tendency of model bias relative to observations,
      less sensitive to outliers than mean.
    - Useful for robust model evaluation in the presence of skewed or non-normal
      error distributions.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the bias.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median normalized bias (percent).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = (mod - obs) / obs
        diff = ensure_single_chunk(diff, dim)
        result = diff.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore") * 100.0
        return _update_history(result, "MdnNB")
    else:
        result = np.ma.median(np.ma.masked_invalid((mod - obs) / obs), axis=axis) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MdnNE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Median Normalized Gross Error (%).

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of model errors relative to observations,
      robust to outliers.
    - Useful for summarizing error magnitude in non-Gaussian or heavy-tailed
      error distributions.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median normalized gross error (percent).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = abs(mod - obs) / obs
        diff = ensure_single_chunk(diff, dim)
        result = diff.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore") * 100.0
        return _update_history(result, "MdnNE")
    else:
        result = np.ma.median(np.ma.masked_invalid(np.ma.abs(mod - obs) / obs), axis=axis) * 100.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NMdnGE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Normalized Median Gross Error (%).

    Typical Use Cases
    -----------------
    - Comparing the typical (median) error magnitude, normalized by the mean
      observation, for robust model evaluation.
    - Useful for inter-comparison of model performance across sites or variables
      with different scales.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Normalized median gross error (percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import NMdnGE
    >>> obs = np.array([1, 2, 3, 4, 100])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1, 105])
    >>> NMdnGE(obs, mod)
    0.45454545454545453
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = abs(mod - obs)
        diff = ensure_single_chunk(diff, dim)
        result = (diff.quantile(q=0.5, dim=dim).drop_vars("quantile", errors="ignore") / obs.mean(dim=dim)) * 100.0
        return _update_history(result, "NMdnGE")
    else:
        result = (
            np.ma.masked_invalid(np.ma.median(np.ma.abs(mod - obs), axis=axis) / np.ma.mean(obs, axis=axis)) * 100.0
        )
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NO(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Optional[Union[np.ndarray, xr.DataArray]] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[int, np.ndarray, xr.DataArray]:
    """
    N Observations (#).

    Typical Use Cases
    -----------------
    - Counting the number of valid (non-masked) observations in a dataset.
    - Used to report sample size for statistical summaries and model evaluation.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray, optional
        Model predicted values (not used for NO but included for signature matching).
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to count.

    Returns
    -------
    int, numpy.ndarray, or xarray.DataArray
        Number of valid observations.
    """
    if isinstance(obs, xr.DataArray):
        dim = _resolve_axis_to_dim(obs, axis)
        return obs.count(dim=dim)
    else:
        result = (~np.ma.getmaskarray(obs)).sum(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NOP(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[int, np.ndarray, xr.DataArray]:
    """
    N Observations/Prediction Pairs (#).

    Typical Use Cases
    -----------------
    - Counting the number of valid observation-prediction pairs for paired
      statistical analysis.
    - Used to ensure sample size consistency in paired model evaluation metrics.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to count.

    Returns
    -------
    int, numpy.ndarray, or xarray.DataArray
        Number of valid pairs.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        # To get pairs where BOTH are not NaN:
        mask = obs.notnull() & mod.notnull()
        return mask.sum(dim=dim)
    else:
        obsc, modc = matchmasks(obs, mod)
        result = (~np.ma.getmaskarray(obsc)).sum(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NP(
    obs: Optional[Union[np.ndarray, xr.DataArray]] = None,
    mod: Union[np.ndarray, xr.DataArray] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[int, np.ndarray, xr.DataArray]:
    """
    N Predictions (#).

    Typical Use Cases
    -----------------
    - Counting the number of valid (non-masked) model predictions in a dataset.
    - Used to report sample size for model output and for filtering invalid
      predictions.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray, optional
        Observed values (not used for NP but included for signature matching).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to count.

    Returns
    -------
    int, numpy.ndarray, or xarray.DataArray
        Number of valid predictions.
    """
    if isinstance(mod, xr.DataArray):
        dim = _resolve_axis_to_dim(mod, axis)
        return mod.count(dim=dim)
    else:
        result = (~np.ma.getmaskarray(mod)).sum(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MO(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Error (MO) - Mean of (model - observation).

    Typical Use Cases
    -----------------
    - Quantifying the average bias between model predictions and observations.
    - Used in model evaluation to assess systematic errors.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the mean error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean error (model - observation) in observation units.
        Returns 0.0 for perfect agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MO
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    >>> MO(obs, mod)
    0.1
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = (mod - obs).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MO")
    else:
        result = np.ma.mean(np.ma.masked_invalid(np.subtract(mod, obs)), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MP(
    obs: Optional[Union[np.ndarray, xr.DataArray]] = None,
    mod: Union[np.ndarray, xr.DataArray] = None,
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Predictions (model unit).

    Typical Use Cases
    -----------------
    - Calculating the average value of model predictions for baseline or
      climatological reference.
    - Used in normalization, anomaly calculation, and summary statistics for
      model output.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray, optional
        Observed values (not used for MP but included for signature matching).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the mean.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean of predictions.
    """
    if isinstance(mod, xr.DataArray):
        dim = _resolve_axis_to_dim(mod, axis)
        result = mod.mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MP")
    else:
        result = np.ma.mean(np.ma.masked_invalid(mod), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MdnO(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Median Error (MdnO) - Median of (model - observation).

    Typical Use Cases
    -----------------
    - Quantifying the typical bias between model predictions and observations,
      robust to outliers.
    - Used in robust model evaluation for non-parametric error assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the median error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median error (model - observation) in observation units.
        Returns 0.0 for perfect agreement.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MdnO
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    >>> MdnO(obs, mod)
    0.1
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = mod - obs
        diff = ensure_single_chunk(diff, dim)
        result = diff.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore")
        return _update_history(result, "MdnO")
    else:
        result = np.median(np.subtract(mod, obs), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MdnP(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Median Error (MdnP) - Median of (model - observation).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the median error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median error (model - observation) in model units.
        Returns 0.0 for perfect agreement.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = mod - obs
        diff = ensure_single_chunk(diff, dim)
        result = diff.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore")
        return _update_history(result, "MdnP")
    else:
        result = np.median(np.subtract(mod, obs), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def RM(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Root Mean Error (RM) - Root of mean squared error.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Root of mean squared error (observation units).
        Returns 0.0 for perfect agreement.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = np.sqrt(((obs - mod) ** 2).mean(dim=dim, keep_attrs=True))
        return _update_history(result, "RM")
    else:
        result = np.sqrt(np.mean((np.subtract(obs, mod)) ** 2, axis=axis))
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def RMdn(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Root Median Error (RMdn) - Root of median squared error.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Root of median squared error (observation units).
        Returns 0.0 for perfect agreement.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff_sq = (obs - mod) ** 2
        diff_sq = ensure_single_chunk(diff_sq, dim)
        result = np.sqrt(diff_sq.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore"))
        return _update_history(result, "RMdn")
    else:
        squared_errors = (np.subtract(obs, mod)) ** 2
        result = np.sqrt(np.median(squared_errors, axis=axis))
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MB(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Bias (MB).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the mean bias.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean bias value(s) = mean(model - observation).
        Positive values indicate model overestimation.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = (mod - obs).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MB")
    else:
        result = np.ma.mean(np.subtract(mod, obs), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MdnB(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Median Bias (MdnB).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the median bias.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median bias value(s) = median(model - observation).
        Positive values indicate model overestimation.
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = mod - obs
        diff = ensure_single_chunk(diff, dim)
        result = diff.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore")
        return _update_history(result, "MdnB")
    else:
        result = np.ma.median(np.subtract(mod, obs), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def WDMB(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Mean Bias (WDMB).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the mean bias.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean wind direction bias (degrees).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = circlebias(mod - obs).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "WDMB")
    else:
        result = np.ma.mean(circlebias(np.subtract(mod, obs)), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


WDMB_m = WDMB


def WDMdnB(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Wind Direction Median Bias (WDMdnB).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed wind direction values (degrees).
    mod : numpy.ndarray or xarray.DataArray
        Model predicted wind direction values (degrees).
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the median bias.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median wind direction bias (degrees).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff = circlebias(mod - obs)
        diff = ensure_single_chunk(diff, dim)
        result = diff.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore")
        return _update_history(result, "WDMdnB")
    else:
        result = np.ma.median(circlebias(np.subtract(mod, obs)), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Squared Error (MSE).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute the error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean squared error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MSE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> MSE(obs, mod)
    0.6666666666666666
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = ((mod - obs) ** 2).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MSE")
    else:
        result = np.ma.mean((np.subtract(mod, obs)) ** 2, axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MAE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Absolute Error (MAE).

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations,
      regardless of direction.
    - Used in model evaluation, forecast verification, and regression analysis.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MAE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MAE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> MAE(obs, mod)
    0.6666666666666666
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = abs(mod - obs).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MAE")
    else:
        result = np.ma.abs(np.subtract(mod, obs)).mean(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MedAE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Median Absolute Error (MedAE).

    Typical Use Cases
    -----------------
    - Evaluating the typical magnitude of errors, robust to outliers and
      non-normal error distributions.
    - Used in robust regression, model evaluation, and forecast verification.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MedAE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Median absolute error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MedAE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> MedAE(obs, mod)
    1.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dim = list(obs.dims)
        diff_abs = abs(mod - obs)
        diff_abs = ensure_single_chunk(diff_abs, dim)
        result = diff_abs.quantile(q=0.5, dim=dim, keep_attrs=True).drop_vars("quantile", errors="ignore")
        return _update_history(result, "MedAE")
    else:
        result = np.ma.median(np.ma.abs(np.subtract(mod, obs)), axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def CRMSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Centered Root Mean Square Error (CRMSE).

    Typical Use Cases
    -----------------
    - Quantifying the error between anomalies (deviations from mean) of model
      and observations.
    - Used in Taylor diagrams, model evaluation, and forecast verification.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute CRMSE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Centered root mean square error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import CRMSE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> CRMSE(obs, mod)
    0.4714045207910317
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        o_ = obs - obs.mean(dim=dim)
        m_ = mod - mod.mean(dim=dim)
        result = ((m_ - o_) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        return _update_history(result, "CRMSE")
    else:
        o_ = np.subtract(obs, np.mean(obs, axis=axis, keepdims=True))
        m_ = np.subtract(mod, np.mean(mod, axis=axis, keepdims=True))
        result = (np.ma.abs(m_ - o_) ** 2).mean(axis=axis) ** 0.5
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MAPE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Absolute Percentage Error (MAPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations
      as a percentage.
    - Used in time series forecasting, regression, and model evaluation for
      percentage-based error assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MAPE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean absolute percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MAPE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> MAPE(obs, mod)
    50.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = (100 * abs(mod - obs) / abs(obs)).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MAPE")
    else:
        result = (100 * np.ma.abs(np.subtract(mod, obs)) / np.ma.abs(obs)).mean(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def sMAPE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations,
      normalized by their mean.
    - Used in time series forecasting, regression, and model evaluation for
      percentage-based error assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute sMAPE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Symmetric mean absolute percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import sMAPE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> sMAPE(obs, mod)
    28.57142857142857
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = (200 * abs(mod - obs) / (abs(mod) + abs(obs))).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "sMAPE")
    else:
        result = (200 * np.ma.abs(np.subtract(mod, obs)) / (np.ma.abs(mod) + np.ma.abs(obs))).mean(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NRMSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Normalized Root Mean Square Error (NRMSE).

    Typical Use Cases
    -----------------
    - Quantifying the relative error between model and observations, normalized
      by the range of observations.
    - Used in model evaluation to compare performance across different variables
      or sites with different scales.
    - Provides dimensionless error metric for cross-comparison.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute NRMSE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Normalized root mean square error (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import NRMSE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NRMSE(obs, mod)
    0.4714045207910317
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        rmse = ((mod - obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        obs_range = obs.max(dim=dim) - obs.min(dim=dim)
        result = xr.where(obs_range == 0, 0, rmse / obs_range)
        return _update_history(result, "NRMSE")
    else:
        rmse = np.ma.sqrt(np.ma.mean((np.subtract(mod, obs)) ** 2, axis=axis))
        obs_range = np.ma.max(obs, axis=axis) - np.ma.min(obs, axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(obs_range == 0, 0, rmse / obs_range)
            return result.item() if np.ndim(result) == 0 else result


def MASE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Absolute Scaled Error (MASE).

    Typical Use Cases
    -----------------
    - Quantifying model error relative to the error of a simple baseline model
      (e.g., naive forecast).
    - Used in time series forecasting and model evaluation.
    - Provides scale-independent comparison across different datasets.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MASE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean absolute scaled error (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MASE
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1])
    >>> MASE(obs, mod)
    0.1
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)

        # Calculate naive forecast error (using previous observation)
        if "time" in obs.dims:
            naive_error = abs(obs - obs.shift(time=1)).mean(dim=dim, skipna=True)
        else:
            # Fallback if time is not named 'time'
            naive_error = abs(obs - obs.shift({obs.dims[0]: 1})).mean(dim=dim, skipna=True)

        model_error = abs(mod - obs).mean(dim=dim, keep_attrs=True)
        result = model_error / naive_error
        return _update_history(result, "MASE")
    else:
        # Calculate naive forecast error (using previous observation)
        if axis is not None:
            naive_diff = np.diff(obs, axis=axis)
            naive_error = np.mean(np.abs(naive_diff), axis=axis)
        else:
            naive_diff = np.diff(obs)
            naive_error = np.mean(np.abs(naive_diff))
        model_error = np.mean(np.abs(np.subtract(mod, obs)), axis=axis)
        result = model_error / naive_error
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MASEm(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Mean Absolute Scaled Error (MASE) - robust to masked arrays.

    Typical Use Cases
    -----------------
    - Quantifying model error relative to the error of a simple baseline model
      (e.g., naive forecast), robust to masked arrays.
    - Used in time series forecasting and model evaluation with missing data.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MASE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean absolute scaled error (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import MASEm
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([1.1, 2.1, 3.1, 4.1])
    >>> MASEm(obs, mod)
    0.1
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        # MASE implementation for xarray already handles NaNs with skipna=True
        return MASE(obs, mod, axis=axis)
    else:
        # Calculate naive forecast error (using previous observation) with masked arrays
        if axis is not None:
            # Use numpy's gradient-like approach for masked arrays
            naive_diff = np.ma.diff(obs, axis=axis)
            naive_error = np.ma.mean(np.ma.abs(naive_diff), axis=axis)
        else:
            naive_diff = np.ma.diff(obs)
            naive_error = np.ma.mean(np.ma.abs(naive_diff))
        model_error = np.ma.mean(np.ma.abs(np.subtract(mod, obs)), axis=axis)
        result = model_error / naive_error
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def RMSPE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Root Mean Square Percentage Error (RMSPE).

    Typical Use Cases
    -----------------
    - Quantifying the average relative error between model and observations as
      a percentage, emphasizing larger errors.
    - Used in time series forecasting, regression, and model evaluation for
      percentage-based error assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute RMSPE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Root mean square percentage error (in percent).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import RMSPE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> RMSPE(obs, mod)
    50.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = (100 * ((mod - obs) / obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        return _update_history(result, "RMSPE")
    else:
        result = 100 * np.ma.sqrt(np.ma.mean(((mod - obs) / obs) ** 2, axis=axis))
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


MAPEm = MAPE  # noqa: N816
sMAPEm = sMAPE  # noqa: N816


def NSC(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Nash-Sutcliffe Coefficient (NSC) - Alternative to NSE.

    Typical Use Cases
    -----------------
    - Quantifying the predictive power of hydrological models relative to
      the mean of observations.
    - Used in hydrology, meteorology, and environmental model evaluation.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute NSC.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Nash-Sutcliffe coefficient (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import NSC
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NSC(obs, mod)
    -0.33333333333333326
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        obs_mean = obs.mean(dim=dim)
        numerator = ((obs - mod) ** 2).sum(dim=dim)
        denominator = ((obs - obs_mean) ** 2).sum(dim=dim)
        result = 1.0 - (numerator / denominator)
        return _update_history(result, "NSC")
    else:
        obs_mean = np.mean(obs, axis=axis, keepdims=True)
        numerator = np.sum((obs - mod) ** 2, axis=axis)
        denominator = np.sum((obs - obs_mean) ** 2, axis=axis)
        result = 1.0 - (numerator / denominator)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NSE_alpha(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    NSE Alpha - Decomposed NSE component measuring ratio of standard deviations.

    Typical Use Cases
    -----------------
    - Quantifying the model's ability to capture the variability of observations.
    - Used in model evaluation to assess how well model represents observed
      variability.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute NSE_alpha.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        NSE alpha component (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import NSE_alpha
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NSE_alpha(obs, mod)
    0.0
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = mod.std(dim=dim) / obs.std(dim=dim)
        return _update_history(result, "NSE_alpha")
    else:
        result = np.std(mod, axis=axis) / np.std(obs, axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def NSE_beta(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    NSE Beta - Decomposed NSE component measuring bias.

    Typical Use Cases
    -----------------
    - Quantifying the systematic bias between model and observations.
    - Used in model evaluation to assess mean differences between model and
      observations.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute NSE_beta.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        NSE beta component (unitless).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import NSE_beta
    >>> obs = np.array([1, 2, 3, 4])
    >>> mod = np.array([2, 2, 2, 2])
    >>> NSE_beta(obs, mod)
    0.5
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = mod.mean(dim=dim) / obs.mean(dim=dim)
        return _update_history(result, "NSE_beta")
    else:
        result = np.mean(mod, axis=axis) / np.mean(obs, axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


# Aliases for masked versions (already handled by base functions)
MAE_m = MAE


MedAE_m = MedAE


def RMSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Root Mean Square Error (RMSE).

    Typical Use Cases
    -----------------
    - Quantifying the average magnitude of errors between model and observations,
      accounting for large errors more heavily than MAE.
    - Used in model evaluation, forecast verification, and regression analysis.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute RMSE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Root mean square error.

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import RMSE
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> RMSE(obs, mod)
    0.816496580927726
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        result = ((mod - obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        return _update_history(result, "RMSE")
    else:
        result = np.ma.sqrt(np.ma.mean((np.subtract(mod, obs)) ** 2, axis=axis))
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


RMSE_m = RMSE


def IOA(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Index of Agreement (IOA).

    Typical Use Cases
    -----------------
    - Quantifying the agreement between model and observations, normalized by
      total deviation.
    - Used in model evaluation for skill assessment.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute IOA.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Index of agreement (unitless, 0-1).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import IOA
    >>> obs = np.array([1, 2, 3])
    >>> mod = np.array([2, 2, 4])
    >>> IOA(obs, mod)
    0.8
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        obs_mean = obs.mean(dim=dim)
        num = ((obs - mod) ** 2).sum(dim=dim)
        denom = ((abs(mod - obs_mean) + abs(obs - obs_mean)) ** 2).sum(dim=dim)
        result = 1.0 - (num / denom)
        return _update_history(result, "IOA")
    else:
        obs_m = np.ma.masked_invalid(obs)
        mod_m = np.ma.masked_invalid(mod)
        obs_mean = np.ma.mean(obs_m, axis=axis, keepdims=True)
        num = np.ma.sum((obs_m - mod_m) ** 2, axis=axis)
        denom = np.ma.sum((np.ma.abs(mod_m - obs_mean) + np.ma.abs(obs_m - obs_mean)) ** 2, axis=axis)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 1.0 - (num / denom)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


IOA_m = IOA


# Add the missing functions from the specification


def MAPE_mod(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Modified Mean Absolute Percentage Error (MAPE).

    This version handles cases where observations might be zero or near zero
    by using a small epsilon to avoid division by zero.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MAPE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean absolute percentage error (in percent).
    """
    # Small epsilon to avoid division by zero
    epsilon = 1e-8

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        # Add epsilon to avoid division by zero
        obs_safe = xr.where(abs(obs) < epsilon, epsilon, obs)
        result = (100 * abs(mod - obs) / abs(obs_safe)).mean(dim=dim, keep_attrs=True)
        return _update_history(result, "MAPE_mod")
    else:
        # Add epsilon to avoid division by zero
        obs_safe = np.where(np.abs(obs) < epsilon, epsilon, obs)
        result = (100 * np.abs(np.subtract(mod, obs)) / np.abs(obs_safe)).mean(axis=axis)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def MASE_mod(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Modified Mean Absolute Scaled Error (MASE).

    This version handles cases where the naive forecast error is zero
    by using a small epsilon to avoid division by zero.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute MASE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Mean absolute scaled error (unitless).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        # Calculate naive forecast error (using previous observation)
        if "time" in obs.dims:
            naive_error = abs(obs - obs.shift(time=1)).mean(dim=dim, skipna=True)
        else:
            naive_error = abs(obs - obs.shift({obs.dims[0]: 1})).mean(dim=dim, skipna=True)

        model_error = abs(mod - obs).mean(dim=dim, keep_attrs=True)
        # Avoid division by zero
        result = xr.where(naive_error == 0, model_error, model_error / naive_error)
        return _update_history(result, "MASE_mod")
    else:
        # Calculate naive forecast error (using previous observation)
        if axis is not None:
            naive_diff = np.diff(obs, axis=axis)
            naive_error = np.mean(np.abs(naive_diff), axis=axis)
        else:
            naive_diff = np.diff(obs)
            naive_error = np.mean(np.abs(naive_diff))
        model_error = np.mean(np.abs(np.subtract(mod, obs)), axis=axis)
        # Avoid division by zero
        result = np.where(naive_error == 0, model_error, model_error / naive_error)
        return result.item() if np.ndim(result) == 0 else result


def RMSE_norm(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Normalized Root Mean Square Error (RMSE_norm).

    Normalizes RMSE by the range of observations.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute normalized RMSE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Normalized root mean square error (unitless).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        rmse = ((mod - obs) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        obs_min = obs.min(dim=dim)
        obs_max = obs.max(dim=dim)
        obs_range = obs_max - obs_min
        # Avoid division by zero
        result = xr.where(obs_range == 0, rmse, rmse / obs_range)
        return _update_history(result, "RMSE_norm")
    else:
        rmse = np.sqrt(np.mean((np.subtract(mod, obs)) ** 2, axis=axis))
        obs_min = np.min(obs, axis=axis)
        obs_max = np.max(obs, axis=axis)
        obs_range = obs_max - obs_min
        # Avoid division by zero
        result = np.where(obs_range == 0, rmse, rmse / obs_range)
        return result.item() if np.ndim(result) == 0 else result


def MAE_norm(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Normalized Mean Absolute Error (MAE_norm).

    Normalizes MAE by the range of observations.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute normalized MAE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Normalized mean absolute error (unitless).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        mae = abs(mod - obs).mean(dim=dim, keep_attrs=True)
        obs_min = obs.min(dim=dim)
        obs_max = obs.max(dim=dim)
        obs_range = obs_max - obs_min
        # Avoid division by zero
        result = xr.where(obs_range == 0, mae, mae / obs_range)
        return _update_history(result, "MAE_norm")
    else:
        mae = np.mean(np.abs(np.subtract(mod, obs)), axis=axis)
        obs_min = np.min(obs, axis=axis)
        obs_max = np.max(obs, axis=axis)
        obs_range = obs_max - obs_min
        # Avoid division by zero
        result = np.where(obs_range == 0, mae, mae / obs_range)
        return result.item() if np.ndim(result) == 0 else result


def bias_fraction(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Bias Fraction (BF).

    Quantifies the fraction of total error that is due to systematic bias.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute bias fraction.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Bias fraction (unitless, 0-1).
    """
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        bias = (mod - obs).mean(dim=dim)
        total_error = np.sqrt(((mod - obs) ** 2).mean(dim=dim, keep_attrs=True))
        # Avoid division by zero
        result = xr.where(total_error == 0, 0, (bias**2) / (total_error**2))
        return _update_history(result, "bias_fraction")
    else:
        bias = np.mean(np.subtract(mod, obs), axis=axis)
        total_error = np.sqrt(np.mean((np.subtract(mod, obs)) ** 2, axis=axis))
        # Avoid division by zero
        result = np.where(total_error == 0, 0, (bias**2) / (total_error**2))
        return result.item() if np.ndim(result) == 0 else result


# Add missing functions from the specification


def NMSE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Normalized Mean Square Error (NMSE).

    Typical Use Cases
    -----------------
    - Quantifying the normalized squared error between model and observations.
    - Used in model evaluation to compare performance across different variables
      or sites with different scales.
    - Provides dimensionless error metric for cross-comparison.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute NMSE.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
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
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        mse = ((mod - obs) ** 2).mean(dim=dim, keep_attrs=True)
        obs_var = obs.var(dim=dim)
        # Handle case where variance is 0 (perfect agreement)
        result = xr.where(obs_var == 0, 0, mse / obs_var)
        return _update_history(result, "NMSE")
    else:
        mse = np.ma.mean(np.ma.masked_invalid(np.subtract(mod, obs)) ** 2, axis=axis)
        obs_var = np.ma.var(np.ma.masked_invalid(obs), axis=axis)
        # Handle case where variance is 0 (perfect agreement)
        result = np.where(obs_var == 0, 0, mse / obs_var)
        return result.item() if np.ndim(result) == 0 else result


def LOG_ERROR(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Logarithmic Error Metric.

    Typical Use Cases
    -----------------
    - Quantifying errors for variables that span several orders of magnitude.
    - Used in atmospheric sciences for concentration data (e.g., pollutants).
    - Helpful when relative rather than absolute errors are important.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values (should be positive).
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values (should be positive).
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute log error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
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
    # Add small epsilon to avoid log(0) and handle negative values
    epsilon = 1e-10

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        # Use abs to handle potential negative values, then add epsilon
        obs_safe = abs(obs) + epsilon
        mod_safe = abs(mod) + epsilon
        obs_log = np.log(obs_safe)
        mod_log = np.log(mod_safe)
        result = ((mod_log - obs_log) ** 2).mean(dim=dim, keep_attrs=True) ** 0.5
        return _update_history(result, "LOG_ERROR")
    else:
        # Use abs to handle potential negative values, then add epsilon
        obs_safe = np.abs(obs) + epsilon
        mod_safe = np.abs(mod) + epsilon
        obs_log = np.log(obs_safe)
        mod_log = np.log(mod_safe)

        result = np.sqrt(np.mean((mod_log - obs_log) ** 2, axis=axis))
        # Return 0 for perfect agreement
        if np.array_equal(obs, mod):
            return 0.0
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def COE(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Center of Mass Error (COE).

    The COE measures the displacement between the centroids (centers of mass)
    of two fields. For spatial data, this represents the shift in the center
    of a feature (e.g., a storm or a pollutant plume).

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values (typically 2D spatial field).
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values (typically 2D spatial field).
    axis : int, str, or iterable of such, optional
        Axis or dimension(s) over which to compute the centroid.
        If None, computes over all axes.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
        Center of mass error (Euclidean distance between centroids).

    Examples
    --------
    >>> import numpy as np
    >>> from monet_stats.error_metrics import COE
    >>> obs = np.zeros((5, 5))
    >>> obs[2, 2] = 1.0  # Peak at center (2, 2)
    >>> mod = np.zeros((5, 5))
    >>> mod[3, 3] = 1.0  # Peak shifted to (3, 3)
    >>> # Displacement is sqrt(1^2 + 1^2) = sqrt(2) approx 1.414
    >>> np.allclose(COE(obs, mod), np.sqrt(2))
    True
    """

    def _get_centroid(da: xr.DataArray, dims: Iterable[str]) -> List[xr.DataArray]:
        """Helper to calculate centroid of a DataArray."""
        total = da.sum(dim=dims)
        # Handle zero sum to avoid division by zero
        total_safe = xr.where(total == 0, 1e-10, total)
        coords_list = []
        for d in dims:
            # Check if coord exists and is numeric
            if d in da.coords and np.issubdtype(da.coords[d].dtype, np.number):
                coord = da.coords[d]
            else:
                # Fallback to dimension indices
                coord = xr.DataArray(np.arange(da.sizes[d]), dims=d, name=d)
            # Weighted mean of coordinate
            c = (da * coord).sum(dim=dims) / total_safe
            coords_list.append(c)
        return coords_list

    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        if dim is None:
            dims = list(obs.dims)
        elif isinstance(dim, str):
            dims = [dim]
        else:
            dims = list(dim)

        c_obs = _get_centroid(obs, dims)
        c_mod = _get_centroid(mod, dims)

        # Euclidean distance
        dist_sq = sum((cm - co) ** 2 for cm, co in zip(c_mod, c_obs))
        result = dist_sq**0.5

        return _update_history(result, "Center of Mass Error (COE)")

    # Fallback to numpy
    obs_arr = np.asanyarray(obs)
    mod_arr = np.asanyarray(mod)

    if axis is None:
        axes = tuple(range(obs_arr.ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    elif isinstance(axis, str):
        # Handle single string axis for consistency with xarray path
        axes = (obs_arr.ndim - 1,)  # Best guess for numpy if only string provided
    else:
        axes = tuple(axis)

    def _get_numpy_centroid(arr: np.ndarray, axes_tuple: Tuple[int, ...]) -> List[np.ndarray]:
        """Helper to calculate centroid of a NumPy array."""
        total = np.sum(arr, axis=axes_tuple)
        total_safe = np.where(total == 0, 1e-10, total)
        c_list = []
        for ax in axes_tuple:
            # Create coordinate array for this axis
            shape = [1] * arr.ndim
            shape[ax] = arr.shape[ax]
            coord = np.arange(arr.shape[ax]).reshape(shape)
            c = np.sum(arr * coord, axis=axes_tuple) / total_safe
            c_list.append(c)
        return c_list

    c_obs_np = _get_numpy_centroid(obs_arr, axes)
    c_mod_np = _get_numpy_centroid(mod_arr, axes)

    dist_sq_np = sum((cm - co) ** 2 for cm, co in zip(c_mod_np, c_obs_np))
    result = dist_sq_np**0.5
    return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def VOLUMETRIC_ERROR(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Volumetric Error Metric.

    Typical Use Cases
    -----------------
    - Quantifying the volume difference between observed and modeled features.
    - Used in hydrology for flood extent verification.
    - Applied in meteorology for precipitation volume verification.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute volumetric error.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
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
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        obs_sum = obs.sum(dim=dim)
        mod_sum = mod.sum(dim=dim)
        result = abs(mod_sum - obs_sum) / abs(obs_sum)
        return _update_history(result, "VOLUMETRIC_ERROR")
    else:
        obs_sum = np.sum(obs, axis=axis)
        mod_sum = np.sum(mod, axis=axis)
        result = np.abs(mod_sum - obs_sum) / np.abs(obs_sum)
        return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result


def CORR_INDEX(
    obs: Union[np.ndarray, xr.DataArray],
    mod: Union[np.ndarray, xr.DataArray],
    axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
) -> Union[np.number, np.ndarray, xr.DataArray]:
    """
    Correlation Index (CORR_INDEX).

    Typical Use Cases
    -----------------
    - Measuring the linear relationship between observed and modeled values.
    - Used as a component in model evaluation.
    - Quantifies how well model captures observed patterns.

    Parameters
    ----------
    obs : numpy.ndarray or xarray.DataArray
        Observed values.
    mod : numpy.ndarray or xarray.DataArray
        Model or predicted values.
    axis : int, str, or iterable of such, optional
        Axis or dimension along which to compute correlation index.

    Returns
    -------
    numpy.number, numpy.ndarray, or xarray.DataArray
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
    if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
        obs, mod = xr.align(obs, mod, join="inner")
        dim = _resolve_axis_to_dim(obs, axis)
        # Using xarray's built-in correlation function
        result = xr.corr(obs, mod, dim=dim)
        return _update_history(result, "CORR_INDEX")
    else:
        # Fallback to numpy-compatible logic
        obs = np.asarray(obs)
        mod = np.asarray(mod)
        if axis is None:
            from scipy.stats import pearsonr

            result = pearsonr(obs.flatten(), mod.flatten())[0]
            return result.item() if hasattr(result, "item") else float(result)
        else:
            # Manual vectorized correlation over axis for robustness across scipy versions
            obs_mean = np.mean(obs, axis=axis, keepdims=True)
            mod_mean = np.mean(mod, axis=axis, keepdims=True)
            obs_std = obs - obs_mean
            mod_std = mod - mod_mean
            num = np.sum(obs_std * mod_std, axis=axis)
            den = np.sqrt(np.sum(obs_std**2, axis=axis) * np.sum(mod_std**2, axis=axis))
            result = num / den
            return result.item() if hasattr(result, "item") and np.ndim(result) == 0 else result
