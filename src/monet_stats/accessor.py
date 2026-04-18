"""
Xarray accessors for the MONET Stats package (Aero Protocol Compliant).
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

from . import (
    analysis,
    contingency_metrics,
    correlation_metrics,
    distribution_metrics,
    efficiency_metrics,
    error_metrics,
    performance,
    relative_metrics,
    spatial_ensemble_metrics,
    spatial_skill_metrics,
    temporal_metrics,
    uncertainty,
)


@xr.register_dataarray_accessor("monet_stats")
class MonetDataArrayAccessor:
    """
    Accessor for xarray.DataArray to provide MONET statistical methods.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def climatology(self, freq: str = "season", method: str = "mean", dim: str = "time") -> xr.DataArray:
        """
        Compute climatological statistics.

        Parameters
        ----------
        freq : str, optional
            Climatology frequency ('season', 'month', 'dayofyear', 'hour').
            Default is 'season'.
        method : str, optional
            Statistical method to apply ('mean', 'std', 'min', 'max', 'median').
            Default is 'mean'.
        dim : str, optional
            Dimension along which to compute climatology. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Climatological statistics.
        """
        return analysis.climatology(self._obj, freq=freq, method=method, dim=dim)

    def resample_data(self, freq: str = "MS", method: str = "mean", dim: str = "time", **kwargs: Any) -> xr.DataArray:
        """
        Resample data to a new temporal frequency.

        Parameters
        ----------
        freq : str, optional
            Resampling frequency (e.g., 'MS', 'W', 'D'). Default is 'MS'.
        method : str, optional
            Statistical method to apply ('mean', 'sum', 'min', 'max', 'std', 'median').
            Default is 'mean'.
        dim : str, optional
            Dimension along which to resample. Default is 'time'.
        **kwargs : Any
            Additional keyword arguments passed to the resample method.

        Returns
        -------
        xarray.DataArray
            Resampled data.
        """
        return analysis.resample_data(self._obj, freq=freq, method=method, dim=dim, **kwargs)

    def kz_filter(self, m: int, k: int, dim: str = "time") -> xr.DataArray:
        """
        Apply Kolmogorov-Zurbenko (KZ) filter.

        Parameters
        ----------
        m : int
            Window size for the moving average.
        k : int
            Number of iterations.
        dim : str, optional
            Dimension along which to apply the filter. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Filtered data.
        """
        return analysis.kz_filter(self._obj, m=m, k=k, dim=dim)

    def diurnal_cycle(self, method: str = "mean", dim: str = "time") -> xr.DataArray:
        """
        Compute the diurnal cycle (average hourly profile).

        Parameters
        ----------
        method : str, optional
            Statistical method to apply ('mean', 'median', 'std'). Default is 'mean'.
        dim : str, optional
            Dimension along which to compute the cycle. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Diurnal cycle.
        """
        return analysis.diurnal_cycle(self._obj, method=method, dim=dim)

    def rolling_mean_8h(self, dim: str = "time", min_periods: int = 6, center: bool = True) -> xr.DataArray:
        """
        Compute rolling 8-hour mean.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute the mean. Default is 'time'.
        min_periods : int, optional
            Minimum number of observations in window. Default is 6.
        center : bool, optional
            If True, set the labels at the center of the window. Default is True.

        Returns
        -------
        xarray.DataArray
            Rolling 8-hour mean.
        """
        return analysis.rolling_mean_8h(self._obj, dim=dim, min_periods=min_periods, center=center)

    def rolling_mean_24h(self, dim: str = "time", min_periods: int = 18, center: bool = True) -> xr.DataArray:
        """
        Compute rolling 24-hour mean.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute the mean. Default is 'time'.
        min_periods : int, optional
            Minimum number of observations in window. Default is 18.
        center : bool, optional
            If True, set the labels at the center of the window. Default is True.

        Returns
        -------
        xarray.DataArray
            Rolling 24-hour mean.
        """
        return analysis.rolling_mean_24h(self._obj, dim=dim, min_periods=min_periods, center=center)

    def mda1(self, dim: str = "time") -> xr.DataArray:
        """
        Compute Maximum Daily 1-hour Average (MDA1).

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            MDA1 values.
        """
        return analysis.mda1(self._obj, dim=dim)

    def mda8(self, dim: str = "time", min_periods: int = 6, center: bool = False) -> xr.DataArray:
        """
        Compute Maximum Daily 8-hour Average (MDA8).

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.
        min_periods : int, optional
            Minimum number of observations for the 8-hour rolling mean. Default is 6.
        center : bool, optional
            Whether to center the 8-hour rolling window. Default is False.

        Returns
        -------
        xarray.DataArray
            MDA8 values.
        """
        return analysis.mda8(self._obj, dim=dim, min_periods=min_periods, center=center)

    def exceedance_count(self, threshold: float, dim: str = "time") -> xr.DataArray:
        """
        Count exceedances of a threshold.

        Parameters
        ----------
        threshold : float
            Value above which an exceedance is counted.
        dim : str, optional
            Dimension along which to count exceedances. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Number of exceedances.
        """
        return analysis.exceedance_count(self._obj, threshold=threshold, dim=dim)

    def percentile(self, q: Union[float, List[float], np.ndarray], dim: str = "time", **kwargs: Any) -> xr.DataArray:
        """
        Compute percentiles.

        Parameters
        ----------
        q : float or list of float
            Percentile(s) to compute (0-100).
        dim : str, optional
            Dimension over which to compute percentiles. Default is 'time'.
        **kwargs : Any
            Additional keyword arguments passed to xarray.quantile.

        Returns
        -------
        xarray.DataArray
            Computed percentiles.
        """
        return analysis.percentile(self._obj, q=q, dim=dim, **kwargs)

    def peak_timing(self, dim: str = "time") -> xr.DataArray:
        """
        Identify the coordinate value of the maximum.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to find the peak. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Coordinate values where the maximum occurs.
        """
        return analysis.peak_timing(self._obj, dim=dim)

    def weighted_spatial_mean(
        self,
        lat_dim: str = "lat",
        lon_dim: str = "lon",
        weights: Optional[Union[xr.DataArray, np.ndarray]] = None,
    ) -> xr.DataArray:
        """
        Compute area-weighted spatial mean.

        Parameters
        ----------
        lat_dim : str, optional
            Name of the latitude dimension. Default is 'lat'.
        lon_dim : str, optional
            Name of the longitude dimension. Default is 'lon'.
        weights : xarray.DataArray or numpy.ndarray, optional
            Custom weights for the mean.

        Returns
        -------
        xarray.DataArray
            Area-weighted spatial mean.
        """
        return analysis.weighted_spatial_mean(self._obj, lat_dim=lat_dim, lon_dim=lon_dim, weights=weights)

    def fft_analysis(self, dim: str = "time", output: str = "psd") -> xr.DataArray:
        """
        Perform Fast Fourier Transform (FFT) analysis.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to perform FFT. Default is 'time'.
        output : str, optional
            Type of output to return ('psd', 'magnitude', 'complex'). Default is 'psd'.

        Returns
        -------
        xarray.DataArray
            FFT results.
        """
        return analysis.fft_analysis(self._obj, dim=dim, output=output)

    def power_spectrum(
        self,
        dim: str = "time",
        fs: float = 1.0,
        window: str = "hann",
        nperseg: Optional[int] = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """
        Compute power spectrum using Welch's method.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute the spectrum. Default is 'time'.
        fs : float, optional
            Sampling frequency. Default is 1.0.
        window : str, optional
            Desired window to use. Default is 'hann'.
        nperseg : int, optional
            Length of each segment.
        **kwargs : Any
            Additional keyword arguments passed to scipy.signal.welch.

        Returns
        -------
        xarray.DataArray
            Power spectral density.
        """
        return analysis.power_spectrum(self._obj, dim=dim, fs=fs, window=window, nperseg=nperseg, **kwargs)

    def seasonal_mean(self, dim: str = "time", weighted: bool = True) -> xr.DataArray:
        """
        Compute seasonal mean (DJF, MAM, JJA, SON).

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute the mean. Default is 'time'.
        weighted : bool, optional
            If True, weight by days in month. Default is True.

        Returns
        -------
        xarray.DataArray
            Seasonal means.
        """
        return analysis.seasonal_mean(self._obj, dim=dim, weighted=weighted)

    def monthly_climatology(self, dim: str = "time", method: str = "mean") -> xr.DataArray:
        """
        Compute monthly climatology.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute the climatology. Default is 'time'.
        method : str, optional
            Statistical method to apply. Default is 'mean'.

        Returns
        -------
        xarray.DataArray
            Monthly climatology.
        """
        return analysis.monthly_climatology(self._obj, dim=dim, method=method)

    def anomalies(self, freq: str = "month", dim: str = "time") -> xr.DataArray:
        """
        Compute anomalies by subtracting the climatology.

        Parameters
        ----------
        freq : str, optional
            Climatology frequency ('season', 'month', 'dayofyear', 'hour').
            Default is 'month'.
        dim : str, optional
            Dimension along which to compute the anomalies. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Anomalies.
        """
        return analysis.anomalies(self._obj, freq=freq, dim=dim)

    def detrend(self, method: str = "linear", dim: str = "time") -> xr.DataArray:
        """
        Remove trend from data.

        Parameters
        ----------
        method : str, optional
            Detrending method ('linear', 'constant'). Default is 'linear'.
        dim : str, optional
            Dimension along which to detrend. Default is 'time'.

        Returns
        -------
        xarray.DataArray
            Detrended data.
        """
        return analysis.detrend(self._obj, method=method, dim=dim)

    def optimize(self, target_mb: float = 100.0) -> xr.DataArray:
        """
        Optimize performance by ensuring laziness and recommended chunks (Aero Protocol).

        Parameters
        ----------
        target_mb : float, optional
            Target size for each chunk in Megabytes. Default is 100.0.

        Returns
        -------
        xarray.DataArray
            Optimized DataArray.
        """
        from .utils_stats import _update_history

        if not performance._has_dask():
            return _update_history(self._obj, "Optimization skipped (Dask not installed)")

        # Ensure data is lazy
        res = performance.apply_lazy_threshold(self._obj, threshold_mb=0.1)
        # Always calculate and apply recommended chunks for the target size
        recommendation = performance.get_chunk_recommendation(res, target_mb=target_mb)
        res = res.chunk(recommendation)

        return _update_history(res, f"Optimized for performance (target={target_mb}MB)")

    def rechunk(self, chunks: Optional[dict] = None) -> xr.DataArray:
        """
        Apply new chunks to the DataArray (Aero Protocol provenance tracking).

        Parameters
        ----------
        chunks : dict, optional
            New chunk sizes. If None, uses optimal recommendations (~100MB).

        Returns
        -------
        xarray.DataArray
            Rechunked DataArray.
        """
        from .utils_stats import _update_history

        if not performance._has_dask():
            return _update_history(self._obj, "Rechunking skipped (Dask not installed)")

        if chunks is None:
            chunks = performance.get_chunk_recommendation(self._obj)

        res = self._obj.chunk(chunks)

        return _update_history(res, f"Rechunked with {chunks}")

    def taylor_statistics(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.Dataset:
        """
        Calculate components required for a Taylor diagram (Aero Protocol).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values (reference).
        dim : str or list of str, optional
            Dimension(s) along which to compute the statistics.

        Returns
        -------
        xarray.Dataset
            Dataset containing:
            - std_obs: Standard deviation of observations.
            - std_mod: Standard deviation of model predictions.
            - correlation: Pearson correlation coefficient.
        """
        obs, mod = xr.align(obs, self._obj, join="inner")
        from .utils_stats import _resolve_axis_to_dim, _update_history

        d = _resolve_axis_to_dim(obs, dim)

        res = xr.Dataset(
            {
                "std_obs": obs.std(dim=d),
                "std_mod": mod.std(dim=d),
                "correlation": correlation_metrics.pearsonr(obs, mod, axis=dim),
            }
        )
        return _update_history(res, "Taylor statistics components")

    def mae(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.DataArray:
        """
        Compute Mean Absolute Error (MAE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted mean.

        Returns
        -------
        xarray.DataArray
            Mean absolute error.
        """
        return error_metrics.MAE(obs, self._obj, axis=dim, weights=weights)

    def rmse(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.DataArray:
        """
        Compute Root Mean Square Error (RMSE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted mean.

        Returns
        -------
        xarray.DataArray
            Root mean square error.
        """
        return error_metrics.RMSE(obs, self._obj, axis=dim, weights=weights)

    def mb(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.DataArray:
        """
        Compute Mean Bias (MB).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted mean.

        Returns
        -------
        xarray.DataArray
            Mean bias.
        """
        return error_metrics.MB(obs, self._obj, axis=dim, weights=weights)

    def ioa(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Index of Agreement (IOA).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Index of agreement.
        """
        return error_metrics.IOA(obs, self._obj, axis=dim)

    def crmse(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Centered Root Mean Square Error (CRMSE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Centered root mean square error.
        """
        return error_metrics.CRMSE(obs, self._obj, axis=dim)

    def mdnb(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Median Bias (MdnB).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Median bias.
        """
        return error_metrics.MdnB(obs, self._obj, axis=dim)

    def nmse(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Normalized Mean Square Error (NMSE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Normalized mean square error.
        """
        return error_metrics.NMSE(obs, self._obj, axis=dim)

    def pearsonr(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Pearson correlation coefficient.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Pearson correlation coefficient.
        """
        return correlation_metrics.pearsonr(obs, self._obj, axis=dim)

    def r2(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Coefficient of Determination (R^2).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Coefficient of determination.
        """
        return correlation_metrics.R2(obs, self._obj, axis=dim)

    def kge(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Kling-Gupta Efficiency (KGE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Kling-Gupta efficiency.
        """
        return correlation_metrics.KGE(obs, self._obj, axis=dim)

    def ccc(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Concordance Correlation Coefficient (CCC).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Concordance correlation coefficient.
        """
        return correlation_metrics.CCC(obs, self._obj, axis=dim)

    def nmb(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.DataArray:
        """
        Compute Normalized Mean Bias (NMB).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted mean.

        Returns
        -------
        xarray.DataArray
            Normalized mean bias.
        """
        return relative_metrics.NMB(obs, self._obj, axis=dim, weights=weights)

    def mg(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.DataArray:
        """
        Compute Geometric Mean Bias (MG).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted mean.

        Returns
        -------
        xarray.DataArray
            Geometric mean bias.
        """
        return relative_metrics.MG(obs, self._obj, axis=dim, weights=weights)

    def vg(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.DataArray:
        """
        Compute Geometric Variance (VG).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted mean.

        Returns
        -------
        xarray.DataArray
            Geometric variance.
        """
        return relative_metrics.VG(obs, self._obj, axis=dim, weights=weights)

    def fb(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Fractional Bias (FB).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Fractional bias.
        """
        return relative_metrics.FB(obs, self._obj, axis=dim)

    def mnb(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Mean Normalized Bias (MNB).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Mean normalized bias.
        """
        return error_metrics.MNB(obs, self._obj, axis=dim)

    def mne(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Mean Normalized Gross Error (MNE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Mean normalized gross error.
        """
        return error_metrics.MNE(obs, self._obj, axis=dim)

    def nse(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Nash-Sutcliffe Efficiency (NSE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Nash-Sutcliffe efficiency.
        """
        return efficiency_metrics.NSE(obs, self._obj, axis=dim)

    def fac2(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute fraction of predictions within a factor of two (FAC2).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            FAC2.
        """
        return error_metrics.FAC2(obs, self._obj, axis=dim)

    def rmsle(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Root Mean Square Logarithmic Error (RMSLE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            RMSLE.
        """
        return error_metrics.RMSLE(obs, self._obj, axis=dim)

    def hss(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Heidke Skill Score (HSS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Heidke Skill Score.
        """
        return contingency_metrics.HSS(obs, self._obj, minval=threshold, axis=dim)

    def ets(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Equitable Threat Score (ETS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Equitable Threat Score.
        """
        return contingency_metrics.ETS(obs, self._obj, minval=threshold, axis=dim)

    def csi(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Critical Success Index (CSI).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Critical Success Index.
        """
        return contingency_metrics.CSI(obs, self._obj, minval=threshold, axis=dim)

    def pod(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Probability of Detection (POD).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Probability of Detection.
        """
        return contingency_metrics.POD(obs, self._obj, minval=threshold, axis=dim)

    def far(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute False Alarm Rate (FAR).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            False Alarm Rate.
        """
        return contingency_metrics.FAR(obs, self._obj, minval=threshold, axis=dim)

    def fbi(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Frequency Bias Index (FBI).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Frequency Bias Index.
        """
        return contingency_metrics.FBI(obs, self._obj, minval=threshold, axis=dim)

    def tss(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute True Skill Statistic (TSS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            True Skill Statistic.
        """
        return contingency_metrics.TSS(obs, self._obj, minval=threshold, axis=dim)

    def bss_binary(
        self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None
    ) -> xr.DataArray:
        """
        Compute Binary Brier Skill Score (BSS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Binary Brier Skill Score.
        """
        return contingency_metrics.BSS_binary(obs, self._obj, threshold=threshold, axis=dim)

    def bs(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Brier Score (BS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed binary outcomes.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Brier Score.
        """
        return contingency_metrics.BS(obs, self._obj, axis=dim)

    def stdo(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Standard deviation of Observation Errors (obs - mod).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Standard deviation of errors.

        Examples
        --------
        >>> stdo = mod.monet_stats.stdo(obs)
        """
        return error_metrics.STDO(obs, self._obj, axis=dim)

    def stdp(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Standard deviation of Prediction Errors (mod - obs).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Standard deviation of errors.

        Examples
        --------
        >>> stdp = mod.monet_stats.stdp(obs)
        """
        return error_metrics.STDP(obs, self._obj, axis=dim)

    def coe(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Center of Mass Error (COE).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed field.
        dim : str or list of str, optional
            Dimension(s) along which to compute the centroid.

        Returns
        -------
        xarray.DataArray
            Center of mass error.

        Examples
        --------
        >>> coe = mod.monet_stats.coe(obs, dim=['lat', 'lon'])
        """
        return error_metrics.COE(obs, self._obj, axis=dim)

    def corr_index(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Correlation Index.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Correlation index.

        Examples
        --------
        >>> r_index = mod.monet_stats.corr_index(obs)
        """
        return error_metrics.CORR_INDEX(obs, self._obj, axis=dim)

    def bias_fraction(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Bias Fraction.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Bias fraction.

        Examples
        --------
        >>> bf = mod.monet_stats.bias_fraction(obs)
        """
        return error_metrics.bias_fraction(obs, self._obj, axis=dim)

    def log_error(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Logarithmic Error.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Logarithmic error.

        Examples
        --------
        >>> log_err = mod.monet_stats.log_error(obs)
        """
        return error_metrics.LOG_ERROR(obs, self._obj, axis=dim)

    def volumetric_error(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Volumetric Error.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metric.

        Returns
        -------
        xarray.DataArray
            Volumetric error.

        Examples
        --------
        >>> vol_err = mod.monet_stats.volumetric_error(obs)
        """
        return error_metrics.VOLUMETRIC_ERROR(obs, self._obj, axis=dim)

    def wasserstein_distance(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Wasserstein Distance.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the distance.

        Returns
        -------
        xarray.DataArray
            Wasserstein distance.
        """
        return distribution_metrics.WassersteinDistance(obs, self._obj, dim=dim)

    def kl_divergence(
        self, obs: xr.DataArray, bins: int = 100, dim: Optional[Union[str, List[str]]] = None
    ) -> xr.DataArray:
        """
        Compute KL Divergence.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        bins : int, optional
            Number of bins.
        dim : str or list of str, optional
            Dimension(s) along which to compute the divergence.

        Returns
        -------
        xarray.DataArray
            KL divergence.
        """
        return distribution_metrics.KLDivergence(obs, self._obj, bins=bins, dim=dim)

    def mutual_information(
        self, obs: xr.DataArray, bins: int = 30, dim: Optional[Union[str, List[str]]] = None
    ) -> xr.DataArray:
        """
        Compute Mutual Information.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        bins : int, optional
            Number of bins.
        dim : str or list of str, optional
            Dimension(s) along which to compute the MI.

        Returns
        -------
        xarray.DataArray
            Mutual information.

        Examples
        --------
        >>> mi = mod.monet_stats.mutual_information(obs, bins=20)
        """
        return distribution_metrics.MutualInformation(obs, self._obj, bins=bins, dim=dim)

    def dtw(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Dynamic Time Warping (DTW) distance.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension along which to compute DTW.

        Returns
        -------
        xarray.DataArray
            DTW distance.
        """
        return temporal_metrics.DynamicTimeWarping(obs, self._obj, dim=dim)

    def xwt(self, obs: xr.DataArray, widths: Optional[np.ndarray] = None, dim: str = "time") -> xr.DataArray:
        """
        Compute Cross-Wavelet Transform (XWT).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        widths : numpy.ndarray, optional
            Wavelet widths.
        dim : str, optional
            Dimension along which to compute XWT.

        Returns
        -------
        xarray.DataArray
            XWT.
        """
        return temporal_metrics.CrossWaveletTransform(obs, self._obj, widths=widths, dim=dim)

    def bootstrap(
        self,
        obs: xr.DataArray,
        metric_func: Any,
        n_boot: int = 1000,
        block_size: int = 1,
        dim: str = "time",
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Perform block-bootstrapping for a metric.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        metric_func : callable
            Metric function.
        n_boot : int, optional
            Number of bootstrap samples.
        block_size : int, optional
            Block size.
        dim : str, optional
            Dimension along which to bootstrap.

        Returns
        -------
        xarray.Dataset
            Bootstrapped results (mean, lower, upper).
        """
        return uncertainty.block_bootstrap(
            obs, self._obj, metric_func=metric_func, n_boot=n_boot, block_size=block_size, dim=dim, **kwargs
        )

    def fss(
        self,
        obs: xr.DataArray,
        window_size: int = 3,
        threshold: Optional[float] = None,
        dim: Optional[Union[str, List[str]]] = None,
    ) -> xr.DataArray:
        """
        Compute Fractions Skill Score (FSS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        window_size : int, optional
            Size of the square window, by default 3.
        threshold : float, optional
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the fractions.

        Returns
        -------
        xarray.DataArray
            Fractions Skill Score.
        """
        return spatial_skill_metrics.FSS(obs, self._obj, window_size=window_size, threshold=threshold, dim=dim)

    def vets(self, obs: xr.DataArray, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Volumetric Equitable Threat Score (VETS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the score.

        Returns
        -------
        xarray.DataArray
            Volumetric Equitable Threat Score.
        """
        return spatial_skill_metrics.VETS(obs, self._obj, dim=dim)

    def crps(self, obs: xr.DataArray, dim: Union[int, str] = 0) -> xr.DataArray:
        """
        Compute Continuous Ranked Probability Score (CRPS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : int or str, optional
            Ensemble dimension. Default is 0.

        Returns
        -------
        xarray.DataArray
            CRPS values.
        """
        return spatial_ensemble_metrics.CRPS(self._obj, obs, axis=dim)

    def sal(
        self,
        obs: xr.DataArray,
        threshold: Optional[float] = None,
        lat_dim: str = "lat",
        lon_dim: str = "lon",
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Compute Structure-Amplitude-Location (SAL) score.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed field.
        threshold : float, optional
            Threshold for object identification.
        lat_dim : str, optional
            Latitude dimension name.
        lon_dim : str, optional
            Longitude dimension name.

        Returns
        -------
        S, A, L : xarray.DataArray
            Structure, Amplitude, and Location components.
        """
        return spatial_ensemble_metrics.SAL(obs, self._obj, threshold=threshold, lat_dim=lat_dim, lon_dim=lon_dim)

    def eds(self, obs: xr.DataArray, threshold: float, dim: Optional[Union[str, List[str]]] = None) -> xr.DataArray:
        """
        Compute Extreme Dependency Score (EDS).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed field.
        threshold : float
            Event threshold.
        dim : str or list of str, optional
            Dimension(s) along which to compute the score.

        Returns
        -------
        xarray.DataArray
            Extreme Dependency Score.
        """
        return spatial_ensemble_metrics.EDS(obs, self._obj, threshold=threshold, dim=dim)

    def spread_error(
        self,
        obs: xr.DataArray,
        ensemble_dim: Union[int, str] = 0,
        dim: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute Spread-Error Relationship.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        ensemble_dim : int or str, optional
            Ensemble dimension. Default is 0.
        dim : str or list of str, optional
            Dimension(s) along which to compute the mean spread and error.

        Returns
        -------
        mean_spread : xarray.DataArray
            Mean ensemble spread.
        mean_error : xarray.DataArray
            Mean absolute error of ensemble mean vs. obs.
        """
        return spatial_ensemble_metrics.spread_error(self._obj, obs, axis=ensemble_dim, dim=dim)

    def reliability_diagram(
        self,
        obs: xr.DataArray,
        threshold: float = 0.5,
        n_bins: int = 10,
        dim: Optional[Union[str, List[str]]] = None,
    ) -> xr.Dataset:
        """
        Compute Reliability Diagram.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        threshold : float, optional
            Event threshold.
        n_bins : int, optional
            Number of probability bins.
        dim : str or list of str, optional
            Dimension(s) along which to aggregate.

        Returns
        -------
        xarray.Dataset
            Reliability diagram components.
        """
        return spatial_ensemble_metrics.reliability_diagram(obs, self._obj, threshold=threshold, n_bins=n_bins, dim=dim)

    def verify(
        self,
        obs: xr.DataArray,
        dim: Optional[Union[str, List[str]]] = None,
        threshold: Optional[float] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> xr.Dataset:
        """
        Calculate a bundle of common evaluation metrics (Aero Protocol).

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        dim : str or list of str, optional
            Dimension(s) along which to compute the metrics.
        threshold : float, optional
            Threshold for categorical metrics (POD, FAR, HSS, etc.).
            If provided, categorical metrics are included in the bundle.
        weights : xarray.DataArray or numpy.ndarray, optional
            Weights for the weighted metrics.

        Returns
        -------
        xarray.Dataset
            Dataset containing common evaluation metrics.
        """
        metrics = {
            "MAE": error_metrics.MAE(obs, self._obj, axis=dim, weights=weights),
            "RMSE": error_metrics.RMSE(obs, self._obj, axis=dim, weights=weights),
            "MB": error_metrics.MB(obs, self._obj, axis=dim, weights=weights),
            "R": correlation_metrics.pearsonr(obs, self._obj, axis=dim),
            "IOA": error_metrics.IOA(obs, self._obj, axis=dim),
            "NMB": relative_metrics.NMB(obs, self._obj, axis=dim, weights=weights),
            "MNB": error_metrics.MNB(obs, self._obj, axis=dim),
            "MNE": error_metrics.MNE(obs, self._obj, axis=dim),
            "NSE": efficiency_metrics.NSE(obs, self._obj, axis=dim),
            "R2": correlation_metrics.R2(obs, self._obj, axis=dim),
            "FAC2": error_metrics.FAC2(obs, self._obj, axis=dim),
        }

        if threshold is not None:
            metrics.update(
                {
                    "POD": contingency_metrics.POD(obs, self._obj, minval=threshold, axis=dim),
                    "FAR": contingency_metrics.FAR(obs, self._obj, minval=threshold, axis=dim),
                    "HSS": contingency_metrics.HSS(obs, self._obj, minval=threshold, axis=dim),
                    "CSI": contingency_metrics.CSI(obs, self._obj, minval=threshold, axis=dim),
                    "TSS": contingency_metrics.TSS(obs, self._obj, minval=threshold, axis=dim),
                    "ETS": contingency_metrics.ETS(obs, self._obj, minval=threshold, axis=dim),
                    "FBI": contingency_metrics.FBI(obs, self._obj, minval=threshold, axis=dim),
                    "BSS_binary": contingency_metrics.BSS_binary(obs, self._obj, threshold=threshold, axis=dim),
                }
            )
        res = xr.Dataset(metrics)
        from .utils_stats import _update_history

        return _update_history(res, "Verification metrics bundle (verify)")


@xr.register_dataset_accessor("monet_stats")
class MonetDatasetAccessor:
    """
    Accessor for xarray.Dataset to provide MONET statistical methods.
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def stats(
        self,
        obs_name: str = "Obs",
        mod_name: str = "Mod",
        threshold: float = 0.0,
        minval: Optional[float] = None,
        maxval: Optional[float] = None,
        dim: Optional[Union[str, List[str]]] = None,
        plugins: Optional[List[str]] = None,
        weights: Optional[Union[np.ndarray, xr.DataArray]] = None,
    ) -> dict:
        """
        Calculate summary statistics for observations and model results.

        Parameters
        ----------
        obs_name : str, optional
            Name of observation variable. Default is 'Obs'.
        mod_name : str, optional
            Name of model variable. Default is 'Mod'.
        threshold : float, optional
            Threshold for contingency scores. Default is 0.0.
        minval : float, optional
            Minimum value for filtering.
        maxval : float, optional
            Maximum value for filtering.
        dim : str or list of str, optional
            Dimension(s) along which to compute the statistics.
        plugins : list of str, optional
            List of registered plugin names to include.

        Returns
        -------
        dict
            Dictionary of calculated statistics.
        """
        from . import stats

        return stats(
            self._obj,
            obs_name=obs_name,
            mod_name=mod_name,
            threshold=threshold,
            minval=minval,
            maxval=maxval,
            axis=dim,
            plugins=plugins,
            weights=weights,
        )

    def climatology(self, freq: str = "season", method: str = "mean", dim: str = "time") -> xr.Dataset:
        """
        Compute climatological statistics.

        Parameters
        ----------
        freq : str, optional
            Climatology frequency. Default is 'season'.
        method : str, optional
            Statistical method. Default is 'mean'.
        dim : str, optional
            Dimension along which to compute. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Climatological statistics.
        """
        return analysis.climatology(self._obj, freq=freq, method=method, dim=dim)

    def resample_data(self, freq: str = "MS", method: str = "mean", dim: str = "time", **kwargs: Any) -> xr.Dataset:
        """
        Resample data to a new temporal frequency.

        Parameters
        ----------
        freq : str, optional
            Resampling frequency. Default is 'MS'.
        method : str, optional
            Statistical method. Default is 'mean'.
        dim : str, optional
            Dimension along which to resample. Default is 'time'.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        xarray.Dataset
            Resampled data.
        """
        return analysis.resample_data(self._obj, freq=freq, method=method, dim=dim, **kwargs)

    def kz_filter(self, m: int, k: int, dim: str = "time") -> xr.Dataset:
        """
        Apply Kolmogorov-Zurbenko (KZ) filter.

        Parameters
        ----------
        m : int
            Window size.
        k : int
            Number of iterations.
        dim : str, optional
            Dimension along which to apply. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Filtered data.
        """
        return analysis.kz_filter(self._obj, m=m, k=k, dim=dim)

    def diurnal_cycle(self, method: str = "mean", dim: str = "time") -> xr.Dataset:
        """
        Compute the diurnal cycle.

        Parameters
        ----------
        method : str, optional
            Statistical method. Default is 'mean'.
        dim : str, optional
            Dimension along which to compute. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Diurnal cycle.
        """
        return analysis.diurnal_cycle(self._obj, method=method, dim=dim)

    def rolling_mean_8h(self, dim: str = "time", min_periods: int = 6, center: bool = True) -> xr.Dataset:
        """
        Compute rolling 8-hour mean.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.
        min_periods : int, optional
            Minimum number of observations. Default is 6.
        center : bool, optional
            If True, center the labels. Default is True.

        Returns
        -------
        xarray.Dataset
            Rolling 8-hour mean.
        """
        return analysis.rolling_mean_8h(self._obj, dim=dim, min_periods=min_periods, center=center)

    def rolling_mean_24h(self, dim: str = "time", min_periods: int = 18, center: bool = True) -> xr.Dataset:
        """
        Compute rolling 24-hour mean.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.
        min_periods : int, optional
            Minimum number of observations. Default is 18.
        center : bool, optional
            If True, center the labels. Default is True.

        Returns
        -------
        xarray.Dataset
            Rolling 24-hour mean.
        """
        return analysis.rolling_mean_24h(self._obj, dim=dim, min_periods=min_periods, center=center)

    def mda1(self, dim: str = "time") -> xr.Dataset:
        """
        Compute Maximum Daily 1-hour Average (MDA1).

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            MDA1 values.
        """
        return analysis.mda1(self._obj, dim=dim)

    def mda8(self, dim: str = "time", min_periods: int = 6, center: bool = False) -> xr.Dataset:
        """
        Compute Maximum Daily 8-hour Average (MDA8).

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.
        min_periods : int, optional
            Minimum number of observations. Default is 6.
        center : bool, optional
            Whether to center the window. Default is False.

        Returns
        -------
        xarray.Dataset
            MDA8 values.
        """
        return analysis.mda8(self._obj, dim=dim, min_periods=min_periods, center=center)

    def exceedance_count(self, threshold: float, dim: str = "time") -> xr.Dataset:
        """
        Count exceedances of a threshold.

        Parameters
        ----------
        threshold : float
            Threshold value.
        dim : str, optional
            Dimension along which to count. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Number of exceedances.
        """
        return analysis.exceedance_count(self._obj, threshold=threshold, dim=dim)

    def percentile(self, q: Union[float, List[float], np.ndarray], dim: str = "time", **kwargs: Any) -> xr.Dataset:
        """
        Compute percentiles.

        Parameters
        ----------
        q : float or list of float
            Percentile(s) (0-100).
        dim : str, optional
            Dimension over which to compute. Default is 'time'.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        xarray.Dataset
            Computed percentiles.
        """
        return analysis.percentile(self._obj, q=q, dim=dim, **kwargs)

    def peak_timing(self, dim: str = "time") -> xr.Dataset:
        """
        Identify the coordinate value of the maximum.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to find peak. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Coordinate values.
        """
        return analysis.peak_timing(self._obj, dim=dim)

    def weighted_spatial_mean(
        self,
        lat_dim: str = "lat",
        lon_dim: str = "lon",
        weights: Optional[Union[xr.DataArray, np.ndarray]] = None,
    ) -> xr.Dataset:
        """
        Compute area-weighted spatial mean.

        Parameters
        ----------
        lat_dim : str, optional
            Latitude dimension name. Default is 'lat'.
        lon_dim : str, optional
            Longitude dimension name. Default is 'lon'.
        weights : xarray.DataArray or numpy.ndarray, optional
            Custom weights.

        Returns
        -------
        xarray.Dataset
            Area-weighted spatial mean.
        """
        return analysis.weighted_spatial_mean(self._obj, lat_dim=lat_dim, lon_dim=lon_dim, weights=weights)

    def seasonal_mean(self, dim: str = "time", weighted: bool = True) -> xr.Dataset:
        """
        Compute seasonal mean (DJF, MAM, JJA, SON).

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.
        weighted : bool, optional
            Weight by days in month. Default is True.

        Returns
        -------
        xarray.Dataset
            Seasonal means.
        """
        return analysis.seasonal_mean(self._obj, dim=dim, weighted=weighted)

    def monthly_climatology(self, dim: str = "time", method: str = "mean") -> xr.Dataset:
        """
        Compute monthly climatology.

        Parameters
        ----------
        dim : str, optional
            Dimension along which to compute. Default is 'time'.
        method : str, optional
            Statistical method. Default is 'mean'.

        Returns
        -------
        xarray.Dataset
            Monthly climatology.
        """
        return analysis.monthly_climatology(self._obj, dim=dim, method=method)

    def anomalies(self, freq: str = "month", dim: str = "time") -> xr.Dataset:
        """
        Compute anomalies by subtracting the climatology.

        Parameters
        ----------
        freq : str, optional
            Climatology frequency ('season', 'month', 'dayofyear', 'hour').
            Default is 'month'.
        dim : str, optional
            Dimension along which to compute the anomalies. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Anomalies.
        """
        return analysis.anomalies(self._obj, freq=freq, dim=dim)

    def detrend(self, method: str = "linear", dim: str = "time") -> xr.Dataset:
        """
        Remove trend from data.

        Parameters
        ----------
        method : str, optional
            Detrending method ('linear', 'constant'). Default is 'linear'.
        dim : str, optional
            Dimension along which to detrend. Default is 'time'.

        Returns
        -------
        xarray.Dataset
            Detrended data.
        """
        return analysis.detrend(self._obj, method=method, dim=dim)

    def optimize(self, target_mb: float = 100.0) -> xr.Dataset:
        """
        Optimize performance by ensuring laziness and recommended chunks (Aero Protocol).

        Parameters
        ----------
        target_mb : float, optional
            Target size for each chunk in Megabytes. Default is 100.0.

        Returns
        -------
        xarray.Dataset
            Optimized Dataset.
        """
        from .utils_stats import _update_history

        if not performance._has_dask():
            return _update_history(self._obj, "Optimization skipped (Dask not installed)")

        # Ensure data is lazy
        res = performance.apply_lazy_threshold(self._obj, threshold_mb=0.1)
        # Always calculate and apply recommended chunks for the target size
        recommendation = performance.get_chunk_recommendation(res, target_mb=target_mb)
        res = res.chunk(recommendation)

        return _update_history(res, f"Optimized for performance (target={target_mb}MB)")

    def rechunk(self, chunks: Optional[dict] = None) -> xr.Dataset:
        """
        Apply new chunks to the Dataset (Aero Protocol provenance tracking).

        Parameters
        ----------
        chunks : dict, optional
            New chunk sizes. If None, uses optimal recommendations (~100MB).

        Returns
        -------
        xarray.Dataset
            Rechunked Dataset.
        """
        from .utils_stats import _update_history

        if not performance._has_dask():
            return _update_history(self._obj, "Rechunking skipped (Dask not installed)")

        if chunks is None:
            chunks = performance.get_chunk_recommendation(self._obj)

        res = self._obj.chunk(chunks)

        return _update_history(res, f"Rechunked with {chunks}")
