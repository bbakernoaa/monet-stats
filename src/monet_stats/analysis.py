"""
Advanced analysis methods for weather and air quality (Aero Protocol Compliant).
"""

import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve1d

from .utils_stats import _update_history


def resample_data(
    data: Union[xr.DataArray, xr.Dataset, pd.Series, pd.DataFrame],
    freq: str = "MS",
    method: str = "mean",
    dim: str = "time",
    **kwargs: Any,
) -> Union[xr.DataArray, xr.Dataset, pd.Series, pd.DataFrame]:
    """
    Resample data to a new temporal frequency (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, pandas.Series, or pandas.DataFrame
        Input data with a time-like index or coordinate.
    freq : str, optional
        Resampling frequency (e.g., 'MS' for monthly start, 'W' for weekly, 'D' for daily).
        Default is 'MS'.
    method : str, optional
        Statistical method to apply ('mean', 'sum', 'min', 'max', 'std', 'median').
        Default is 'mean'.
    dim : str, optional
        Dimension along which to resample (xarray only). Default is 'time'.
    **kwargs : Any
        Additional keyword arguments passed to the resample method.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset, pd.Series, pd.DataFrame]
        Resampled data.

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> times = pd.date_range("2020-01-01", periods=100, freq="D")
    >>> da = xr.DataArray(np.random.rand(100), coords={"time": times}, dims="time")
    >>> monthly_mean = resample_data(da, freq="MS", method="mean")
    """
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        resampled = data.resample({dim: freq}, **kwargs)
        result = getattr(resampled, method)()
        return _update_history(result, f"Resampled to {freq} using {method}")
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        result = data.resample(freq, **kwargs).agg(method)
        return result
    else:
        raise TypeError("data must be an xarray or pandas object with a time index")


def climatology(
    data: Union[xr.DataArray, xr.Dataset],
    freq: str = "season",
    method: str = "mean",
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute climatological statistics (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data with a time-like coordinate.
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
    Union[xr.DataArray, xr.Dataset]
        Climatological statistics.

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> times = pd.date_range("2020-01-01", periods=365*2, freq="D")
    >>> da = xr.DataArray(np.random.rand(730), coords={"time": times}, dims="time")
    >>> seasonal_climo = climatology(da, freq="season", method="mean")
    """
    group = f"{dim}.{freq}"
    # Use native xarray groupby methods for better performance and Dask integration
    grouped = data.groupby(group)
    if hasattr(grouped, method):
        result = getattr(grouped, method)(dim=dim)
    else:
        # Fallback for methods not directly implemented on GroupBy
        result = grouped.reduce(getattr(np, f"nan{method}" if "nan" not in method else method), dim=dim)

    return _update_history(result, f"Climatology ({freq}) using {method}")


def kz_filter(
    data: Union[xr.DataArray, xr.Dataset, np.ndarray],
    m: int,
    k: int,
    dim: str = "time",
    axis: int = -1,
) -> Union[xr.DataArray, xr.Dataset, np.ndarray]:
    """
    Kolmogorov-Zurbenko (KZ) filter (Aero Protocol).

    The KZ filter is a low-pass filter implemented as k iterations of a moving
    average of window size m.

    Parameters
    ----------
    data : xarray.DataArray, xarray.Dataset, or numpy.ndarray
        Input data.
    m : int
        Window size for the moving average (must be an odd integer for symmetry).
    k : int
        Number of iterations.
    dim : str, optional
        Dimension along which to apply the filter (xarray only). Default is 'time'.
    axis : int, optional
        Axis along which to apply the filter (numpy only). Default is -1.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset, np.ndarray]
        Filtered data.

    Notes
    -----
    The KZ filter is widely used in air quality analysis to separate different
    time scales in a time series (e.g., seasonal, long-term, and short-term).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(100)
    >>> filtered = kz_filter(x, m=5, k=3)
    """
    if m % 2 == 0:
        warnings.warn("KZ filter window size m should ideally be odd for symmetry.", stacklevel=2)

    # A KZ filter (m, k) is equivalent to a convolution with a kernel
    # that is the k-fold convolution of a boxcar of width m.
    boxcar = np.ones(m) / m
    kernel = boxcar
    for _ in range(k - 1):
        kernel = np.convolve(kernel, boxcar)

    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        # Fast path for NumPy using single convolution
        res_np = np.asanyarray(data, dtype=float)
        # Use scipy.ndimage.convolve1d for multi-dimensional support and speed.
        # mode='constant', cval=np.nan ensures that edges where the kernel
        # overlaps with the boundary become NaN, matching Xarray's rolling behavior.
        return convolve1d(res_np, kernel, axis=axis, mode="constant", cval=np.nan)

    if isinstance(data, xr.Dataset):
        # Apply to each variable in the dataset
        res = data.map(kz_filter, m=m, k=k, dim=dim)
        return _update_history(res, f"KZ filter (m={m}, k={k})")

    # Xarray DataArray path (handles Dask natively via apply_ufunc)
    def _kz_wrapper(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return convolve1d(x, kernel, axis=-1, mode="constant", cval=np.nan)

    # Ensure dimension is in a single chunk for Dask-backed arrays
    if hasattr(data.data, "chunks"):
        data = data.chunk({dim: -1})

    res = xr.apply_ufunc(
        _kz_wrapper,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={"kernel": kernel},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    return _update_history(res, f"KZ filter (m={m}, k={k})")


def diurnal_cycle(
    data: Union[xr.DataArray, xr.Dataset],
    method: str = "mean",
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute the diurnal cycle (average hourly profile) (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data with a time-like coordinate.
    method : str, optional
        Statistical method to apply ('mean', 'median', 'std'). Default is 'mean'.
    dim : str, optional
        Dimension along which to compute the cycle. Default is 'time'.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Diurnal cycle (24 values, one for each hour).

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> times = pd.date_range("2020-01-01", periods=24*10, freq="h")
    >>> da = xr.DataArray(np.random.rand(240), coords={"time": times}, dims="time")
    >>> cycle = diurnal_cycle(da, method="mean")
    """
    return climatology(data, freq="hour", method=method, dim=dim)


def rolling_mean_8h(
    data: Union[xr.DataArray, xr.Dataset],
    dim: str = "time",
    min_periods: int = 6,
    center: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute rolling 8-hour mean (commonly for Ozone) (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data. Must have hourly frequency.
    dim : str, optional
        Dimension along which to compute the mean. Default is 'time'.
    min_periods : int, optional
        Minimum number of observations in window required to have a value.
        Default is 6 (75% of 8 hours).
    center : bool, optional
        If True, set the labels at the center of the window. Default is True.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Rolling 8-hour mean.
    """
    res = data.rolling({dim: 8}, min_periods=min_periods, center=center).mean()
    return _update_history(res, "Rolling 8-hour mean")


def rolling_mean_24h(
    data: Union[xr.DataArray, xr.Dataset],
    dim: str = "time",
    min_periods: int = 18,
    center: bool = True,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute rolling 24-hour mean (commonly for PM2.5) (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data. Must have hourly frequency.
    dim : str, optional
        Dimension along which to compute the mean. Default is 'time'.
    min_periods : int, optional
        Minimum number of observations in window required to have a value.
        Default is 18 (75% of 24 hours).
    center : bool, optional
        If True, set the labels at the center of the window. Default is True.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Rolling 24-hour mean.
    """
    res = data.rolling({dim: 24}, min_periods=min_periods, center=center).mean()
    return _update_history(res, "Rolling 24-hour mean")


def mda8(
    data: Union[xr.DataArray, xr.Dataset],
    dim: str = "time",
    min_periods: int = 6,
    center: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute Maximum Daily 8-hour Average (MDA8) (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data. Must have hourly frequency.
    dim : str, optional
        Dimension along which to compute. Default is 'time'.
    min_periods : int, optional
        Minimum number of observations for the 8-hour rolling mean. Default is 6.
    center : bool, optional
        Whether to center the 8-hour rolling window. Regulatory MDA8 (e.g., EPA)
        typically uses a non-centered (trailing) window. Default is False.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        MDA8 values (one per day).

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> import numpy as np
    >>> times = pd.date_range("2020-01-01", periods=24*5, freq="h")
    >>> da = xr.DataArray(np.random.rand(120), coords={"time": times}, dims="time")
    >>> ozone_mda8 = mda8(da)
    """
    rolling_8h = rolling_mean_8h(data, dim=dim, min_periods=min_periods, center=center)
    # Group by day and take maximum
    res = rolling_8h.resample({dim: "D"}).max()
    return _update_history(res, "MDA8 (Maximum Daily 8-hour Average)")


def exceedance_count(
    data: Union[xr.DataArray, xr.Dataset],
    threshold: float,
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Count exceedances of a threshold (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data.
    threshold : float
        Value above which an exceedance is counted.
    dim : str, optional
        Dimension along which to count exceedances. Default is 'time'.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Number of exceedances.

    Examples
    --------
    >>> import xarray as xr
    >>> da = xr.DataArray([1, 5, 2, 6, 3])
    >>> exceedance_count(da, threshold=4)
    <xarray.DataArray ()>
    array(2)
    """
    res = (data > threshold).sum(dim=dim)
    return _update_history(res, f"Exceedance count (threshold={threshold})")


def percentile(
    data: Union[xr.DataArray, xr.Dataset],
    q: Union[float, list, np.ndarray],
    dim: str = "time",
    **kwargs: Any,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute percentiles (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data.
    q : float or list of float
        Percentile(s) to compute (0-100).
    dim : str, optional
        Dimension(s) over which to compute percentiles. Default is 'time'.
    **kwargs : Any
        Additional keyword arguments passed to xarray.quantile.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Computed percentiles.
    """
    # Ensure dimension is in a single chunk for Dask-backed arrays
    if hasattr(data.data, "chunks"):
        data = data.chunk({dim: -1})

    # xarray uses 0-1 for quantile, so divide by 100
    res = data.quantile(np.asanyarray(q) / 100.0, dim=dim, **kwargs)
    return _update_history(res, f"Percentile (q={q})")


def peak_timing(
    data: Union[xr.DataArray, xr.Dataset],
    dim: str = "time",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Identify the coordinate value of the maximum (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data.
    dim : str, optional
        Dimension along which to find the peak. Default is 'time'.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Coordinate values where the maximum occurs.

    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> times = pd.date_range("2020-01-01", periods=24, freq="h")
    >>> da = xr.DataArray(np.random.rand(24), coords={"time": times}, dims="time")
    >>> peak_hour = peak_timing(da, dim="time")
    """
    # Ensure dimension is in a single chunk for Dask-backed arrays
    if hasattr(data.data, "chunks"):
        data = data.chunk({dim: -1})

    # idxmax returns the coordinate of the maximum
    res = data.idxmax(dim=dim)
    return _update_history(res, f"Peak timing along {dim}")


def weighted_spatial_mean(
    data: Union[xr.DataArray, xr.Dataset],
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Compute area-weighted spatial mean for lat/lon grids (Aero Protocol).

    Assumes a regular lat/lon grid where weights are proportional to cos(lat).

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data with latitude and longitude coordinates.
    lat_dim : str, optional
        Name of the latitude dimension. Default is 'lat'.
    lon_dim : str, optional
        Name of the longitude longitude. Default is 'lon'.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        Area-weighted spatial mean.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> lats = np.arange(-90, 91, 1)
    >>> lons = np.arange(-180, 181, 1)
    >>> da = xr.DataArray(np.ones((len(lats), len(lons))),
    ...                   coords={"lat": lats, "lon": lons},
    ...                   dims=("lat", "lon"))
    >>> spatial_mean = weighted_spatial_mean(da)
    """
    weights = np.cos(np.deg2rad(data[lat_dim]))
    weights.name = "weights"
    weighted_data = data.weighted(weights)
    res = weighted_data.mean(dim=(lat_dim, lon_dim))
    return _update_history(res, "Weighted spatial mean")


def fft_analysis(
    data: xr.DataArray,
    dim: str = "time",
    output: str = "psd",
) -> xr.DataArray:
    """
    Perform Fast Fourier Transform (FFT) analysis (Aero Protocol).

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    dim : str, optional
        Dimension along which to perform FFT. Default is 'time'.
    output : str, optional
        Type of output to return:
        - 'psd': Power Spectral Density (magnitude squared of FFT).
        - 'magnitude': Magnitude of FFT.
        - 'complex': Complex FFT results.
        Default is 'psd'.

    Returns
    -------
    xarray.DataArray
        FFT results. The coordinate for 'dim' is replaced by frequency indices.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 100)
    >>> signal = np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz signal
    >>> da = xr.DataArray(signal, coords={"time": t}, dims="time")
    >>> psd = fft_analysis(da, dim="time", output="psd")
    """

    def _fft_wrapper(x: np.ndarray) -> np.ndarray:
        return np.fft.fft(x, axis=-1)

    # Core dimensions for apply_ufunc must be a single chunk if using dask
    if hasattr(data.data, "chunks"):
        data = data.chunk({dim: -1})

    res_complex = xr.apply_ufunc(
        _fft_wrapper,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[np.complex128],
    )

    if output == "complex":
        res = res_complex
    elif output == "magnitude":
        res = np.abs(res_complex)
    elif output == "psd":
        res = np.abs(res_complex) ** 2
    else:
        raise ValueError(f"Unknown output type: {output}")

    # Update coordinate to frequency index
    n = data.sizes[dim]
    res = res.assign_coords({dim: np.arange(n)})

    return _update_history(res, f"FFT analysis (output={output})")


def power_spectrum(
    data: xr.DataArray,
    dim: str = "time",
    fs: float = 1.0,
    window: str = "hann",
    nperseg: Optional[int] = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Compute power spectrum using Welch's method (Aero Protocol).

    Welch's method computes an estimate of the power spectral density by
    dividing the data into overlapping segments, computing a periodogram for
    each segment and averaging the results.

    Parameters
    ----------
    data : xarray.DataArray
        Input data.
    dim : str, optional
        Dimension along which to compute the spectrum. Default is 'time'.
    fs : float, optional
        Sampling frequency. Default is 1.0.
    window : str, optional
        Desired window to use. Default is 'hann'.
    nperseg : int, optional
        Length of each segment. Default is None (256).
    **kwargs : Any
        Additional keyword arguments passed to scipy.signal.welch.

    Returns
    -------
    xarray.DataArray
        Power spectral density. The 'dim' dimension is replaced by 'frequency'.

    Examples
    --------
    >>> import xarray as xr
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 1000)
    >>> signal = np.sin(2 * np.pi * 0.1 * t) + np.random.randn(1000) * 2
    >>> da = xr.DataArray(signal, coords={"time": t}, dims="time")
    >>> psd = power_spectrum(da, dim="time", fs=10.0)
    """
    from scipy.signal import welch

    # Core dimensions for apply_ufunc must be a single chunk if using dask
    if hasattr(data.data, "chunks"):
        data = data.chunk({dim: -1})

    def _welch_wrapper(x: np.ndarray, fs: float, window: str, nperseg: int, **kwargs: Any) -> np.ndarray:
        f, psd = welch(x, fs=fs, window=window, nperseg=nperseg, axis=-1, **kwargs)
        return psd

    # Get the number of frequency bins to set output_core_dims size
    # For real FFT, it's nperseg // 2 + 1
    if nperseg is None:
        nperseg = min(data.sizes[dim], 256)

    n_freq = nperseg // 2 + 1

    res = xr.apply_ufunc(
        _welch_wrapper,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[["frequency"]],
        kwargs={"fs": fs, "window": window, "nperseg": nperseg, **kwargs},
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"output_sizes": {"frequency": n_freq}},
    )

    # Assign frequency coordinates
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    res = res.assign_coords(frequency=freqs)

    return _update_history(res, "Power spectrum (Welch method)")
