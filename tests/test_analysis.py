import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats.analysis import (
    climatology,
    diurnal_cycle,
    exceedance_count,
    fft_analysis,
    kz_filter,
    mda8,
    peak_timing,
    percentile,
    power_spectrum,
    resample_data,
    rolling_mean_8h,
    rolling_mean_24h,
    weighted_spatial_mean,
)

HAS_DASK = True
try:
    import dask.array  # noqa: F401
except ImportError:
    HAS_DASK = False


@pytest.mark.parametrize("lazy", [False, True])
def test_resample_data_xarray(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=60, freq="D")
    data = np.arange(60).astype(float)
    da = xr.DataArray(data, coords={"time": times}, dims="time")

    if lazy:
        da = da.chunk({"time": 10})

    res = resample_data(da, freq="MS", method="mean")
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()

    assert len(res) == 2
    assert np.isclose(res.values[0], np.mean(data[:31]))


def test_resample_data_pandas():
    times = pd.date_range("2020-01-01", periods=60, freq="D")
    df = pd.DataFrame({"Mod": np.arange(60).astype(float)}, index=times)

    res = resample_data(df, freq="MS", method="mean")
    assert len(res) == 2
    assert np.isclose(res["Mod"].iloc[0], np.mean(np.arange(31)))


@pytest.mark.parametrize("lazy", [False, True])
def test_climatology(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=366, freq="D")
    da = xr.DataArray(np.random.rand(366), coords={"time": times}, dims="time")

    if lazy:
        da = da.chunk({"time": 100})

    # Monthly climatology
    res = climatology(da, freq="month", method="mean")
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()

    assert "month" in res.coords
    assert len(res.month) == 12

    # Check January mean
    jan_mean = (
        da.where(da.time.dt.month == 1, drop=True).mean().compute()
        if lazy
        else da.where(da.time.dt.month == 1, drop=True).mean()
    )
    assert np.isclose(res.sel(month=1), jan_mean)

    # Seasonal climatology
    res_seasonal = climatology(da, freq="season", method="mean")
    if lazy:
        res_seasonal = res_seasonal.compute()
    assert "season" in res_seasonal.coords
    assert len(res_seasonal.season) == 4


@pytest.mark.parametrize("lazy", [False, True])
def test_kz_filter(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    data = np.arange(10).astype(float)
    m = 3
    k = 1
    expected = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan])

    da = xr.DataArray(data, dims="x")
    if lazy:
        da = da.chunk({"x": 5})

    res_xr = kz_filter(da, m=m, k=k, dim="x")
    if lazy:
        assert hasattr(res_xr.data, "chunks")
        res_xr = res_xr.compute()

    np.testing.assert_allclose(res_xr.values, expected)

    # Also test numpy path (which wraps in xarray internally now)
    res_np = kz_filter(data, m=m, k=k)
    np.testing.assert_allclose(res_np, expected)


@pytest.mark.parametrize("lazy", [False, True])
def test_diurnal_cycle(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=24 * 10, freq="h")
    data = np.sin(2 * np.pi * np.arange(240) / 24)
    da = xr.DataArray(data, coords={"time": times}, dims="time")
    if lazy:
        da = da.chunk({"time": 24})

    res = diurnal_cycle(da, method="mean")
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()

    assert len(res) == 24
    assert np.allclose(res.values, data[:24], atol=1e-10)


@pytest.mark.parametrize("lazy", [False, True])
def test_rolling_means(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=48, freq="h")
    data = np.ones(48)
    da = xr.DataArray(data, coords={"time": times}, dims="time")
    if lazy:
        da = da.chunk({"time": 10})

    res8 = rolling_mean_8h(da)
    if lazy:
        assert hasattr(res8.data, "chunks")
        res8 = res8.compute()
    assert np.allclose(res8.dropna("time"), 1.0)

    res24 = rolling_mean_24h(da)
    if lazy:
        res24 = res24.compute()
    assert np.allclose(res24.dropna("time"), 1.0)


@pytest.mark.parametrize("lazy", [False, True])
def test_mda8(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=24 * 3, freq="h")
    da = xr.DataArray(np.random.rand(24 * 3), coords={"time": times}, dims="time")
    if lazy:
        da = da.chunk({"time": 24})

    res = mda8(da)
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()
    assert len(res) == 3


@pytest.mark.parametrize("lazy", [False, True])
def test_exceedance_count(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    da = xr.DataArray([1, 5, 2, 6, 3], dims="time")
    if lazy:
        da = da.chunk({"time": 2})

    res = exceedance_count(da, threshold=4)
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()
    assert int(res) == 2


@pytest.mark.parametrize("lazy", [False, True])
def test_percentile(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    da = xr.DataArray(np.arange(101), dims="time")
    if lazy:
        da = da.chunk({"time": 50})

    res = percentile(da, q=95)
    if lazy:
        # quantile usually triggers some computation or has specific dask behavior
        res = res.compute()
    assert float(res) == 95.0


@pytest.mark.parametrize("lazy", [False, True])
def test_peak_timing(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=24, freq="h")
    data = np.zeros(24)
    data[14] = 10.0
    da = xr.DataArray(data, coords={"time": times}, dims="time")
    if lazy:
        da = da.chunk({"time": 12})

    res = peak_timing(da, dim="time")
    if lazy:
        res = res.compute()
    assert res.values == times[14]


@pytest.mark.parametrize("lazy", [False, True])
def test_weighted_spatial_mean(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    lats = np.array([-60, 0, 60])
    lons = np.array([0, 120, 240])
    data = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])
    da = xr.DataArray(data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))
    if lazy:
        da = da.chunk({"lat": 1})

    res = weighted_spatial_mean(da)
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()
    assert np.isclose(float(res), 1.5)


@pytest.mark.parametrize("lazy", [False, True])
def test_fft_analysis(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    t = np.linspace(0, 1, 100)
    # 10 Hz signal
    data = np.sin(2 * np.pi * 10 * t)
    da = xr.DataArray(data, coords={"time": t}, dims="time")

    if lazy:
        da = da.chunk({"time": 50})

    # Test PSD
    psd = fft_analysis(da, dim="time", output="psd")
    if lazy:
        assert hasattr(psd.data, "chunks")
        psd = psd.compute()

    assert psd.dims == ("time",)
    assert psd.size == 100
    # Peak should be at frequency index 10 (for 1 second duration)
    assert np.argmax(psd.values) == 10

    # Test Magnitude
    mag = fft_analysis(da, dim="time", output="magnitude")
    if lazy:
        mag = mag.compute()
    assert np.allclose(mag, np.sqrt(psd))

    # Test Complex
    comp = fft_analysis(da, dim="time", output="complex")
    if lazy:
        comp = comp.compute()
    assert np.iscomplexobj(comp.values)


@pytest.mark.parametrize("lazy", [False, True])
def test_power_spectrum(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    t = np.linspace(0, 10, 1000)
    # 5 Hz signal
    data = np.sin(2 * np.pi * 5 * t)
    da = xr.DataArray(data, coords={"time": t}, dims="time")

    if lazy:
        da = da.chunk({"time": 250})

    # Sampling frequency is 100 Hz (1000 points over 10 seconds)
    fs = 100.0
    nperseg = 200
    psd = power_spectrum(da, dim="time", fs=fs, nperseg=nperseg)

    if lazy:
        assert hasattr(psd.data, "chunks")
        psd = psd.compute()

    assert psd.dims == ("frequency",)
    assert psd.size == nperseg // 2 + 1
    # Peak should be at 5 Hz
    peak_freq = psd.frequency[np.argmax(psd.values)]
    assert np.isclose(peak_freq, 5.0)
