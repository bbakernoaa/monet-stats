import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats.analysis import (
    mda1,
    monthly_climatology,
    seasonal_mean,
    weighted_spatial_mean,
)

HAS_DASK = True
try:
    import dask.array  # noqa: F401
except ImportError:
    HAS_DASK = False


@pytest.mark.parametrize("lazy", [False, True])
def test_weighted_spatial_mean_pro(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    lats = np.array([-60, 0, 60])
    lons = np.array([0, 120, 240])
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    da = xr.DataArray(data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"))

    if lazy:
        da = da.chunk({"lat": 1})

    # 1. Test default (cosine lat)
    # weights: cos(-60)=0.5, cos(0)=1, cos(60)=0.5
    # sum(weights) = 3*0.5 + 3*1 + 3*0.5 = 6
    # weighted_sum = 3*(1*0.5) + 3*(2*1) + 3*(3*0.5) = 1.5 + 6 + 4.5 = 12
    # mean = 12 / 6 = 2.0
    res_default = weighted_spatial_mean(da)
    if lazy:
        res_default = res_default.compute()
    assert np.isclose(float(res_default), 2.0)

    # 2. Test with cell_area coordinate
    cell_area = xr.DataArray(
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), coords={"lat": lats, "lon": lons}, dims=("lat", "lon")
    )
    da_with_area = da.assign_coords(cell_area=cell_area)
    # mean should be (1+2+3)/3 = 2.0 (since all areas are 1)
    res_area = weighted_spatial_mean(da_with_area)
    if lazy:
        res_area = res_area.compute()
    assert np.isclose(float(res_area), 2.0)

    # 3. Test with custom weights array
    custom_weights = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    res_custom = weighted_spatial_mean(da, weights=custom_weights)
    if lazy:
        res_custom = res_custom.compute()
    # only middle row counts: mean = 2.0
    assert np.isclose(float(res_custom), 2.0)


@pytest.mark.parametrize("lazy", [False, True])
def test_seasonal_mean_pro(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    # Create 1 year of daily data (2020 is a leap year)
    times = pd.date_range("2020-01-01", periods=366, freq="D")
    # Set all values to 1.0
    da = xr.DataArray(np.ones(366), coords={"time": times}, dims="time")

    if lazy:
        da = da.chunk({"time": 100})

    # Weighted seasonal mean of all ones should be one
    res = seasonal_mean(da, weighted=True)
    if lazy:
        assert hasattr(res.data, "chunks")
        res = res.compute()

    assert "season" in res.coords
    assert len(res.season) == 4
    np.testing.assert_allclose(res.values, 1.0)

    # Test with varying values to verify weighting
    # Let's make February (29 days in 2020) have value 2.0, others 1.0
    da_var = da.copy()
    da_var.loc[da_var.time.dt.month == 2] = 2.0

    # DJF: Dec(1.0), Jan(1.0), Feb(2.0)
    # Days: Dec=31, Jan=31, Feb=29
    # Weighted Mean = (31*1.0 + 31*1.0 + 29*2.0) / (31 + 31 + 29)
    # = (31 + 31 + 58) / 91 = 120 / 91 approx 1.31868

    res_weighted = seasonal_mean(da_var, weighted=True)
    if lazy:
        res_weighted = res_weighted.compute()

    expected_djf = (31 + 31 + 58) / 91
    assert np.isclose(res_weighted.sel(season="DJF"), expected_djf)

    # Unweighted Mean for comparison
    # Groupby('season').mean() treats all days equally (same as weighted=True for daily data)
    # Wait, climatology(method='mean') uses groupby(season).mean()
    # In xarray, .mean() on a group of daily values IS a weighted mean of months if you think about it,
    # because it's sum(values)/sum(days).
    # BUT if the data was already monthly, it wouldn't be.

    # Let's test with monthly data
    da_monthly = da_var.resample(time="MS").mean()
    # DJF monthly means: Dec=1.0, Jan=1.0, Feb=2.0
    # Unweighted seasonal mean = (1.0 + 1.0 + 2.0) / 3 = 1.333
    # Weighted seasonal mean = (31*1.0 + 31*1.0 + 29*2.0) / (31+31+29) = 120/91 approx 1.318

    res_m_unweighted = seasonal_mean(da_monthly, weighted=False)
    assert np.isclose(res_m_unweighted.sel(season="DJF"), 4 / 3)

    res_m_weighted = seasonal_mean(da_monthly, weighted=True)
    assert np.isclose(res_m_weighted.sel(season="DJF"), 120 / 91)


@pytest.mark.parametrize("lazy", [False, True])
def test_mda1_pro(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    times = pd.date_range("2020-01-01", periods=24 * 2, freq="h")
    data = np.zeros(48)
    data[10] = 5.0  # Day 1
    data[30] = 10.0  # Day 2
    da = xr.DataArray(data, coords={"time": times}, dims="time")

    if lazy:
        da = da.chunk({"time": 24})

    res = mda1(da)
    if lazy:
        res = res.compute()

    assert len(res) == 2
    assert res.values[0] == 5.0
    assert res.values[1] == 10.0


@pytest.mark.parametrize("lazy", [False, True])
def test_monthly_climatology_pro(lazy):
    if lazy and not HAS_DASK:
        pytest.skip("Dask not available")

    # 2 years of daily data
    times = pd.date_range("2020-01-01", periods=366 + 365, freq="D")
    data = np.random.rand(len(times))
    da = xr.DataArray(data, coords={"time": times}, dims="time")

    if lazy:
        da = da.chunk({"time": 100})

    res = monthly_climatology(da)
    if lazy:
        res = res.compute()

    assert "month" in res.coords
    assert len(res.month) == 12
    # Verify January mean
    jan_mean = da.where(da.time.dt.month == 1, drop=True).mean().values
    assert np.isclose(res.sel(month=1), jan_mean)
