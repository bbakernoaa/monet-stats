import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    import dask.array as da  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

import monet_stats  # noqa: F401


@pytest.fixture
def sample_da():
    times = pd.date_range("2020-01-01", periods=24, freq="h")
    lats = np.arange(10)
    lons = np.arange(10)
    data = np.random.rand(24, 10, 10)
    return xr.DataArray(
        data,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=("time", "lat", "lon"),
        name="test_data",
    )


@pytest.fixture
def sample_ds(sample_da):
    return xr.Dataset({"Obs": sample_da, "Mod": sample_da + 0.1})


def test_dataarray_accessor_climatology(sample_da):
    # Test climatology via accessor
    res = sample_da.monet.climatology(freq="hour")
    assert res.sizes["hour"] == 24
    assert "history" in res.attrs
    assert "Climatology" in res.attrs["history"]


def test_dataarray_accessor_mda8(sample_da):
    # Test MDA8 via accessor
    res = sample_da.monet.mda8()
    assert res.sizes["time"] == 1  # 24 hours = 1 day
    assert "MDA8" in res.attrs["history"]


def test_dataarray_accessor_plot(sample_da):
    # Test plot_spatial via accessor (just verify it calls and returns something)
    # We use a small subset to speed up
    subset = sample_da.isel(time=0)
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import matplotlib.pyplot as plt

        ax = subset.monet.plot_spatial(method="matplotlib")
        assert isinstance(ax, plt.Axes)
        plt.close()
    except ImportError:
        pytest.skip("matplotlib/cartopy not installed")


def test_dataset_accessor_stats(sample_ds):
    # Test stats via accessor
    res = sample_ds.monet.stats()
    assert isinstance(res, dict)
    assert "MB" in res
    assert np.isclose(res["MB"], 0.1)


def test_dataset_accessor_weighted_spatial_mean(sample_ds):
    # Test weighted_spatial_mean via accessor
    res = sample_ds.monet.weighted_spatial_mean()
    assert "time" in res.dims
    assert "lat" not in res.dims
    assert "lon" not in res.dims
    assert "Weighted spatial mean" in res.attrs["history"]


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_accessor_laziness(sample_da):
    # Ensure dask laziness is maintained
    lazy_da = sample_da.chunk({"time": 12})
    res = lazy_da.monet.climatology(freq="hour")
    assert hasattr(res.data, "chunks")
    # Result should still be lazy
    assert res.data.chunks is not None
