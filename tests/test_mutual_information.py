import numpy as np
import pytest
import xarray as xr

from monet_stats.distribution_metrics import MutualInformation


def test_mutual_information_numpy():
    # Independent variables should have MI close to 0
    np.random.seed(42)
    obs = np.random.normal(0, 1, 1000)
    mod = np.random.normal(0, 1, 1000)
    mi_indep = MutualInformation(obs, mod, bins=10)
    assert mi_indep < 0.1

    # Perfectly correlated variables should have high MI
    mi_perf = MutualInformation(obs, obs, bins=10)
    assert mi_perf > 1.0
    assert mi_perf > mi_indep


def test_mutual_information_numpy_multidim():
    # Test multi-dimensional numpy with axis
    np.random.seed(42)
    obs = np.random.normal(0, 1, (10, 100))
    mod = obs + np.random.normal(0, 0.1, (10, 100))

    # Reduce along last axis
    mi = MutualInformation(obs, mod, axis=1, bins=10)
    assert mi.shape == (10,)
    assert (mi > 0.5).all()

    # Check consistency with 1D
    mi_single = MutualInformation(obs[0], mod[0], bins=10)
    assert np.allclose(mi[0], mi_single)


def test_mutual_information_xarray():
    # Test with xarray DataArrays
    obs = xr.DataArray(np.random.normal(0, 1, (10, 10)), dims=("lat", "lon"))
    mod = xr.DataArray(np.random.normal(0, 1, (10, 10)), dims=("lat", "lon"))

    # Global MI
    mi_global = MutualInformation(obs, mod, bins=5)
    assert isinstance(mi_global, xr.DataArray)
    assert mi_global.ndim == 0

    # MI along a dimension
    mi_lat = MutualInformation(obs, mod, dim="lon", bins=5)
    assert isinstance(mi_lat, xr.DataArray)
    assert mi_lat.dims == ("lat",)
    assert len(mi_lat) == 10


def test_mutual_information_dask():
    # Test laziness and correctness with Dask
    pytest.importorskip("dask")

    obs = xr.DataArray(np.random.normal(0, 1, (20, 20)), dims=("x", "y")).chunk({"x": 10})
    mod = xr.DataArray(np.random.normal(0, 1, (20, 20)), dims=("x", "y")).chunk({"x": 10})

    mi = MutualInformation(obs, mod, dim="y", bins=5)

    # Check it's still lazy
    assert hasattr(mi.data, "dask")

    # Compute and check
    res = mi.compute()
    assert res.dims == ("x",)
    assert not np.isnan(res).all()


def test_mutual_information_provenance():
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="t")
    mod = xr.DataArray([1.1, 1.9, 3.2], dims="t")

    mi = MutualInformation(obs, mod)
    assert "Mutual Information" in mi.attrs.get("history", "")


def test_mutual_information_nan_handling():
    obs = np.array([1, 2, np.nan, 4])
    mod = np.array([1.1, np.nan, 2.9, 4.1])

    # Should handle NaNs by ignoring them in joint histogram
    mi = MutualInformation(obs, mod, bins=5)
    assert not np.isnan(mi)

    # All NaN should return NaN
    mi_all_nan = MutualInformation(np.array([np.nan]), np.array([np.nan]))
    assert np.isnan(mi_all_nan)


def test_accessor_mutual_information():
    obs = xr.DataArray(np.random.rand(100), dims="time")
    mod = xr.DataArray(np.random.rand(100), dims="time")

    # Test via accessor
    mi = mod.monet_stats.mutual_information(obs, bins=5)
    assert isinstance(mi, xr.DataArray)
    assert mi.ndim == 0
