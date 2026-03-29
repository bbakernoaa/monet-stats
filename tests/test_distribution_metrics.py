import numpy as np
import pytest
import xarray as xr

from monet_stats.distribution_metrics import KLDivergence, WassersteinDistance


def test_wasserstein_xarray_lazy():
    pytest.importorskip("dask.array")
    import dask.array as da

    obs = xr.DataArray(da.from_array(np.array([1, 2, 3]), chunks=3), dims="x")
    mod = xr.DataArray(da.from_array(np.array([2, 3, 4]), chunks=3), dims="x")
    res = WassersteinDistance(obs, mod)
    assert hasattr(res.data, "chunks")
    assert np.isclose(res.compute(), 1.0)


def test_kl_xarray_lazy():
    pytest.importorskip("dask.array")
    import dask.array as da

    obs_data = np.random.normal(0, 1, 1000)
    mod_data = np.random.normal(0.5, 1, 1000)
    obs = xr.DataArray(da.from_array(obs_data, chunks=500), dims="x")
    mod = xr.DataArray(da.from_array(mod_data, chunks=500), dims="x")
    res = KLDivergence(obs, mod, bins=10)
    assert hasattr(res.data, "chunks")
    val = res.compute()
    assert val > 0


def test_wasserstein_numpy():
    obs = np.array([1, 2, 3])
    mod = np.array([2, 3, 4])
    # Wasserstein distance between [1,2,3] and [2,3,4] is 1.0
    res = WassersteinDistance(obs, mod)
    assert np.isclose(res, 1.0)


def test_kl_numpy():
    obs = np.random.normal(0, 1, 1000)
    mod = np.random.normal(0, 1, 1000)
    res = KLDivergence(obs, mod, bins=10)
    # KL divergence of identical distributions should be small
    assert res < 0.1


if __name__ == "__main__":
    test_wasserstein_numpy()
    try:
        test_wasserstein_xarray_lazy()
    except pytest.skip.Exception:
        pass
    test_kl_numpy()
    try:
        test_kl_xarray_lazy()
    except pytest.skip.Exception:
        pass
    print("Distribution metrics tests passed!")
