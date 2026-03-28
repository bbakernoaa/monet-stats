import numpy as np
import pytest
import xarray as xr

from monet_stats import stats
from monet_stats.error_metrics import MAE, MB, MSE, RMSE
from monet_stats.relative_metrics import NMB


def test_numpy_weighted_metrics():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 2.2, 3.3])

    # Weight 1.0 on index 0 (diff 0.1, obs 1.0), weight 0.0 on others
    weights = np.array([1.0, 0.0, 0.0])
    assert np.allclose(MB(obs, mod, weights=weights), 0.1)
    assert np.allclose(MAE(obs, mod, weights=weights), 0.1)
    assert np.allclose(MSE(obs, mod, weights=weights), 0.01)
    assert np.allclose(RMSE(obs, mod, weights=weights), 0.1)
    assert np.allclose(NMB(obs, mod, weights=weights), 10.0)


def test_xarray_weighted_metrics_eager():
    obs = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": [0, 45]})
    mod = xr.DataArray([1.1, 2.2], dims="lat", coords={"lat": [0, 45]})
    weights = xr.DataArray([1.0, 0.0], dims="lat", coords={"lat": [0, 45]})

    # Weight 1.0 on lat 0 (diff 0.1, obs 1.0), weight 0.0 on lat 45
    assert np.allclose(MB(obs, mod, weights=weights), 0.1)
    assert np.allclose(NMB(obs, mod, weights=weights), 10.0)


def test_xarray_weighted_metrics_lazy():
    pytest.importorskip("dask")
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="lat").chunk({"lat": 1})
    mod = xr.DataArray([1.1, 2.2, 3.3], dims="lat").chunk({"lat": 1})
    weights = xr.DataArray([1.0, 0.0, 0.0], dims="lat").chunk({"lat": 1})

    res = MB(obs, mod, weights=weights)
    assert hasattr(res.data, "dask")
    assert np.allclose(res.compute(), 0.1)


def test_stats_weighted():
    ds = xr.Dataset({"Obs": (("lat",), [1.0, 2.0]), "Mod": (("lat",), [1.1, 2.2])}, coords={"lat": [0, 45]})
    weights = xr.DataArray([1.0, 0.0], dims="lat", coords={"lat": [0, 45]})

    results = stats(ds, weights=weights)

    # With weight [1, 0], Obs mean should be 1.0, Mod mean 1.1, MB 0.1
    assert np.allclose(results["Obs"], 1.0)
    assert np.allclose(results["Mod"], 1.1)
    assert np.allclose(results["MB"], 0.1)
    assert np.allclose(results["NMB"], 10.0)
