
import numpy as np
import xarray as xr
import dask.array as da
from monet_stats.uncertainty import block_bootstrap
from monet_stats.error_metrics import MAE

def test_bootstrap_numpy():
    obs = np.random.normal(0, 1, 100)
    mod = obs + np.random.normal(0, 0.1, 100)
    res = block_bootstrap(obs, mod, metric_func=MAE, n_boot=100)
    assert "mean" in res
    assert res["lower"] < res["mean"] < res["upper"]

def test_bootstrap_xarray_lazy():
    obs_data = np.random.normal(0, 1, 100)
    mod_data = obs_data + np.random.normal(0, 0.1, 100)
    obs = xr.DataArray(da.from_array(obs_data, chunks=50), dims="time")
    mod = xr.DataArray(da.from_array(mod_data, chunks=50), dims="time")
    res = block_bootstrap(obs, mod, metric_func=MAE, n_boot=100)
    assert "mean" in res.data_vars
    assert hasattr(res["mean"].data, "chunks")
    val = res.compute()
    assert val["lower"] < val["mean"] < val["upper"]

if __name__ == "__main__":
    test_bootstrap_numpy()
    test_bootstrap_xarray_lazy()
    print("Uncertainty metrics tests passed!")
