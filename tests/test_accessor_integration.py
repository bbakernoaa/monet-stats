
import numpy as np
import xarray as xr
import monet_stats
from monet_stats.error_metrics import MAE

def test_accessor_new_metrics():
    obs = xr.DataArray(np.random.normal(0, 1, 100), dims="time", coords={"time": np.arange(100)})
    mod = obs + np.random.normal(0, 0.1, 100)

    # Test Wasserstein
    res_wd = mod.monet_stats.wasserstein_distance(obs, dim="time")
    assert res_wd > 0

    # Test KL
    res_kl = mod.monet_stats.kl_divergence(obs, dim="time")
    assert res_kl > 0

    # Test DTW
    res_dtw = mod.monet_stats.dtw(obs, dim="time")
    assert res_dtw >= 0

    # Test XWT
    res_xwt = mod.monet_stats.xwt(obs, dim="time")
    assert "scale" in res_xwt.dims

    # Test Bootstrap
    res_boot = mod.monet_stats.bootstrap(obs, metric_func=MAE, n_boot=10, dim="time")
    assert "mean" in res_boot.data_vars

    # Test Reliability
    res_rel = mod.monet_stats.reliability_diagram(obs, threshold=0, dim="time")
    assert "observed_freq" in res_rel.data_vars

if __name__ == "__main__":
    test_accessor_new_metrics()
    print("Accessor integration tests passed!")
