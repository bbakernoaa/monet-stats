import numpy as np
import xarray as xr

from monet_stats.spatial_ensemble_metrics import reliability_diagram


def test_reliability_numpy():
    obs = np.array([0, 1, 0, 1, 0])
    mod_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    res = reliability_diagram(obs, mod_prob, n_bins=2)
    assert len(res["forecast_prob"]) == 2
    # First bin (0-0.5) has probs [0.1, 0.2, 0.3] and obs [0, 0, 0] -> freq 0
    assert np.isclose(res["observed_freq"][0], 0.0)
    # Second bin (0.5-1.0) has probs [0.9, 0.8] and obs [1, 1] -> freq 1
    assert np.isclose(res["observed_freq"][1], 1.0)


def test_reliability_xarray():
    obs = xr.DataArray([0, 1, 0, 1, 0], dims="x")
    mod_prob = xr.DataArray([0.1, 0.9, 0.2, 0.8, 0.3], dims="x")
    res = reliability_diagram(obs, mod_prob, n_bins=2, dim="x")
    assert "bin" in res.coords
    assert np.isclose(res["observed_freq"].isel(bin=0), 0.0)
    assert np.isclose(res["observed_freq"].isel(bin=1), 1.0)


if __name__ == "__main__":
    test_reliability_numpy()
    test_reliability_xarray()
    print("Reliability diagram tests passed!")
