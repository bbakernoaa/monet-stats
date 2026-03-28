import numpy as np
import xarray as xr

from monet_stats import stats
from monet_stats.error_metrics import MAE, MB, RMSE


def test_numpy_weighted_metrics():
    """Test NumPy weighted metrics for mathematical accuracy."""
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])
    weights = np.array([1.0, 0.5, 0.1])

    # Expected weighted MB: (0.1 * 1.0 + -0.1 * 0.5 + 0.2 * 0.1) / (1.0 + 0.5 + 0.1)
    # = (0.1 - 0.05 + 0.02) / 1.6 = 0.07 / 1.6 = 0.04375
    expected_mb = 0.04375
    assert np.allclose(MB(obs, mod, weights=weights), expected_mb)

    # MAE: (abs(0.1) * 1.0 + abs(-0.1) * 0.5 + abs(0.2) * 0.1) / 1.6
    # = (0.1 + 0.05 + 0.02) / 1.6 = 0.17 / 1.6 = 0.10625
    expected_mae = 0.10625
    assert np.allclose(MAE(obs, mod, weights=weights), expected_mae)


def test_xarray_weighted_metrics_eager():
    """Test Xarray weighted metrics on eager DataArrays."""
    lats = np.array([0, 60])
    obs = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": lats})
    mod = xr.DataArray([1.1, 2.2], dims="lat", coords={"lat": lats})
    weights = xr.DataArray(np.cos(np.deg2rad(lats)), dims="lat", coords={"lat": lats})

    # weights: [1.0, 0.5]
    # diff: [0.1, 0.2]
    # Weighted MB: (0.1 * 1.0 + 0.2 * 0.5) / (1.0 + 0.5) = (0.1 + 0.1) / 1.5 = 0.2 / 1.5 = 0.133333
    res = MB(obs, mod, weights=weights)
    assert isinstance(res, xr.DataArray)
    assert np.allclose(res.values, 0.2 / 1.5)
    assert "history" in res.attrs
    assert "Weighted MB" in res.attrs["history"]


def test_xarray_weighted_metrics_lazy():
    """Test Xarray weighted metrics on Dask-backed DataArrays (Aero Protocol)."""
    lats = np.array([0, 60])
    obs = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": lats}).chunk({"lat": 1})
    mod = xr.DataArray([1.1, 2.2], dims="lat", coords={"lat": lats}).chunk({"lat": 1})
    weights = xr.DataArray(np.cos(np.deg2rad(lats)), dims="lat", coords={"lat": lats}).chunk({"lat": 1})

    res = RMSE(obs, mod, weights=weights)

    # Check laziness
    assert hasattr(res.data, "chunks")

    # Compute and verify
    # diff_sq: [0.01, 0.04]
    # weighted_mse: (0.01 * 1.0 + 0.04 * 0.5) / 1.5 = (0.01 + 0.02) / 1.5 = 0.03 / 1.5 = 0.02
    # weighted_rmse: sqrt(0.02) = 0.141421
    val = res.compute()
    assert np.allclose(val, np.sqrt(0.02))
    assert "Weighted RMSE" in val.attrs["history"]


def test_top_level_stats_weighted():
    """Test the central stats() function with weights."""
    ds = xr.Dataset({"Obs": (["lat"], [1.0, 2.0]), "Mod": (["lat"], [1.1, 2.2])}, coords={"lat": [0, 60]})
    weights = xr.DataArray([1.0, 0.5], dims="lat")

    res = stats(ds, weights=weights)

    assert "MB" in res
    assert np.allclose(res["MB"], 0.2 / 1.5)
    assert "MAE" in res
    assert np.allclose(res["MAE"], 0.2 / 1.5)
    assert "RMSE" in res
    assert np.allclose(res["RMSE"], np.sqrt(0.03 / 1.5))

    # Check that Obs and Mod means are also weighted
    # Weighted Obs: (1.0 * 1.0 + 2.0 * 0.5) / 1.5 = 2.0 / 1.5 = 1.333333
    # Weighted Mod: (1.1 * 1.0 + 2.2 * 0.5) / 1.5 = 2.2 / 1.5 = 1.466667
    assert np.allclose(res["Obs"], 2.0 / 1.5)
    assert np.allclose(res["Mod"], 2.2 / 1.5)
    # MB should be Mod - Obs
    assert np.allclose(res["MB"], res["Mod"] - res["Obs"])


def test_accessor_verify_weighted():
    """Test the Xarray accessor's verify method with weights."""
    obs = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": [0, 60]}, name="obs")
    mod = xr.DataArray([1.1, 2.2], dims="lat", coords={"lat": [0, 60]}, name="mod")
    weights = xr.DataArray([1.0, 0.5], dims="lat")

    res = mod.monet_stats.verify(obs, weights=weights)

    assert isinstance(res, xr.Dataset)
    assert np.allclose(res.MB.values, 0.2 / 1.5)
    assert "history" in res.attrs
    assert "Verification metrics bundle" in res.attrs["history"]
