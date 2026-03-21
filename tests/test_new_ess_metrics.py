import numpy as np
import pytest
import xarray as xr

from monet_stats import BS, FAC2, MG, RMSLE, VG, stats


def import_pytest():
    return pytest


def test_fac2():
    obs = np.array([1, 2, 3, 4])
    mod = np.array([1.5, 5, 2.5, 1])
    # ratios: 1.5, 2.5, 0.833, 0.25
    # in range: Yes, No, Yes, No
    expected = 0.5
    assert FAC2(obs, mod) == expected

    # Test xarray
    obs_da = xr.DataArray(obs, dims="x")
    mod_da = xr.DataArray(mod, dims="x")
    res = FAC2(obs_da, mod_da)
    assert float(res) == expected
    assert "history" in res.attrs


def test_rmsle():
    obs = np.array([1, 10, 100])
    mod = np.array([1.1, 15, 80])
    res = RMSLE(obs, mod)
    assert res > 0

    # Test xarray
    obs_da = xr.DataArray(obs, dims="x")
    mod_da = xr.DataArray(mod, dims="x")
    res_xr = RMSLE(obs_da, mod_da)
    assert np.allclose(float(res_xr), res)
    assert "history" in res_xr.attrs


def test_geometric_metrics():
    obs = np.array([1, 2, 3])
    mod = np.array([1.1, 2.2, 3.3])
    mg = MG(obs, mod)
    vg = VG(obs, mod)
    assert mg > 1.0
    assert vg > 1.0

    # Test xarray
    obs_da = xr.DataArray(obs, dims="x")
    mod_da = xr.DataArray(mod, dims="x")
    mg_xr = MG(obs_da, mod_da)
    vg_xr = VG(obs_da, mod_da)
    assert np.allclose(float(mg_xr), mg)
    assert np.allclose(float(vg_xr), vg)
    assert "history" in mg_xr.attrs
    assert "history" in vg_xr.attrs


def test_brier_score():
    obs = np.array([0, 1, 1, 0])
    mod = np.array([0.1, 0.9, 0.2, 0.3])
    res = BS(obs, mod)
    expected = np.mean((mod - obs) ** 2)
    assert np.allclose(res, expected)

    # Test xarray
    obs_da = xr.DataArray(obs, dims="x")
    mod_da = xr.DataArray(mod, dims="x")
    res_xr = BS(obs_da, mod_da)
    assert np.allclose(float(res_xr), expected)
    assert "history" in res_xr.attrs


def test_stats_with_fac2():
    df = xr.Dataset({"Obs": (("x",), [1, 2, 3, 4]), "Mod": (("x",), [1.5, 5, 2.5, 1])})
    res = stats(df)
    assert "FAC2" in res
    assert res["FAC2"] == 0.5


def test_fac2_with_nans():
    obs = np.array([1, 2, 3, np.nan])
    mod = np.array([1.5, 5, 2.5, 1])
    # ratios: 1.5, 2.5, 0.833, NaN
    # in range: Yes, No, Yes, (Excluded)
    # expected: 2/3 approx 0.666
    expected = 2 / 3.0
    assert np.allclose(FAC2(obs, mod), expected)

    # Test xarray
    obs_da = xr.DataArray(obs, dims="x")
    mod_da = xr.DataArray(mod, dims="x")
    res = FAC2(obs_da, mod_da)
    assert np.allclose(float(res), expected)


def test_dask_laziness():
    pytest = import_pytest()
    da = pytest.importorskip("dask.array")

    obs = da.from_array([1.0, 2.0, 3.0, 4.0], chunks=2)
    mod = da.from_array([1.5, 5.0, 2.5, 1.0], chunks=2)
    obs_da = xr.DataArray(obs, dims="x")
    mod_da = xr.DataArray(mod, dims="x")

    # FAC2
    res = FAC2(obs_da, mod_da)
    assert hasattr(res.data, "dask")
    assert float(res.compute()) == 0.5

    # RMSLE
    res = RMSLE(obs_da, mod_da)
    assert hasattr(res.data, "dask")
    assert float(res.compute()) > 0

    # MG
    res = MG(obs_da, mod_da)
    assert hasattr(res.data, "dask")
    assert float(res.compute()) > 0

    # BS
    res = BS(obs_da, mod_da)
    assert hasattr(res.data, "dask")
    assert float(res.compute()) > 0


def test_accessor_new_metrics():
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="x", name="Obs")
    mod = xr.DataArray([1.1, 2.2, 3.3], dims="x", name="Mod")

    # Test accessor
    assert float(mod.monet_stats.fac2(obs)) == 1.0
    assert float(mod.monet_stats.mg(obs)) > 1.0
    assert float(mod.monet_stats.vg(obs)) > 1.0
    assert float(mod.monet_stats.rmsle(obs)) > 0

    # Test verify bundle
    bundle = mod.monet_stats.verify(obs)
    assert "FAC2" in bundle
    assert float(bundle["FAC2"]) == 1.0
