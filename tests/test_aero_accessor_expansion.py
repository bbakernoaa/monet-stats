import numpy as np
import pytest
import xarray as xr
import dask.array as da
import monet_stats  # Ensure accessors are registered

def test_accessor_expansion_numpy():
    obs = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": [0, 1, 2, 3]})
    mod = xr.DataArray([1.1, 1.9, 3.2, 3.8], dims="time", coords={"time": [0, 1, 2, 3]})

    # Test a subset of new metrics
    assert np.isclose(mod.monet_stats.mse(obs), 0.025)
    # MAPE: 100 * [|1.1-1.0|/1 + |1.9-2.0|/2 + |3.2-3.0|/3 + |3.8-4.0|/4] / 4
    # = 100 * [0.1 + 0.05 + 0.0666... + 0.05] / 4 = 100 * 0.2666... / 4 = 26.66... / 4 = 6.66...
    assert np.isclose(mod.monet_stats.mape(obs), 6.666666666666667)

    # PC: Using more robust data to avoid float precision edges
    obs_pc = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time")
    mod_pc = xr.DataArray([1.05, 1.95, 3.05, 4.05], dims="time")
    assert np.isclose(mod_pc.monet_stats.pc(obs_pc, tolerance=0.1), 100.0)

    assert np.isclose(mod.monet_stats.spearmanr(obs), 1.0)
    assert np.isclose(mod.monet_stats.nme(obs), 6.0)
    assert np.isclose(mod.monet_stats.phase_error(obs), 0.0)

def test_accessor_expansion_dask():
    obs_data = da.from_array([1.0, 2.0, 3.0, 4.0], chunks=2)
    mod_data = da.from_array([1.1, 1.9, 3.2, 3.8], chunks=2)
    obs = xr.DataArray(obs_data, dims="time", coords={"time": [0, 1, 2, 3]})
    mod = xr.DataArray(mod_data, dims="time", coords={"time": [0, 1, 2, 3]})

    # Test laziness preservation
    res_mse = mod.monet_stats.mse(obs)
    assert hasattr(res_mse.data, "dask")
    assert np.isclose(res_mse.compute(), 0.025)

    res_spearman = mod.monet_stats.spearmanr(obs)
    assert hasattr(res_spearman.data, "dask")
    assert np.isclose(res_spearman.compute(), 1.0)

def test_accessor_verify_expansion():
    obs = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": [0, 1, 2, 3]})
    mod = xr.DataArray([1.1, 1.9, 3.2, 3.8], dims="time", coords={"time": [0, 1, 2, 3]})

    ds = mod.monet_stats.verify(obs)
    assert "MSE" in ds
    assert "SpearmanR" in ds
    assert "NME" in ds
    assert np.isclose(ds["MSE"], 0.025)
    assert np.isclose(ds["SpearmanR"], 1.0)
    assert np.isclose(ds["NME"], 6.0)

    # Verify history is present in the bundle
    assert "history" in ds.attrs

def test_accessor_peak_metrics():
    obs = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=("site", "time"))
    mod = xr.DataArray([[1.1, 2.1, 2.9], [3.9, 4.9, 6.1]], dims=("site", "time"))

    # MPE (mean over everything by default)
    res_mpe = mod.monet_stats.mpe(obs)
    # peaks: obs=[3, 6], mod=[2.9, 6.1] -- Wait, MPE uses global max by default if dim=None
    # If dim=None, mod.max() = 6.1, obs.max() = 6.
    # Error = |6.1-6|/6 = 0.1/6 = 0.0166...
    # 0.0166... * 100 = 1.666...
    assert np.isclose(res_mpe, 1.666666666666667)

    # NMPE (normalized peak error)
    # paxis: site (0) or time (1). Let's use time (1) to get peak over time per site.
    res_nmpe = mod.monet_stats.nmpe(obs, paxis="time")
    # Peak over time per site: obs=[3, 6], mod=[2.9, 6.1]
    # |mod_max - obs_max| = [0.1, 0.1]
    # obs_max = [3, 6]
    # mean(|diff|) / mean(obs_max) = 0.1 / 4.5 approx 0.0222
    # 0.0222 * 100 approx 2.222
    assert np.isclose(res_nmpe, 2.222222222)
