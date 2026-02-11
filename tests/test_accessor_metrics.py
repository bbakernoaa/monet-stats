import dask.array as da
import numpy as np
import xarray as xr

from monet_stats.correlation_metrics import CCC, KGE, R2, pearsonr
from monet_stats.efficiency_metrics import NSE
from monet_stats.error_metrics import CRMSE, IOA, MAE, MB, MNB, MNE, NMSE, RMSE, MdnB
from monet_stats.relative_metrics import FB, NMB


def test_accessor_error_metrics():
    obs = xr.DataArray(np.array([1.0, 2.0, 3.0]), coords={"time": [1, 2, 3]}, dims="time", name="obs")
    mod = xr.DataArray(np.array([1.1, 1.9, 3.2]), coords={"time": [1, 2, 3]}, dims="time", name="mod")

    # MAE
    assert np.allclose(mod.monet_stats.mae(obs), MAE(obs, mod))

    # RMSE
    assert np.allclose(mod.monet_stats.rmse(obs), RMSE(obs, mod))

    # MB
    assert np.allclose(mod.monet_stats.mb(obs), MB(obs, mod))

    # IOA
    assert np.allclose(mod.monet_stats.ioa(obs), IOA(obs, mod))

    # CRMSE
    assert np.allclose(mod.monet_stats.crmse(obs), CRMSE(obs, mod))

    # MdnB
    assert np.allclose(mod.monet_stats.mdnb(obs), MdnB(obs, mod))

    # NMSE
    assert np.allclose(mod.monet_stats.nmse(obs), NMSE(obs, mod))

    # MNB
    assert np.allclose(mod.monet_stats.mnb(obs), MNB(obs, mod))

    # MNE
    assert np.allclose(mod.monet_stats.mne(obs), MNE(obs, mod))

    # NSE
    assert np.allclose(mod.monet_stats.nse(obs), NSE(obs, mod))


def test_accessor_correlation_metrics():
    obs = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), coords={"time": [1, 2, 3, 4]}, dims="time")
    mod = xr.DataArray(np.array([1.1, 1.9, 3.2, 3.8]), coords={"time": [1, 2, 3, 4]}, dims="time")

    # Pearson R
    assert np.allclose(mod.monet_stats.pearsonr(obs), pearsonr(obs, mod))

    # R2
    assert np.allclose(mod.monet_stats.r2(obs), R2(obs, mod))

    # KGE
    assert np.allclose(mod.monet_stats.kge(obs), KGE(obs, mod))

    # CCC
    assert np.allclose(mod.monet_stats.ccc(obs), CCC(obs, mod))


def test_accessor_relative_metrics():
    obs = xr.DataArray(np.array([1.0, 2.0, 3.0]), coords={"time": [1, 2, 3]}, dims="time")
    mod = xr.DataArray(np.array([1.1, 2.2, 3.3]), coords={"time": [1, 2, 3]}, dims="time")

    # NMB
    assert np.allclose(mod.monet_stats.nmb(obs), NMB(obs, mod))

    # FB
    assert np.allclose(mod.monet_stats.fb(obs), FB(obs, mod))


def test_accessor_verify_bundle():
    obs = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0]), coords={"time": [1, 2, 3, 4]}, dims="time")
    mod = xr.DataArray(np.array([1.1, 1.9, 3.2, 3.8]), coords={"time": [1, 2, 3, 4]}, dims="time")

    ds = mod.monet_stats.verify(obs)

    assert isinstance(ds, xr.Dataset)
    assert "MAE" in ds
    assert "RMSE" in ds
    assert "MB" in ds
    assert "R" in ds
    assert "IOA" in ds
    assert "NMB" in ds
    assert "MNB" in ds
    assert "MNE" in ds
    assert "NSE" in ds
    assert "R2" in ds

    assert np.allclose(ds["MAE"], MAE(obs, mod))
    assert np.allclose(ds["RMSE"], RMSE(obs, mod))
    assert np.allclose(ds["R"], pearsonr(obs, mod))


def test_accessor_dask_lazy():
    obs_data = da.from_array(np.random.rand(100), chunks=50)
    mod_data = da.from_array(np.random.rand(100), chunks=50)

    obs = xr.DataArray(obs_data, coords={"time": np.arange(100)}, dims="time")
    mod = xr.DataArray(mod_data, coords={"time": np.arange(100)}, dims="time")

    # Accessor call should remain lazy
    rmse_lazy = mod.monet_stats.rmse(obs)
    assert hasattr(rmse_lazy.data, "chunks")

    # Verify result
    assert np.allclose(rmse_lazy.compute(), RMSE(obs.compute(), mod.compute()))

    # Verify bundle is lazy
    ds_lazy = mod.monet_stats.verify(obs)
    for var in ds_lazy.data_vars:
        assert hasattr(ds_lazy[var].data, "chunks")

    ds_computed = ds_lazy.compute()
    assert "history" in ds_computed.attrs
