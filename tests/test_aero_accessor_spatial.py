import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_ensemble_metrics import CRPS, EDS, SAL, spread_error
from monet_stats.spatial_skill_metrics import FSS, VETS


def test_accessor_spatial_skill():
    obs = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"], name="obs")
    mod = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"], name="mod")

    # FSS
    assert np.allclose(mod.monet_stats.fss(obs, window_size=3), FSS(obs, mod, window_size=3))

    # VETS
    assert np.allclose(mod.monet_stats.vets(obs), VETS(obs, mod))


def test_accessor_ensemble_metrics():
    # Ensemble: (member, lat, lon)
    ens = xr.DataArray(np.random.rand(5, 4, 4), dims=["member", "lat", "lon"], name="ens")
    obs = xr.DataArray(np.random.rand(4, 4), dims=["lat", "lon"], name="obs")

    # CRPS
    assert np.allclose(ens.monet_stats.crps(obs, dim="member"), CRPS(ens, obs, axis="member"))

    # Spread-Error
    m_spread, m_error = ens.monet_stats.spread_error(obs, ensemble_dim="member")
    ref_spread, ref_error = spread_error(ens, obs, axis="member")
    assert np.allclose(m_spread, ref_spread)
    assert np.allclose(m_error, ref_error)


def test_accessor_eds():
    obs = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"])
    mod = xr.DataArray(np.random.rand(10, 10), dims=["lat", "lon"])
    threshold = 0.5

    assert np.allclose(mod.monet_stats.eds(obs, threshold=threshold), EDS(obs, mod, threshold=threshold))


def test_accessor_sal():
    obs = xr.DataArray(np.random.rand(20, 20), dims=["lat", "lon"], coords={"lat": np.arange(20), "lon": np.arange(20)})
    mod = xr.DataArray(np.random.rand(20, 20), dims=["lat", "lon"], coords={"lat": np.arange(20), "lon": np.arange(20)})

    res = mod.monet_stats.sal(obs)
    ref = SAL(obs, mod)

    for r, f in zip(res, ref):
        assert np.allclose(r, f)


def test_accessor_spatial_dask_lazy():
    da = pytest.importorskip("dask.array")
    obs_data = da.random.random((10, 10), chunks=(5, 5))
    mod_data = da.random.random((10, 10), chunks=(5, 5))

    obs = xr.DataArray(obs_data, dims=["lat", "lon"], name="obs")
    mod = xr.DataArray(mod_data, dims=["lat", "lon"], name="mod")

    # FSS lazy check
    fss_lazy = mod.monet_stats.fss(obs, window_size=3)
    assert hasattr(fss_lazy.data, "chunks")
    assert np.allclose(fss_lazy.compute(), FSS(obs.compute(), mod.compute(), window_size=3))

    # VETS lazy check
    vets_lazy = mod.monet_stats.vets(obs)
    assert hasattr(vets_lazy.data, "chunks")
    assert np.allclose(vets_lazy.compute(), VETS(obs.compute(), mod.compute()))

    # CRPS lazy check
    ens_data = da.random.random((5, 10, 10), chunks=(5, 5, 5))
    ens = xr.DataArray(ens_data, dims=["member", "lat", "lon"])
    obs_2d = xr.DataArray(da.random.random((10, 10), chunks=(5, 5)), dims=["lat", "lon"])

    crps_lazy = ens.monet_stats.crps(obs_2d, dim="member")
    assert hasattr(crps_lazy.data, "chunks")
    assert np.allclose(crps_lazy.compute(), CRPS(ens.compute(), obs_2d.compute(), axis="member"))
