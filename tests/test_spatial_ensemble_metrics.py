import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_ensemble_metrics import (
    BSS,
    CRPS,
    EDS,
    SAL,
    ensemble_mean,
    ensemble_std,
    rank_histogram,
    spread_error,
)

try:
    import dask.array as da  # noqa: F401

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


@pytest.fixture
def sample_ensemble_data():
    np.random.seed(42)
    ens = np.random.rand(5, 10, 10)  # (ens, y, x)
    obs = np.random.rand(10, 10)
    return ens, obs


def test_eds_eager_lazy():
    np.random.seed(42)
    obs = np.random.rand(20, 20)
    mod = np.random.rand(20, 20)
    threshold = 0.5

    res_eager = EDS(obs, mod, threshold)

    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    mod_xr = xr.DataArray(mod, dims=["y", "x"])
    if HAS_DASK:
        obs_xr = obs_xr.chunk({"y": 10})
        mod_xr = mod_xr.chunk({"y": 10})

    res_lazy = EDS(obs_xr, mod_xr, threshold)

    if HAS_DASK:
        assert hasattr(res_lazy.data, "dask")

    np.testing.assert_allclose(res_eager, res_lazy.compute(), atol=1e-7)
    assert "history" in res_lazy.attrs
    assert "EDS" in res_lazy.attrs["history"]


def test_crps_eager_lazy(sample_ensemble_data):
    ens, obs = sample_ensemble_data

    res_eager = CRPS(ens, obs, axis=0)

    ens_xr = xr.DataArray(ens, dims=["ensemble", "y", "x"])
    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    if HAS_DASK:
        ens_xr = ens_xr.chunk({"ensemble": -1, "y": 5})
        obs_xr = obs_xr.chunk({"y": 5})

    res_lazy = CRPS(ens_xr, obs_xr, axis="ensemble")

    if HAS_DASK:
        assert hasattr(res_lazy.data, "dask")

    np.testing.assert_allclose(res_eager, res_lazy.compute(), atol=1e-7)
    assert "history" in res_lazy.attrs
    assert "CRPS" in res_lazy.attrs["history"]


def test_spread_error_eager_lazy(sample_ensemble_data):
    ens, obs = sample_ensemble_data

    s_eager, e_eager = spread_error(ens, obs, axis=0)

    ens_xr = xr.DataArray(ens, dims=["ensemble", "y", "x"])
    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    if HAS_DASK:
        ens_xr = ens_xr.chunk({"ensemble": -1})

    s_lazy, e_lazy = spread_error(ens_xr, obs_xr, axis="ensemble")

    np.testing.assert_allclose(s_eager, s_lazy.compute(), atol=1e-7)
    np.testing.assert_allclose(e_eager, e_lazy.compute(), atol=1e-7)


def test_bss_eager_lazy():
    np.random.seed(42)
    obs = np.random.choice([0, 1], size=(10, 10)).astype(float)
    mod = np.random.rand(10, 10)
    threshold = 0.5

    res_eager = BSS(obs, mod, threshold)

    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    mod_xr = xr.DataArray(mod, dims=["y", "x"])

    res_lazy = BSS(obs_xr, mod_xr, threshold)

    np.testing.assert_allclose(res_eager, res_lazy, atol=1e-7)


def test_ensemble_stats_eager_lazy(sample_ensemble_data):
    ens, _ = sample_ensemble_data

    m_eager = ensemble_mean(ens, axis=0)
    std_eager = ensemble_std(ens, axis=0)

    ens_xr = xr.DataArray(ens, dims=["ensemble", "y", "x"])
    m_lazy = ensemble_mean(ens_xr, axis="ensemble")
    std_lazy = ensemble_std(ens_xr, axis="ensemble")

    np.testing.assert_allclose(m_eager, m_lazy, atol=1e-7)
    np.testing.assert_allclose(std_eager, std_lazy, atol=1e-7)


def test_rank_histogram_eager_lazy(sample_ensemble_data):
    ens, obs = sample_ensemble_data

    hist_eager = rank_histogram(ens, obs, axis=0)

    ens_xr = xr.DataArray(ens, dims=["ensemble", "y", "x"])
    obs_xr = xr.DataArray(obs, dims=["y", "x"])

    hist_lazy = rank_histogram(ens_xr, obs_xr, axis="ensemble")

    np.testing.assert_allclose(hist_eager, hist_lazy, atol=1e-7)
    assert "rank" in hist_lazy.dims


def test_sal_eager():
    np.random.seed(42)
    obs = np.zeros((20, 20))
    obs[5:10, 5:10] = 1.0
    mod = np.zeros((20, 20))
    mod[6:11, 6:11] = 1.0  # Shifted

    s_e, a_e, l_e = SAL(obs, mod, threshold=0.5)

    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    mod_xr = xr.DataArray(mod, dims=["y", "x"])

    s_l, a_l, l_l = SAL(obs_xr, mod_xr, threshold=0.5)

    np.testing.assert_allclose(s_e, s_l, atol=1e-7)
    np.testing.assert_allclose(a_e, a_l, atol=1e-7)
    np.testing.assert_allclose(l_e, l_l, atol=1e-7)
