import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

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


def test_eds_numpy():
    obs = np.array([0, 1, 0, 1])
    mod = np.array([0, 1, 1, 0])
    # hits=1, n_obs=2, n_mod=2, n=4
    # p=0.5, q=0.5, hits/n=0.25
    # EDS = log(0.25) / log(0.5*0.5) = log(0.25) / log(0.25) = 1.0
    res = EDS(obs, mod, threshold=0.5)
    assert np.isclose(res, 1.0)


def test_eds_xarray_dask():
    if not HAS_DASK:
        pytest.skip("Dask not installed")
    obs = xr.DataArray(da.from_array([0, 1, 0, 1], chunks=2), dims="x")
    mod = xr.DataArray(da.from_array([0, 1, 1, 0], chunks=2), dims="x")
    res = EDS(obs, mod, threshold=0.5)
    assert isinstance(res.data, da.Array)
    assert np.isclose(res.compute(), 1.0)
    assert "history" in res.attrs


def test_crps_numpy():
    ens = np.array([[1, 2], [2, 3], [3, 4]])
    obs = np.array([2, 3])
    # ens_sorted (axis 0): [[1, 2], [2, 3], [3, 4]]
    # n=3
    # cdf_ens: [1/3, 2/3, 3/3]
    # For obs=2: cdf_obs = [0, 1, 1] (is 1, 2, 3 >= 2?) -> [0, 1, 1]
    # (1/3-0)^2 + (2/3-1)^2 + (3/3-1)^2 = 1/9 + 1/9 + 0 = 2/9
    # Same for obs=3: cdf_obs = [0, 0, 1] (is 2, 3, 4 >= 3?) -> [0, 1, 1] Wait.
    # ens_sorted for second col: [2, 3, 4]. Is [2, 3, 4] >= 3? -> [False, True, True] -> [0, 1, 1]
    # (1/3-0)^2 + (2/3-1)^2 + (3/3-1)^2 = 2/9
    res = CRPS(ens, obs, axis=0)
    assert np.allclose(res, [2 / 9, 2 / 9])


def test_crps_xarray_dask():
    if not HAS_DASK:
        pytest.skip("Dask not installed")
    ens = xr.DataArray(da.from_array([[1, 2], [2, 3], [3, 4]], chunks=(2, 2)), dims=("member", "x"))
    obs = xr.DataArray(da.from_array([2, 3], chunks=2), dims="x")
    res = CRPS(ens, obs, axis="member")
    assert isinstance(res.data, da.Array)
    assert np.allclose(res.compute(), [2 / 9, 2 / 9])
    assert "history" in res.attrs


def test_bss_numpy():
    obs = np.array([0, 1, 1, 0])
    mod = np.array([0.2, 0.8, 0.9, 0.3])
    # threshold 0.5 -> mod_bin = [0, 1, 1, 0]
    # bs = mean([0, 0, 0, 0]) = 0
    # obs_clim = 0.5. bs_ref = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    # BSS = 1 - 0/0.25 = 1.0
    res = BSS(obs, mod, threshold=0.5)
    assert res == 1.0


def test_sal_numpy():
    obs = np.zeros((10, 10))
    obs[2:5, 2:5] = 1.0
    mod = np.zeros((10, 10))
    mod[2:5, 3:6] = 1.0
    # Should be similar but shifted
    s, a, location_score = SAL(obs, mod, threshold=0.5)
    assert np.isclose(a, 0.0)
    assert np.isclose(s, 0.0)
    assert location_score > 0


def test_ensemble_mean_std_xarray():
    ens = xr.DataArray(np.random.rand(3, 10, 10), dims=("member", "x", "y"))
    m = ensemble_mean(ens, axis="member")
    s = ensemble_std(ens, axis="member")
    assert m.dims == ("x", "y")
    assert s.dims == ("x", "y")
    assert "history" in m.attrs
    assert "history" in s.attrs


def test_rank_histogram_xarray():
    ens = xr.DataArray(np.array([[1, 5], [2, 6], [3, 7]]), dims=("member", "x"))
    obs = xr.DataArray(np.array([2, 6.5]), dims="x")
    # For x=0: rank is 2.
    # For x=1: rank is 2.
    # Histogram of ranks [2, 2] with bins [0, 1, 2, 3] should be [0, 0, 2, 0]
    res = rank_histogram(ens, obs, axis="member")
    # res is now a histogram DataArray
    assert res.dims == ("rank",)
    assert np.allclose(res.values, [0, 0, 2, 0])
    assert "history" in res.attrs


def test_spread_error_xarray():
    ens = xr.DataArray(np.random.rand(5, 10, 10), dims=("member", "x", "y"))
    obs = xr.DataArray(np.random.rand(10, 10), dims=("x", "y"))
    m_spread, m_error = spread_error(ens, obs, axis="member")
    assert isinstance(m_spread, xr.DataArray)
    assert isinstance(m_error, xr.DataArray)
    assert "history" in m_spread.attrs
    assert "history" in m_error.attrs
