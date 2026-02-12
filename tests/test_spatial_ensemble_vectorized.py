import numpy as np
import xarray as xr

from monet_stats.spatial_ensemble_metrics import BSS, EDS, spread_error


def test_eds_vectorized():
    # Create synthetic data with a time dimension
    # (time, space)
    obs = xr.DataArray(np.array([[1, 0], [1, 1]]), dims=["time", "space"], coords={"time": [0, 1], "space": [0, 1]})
    mod = xr.DataArray(np.array([[1, 1], [0, 1]]), dims=["time", "space"], coords={"time": [0, 1], "space": [0, 1]})

    # 1. Global reduction
    res_global = EDS(obs, mod, threshold=0.5)
    assert res_global.ndim == 0

    # 2. Reduction over space (should have 'time' dimension)
    res_time = EDS(obs, mod, threshold=0.5, dim="space")
    assert "time" in res_time.dims
    assert res_time.shape == (2,)


def test_bss_vectorized():
    # (time, space)
    obs = xr.DataArray(np.array([[1, 0], [1, 1]]), dims=["time", "space"], coords={"time": [0, 1], "space": [0, 1]})
    mod = xr.DataArray(
        np.array([[0.8, 0.2], [0.1, 0.9]]), dims=["time", "space"], coords={"time": [0, 1], "space": [0, 1]}
    )

    # 1. Global reduction
    res_global = BSS(obs, mod, threshold=0.5)
    assert res_global.ndim == 0

    # 2. Reduction over space
    res_time = BSS(obs, mod, threshold=0.5, dim="space")
    assert "time" in res_time.dims
    assert res_time.shape == (2,)


def test_spread_error_vectorized():
    # (ensemble, time, space)
    ens = xr.DataArray(np.random.rand(5, 10, 10), dims=["ens", "time", "space"])
    obs = xr.DataArray(np.random.rand(10, 10), dims=["time", "space"])

    # 1. Global reduction
    # Passing axis as positional should go to 'axis' (ensemble dim)
    m_spread, m_error = spread_error(ens, obs, "ens")
    assert m_spread.ndim == 0
    assert m_error.ndim == 0

    # 2. Reduction over space (should have 'time' dimension)
    m_spread_t, m_error_t = spread_error(ens, obs, axis="ens", dim="space")
    assert "time" in m_spread_t.dims
    assert m_spread_t.shape == (10,)
    assert "time" in m_error_t.dims
    assert m_error_t.shape == (10,)


def test_spread_error_backward_compat():
    # Mimic the failing CI test
    obs = np.random.rand(5, 5)
    ens = np.random.rand(10, 5, 5)

    # In old code, axis=0 meant ensemble axis.
    # In my new code, axis=0 also means ensemble axis.
    # And since reduce_axis is None, it should reduce over all other axes.
    s, e = spread_error(ens, obs, axis=0)
    assert np.shape(s) == ()
    assert np.shape(e) == ()
    assert isinstance(s, float)


def test_numpy_paths():
    obs = np.array([1, 0, 1, 1])
    mod = np.array([1, 1, 0, 1])

    # EDS
    res_eds = EDS(obs, mod, threshold=0.5)
    assert isinstance(res_eds, (float, np.number))

    # BSS
    res_bss = BSS(obs, mod, threshold=0.5)
    assert isinstance(res_bss, (float, np.number))

    # spread_error
    ens = np.random.rand(5, 4)
    obs_1d = np.random.rand(4)
    # axis=0 is ensemble dimension. reduce_axis=None means full reduction.
    m_spread, m_error = spread_error(ens, obs_1d, axis=0)
    assert isinstance(m_spread, float)
