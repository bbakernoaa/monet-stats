import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

from monet_stats.error_metrics import (
    CORR_INDEX,
    MAE,
    MB,
    MNB,
    MNE,
    MO,
    MP,
    NRMSE,
    NSC,
    RM,
    RMSE,
    STDO,
    STDP,
    NSE_alpha,
    NSE_beta,
    WDMdnB,
    bias_fraction,
    sMAPE,
)


def test_stdo_numpy():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])
    expected = np.std(obs - mod)
    result = STDO(obs, mod)
    assert np.isclose(result, expected)


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_stdo_xarray_dask():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)
    obs = xr.DataArray(da.from_array(obs_data, chunks=(5, 5)), dims=("x", "y"), name="obs")
    mod = xr.DataArray(da.from_array(mod_data, chunks=(5, 5)), dims=("x", "y"), name="mod")
    result = STDO(obs, mod)
    assert isinstance(result.data, da.Array)
    expected = np.std(obs_data - mod_data)
    xr.testing.assert_allclose(result.compute(), xr.DataArray(expected))
    assert "history" in result.attrs
    assert "STDO computed at" in result.attrs["history"]


def test_stdo_xarray_eager():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)
    obs = xr.DataArray(obs_data, dims=("x", "y"), name="obs")
    mod = xr.DataArray(mod_data, dims=("x", "y"), name="mod")
    result = STDO(obs, mod)
    assert not hasattr(result.data, "dask")
    expected = np.std(obs_data - mod_data)
    xr.testing.assert_allclose(result, xr.DataArray(expected))
    assert "history" in result.attrs
    assert "STDO computed at" in result.attrs["history"]


def test_stdp_xarray_dims():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)
    coords = {"x": np.arange(10), "y": np.arange(10)}
    obs = xr.DataArray(obs_data, dims=("x", "y"), coords=coords, name="obs")
    mod = xr.DataArray(mod_data, dims=("x", "y"), coords=coords, name="mod")
    result = STDP(obs, mod, axis=0)
    assert result.dims == ("y",)
    expected = np.std(mod_data - obs_data, axis=0)
    xr.testing.assert_allclose(result, xr.DataArray(expected, dims=("y",), coords={"y": obs.y}))
    result_dim = STDP(obs, mod, axis="y")
    assert result_dim.dims == ("x",)
    expected_dim = np.std(mod_data - obs_data, axis=1)
    xr.testing.assert_allclose(result_dim, xr.DataArray(expected_dim, dims=("x",), coords={"x": obs.x}))


@pytest.mark.skipif(not HAS_DASK, reason="Dask not installed")
def test_generic_metrics_xarray_dask():
    obs_data = np.random.rand(10, 10) + 1.0
    mod_data = np.random.rand(10, 10) + 1.0
    coords = {"x": np.arange(10), "y": np.arange(10)}
    obs = xr.DataArray(da.from_array(obs_data, chunks=(5, 5)), dims=("x", "y"), coords=coords, name="obs")
    mod = xr.DataArray(da.from_array(mod_data, chunks=(5, 5)), dims=("x", "y"), coords=coords, name="mod")
    metrics = [
        (MNB, "MNB"),
        (MNE, "MNE"),
        (MO, "MO"),
        (MP, "MP"),
        (RM, "RM"),
        (MB, "MB"),
        (MAE, "MAE"),
        (sMAPE, "sMAPE"),
        (RMSE, "RMSE"),
        (NRMSE, "NRMSE"),
        (bias_fraction, "bias_fraction"),
    ]
    for metric_func, name in metrics:
        result = metric_func(obs, mod, axis="x")
        assert isinstance(result.data, da.Array), f"{name} failed dask check"
        assert "history" in result.attrs, f"{name} missing history"
        assert f"{name} computed at" in result.attrs["history"]
        assert "y" in result.coords
        assert result.dims == ("y",)


def test_generic_metrics_xarray_eager():
    obs_data = np.random.rand(10, 10) + 1.0
    mod_data = np.random.rand(10, 10) + 1.0
    coords = {"x": np.arange(10), "y": np.arange(10)}
    obs = xr.DataArray(obs_data, dims=("x", "y"), coords=coords, name="obs")
    mod = xr.DataArray(mod_data, dims=("x", "y"), coords=coords, name="mod")
    metrics = [
        (MNB, "MNB"),
        (MNE, "MNE"),
        (MO, "MO"),
        (MP, "MP"),
        (RM, "RM"),
        (MB, "MB"),
        (MAE, "MAE"),
        (sMAPE, "sMAPE"),
        (RMSE, "RMSE"),
        (NRMSE, "NRMSE"),
        (bias_fraction, "bias_fraction"),
    ]
    for metric_func, name in metrics:
        result = metric_func(obs, mod, axis="x")
        assert not hasattr(result.data, "dask")
        assert "history" in result.attrs, f"{name} missing history"
        assert f"{name} computed at" in result.attrs["history"]
        assert "y" in result.coords
        assert result.dims == ("y",)


def test_nsc_nse_xarray():
    obs_data = np.random.rand(10) + 1.0
    mod_data = np.random.rand(10) + 1.0
    obs = xr.DataArray(obs_data, dims=("time",), name="obs")
    mod = xr.DataArray(mod_data, dims=("time",), name="mod")
    for func in [NSC, NSE_alpha, NSE_beta]:
        result = func(obs, mod)
        assert isinstance(result, xr.DataArray)
        assert "history" in result.attrs


def test_wind_metrics_xarray():
    obs = xr.DataArray(np.array([10.0, 350.0]), dims=("time",), name="obs")
    mod = xr.DataArray(np.array([20.0, 10.0]), dims=("time",), name="mod")
    result = WDMdnB(obs, mod)
    assert isinstance(result, xr.DataArray)
    assert np.isclose(result.values, 15.0)


def test_corr_index():
    # Numpy test
    obs = np.array([1, 2, 3])
    mod = np.array([2, 4, 6])
    assert np.isclose(CORR_INDEX(obs, mod), 1.0)

    # Xarray Eager test
    obs_da = xr.DataArray(np.random.rand(10, 10), dims=("x", "y"))
    mod_da = obs_da * 2.0
    result = CORR_INDEX(obs_da, mod_da, axis="x")
    xr.testing.assert_allclose(result, xr.DataArray(np.ones(10), dims=("y",)))

    # Xarray Dask test
    if HAS_DASK:
        obs_da_dask = xr.DataArray(da.from_array(np.random.rand(10, 10), chunks=(5, 5)), dims=("x", "y"))
        mod_da_dask = obs_da_dask * 2.0
        result_dask = CORR_INDEX(obs_da_dask, mod_da_dask, axis="x")
        assert isinstance(result_dask.data, da.Array)
        xr.testing.assert_allclose(result_dask.compute(), xr.DataArray(np.ones(10), dims=("y",)))
