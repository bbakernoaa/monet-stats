import dask.array as da
import numpy as np
import xarray as xr

from monet_stats.error_metrics import STDO, STDP


def test_stdo_numpy():
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])
    # (1.0-1.1), (2.0-1.9), (3.0-3.2) = -0.1, 0.1, -0.2
    # mean = -0.2/3 = -0.0666...
    # std = sqrt( ((-0.1 - -0.066)^2 + (0.1 - -0.066)^2 + (-0.2 - -0.066)^2) / 3 )
    # std = sqrt( (0.033^2 + 0.166^2 + 0.133^2) / 3 )
    expected = np.std(obs - mod)
    result = STDO(obs, mod)
    assert np.isclose(result, expected)


def test_stdo_xarray_dask():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)

    obs = xr.DataArray(da.from_array(obs_data, chunks=(5, 5)), dims=("x", "y"), name="obs")
    mod = xr.DataArray(da.from_array(mod_data, chunks=(5, 5)), dims=("x", "y"), name="mod")

    result = STDO(obs, mod)

    # Verify it's still a dask array (lazy evaluation)
    assert isinstance(result.data, da.Array)

    # Verify values
    expected = np.std(obs_data - mod_data)
    xr.testing.assert_allclose(result.compute(), xr.DataArray(expected))

    # Verify history
    assert "history" in result.attrs
    assert "STDO computed at" in result.attrs["history"]


def test_stdp_xarray_dims():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)

    coords = {"x": np.arange(10), "y": np.arange(10)}
    obs = xr.DataArray(obs_data, dims=("x", "y"), coords=coords, name="obs")
    mod = xr.DataArray(mod_data, dims=("x", "y"), coords=coords, name="mod")

    # Compute along x-axis
    result = STDP(obs, mod, axis=0)

    assert result.dims == ("y",)
    expected = np.std(mod_data - obs_data, axis=0)
    xr.testing.assert_allclose(result, xr.DataArray(expected, dims=("y",), coords={"y": obs.y}))

    # Compute along y-dimension by name
    result_dim = STDP(obs, mod, axis="y")
    assert result_dim.dims == ("x",)
    expected_dim = np.std(mod_data - obs_data, axis=1)
    xr.testing.assert_allclose(result_dim, xr.DataArray(expected_dim, dims=("x",), coords={"x": obs.x}))
