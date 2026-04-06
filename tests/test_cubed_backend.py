import numpy as np
import pytest
import xarray as xr

from monet_stats.error_metrics import MB, RMSE
from monet_stats.utils_stats import ensure_single_chunk, is_lazy

try:
    import cubed

    HAS_CUBED = True
except ImportError:
    HAS_CUBED = False


@pytest.mark.skipif(not HAS_CUBED, reason="Cubed not installed")
def test_cubed_lazy_detection():
    data = np.random.rand(10, 10)
    c_arr = cubed.from_array(data, chunks=(5, 5))
    da = xr.DataArray(c_arr, dims=["x", "y"])

    assert is_lazy(da)
    assert isinstance(da.data, cubed.array_api.array_object.Array)


@pytest.mark.skipif(not HAS_CUBED, reason="Cubed not installed")
def test_cubed_ensure_single_chunk():
    data = np.random.rand(10, 10)
    c_arr = cubed.from_array(data, chunks=(5, 5))
    da = xr.DataArray(c_arr, dims=["x", "y"])

    da_rechunked = ensure_single_chunk(da, "x")
    assert da_rechunked.chunks == ((10,), (5, 5))
    assert isinstance(da_rechunked.data, cubed.array_api.array_object.Array)


@pytest.mark.skipif(not HAS_CUBED, reason="Cubed not installed")
def test_cubed_metrics_execution():
    # Test that basic metrics can execute with cubed arrays.
    # Note: cubed's compute() might behave differently than dask's,
    # but here we just check if the xarray operations triggered by metrics work.

    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)

    obs_c = cubed.from_array(obs_data, chunks=(10, 10))
    mod_c = cubed.from_array(mod_data, chunks=(10, 10))

    obs = xr.DataArray(obs_c, dims=["x", "y"])
    mod = xr.DataArray(mod_c, dims=["x", "y"])

    # MB should return a cubed-backed DataArray if axis is provided
    bias = MB(obs, mod, axis="x")
    assert is_lazy(bias)
    assert isinstance(bias.data, cubed.array_api.array_object.Array)

    # Compute the result
    result = bias.compute()
    expected = (mod_data - obs_data).mean(axis=0)
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.skipif(not HAS_CUBED, reason="Cubed not installed")
def test_cubed_full_reduction():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)

    obs = xr.DataArray(cubed.from_array(obs_data, chunks=(10, 10)), dims=["x", "y"])
    mod = xr.DataArray(cubed.from_array(mod_data, chunks=(10, 10)), dims=["x", "y"])

    # Full reduction
    rmse_val = RMSE(obs, mod)
    assert is_lazy(rmse_val)

    result = rmse_val.compute()
    expected = np.sqrt(np.mean((mod_data - obs_data) ** 2))
    np.testing.assert_allclose(result.values, expected)
