import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_ensemble_metrics import SAL


def test_sal_numpy_basic():
    # Simple case: identical fields
    obs = np.ones((10, 10))
    mod = np.ones((10, 10))
    # Add an "object"
    obs[3:6, 3:6] = 2.0
    mod[3:6, 3:6] = 2.0

    S, A, L = SAL(obs, mod, threshold=1.5)

    # Perfectly identical fields should have SAL = (0, 0, 0)
    assert np.isclose(S, 0.0)
    assert np.isclose(A, 0.0)
    assert np.isclose(L, 0.0)


def test_sal_numpy_amplitude():
    obs = np.ones((10, 10))
    mod = np.ones((10, 10)) * 2.0

    S, A, L = SAL(obs, mod)

    # A should be positive since mod > obs
    # A = 2 * (mean(mod) - mean(obs)) / (mean(mod) + mean(obs))
    # A = 2 * (2 - 1) / (2 + 1) = 2/3 approx 0.666
    assert np.isclose(A, 2 / 3)
    assert np.isclose(S, 0.0)
    assert np.isclose(L, 0.0)


def test_sal_xarray_multidim():
    # Test vectorization over time dimension
    times = [0, 1]
    obs_data = np.zeros((2, 10, 10))
    mod_data = np.zeros((2, 10, 10))

    # Time 0: identical
    obs_data[0, 2:4, 2:4] = 1.0
    mod_data[0, 2:4, 2:4] = 1.0

    # Time 1: mod shifted
    obs_data[1, 2:4, 2:4] = 1.0
    mod_data[1, 5:7, 5:7] = 1.0

    obs = xr.DataArray(obs_data, dims=["time", "lat", "lon"], coords={"time": times})
    mod = xr.DataArray(mod_data, dims=["time", "lat", "lon"], coords={"time": times})

    S, A, L = SAL(obs, mod, threshold=0.5)

    assert isinstance(S, xr.DataArray)
    assert S.shape == (2,)

    # Time 0: SAL should be 0
    assert np.isclose(S.values[0], 0.0)
    assert np.isclose(A.values[0], 0.0)
    assert np.isclose(L.values[0], 0.0)

    # Time 1: Location should be > 0, A should be 0, S should be 0 (same structure)
    assert np.isclose(A.values[1], 0.0)
    assert np.isclose(S.values[1], 0.0)
    assert L.values[1] > 0.0

    # Check history
    assert "SAL: Structure component" in S.attrs["history"]


def test_sal_dask_lazy():
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    # This should be lazy for non-spatial dims, but SAL is spatial
    # Since we use apply_ufunc with lat/lon as core dims,
    # it will be computed for those dims.
    # If we had a time dim that was chunked, it would remain lazy.

    obs_3d = xr.DataArray(da.random.random((2, 10, 10), chunks=(1, 10, 10)), dims=["time", "lat", "lon"])
    mod_3d = xr.DataArray(da.random.random((2, 10, 10), chunks=(1, 10, 10)), dims=["time", "lat", "lon"])

    S, A, L = SAL(obs_3d, mod_3d, threshold=0.5)

    # Check that it is a dask array
    assert hasattr(S.data, "dask")

    # Compute and check shape
    s_val = S.compute()
    assert s_val.shape == (2,)


def test_sal_no_objects():
    obs = np.zeros((10, 10))
    mod = np.zeros((10, 10))

    # With threshold=0.5, no objects will be found
    S, A, L = SAL(obs, mod, threshold=0.5)

    assert np.isclose(S, 0.0)
    assert np.isclose(A, 0.0)
    assert np.isclose(L, 0.0)
