import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_ensemble_metrics import CRPS


def test_crps_numpy_benchmarks():
    """Verify CRPS against manual analytical benchmarks."""
    # Case 1: ens=[1, 2, 3], obs=2 -> CRPS=2/9
    ens1 = np.array([[1.0, 2.0, 3.0]]).T
    obs1 = np.array([2.0])
    res1 = CRPS(ens1, obs1, axis=0)
    assert np.allclose(res1, 2 / 9)

    # Case 2: ens=[1, 5], obs=3 -> CRPS=1.0
    ens2 = np.array([[1.0, 5.0]]).T
    obs2 = np.array([3.0])
    res2 = CRPS(ens2, obs2, axis=0)
    assert np.allclose(res2, 1.0)


def test_crps_xarray_integration():
    """Verify CRPS with xarray.DataArray and dimension handling."""
    ens = xr.DataArray(
        np.array([[1, 2, 3], [4, 5, 6]]).T,
        dims=["member", "site"],
        coords={"member": [1, 2, 3], "site": ["A", "B"]},
        name="forecast",
    )
    obs = xr.DataArray(np.array([2, 5]), dims=["site"], coords={"site": ["A", "B"]}, name="observation")

    # Reduction over 'member' dimension
    res = CRPS(ens, obs, axis="member")

    assert isinstance(res, xr.DataArray)
    assert res.dims == ("site",)
    assert np.allclose(res.sel(site="A"), 2 / 9)
    assert np.allclose(res.sel(site="B"), 2 / 9)
    assert "history" in res.attrs
    assert "Continuous Ranked Probability Score (CRPS)" in res.attrs["history"]


def test_crps_dask_laziness():
    """Verify CRPS remains lazy with Dask-backed arrays."""
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    ens = xr.DataArray(da.from_array(np.random.rand(10, 100), chunks=(10, 10)), dims=["member", "time"])
    obs = xr.DataArray(da.from_array(np.random.rand(100), chunks=(10,)), dims=["time"])

    res = CRPS(ens, obs, axis="member")

    # Check if the result is still a dask array
    assert hasattr(res.data, "chunks")

    # Compute and verify
    computed = res.compute()
    assert isinstance(computed.data, np.ndarray)
    assert computed.shape == (100,)


def test_crps_empty_ensemble():
    """Verify CRPS handles empty ensembles gracefully."""
    ens = np.array([]).reshape(0, 5)
    obs = np.array([1, 2, 3, 4, 5])
    res = CRPS(ens, obs, axis=0)
    assert np.all(np.isnan(res))
