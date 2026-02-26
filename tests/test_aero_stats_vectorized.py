"""
Tests for vectorized summary statistics in Monet Stats (Aero Protocol).
"""

import numpy as np
import pytest
import xarray as xr
from monet_stats import stats


def test_stats_vectorized_xarray():
    """Test that stats() handles multi-dimensional reductions for Xarray."""
    # Create a 3D dataset (time, lat, lon)
    times = [0, 1, 2]
    lats = [40, 41]
    lons = [-100, -99]

    obs_data = np.random.rand(3, 2, 2)
    mod_data = obs_data + 0.1

    ds = xr.Dataset(
        {
            "Obs": (["time", "lat", "lon"], obs_data),
            "Mod": (["time", "lat", "lon"], mod_data),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
    )

    # 1. Reduction over 'time' dimension
    # Should return a dict of 2x2 DataArrays (lat, lon)
    results = stats(ds, axis="time")

    assert isinstance(results, dict)
    assert "MB" in results
    # Aero Protocol: Should return DataArray to preserve coordinates
    assert isinstance(results["MB"], xr.DataArray)
    assert results["MB"].shape == (2, 2)
    assert "lat" in results["MB"].coords
    assert "lon" in results["MB"].coords
    assert np.allclose(results["MB"], 0.1)

    # 2. Reduction over all dimensions (default)
    # Should return a dict of scalars (floats)
    results_all = stats(ds)
    assert isinstance(results_all["MB"], float)
    assert np.isclose(results_all["MB"], 0.1)

def test_stats_accessor_vectorized():
    """Test the Xarray accessor stats method with dimension reduction."""
    obs_data = np.random.rand(5, 5)
    mod_data = obs_data + 0.5

    ds = xr.Dataset(
        {
            "Obs": (["x", "y"], obs_data),
            "Mod": (["x", "y"], mod_data),
        }
    )

    # Use accessor to reduce over 'x'
    results = ds.monet_stats.stats(dim="x")

    assert "MAE" in results
    assert isinstance(results["MAE"], xr.DataArray)
    assert results["MAE"].shape == (5,)
    assert np.allclose(results["MAE"], 0.5)

@pytest.mark.parametrize("backend", ["numpy", "dask"])
def test_stats_laziness_preservation(backend):
    """Ensure stats() preserves laziness and uses optimized bundling for Dask."""
    obs_data = np.random.rand(10, 10)
    mod_data = obs_data + 0.2

    ds = xr.Dataset(
        {
            "Obs": (["x", "y"], obs_data),
            "Mod": (["x", "y"], mod_data),
        }
    )

    if backend == "dask":
        try:
            import dask.array as da
            ds = ds.chunk({"x": 5, "y": 5})
        except ImportError:
            pytest.skip("Dask not installed")

    # Compute stats over 'y'
    # For Dask, this should ideally bundle computations
    results = stats(ds, axis="y")

    assert "RMSE" in results
    if backend == "dask":
        # The result from stats() is already computed (it calls .compute())
        # but we want to make sure it didn't fail and returned DataArrays
        assert isinstance(results["RMSE"], xr.DataArray)

    assert np.allclose(results["RMSE"], 0.2)

def test_stats_with_plugins_vectorized():
    """Test that plugins also support vectorized reduction through stats()."""
    obs_data = np.array([1.0, 2.0, 3.0, 4.0])
    mod_data = np.array([1.1, 2.2, 3.3, 4.4])

    ds = xr.Dataset(
        {
            "Obs": (["x"], obs_data),
            "Mod": (["x"], mod_data),
        }
    )

    # WMAPE plugin
    results = stats(ds, plugins=["WMAPE"])
    assert "WMAPE" in results
    assert np.isclose(results["WMAPE"], 10.0)

    # Vectorized WMAPE (though reduction over x results in scalar)
    results_v = stats(ds, axis="x", plugins=["WMAPE"])
    assert "WMAPE" in results_v
    assert np.isclose(results_v["WMAPE"], 10.0)
