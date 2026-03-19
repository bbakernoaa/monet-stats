import numpy as np
import pytest
import xarray as xr

from monet_stats.error_metrics import MdnB, MdnNB, MdnNE, MdnO, MdnP, MedAE, NMdnGE, RMdn, WDMdnB


@pytest.mark.parametrize(
    "metric_func, name",
    [
        (MdnNB, "MdnNB"),
        (MdnNE, "MdnNE"),
        (NMdnGE, "NMdnGE"),
        (MdnO, "MdnO"),
        (MdnP, "MdnP"),
        (RMdn, "RMdn"),
        (MdnB, "MdnB"),
        (WDMdnB, "WDMdnB"),
        (MedAE, "MedAE"),
    ],
)
def test_median_metrics_aero_protocol(metric_func, name):
    """
    Verify median-based metrics follow Aero Protocol:
    1. Backend agnostic (NumPy/Dask).
    2. Lazy preservation for Dask.
    3. Correctness (Eager vs Lazy identity).
    4. Provenance tracking.
    """
    # Setup data
    # Use values that avoid edges for wind direction and zero for denominators
    obs_data = np.random.uniform(10, 50, (10, 10))
    mod_data = obs_data + np.random.normal(0, 1, (10, 10))

    obs = xr.DataArray(obs_data, dims=["x", "y"], name="obs", attrs={"units": "ug/m3"})
    mod = xr.DataArray(mod_data, dims=["x", "y"], name="mod", attrs={"units": "ug/m3"})

    # 1. Eager Execution (NumPy)
    res_eager = metric_func(obs, mod)
    assert isinstance(res_eager, (xr.DataArray, np.number, float))

    # 2. Lazy Execution (Dask)
    # Intentionally chunk along the potential reduction dimensions to force rechunking
    obs_lazy = obs.chunk({"x": 5, "y": 5})
    mod_lazy = mod.chunk({"x": 5, "y": 5})

    res_lazy = metric_func(obs_lazy, mod_lazy)

    # Verify Laziness: Result should still be Dask-backed if reduction didn't consume all dims
    # Or even if it did, Xarray's quantile returns a DataArray with dask backend if input was dask
    assert hasattr(res_lazy.data, "chunks"), f"{name} lost laziness"

    # 3. Correctness: Eager and Lazy results must be identical
    np.testing.assert_allclose(res_eager, res_lazy.compute(), err_msg=f"{name} mismatch between Eager and Lazy")

    # 4. Provenance: History attribute should be updated
    assert "history" in res_lazy.attrs
    assert name in res_lazy.attrs["history"]
    # Verify it doesn't drop other attributes
    assert res_lazy.attrs.get("units") == "ug/m3"


def test_median_metrics_partial_reduction():
    """Verify metrics work with partial dimension reduction and maintain laziness."""
    obs_data = np.random.rand(10, 10, 10) + 1.0
    mod_data = obs_data + 0.1
    obs = xr.DataArray(obs_data, dims=["time", "x", "y"], name="obs")
    mod = xr.DataArray(mod_data, dims=["time", "x", "y"], name="mod")

    # Chunk time and x/y
    obs_lazy = obs.chunk({"time": 5, "x": 5, "y": 5})
    mod_lazy = mod.chunk({"time": 5, "x": 5, "y": 5})

    # Reduce over x and y, leaving time
    res = MdnO(obs_lazy, mod_lazy, axis=["x", "y"])

    assert res.dims == ("time",)
    assert res.shape == (10,)
    assert hasattr(res.data, "chunks")
    # Reduction dimensions (x, y) should now be single-chunked in the graph
    # but the result dimension (time) should still be chunked
    assert res.data.chunks[0] == (5, 5)

    # Verify values
    res_eager = MdnO(obs, mod, axis=["x", "y"])
    np.testing.assert_allclose(res_eager, res.compute())


def test_median_metrics_numpy_fallback():
    """Verify fallback to NumPy works correctly for non-Xarray inputs."""
    obs = np.array([1.0, 2.0, 3.0])
    mod = np.array([1.1, 1.9, 3.2])

    res = MdnO(obs, mod)
    assert isinstance(res, (float, np.number))
    np.testing.assert_allclose(res, np.median(mod - obs))
