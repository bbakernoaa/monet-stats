import numpy as np
import pytest
import xarray as xr

from monet_stats import relative_metrics


def test_mg_vg_weights():
    """Verify MG and VG with weights support."""
    obs_np = np.array([1.0, 2.0, 3.0])
    mod_np = np.array([1.1, 2.5, 3.3])
    weights_np = np.array([1.0, 10.0, 1.0])

    # NumPy path
    mg_no_w = relative_metrics.MG(obs_np, mod_np)
    mg_w = relative_metrics.MG(obs_np, mod_np, weights=weights_np)
    assert not np.isclose(mg_no_w, mg_w)

    vg_no_w = relative_metrics.VG(obs_np, mod_np)
    vg_w = relative_metrics.VG(obs_np, mod_np, weights=weights_np)
    assert vg_no_w != vg_w

    # Xarray path
    obs_xr = xr.DataArray(obs_np, dims="x")
    mod_xr = xr.DataArray(mod_np, dims="x")
    weights_xr = xr.DataArray(weights_np, dims="x")

    mg_xr_w = relative_metrics.MG(obs_xr, mod_xr, weights=weights_xr)
    assert "Weighted Geometric Mean Bias (MG)" in mg_xr_w.attrs["history"]
    assert np.isclose(mg_xr_w.values, mg_w)

    vg_xr_w = relative_metrics.VG(obs_xr, mod_xr, weights=weights_xr)
    assert "Weighted Geometric Variance (VG)" in vg_xr_w.attrs["history"]
    assert np.isclose(vg_xr_w.values, vg_w)


def test_mg_vg_dask_laziness():
    """Verify MG and VG preserve Dask laziness even with weights."""
    pytest.importorskip("dask.array")
    import dask.array as da

    obs = xr.DataArray(np.random.rand(10, 10) + 0.1, dims=("x", "y")).chunk({"x": 5})
    mod = xr.DataArray(np.random.rand(10, 10) + 0.1, dims=("x", "y")).chunk({"x": 5})
    weights = xr.DataArray(np.random.rand(10, 10), dims=("x", "y")).chunk({"x": 5})

    mg = relative_metrics.MG(obs, mod, weights=weights, axis="y")
    assert isinstance(mg.data, da.Array)
    assert mg.dims == ("x",)

    vg = relative_metrics.VG(obs, mod, weights=weights, axis="y")
    assert isinstance(vg.data, da.Array)
    assert vg.dims == ("x",)


def test_median_metrics_rechunking_robustness():
    """Verify that median-based metrics use ensure_single_chunk correctly."""
    pytest.importorskip("dask.array")

    # Create a lazy array with multiple chunks along the reduction dimension
    obs = xr.DataArray(np.random.rand(10), dims="x").chunk({"x": 2})
    mod = xr.DataArray(np.random.rand(10), dims="x").chunk({"x": 2})

    # This calls ensure_single_chunk internally
    res = relative_metrics.NMdnB(obs, mod, axis="x")

    # Check that it computed successfully (which means it rechunked internally as needed)
    val = res.compute()
    assert not np.isnan(val)


def test_peak_metrics_rechunking():
    """Verify that peak metrics use ensure_single_chunk correctly."""
    pytest.importorskip("dask.array")

    obs = xr.DataArray(np.random.rand(10, 5), dims=("time", "site")).chunk({"time": 2, "site": -1})
    mod = xr.DataArray(np.random.rand(10, 5), dims=("time", "site")).chunk({"time": 2, "site": -1})

    # MdnNPB reduces over 'axis' after taking max over 'paxis'
    # Here paxis="time", axis="site". result of mod.max(dim="time") is dims=("site",)
    # It will rechunk "site" if it was chunked.
    res = relative_metrics.MdnNPB(obs, mod, paxis="time", axis="site")
    assert not np.isnan(res.compute())
