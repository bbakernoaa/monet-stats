import numpy as np
import pytest
import xarray as xr

from monet_stats import relative_metrics


def test_relative_metrics_dask_compliance():
    """Verify that relative metrics work identically for NumPy and Dask backends."""
    dask = pytest.importorskip("dask.array")

    # Create synthetic data
    data_obs = np.random.rand(10, 10)
    data_mod = np.random.rand(10, 10)

    da_obs_eager = xr.DataArray(data_obs, dims=("x", "y"), name="obs")
    da_mod_eager = xr.DataArray(data_mod, dims=("x", "y"), name="mod")

    da_obs_lazy = da_obs_eager.chunk({"x": 5})
    da_mod_lazy = da_mod_eager.chunk({"x": 5})

    # Test NMB
    res_eager = relative_metrics.NMB(da_obs_eager, da_mod_eager)
    res_lazy = relative_metrics.NMB(da_obs_lazy, da_mod_lazy)

    assert isinstance(res_lazy.data, dask.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated Normalized Mean Bias (NMB) using monet-stats." in res_lazy.attrs["history"]

    # Test ME
    res_eager = relative_metrics.ME(da_obs_eager, da_mod_eager, axis=0)
    res_lazy = relative_metrics.ME(da_obs_lazy, da_mod_lazy, axis=0)

    assert isinstance(res_lazy.data, dask.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated Mean Gross Error (ME) using monet-stats." in res_lazy.attrs["history"]

    # Test USUTPB (Peak Bias)
    res_eager = relative_metrics.USUTPB(da_obs_eager, da_mod_eager)
    res_lazy = relative_metrics.USUTPB(da_obs_lazy, da_mod_lazy)

    assert isinstance(res_lazy.data, dask.Array)
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    assert "Calculated Unpaired Space/Unpaired Time Peak Bias (USUTPB) using monet-stats." in res_lazy.attrs["history"]


def test_relative_metrics_numpy_fallback():
    """Verify that relative metrics still work with raw NumPy arrays."""
    obs = np.array([1, 2, 3])
    mod = np.array([1.1, 2.2, 3.3])

    res = relative_metrics.NMB(obs, mod)
    assert np.isclose(res, 10.0)


def test_relative_metrics_axis_string():
    """Verify that dimension names can be used for axis."""
    data_obs = np.random.rand(10, 10)
    data_mod = np.random.rand(10, 10)

    da_obs = xr.DataArray(data_obs, dims=("time", "site"), name="obs")
    da_mod = xr.DataArray(data_mod, dims=("time", "site"), name="mod")

    res = relative_metrics.NMB(da_obs, da_mod, axis="time")
    assert res.dims == ("site",)
    assert "Calculated Normalized Mean Bias (NMB) using monet-stats." in res.attrs["history"]


def test_all_relative_metrics_coverage():
    """Systematically call all functions in relative_metrics to ensure coverage."""
    # Setup data
    obs_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    mod_np = np.array([[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]])

    obs_xr = xr.DataArray(obs_np, dims=("y", "x"), name="obs")
    mod_xr = xr.DataArray(mod_np, dims=("y", "x"), name="mod")

    metrics = [
        "NMB",
        "WDNMB_m",
        "NMB_ABS",
        "NMdnB",
        "FB",
        "ME",
        "MdnE",
        "WDME_m",
        "WDME",
        "WDMdnE",
        "NME_m",
        "NME_m_ABS",
        "NME",
        "NMdnE",
        "FE",
        "USUTPB",
        "USUTPE",
        "PSUTMNPB",
        "PSUTMdnNPB",
        "PSUTMNPE",
        "PSUTMdnNPE",
        "PSUTNMPB",
        "PSUTNMPE",
        "PSUTNMdnPB",
        "PSUTNMdnPE",
        "MPE",
        "MdnPE",
    ]

    peak_metrics_with_paxis = ["MNPB", "MdnNPB", "MNPE", "MdnNPE", "NMPB", "NMdnPB", "NMPE", "NMdnPE"]

    # Test simple metrics
    for m_name in metrics:
        func = getattr(relative_metrics, m_name)
        # Test NumPy
        func(obs_np, mod_np)
        func(obs_np, mod_np, axis=0)

        # Test Xarray
        func(obs_xr, mod_xr)
        func(obs_xr, mod_xr, axis=0)  # int axis
        func(obs_xr, mod_xr, axis="y")  # str axis

    # Test peak metrics with paxis
    for m_name in peak_metrics_with_paxis:
        func = getattr(relative_metrics, m_name)
        # Test NumPy
        func(obs_np, mod_np, paxis=1)
        func(obs_np, mod_np, paxis=1, axis=0)

        # Test Xarray
        func(obs_xr, mod_xr, paxis="x")
        func(obs_xr, mod_xr, paxis=1)  # int paxis
        func(obs_xr, mod_xr, paxis="x", axis="y")
        func(obs_xr, mod_xr, paxis=1, axis=0)  # int paxis and axis


def test_wind_direction_masked():
    """Test wind direction metrics with masked arrays."""
    obs = np.ma.array([350, 10, 20], mask=[0, 1, 0])
    mod = np.ma.array([345, 15, 25], mask=[0, 0, 1])

    relative_metrics.WDNMB_m(obs, mod)
    relative_metrics.WDME_m(obs, mod)
    relative_metrics.WDME(obs, mod)
    relative_metrics.WDMdnE(obs, mod)


def test_masked_fallback_paths():
    """Test paths that use masked arrays for non-DataArray inputs."""
    obs = np.array([1, 2, np.nan, 4])
    mod = np.array([1.1, 2.1, 3.1, 4.1])

    relative_metrics.NMdnB(obs, mod)
    relative_metrics.FB(obs, mod)
    relative_metrics.MdnE(obs, mod)
    relative_metrics.WDME(obs, mod)
    relative_metrics.WDMdnE(obs, mod)
    relative_metrics.NME(obs, mod)
    relative_metrics.NMdnE(obs, mod)
    relative_metrics.FE(obs, mod)
    relative_metrics.USUTPB(obs, mod)
    relative_metrics.USUTPE(obs, mod)


def test_aero_multi_dim_axis_tuple():
    """Verify that multi-dimensional axis (tuple) works correctly (Aero Protocol)."""
    data_obs = np.random.rand(5, 5, 5)
    data_mod = np.random.rand(5, 5, 5)

    da_obs = xr.DataArray(data_obs, dims=("time", "y", "x"), name="obs")
    da_mod = xr.DataArray(data_mod, dims=("time", "y", "x"), name="mod")

    # Test with tuple axis
    res = relative_metrics.NMB(da_obs, da_mod, axis=("y", "x"))
    assert res.dims == ("time",)
    assert "Calculated Normalized Mean Bias (NMB) using monet-stats." in res.attrs["history"]

    # Test with list axis
    res_list = relative_metrics.NMB(da_obs, da_mod, axis=["y", "x"])
    xr.testing.assert_allclose(res, res_list)


def test_aero_dask_median_chunking():
    """Verify that median-based metrics handle Dask chunking correctly for multi-dim axis."""
    dask = pytest.importorskip("dask.array")

    data_obs = np.random.rand(10, 10)
    data_mod = np.random.rand(10, 10)

    da_obs = xr.DataArray(data_obs, dims=("y", "x"), name="obs").chunk({"y": 5, "x": 5})
    da_mod = xr.DataArray(data_mod, dims=("y", "x"), name="mod").chunk({"y": 5, "x": 5})

    # This should not raise an error and should preserve laziness
    res = relative_metrics.NMdnB(da_obs, da_mod, axis=("y", "x"))
    assert isinstance(res.data, dask.Array)
    assert "Calculated Normalized Median Bias (NMdnB) using monet-stats." in res.attrs["history"]
    assert res.compute() is not None


def test_aero_peak_metrics_provenance():
    """Verify that peak metrics have correct provenance strings."""
    data_obs = np.random.rand(5, 5)
    data_mod = np.random.rand(5, 5)

    da_obs = xr.DataArray(data_obs, dims=("time", "site"), name="obs")
    da_mod = xr.DataArray(data_mod, dims=("time", "site"), name="mod")

    res = relative_metrics.MNPB(da_obs, da_mod, paxis="time", axis="site")
    assert "Calculated Mean Normalized Peak Bias (MNPB) using monet-stats." in res.attrs["history"]

    res2 = relative_metrics.NMdnPB(da_obs, da_mod, paxis="time")
    assert "Calculated Normalized Median Peak Bias (NMdnPB) using monet-stats." in res2.attrs["history"]
