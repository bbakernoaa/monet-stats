import numpy as np
import pytest
import xarray as xr

try:
    import dask.array as da
except ImportError:
    da = None

from monet_stats.contingency_metrics import (
    HSS,
    POD,
    BSS_binary,
    CSI_max_threshold,
    TSS_max_threshold,
    scores,
)


def test_csi_max_threshold():
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    opt_t, max_val = CSI_max_threshold(obs, mod, 1, 5, 0.5)
    assert 1 <= opt_t <= 5
    assert 0 <= max_val <= 1


def test_tss_max_threshold():
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    opt_t, max_val = TSS_max_threshold(obs, mod, 1, 5, 0.5)
    assert 1 <= opt_t <= 5
    assert -1 <= max_val <= 1


def test_bss_binary_optimized():
    obs = np.array([1, 0, 1, 0])
    mod = np.array([0.8, 0.2, 0.7, 0.1])
    # threshold 0.5
    # obs_bin = [1, 0, 1, 0]
    # mod_bin = [1, 0, 1, 0]
    # Perfect match for binary
    res = BSS_binary(obs, mod, threshold=0.5)
    assert res == 1.0


@pytest.mark.skipif(da is None, reason="Dask not installed")
def test_bss_binary_dask():
    obs_data = np.random.rand(10, 10)
    mod_data = np.random.rand(10, 10)
    obs = xr.DataArray(da.from_array(obs_data, chunks=(5, 5)), dims=["x", "y"])
    mod = xr.DataArray(da.from_array(mod_data, chunks=(5, 5)), dims=["x", "y"])

    res = BSS_binary(obs, mod, threshold=0.5, axis=0)
    assert isinstance(res.data, da.Array)
    res_val = res.compute()
    assert res_val.shape == (10,)


def test_contingency_provenance():
    obs = xr.DataArray([1, 0, 1], dims=["time"], attrs={"history": "orig"})
    mod = xr.DataArray([1, 1, 0], dims=["time"])

    res = HSS(obs, mod, minval=0.5)
    assert "Heidke Skill Score (HSS)" in res.attrs["history"]
    assert "orig" in res.attrs["history"]


def test_scores_provenance():
    obs = xr.DataArray([1, 0, 1], dims=["time"])
    mod = xr.DataArray([1, 1, 0], dims=["time"])

    a, b, c, d = scores(obs, mod, minval=0.5)
    assert "Hits (a)" in a.attrs["history"]
    assert "Misses (b)" in b.attrs["history"]


def test_multi_dim_axis_resolution():
    obs = xr.DataArray(np.random.rand(4, 5, 6), dims=["time", "lat", "lon"])
    mod = xr.DataArray(np.random.rand(4, 5, 6), dims=["time", "lat", "lon"])

    # Reduce over lat and lon
    res = POD(obs, mod, minval=0.5, axis=(1, 2))
    assert res.dims == ("time",)
    assert res.shape == (4,)
