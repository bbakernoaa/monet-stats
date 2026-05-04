"""
Tests for the categorical metrics in MonetDataArrayAccessor.
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.accessor import MonetDataArrayAccessor  # noqa: F401


def test_accessor_categorical_metrics():
    """Test individual categorical metric methods in the accessor."""
    obs = xr.DataArray([0, 1, 1, 0], dims="x", name="obs")
    mod = xr.DataArray([1, 1, 0, 0], dims="x", name="mod")

    threshold = 0.5

    # Test methods
    assert mod.monet_stats.pod(obs, threshold).values.item() == 0.5
    assert mod.monet_stats.far(obs, threshold).values.item() == 0.5
    assert mod.monet_stats.fbi(obs, threshold).values.item() == 1.0
    assert mod.monet_stats.csi(obs, threshold).values.item() == pytest.approx(0.33333333)
    assert mod.monet_stats.hss(obs, threshold).values.item() == 0.0
    assert mod.monet_stats.tss(obs, threshold).values.item() == 0.0
    assert mod.monet_stats.ets(obs, threshold).values.item() == 0.0
    assert mod.monet_stats.bss_binary(obs, threshold).values.item() == -1.0


def test_accessor_verify_with_threshold():
    """Test verify() method with an optional threshold."""
    obs = xr.DataArray([0, 1, 1, 0], dims="x", name="obs")
    mod = xr.DataArray([1, 1, 0, 0], dims="x", name="mod")

    # Without threshold
    res_no_threshold = mod.monet_stats.verify(obs)
    assert "POD" not in res_no_threshold
    assert "MAE" in res_no_threshold

    # With threshold
    res = mod.monet_stats.verify(obs, threshold=0.5)

    expected_categorical = ["POD", "FAR", "HSS", "CSI", "TSS", "ETS", "FBI", "BSS_binary"]
    for metric in expected_categorical:
        assert metric in res
        assert not np.isnan(res[metric].values.item())

    assert res["POD"].values.item() == 0.5
    assert res["FBI"].values.item() == 1.0


def test_accessor_categorical_lazy():
    """Test categorical metrics with Dask-backed arrays."""
    pytest.importorskip("dask")

    obs = xr.DataArray([0, 1, 1, 0], dims="x", name="obs").chunk({"x": 2})
    mod = xr.DataArray([1, 1, 0, 0], dims="x", name="mod").chunk({"x": 2})

    threshold = 0.5

    # Method call should return a lazy DataArray
    pod_lazy = mod.monet_stats.pod(obs, threshold)
    assert hasattr(pod_lazy.data, "chunks")

    # Compute and check
    assert pod_lazy.compute().values.item() == 0.5

    # Verify bundle should also be lazy
    res_lazy = mod.monet_stats.verify(obs, threshold=threshold)
    assert hasattr(res_lazy["POD"].data, "chunks")
    assert res_lazy.compute()["POD"].values.item() == 0.5
