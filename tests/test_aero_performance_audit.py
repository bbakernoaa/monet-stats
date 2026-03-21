"""
Validation tests for Aero Performance Audit (Aero Protocol Compliant).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.correlation_metrics import kendalltau, spearmanr
from monet_stats.performance import chunk_array
from monet_stats.utils_stats import is_lazy


def test_chunk_array_robustness():
    """Verify chunk_array handles various sizes without empty outputs."""
    # Test 1D array
    arr1d = np.arange(10)
    chunks = chunk_array(arr1d, chunk_size=3)
    assert len(chunks) == 4
    assert all(len(c) > 0 for c in chunks)
    assert np.array_equal(np.concatenate(chunks), arr1d)

    # Test 2D array
    arr2d = np.ones((5, 10))  # 50 elements
    # Requested chunk_size smaller than one row (10 elements)
    chunks = chunk_array(arr2d, chunk_size=5)
    # Should get 5 chunks of 1 row each (10 elements)
    assert len(chunks) == 5
    assert all(c.shape == (1, 10) for c in chunks)

    # Test 2D array with large chunk_size
    chunks = chunk_array(arr2d, chunk_size=100)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], arr2d)

    # Test empty array
    assert chunk_array(np.array([])) == []


def test_spearmanr_global_laziness():
    """Verify spearmanr remains lazy when axis=None."""
    try:
        import dask.array as da  # noqa: F401
    except ImportError:
        pytest.skip("Dask not installed")

    obs = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"], name="obs").chunk({"x": 5, "y": 5})
    mod = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"], name="mod").chunk({"x": 5, "y": 5})

    # Global reduction (axis=None)
    res = spearmanr(obs, mod, axis=None)

    assert is_lazy(res)
    assert "spearmanr" in res.attrs.get("history", "")
    assert res.compute() is not None


def test_spearmanr_mixed_inputs():
    """Verify spearmanr handles mixed DataArray and NumPy inputs lazily."""
    obs = xr.DataArray(np.random.rand(10, 10), dims=["x", "y"], name="obs").chunk({"x": 5, "y": 5})
    mod_np = np.random.rand(10, 10)

    # Integer axis
    res = spearmanr(obs, mod_np, axis=0)
    assert is_lazy(res)
    assert res.dims == ("y",)
    assert "spearmanr" in res.attrs.get("history", "")

    # String dimension
    res2 = spearmanr(obs, mod_np, axis="y")
    assert is_lazy(res2)
    assert res2.dims == ("x",)


def test_kendalltau_global_laziness():
    """Verify kendalltau remains lazy when axis=None."""
    try:
        import dask.array as da  # noqa: F401
    except ImportError:
        pytest.skip("Dask not installed")

    obs = xr.DataArray(np.random.rand(4, 4), dims=["x", "y"], name="obs").chunk({"x": 2, "y": 2})
    mod = xr.DataArray(np.random.rand(4, 4), dims=["x", "y"], name="mod").chunk({"x": 2, "y": 2})

    # Global reduction (axis=None)
    res = kendalltau(obs, mod, axis=None)

    assert is_lazy(res)
    assert "kendalltau" in res.attrs.get("history", "")
    assert res.compute() is not None


def test_kendalltau_mixed_inputs():
    """Verify kendalltau handles mixed DataArray and NumPy inputs lazily."""
    obs = xr.DataArray(np.random.rand(4, 4), dims=["x", "y"], name="obs").chunk({"x": 2, "y": 2})
    mod_np = np.random.rand(4, 4)

    # Integer axis
    res = kendalltau(obs, mod_np, axis=1)
    assert is_lazy(res)
    assert res.dims == ("x",)
    assert "kendalltau" in res.attrs.get("history", "")

    # Global reduction
    res2 = kendalltau(obs, mod_np, axis=None)
    assert is_lazy(res2)


def test_spearmanr_math_parity():
    """Verify spearmanr lazy result matches eager result."""
    obs_raw = np.random.rand(20)
    mod_raw = obs_raw + np.random.normal(0, 0.1, 20)

    obs_xr = xr.DataArray(obs_raw, dims="t")
    mod_xr = xr.DataArray(mod_raw, dims="t")

    res_eager = spearmanr(obs_xr, mod_xr, axis=None)

    obs_lazy = obs_xr.chunk({"t": 5})
    mod_lazy = mod_xr.chunk({"t": 5})
    res_lazy = spearmanr(obs_lazy, mod_lazy, axis=None)

    assert np.isclose(res_eager, res_lazy.compute())
