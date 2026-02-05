"""
Comprehensive unit tests for optimized KZ filter.
"""

import numpy as np
import pytest
import xarray as xr
from monet_stats.analysis import kz_filter

def test_kz_filter_accuracy():
    """Verify that the optimized convolution-based KZ filter matches rolling means."""
    data = np.random.rand(100)
    m = 5
    k = 3

    # Baseline iterative rolling mean (slow but known correct)
    da = xr.DataArray(data, dims="time")
    expected = da
    for _ in range(k):
        expected = expected.rolling(time=m, center=True).mean()

    # Optimized filter
    result = kz_filter(da, m=m, k=k, dim="time")

    np.testing.assert_allclose(result.values, expected.values)

def test_kz_filter_dask_lazy():
    """Verify that the optimized KZ filter preserves Dask laziness."""
    dask = pytest.importorskip("dask.array")

    data = dask.random.random(100, chunks=20)
    da = xr.DataArray(data, dims="time")

    result = kz_filter(da, m=5, k=3, dim="time")

    assert hasattr(result.data, "chunks")
    # Verify computation works
    computed = result.compute()
    assert not np.all(np.isnan(computed.values[10:90]))

def test_kz_filter_dataset():
    """Verify that KZ filter works on Datasets."""
    ds = xr.Dataset({
        "var1": (("time"), np.random.rand(50)),
        "var2": (("time"), np.random.rand(50))
    })

    result = kz_filter(ds, m=3, k=2, dim="time")

    assert isinstance(result, xr.Dataset)
    assert "var1" in result
    assert "var2" in result
    # Check history on the Dataset
    assert "KZ filter" in result.attrs["history"]
