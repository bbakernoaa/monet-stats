"""
Tests for the stats() function in monet_stats/__init__.py.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats import stats


def test_stats_pandas():
    """Test stats function with Pandas DataFrame."""
    df = pd.DataFrame({"Obs": [1, 2, 3, 4, 5], "Mod": [1.1, 1.9, 3.2, 3.8, 5.5]})

    results = stats(df, threshold=3.0)

    # Check that all expected metrics are present
    expected_metrics = [
        "N",
        "Obs",
        "Mod",
        "MB",
        "MAE",
        "RMSE",
        "R",
        "IOA",
        "NMB",
        "MNB",
        "NSE",
        "POD",
        "FAR",
        "HSS",
        "CRMSE",
        "MdnB",
        "KGE",
        "R2",
        "CCC",
        "MNE",
        "NMSE",
    ]
    for metric in expected_metrics:
        assert metric in results
        assert not np.isnan(results[metric])

    # Basic value checks
    assert results["N"] == 5
    assert results["Obs"] == 3.0
    assert results["Mod"] == 3.1
    assert results["MB"] == pytest.approx(0.1)


def test_stats_xarray():
    """Test stats function with Xarray Dataset."""
    ds = xr.Dataset({"Obs": (["x"], [1, 2, 3, 4, 5]), "Mod": (["x"], [1.1, 1.9, 3.2, 3.8, 5.5])})

    results = stats(ds, threshold=3.0)

    # Check that all expected metrics are present
    expected_metrics = [
        "N",
        "Obs",
        "Mod",
        "MB",
        "MAE",
        "RMSE",
        "R",
        "IOA",
        "NMB",
        "MNB",
        "NSE",
        "POD",
        "FAR",
        "HSS",
        "CRMSE",
        "MdnB",
        "KGE",
        "R2",
        "CCC",
        "MNE",
        "NMSE",
    ]
    for metric in expected_metrics:
        assert metric in results
        assert not np.isnan(results[metric])

    # Basic value checks
    assert results["N"] == 5
    assert results["Obs"] == 3.0
    assert results["Mod"] == 3.1
    assert results["MB"] == pytest.approx(0.1)


def test_stats_xarray_dask():
    """Test stats function with Dask-backed Xarray Dataset."""
    pytest.importorskip("dask")

    ds = xr.Dataset({"Obs": (["x"], [1, 2, 3, 4, 5]), "Mod": (["x"], [1.1, 1.9, 3.2, 3.8, 5.5])}).chunk({"x": 2})

    results = stats(ds, threshold=3.0)

    # Check that all expected metrics are present
    expected_metrics = [
        "N",
        "Obs",
        "Mod",
        "MB",
        "MAE",
        "RMSE",
        "R",
        "IOA",
        "NMB",
        "MNB",
        "NSE",
        "POD",
        "FAR",
        "HSS",
        "CRMSE",
        "MdnB",
        "KGE",
        "R2",
        "CCC",
        "MNE",
        "NMSE",
    ]
    for metric in expected_metrics:
        assert metric in results
        assert not np.isnan(results[metric])


def test_stats_filtering():
    """Test stats function with minval/maxval filtering."""
    df = pd.DataFrame({"Obs": [1, 2, 3, 4, 5], "Mod": [1, 2, 3, 4, 5]})

    results = stats(df, minval=2, maxval=4)

    assert results["N"] == 3
    assert results["Obs"] == 3.0
    assert results["Mod"] == 3.0


def test_stats_invalid_type():
    """Test stats function with invalid data type."""
    with pytest.raises(TypeError):
        stats([1, 2, 3])
