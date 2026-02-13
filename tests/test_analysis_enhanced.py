"""
Tests for enhanced analysis and visualization methods (Aero Protocol).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats.analysis import anomalies, detrend


def test_anomalies():
    """Test computation of anomalies relative to monthly climatology."""
    # Create 2 years of daily data with a clear seasonal cycle
    times = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    # Sine wave with period of ~365 days
    cycle = np.sin(2 * np.pi * times.dayofyear / 365.25)
    data = cycle + np.random.normal(0, 0.01, len(times))

    da = xr.DataArray(
        data, coords={"time": times}, dims="time", name="test", attrs={"history": "initial"}
    )

    # Monthly anomalies
    anom = anomalies(da, freq="month")

    assert isinstance(anom, xr.DataArray)
    assert anom.shape == da.shape
    # The mean of anomalies for each month should be very close to 0
    monthly_mean_anom = anom.groupby("time.month").mean()
    np.testing.assert_allclose(monthly_mean_anom.values, 0, atol=1e-10)
    assert "Anomalies (month)" in anom.attrs["history"]


def test_detrend_linear():
    """Test linear detrending logic."""
    times = pd.date_range("2020-01-01", periods=100, freq="D")
    # Linear trend: y = 0.5x
    trend = 0.5 * np.arange(100)
    data = trend + np.random.normal(0, 0.01, 100)

    da = xr.DataArray(
        data, coords={"time": times}, dims="time", name="test", attrs={"history": "initial"}
    )

    detrended = detrend(da, method="linear")

    assert isinstance(detrended, xr.DataArray)
    # After linear detrend, the result should have near-zero mean and no slope
    assert "Detrended (linear)" in detrended.attrs["history"]
    assert np.abs(detrended.mean()) < 0.1


def test_detrend_constant():
    """Test constant detrending (mean subtraction)."""
    da = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": [1, 2, 3]}, attrs={"history": "initial"})
    detrended = detrend(da, method="constant", dim="time")

    expected = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(detrended.values, expected)
    assert "Detrended (constant)" in detrended.attrs["history"]


def test_accessor_anomalies_detrend():
    """Test Xarray DataArray accessor integration for anomalies and detrend."""
    times = pd.date_range("2020-01-01", periods=100, freq="D")
    da = xr.DataArray(
        np.random.rand(100), coords={"time": times}, dims="time", name="test", attrs={"history": "initial"}
    )

    # Test DataArray accessor
    anom = da.monet_stats.anomalies(freq="month")
    assert isinstance(anom, xr.DataArray)
    assert "Anomalies (month)" in anom.attrs["history"]

    detr = da.monet_stats.detrend(method="constant")
    assert isinstance(detr, xr.DataArray)
    np.testing.assert_allclose(detr.mean(), 0, atol=1e-10)
    assert "Detrended (constant)" in detr.attrs["history"]


def test_dataset_accessor():
    """Test Xarray Dataset accessor integration."""
    times = pd.date_range("2020-01-01", periods=100, freq="D")
    ds = xr.Dataset(
        {"var1": (("time"), np.random.rand(100)), "var2": (("time"), np.random.rand(100))},
        coords={"time": times},
        attrs={"history": "initial"},
    )

    anom = ds.monet_stats.anomalies(freq="month")
    assert isinstance(anom, xr.Dataset)
    assert "var1" in anom
    assert "var2" in anom
    assert "Anomalies (month)" in anom.attrs["history"]

    detr = ds.monet_stats.detrend(method="linear")
    assert isinstance(detr, xr.Dataset)
    assert "Detrended (linear)" in detr.attrs["history"]


def test_plot_diurnal_cycle_logic():
    """Test the diurnal cycle plotting logic (Track A)."""
    plt = pytest.importorskip("matplotlib.pyplot")

    times = pd.date_range("2020-01-01", periods=100, freq="h")
    da = xr.DataArray(
        np.random.rand(100), coords={"time": times}, dims="time", name="test", attrs={"history": "initial"}
    )

    ax = da.monet_stats.plot_diurnal_cycle(method="matplotlib", title="Test Diurnal")
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Test Diurnal"
    assert "Plotted diurnal cycle using Track A" in da.attrs["history"]
    plt.close()


def test_detrend_invalid_method():
    """Test that an invalid detrending method raises ValueError."""
    da = xr.DataArray([1, 2, 3], dims="time")
    with pytest.raises(ValueError, match="Unknown detrending method"):
        detrend(da, method="invalid")
