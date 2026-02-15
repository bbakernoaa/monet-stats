"""
Unit tests for the visualization module (Aero Protocol).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats.visualize import plot_diurnal_cycle, plot_power_spectrum, plot_spatial, plot_timeseries


@pytest.fixture
def sample_da():
    """Create a sample DataArray for testing."""
    lon = np.linspace(-130, -60, 10)
    lat = np.linspace(20, 50, 10)
    data = np.random.rand(10, 10)
    da = xr.DataArray(
        data,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="test_data",
        attrs={"units": "test_units", "history": "initial history"},
    )
    return da


def test_plot_spatial_invalid_method(sample_da):
    """Test that an invalid method raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown plotting method"):
        plot_spatial(sample_da, method="invalid_method")


def test_plot_spatial_matplotlib(sample_da):
    """Test Track A (matplotlib/cartopy) plot generation."""
    plt = pytest.importorskip("matplotlib.pyplot")
    pytest.importorskip("cartopy.crs")

    ax = plot_spatial(sample_da, method="matplotlib", title="Test Plot")

    assert isinstance(ax, plt.Axes)
    assert hasattr(ax, "coastlines")
    assert ax.get_title() == "Test Plot"
    assert "Plotted spatial data using Track A" in sample_da.attrs["history"]
    plt.close()


def test_plot_spatial_matplotlib_custom_ax(sample_da):
    """Test Track A with a pre-provided axis."""
    plt = pytest.importorskip("matplotlib.pyplot")
    ccrs = pytest.importorskip("cartopy.crs")

    fig = plt.figure()
    custom_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax = plot_spatial(sample_da, method="matplotlib", ax=custom_ax)

    assert ax is custom_ax
    plt.close()


def test_plot_spatial_hvplot(sample_da):
    """Test Track B (hvplot) plot generation."""
    hv = pytest.importorskip("holoviews")
    pytest.importorskip("hvplot.xarray")

    plot = plot_spatial(sample_da, method="hvplot", title="Interactive Test")
    # hvplot with rasterize=True can return a DynamicMap or Element
    assert isinstance(plot, (hv.Element, hv.DynamicMap))
    assert "Plotted spatial data using Track B" in sample_da.attrs["history"]


def test_plot_spatial_matplotlib_missing(sample_da, monkeypatch):
    """Test handling of missing matplotlib/cartopy dependencies."""
    import builtins
    import sys

    # Skip this test if cartopy is actually present and we can't easily mock its absence
    # Actually, we ARE mocking its absence here.

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ["cartopy", "cartopy.crs"]:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    # Remove from sys.modules to force re-import attempt
    saved_cartopy = sys.modules.get("cartopy")
    saved_cartopy_crs = sys.modules.get("cartopy.crs")

    if "cartopy" in sys.modules:
        del sys.modules["cartopy"]
    if "cartopy.crs" in sys.modules:
        del sys.modules["cartopy.crs"]

    monkeypatch.setattr(builtins, "__import__", mock_import)

    try:
        with pytest.raises(ImportError, match=r"Track A \(matplotlib\) requires"):
            plot_spatial(sample_da, method="matplotlib")
    finally:
        # Restore sys.modules
        if saved_cartopy:
            sys.modules["cartopy"] = saved_cartopy
        if saved_cartopy_crs:
            sys.modules["cartopy.crs"] = saved_cartopy_crs


def test_plot_diurnal_cycle_matplotlib():
    """Test diurnal cycle plot with matplotlib."""
    plt = pytest.importorskip("matplotlib.pyplot")

    # Create 2 days of hourly data
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    data = np.random.rand(48)
    da = xr.DataArray(data, coords={"time": times}, dims="time", name="test_data", attrs={"history": "initial"})

    ax = plot_diurnal_cycle(da, method="matplotlib", title="Diurnal Test")

    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Diurnal Test"
    assert ax.get_xlabel() == "Hour of Day"
    assert "Plotted diurnal cycle using Track A" in da.attrs["history"]
    plt.close()


def test_plot_diurnal_cycle_hvplot():
    """Test diurnal cycle plot with hvplot."""
    hv = pytest.importorskip("holoviews")
    pytest.importorskip("hvplot.xarray")

    # Create 2 days of hourly data
    times = pd.date_range("2020-01-01", periods=48, freq="h")
    data = np.random.rand(48)
    da = xr.DataArray(data, coords={"time": times}, dims="time", name="test_data", attrs={"history": "initial"})

    plot = plot_diurnal_cycle(da, method="hvplot")
    assert isinstance(plot, (hv.Element, hv.DynamicMap))
    assert "Plotted diurnal cycle using Track B" in da.attrs["history"]


def test_accessor_plot_diurnal_cycle():
    """Test that the accessor correctly calls the plotting function."""
    plt = pytest.importorskip("matplotlib.pyplot")

    times = pd.date_range("2020-01-01", periods=48, freq="h")
    da = xr.DataArray(np.random.rand(48), coords={"time": times}, dims="time", name="test")

    ax = da.monet_stats.plot_diurnal_cycle(method="matplotlib")
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_timeseries_matplotlib():
    """Test timeseries plot with matplotlib."""
    plt = pytest.importorskip("matplotlib.pyplot")
    da = xr.DataArray(np.random.rand(10), dims="time", coords={"time": np.arange(10)})
    ax = plot_timeseries(da, method="matplotlib")
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_power_spectrum_matplotlib():
    """Test power spectrum plot with matplotlib."""
    plt = pytest.importorskip("matplotlib.pyplot")
    da = xr.DataArray(np.random.rand(10), dims="frequency", coords={"frequency": np.linspace(0, 1, 10)})
    ax = plot_power_spectrum(da, method="matplotlib")
    assert isinstance(ax, plt.Axes)
    plt.close()
