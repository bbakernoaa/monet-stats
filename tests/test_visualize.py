"""
Unit tests for the visualization module (Aero Protocol).
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from monet_stats.visualize import plot_spatial


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
    ax = plot_spatial(sample_da, method="matplotlib", title="Test Plot")

    assert isinstance(ax, plt.Axes)
    assert hasattr(ax, "coastlines")
    assert ax.get_title() == "Test Plot"
    assert "Plotted spatial data using Track A" in sample_da.attrs["history"]
    plt.close()


def test_plot_spatial_matplotlib_custom_ax(sample_da):
    """Test Track A with a pre-provided axis."""
    fig = plt.figure()
    custom_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax = plot_spatial(sample_da, method="matplotlib", ax=custom_ax)

    assert ax is custom_ax
    plt.close()


def test_plot_spatial_hvplot(sample_da):
    """Test Track B (hvplot) plot generation."""
    import holoviews as hv

    plot = plot_spatial(sample_da, method="hvplot", title="Interactive Test")
    # hvplot with rasterize=True can return a DynamicMap or Element
    assert isinstance(plot, (hv.Element, hv.DynamicMap))
    assert "Plotted spatial data using Track B" in sample_da.attrs["history"]


def test_plot_spatial_matplotlib_missing(sample_da, monkeypatch):
    """Test handling of missing matplotlib/cartopy dependencies."""
    import builtins
    import sys

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
