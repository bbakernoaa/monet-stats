"""
Tests for plotting.py module.
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.plotting import (
    plot_spatial_interactive,
    plot_spatial_static,
    plot_taylor,
    plot_taylor_interactive,
)


@pytest.fixture
def sample_da():
    """Create a sample 2D DataArray."""
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)
    data = np.random.rand(10, 20)
    return xr.DataArray(
        data,
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
        name="test_data",
    )


def test_plot_spatial_static(sample_da):
    """Test that static plot returns a matplotlib axes."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("cartopy")
    import matplotlib.axes

    ax = plot_spatial_static(sample_da)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_spatial_interactive(sample_da):
    """Test that interactive plot returns a holoviews element or DynamicMap."""
    pytest.importorskip("hvplot")
    pytest.importorskip("geoviews")
    pytest.importorskip("holoviews")
    import holoviews as hv

    plot = plot_spatial_interactive(sample_da)
    # rasterize=True returns a DynamicMap
    assert isinstance(plot, (hv.core.Element, hv.core.DynamicMap))


@pytest.fixture
def sample_taylor_data():
    """Generate sample data for Taylor Diagram testing."""
    np.random.seed(42)
    obs = np.random.normal(10, 2, 100)
    mod1 = obs + np.random.normal(0, 0.5, 100)
    mod2 = obs + np.random.normal(0, 1.5, 100)

    obs_da = xr.DataArray(obs, dims="time", name="obs")
    mod_da = xr.DataArray(mod1, dims="time", name="mod")

    return obs, [mod1, mod2], obs_da, mod_da


def test_plot_taylor_numpy(sample_taylor_data):
    """Test plot_taylor with numpy input."""
    pytest.importorskip("matplotlib")
    obs, mods, _, _ = sample_taylor_data
    import matplotlib.figure

    fig = plot_taylor(obs, mods, labels=["Model 1", "Model 2"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_taylor_xarray(sample_taylor_data):
    """Test plot_taylor with xarray input."""
    pytest.importorskip("matplotlib")
    _, _, obs_da, mod_da = sample_taylor_data
    import matplotlib.figure

    fig = plot_taylor(obs_da, mod_da)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert "Taylor Diagram comparison" in obs_da.attrs.get("history", "")


def test_plot_taylor_interactive(sample_taylor_data):
    """Test plot_taylor_interactive returns a holoviews element."""
    pytest.importorskip("hvplot")
    import holoviews as hv

    _, _, obs_da, mod_da = sample_taylor_data
    plot = plot_taylor_interactive(obs_da, mod_da)
    # hvplot with 'by' returns an NdOverlay
    assert isinstance(plot, (hv.core.Element, hv.core.overlay.NdOverlay))
    assert "Created interactive Taylor exploration plot" in obs_da.attrs.get("history", "")
