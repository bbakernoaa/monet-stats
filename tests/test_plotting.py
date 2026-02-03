"""
Tests for plotting.py module.
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.plotting import plot_spatial_interactive, plot_spatial_static


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
    # We use a mock or just check the type
    import matplotlib.axes

    ax = plot_spatial_static(sample_da)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_spatial_interactive(sample_da):
    """Test that interactive plot returns a holoviews element or DynamicMap."""
    import holoviews as hv

    plot = plot_spatial_interactive(sample_da)
    # rasterize=True returns a DynamicMap
    assert isinstance(plot, (hv.core.Element, hv.core.DynamicMap))
