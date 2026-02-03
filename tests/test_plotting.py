"""
Tests for the plotting module.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    from holoviews.core import Dimensioned

    HAS_HV = True
except ImportError:
    HAS_HV = False

from monet_stats.plotting import plot_spatial_interactive, plot_spatial_static


@pytest.fixture
def sample_da():
    """Create a sample DataArray with lat/lon."""
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)
    data = np.random.rand(10, 20)
    da = xr.DataArray(
        data,
        coords={"lat": lats, "lon": lons},
        dims=("lat", "lon"),
        name="test_data",
    )
    da.attrs["units"] = "K"
    return da


def test_plot_spatial_static(sample_da):
    """Test static spatial plot returns GeoAxes."""
    if not HAS_CARTOPY:
        pytest.skip("Cartopy not available")
    ax = plot_spatial_static(sample_da, projection=ccrs.PlateCarree())
    assert hasattr(ax, "coastlines")
    # Check that it's a GeoAxes
    from cartopy.mpl.geoaxes import GeoAxes

    assert isinstance(ax, GeoAxes)
    plt.close()


def test_plot_spatial_interactive(sample_da):
    """Test interactive spatial plot returns HoloViews object."""
    if not HAS_HV:
        pytest.skip("HoloViews not available")
    plot = plot_spatial_interactive(sample_da)
    assert isinstance(plot, Dimensioned)
    # Check that geo is enabled in opts if possible to inspect
    # But checking type is usually enough for a basic unit test
