import numpy as np
import pytest
import xarray as xr

from monet_stats.visualize import plot_spatial, plot_taylor


def test_plot_spatial_static():
    pytest.importorskip("cartopy")
    pytest.importorskip("matplotlib")

    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    """Test static spatial plot (Track A)."""
    lon = np.linspace(-130, -60, 10)
    lat = np.linspace(20, 50, 10)
    data = np.random.rand(10, 10)
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="test")

    ax = plot_spatial(da, method="static", projection=ccrs.PlateCarree())
    assert isinstance(ax, plt.Axes)
    assert "Plotted using static spatial plot" in da.attrs["history"]
    plt.close()


def test_plot_spatial_interactive():
    """Test interactive spatial plot (Track B)."""
    pytest.importorskip("hvplot")
    pytest.importorskip("geoviews")

    # Note: testing hvplot might return different objects depending on backend
    lon = np.linspace(-130, -60, 10)
    lat = np.linspace(20, 50, 10)
    data = np.random.rand(10, 10)
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="test")

    plot = plot_spatial(da, method="interactive")
    # HvPlot objects are typically from holoviews
    import holoviews as hv

    assert isinstance(plot, hv.core.dimension.Dimensioned)


def test_plot_taylor_numpy():
    """Test Taylor diagram with numpy inputs."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    obs = np.random.rand(100)
    mod = obs + np.random.normal(0, 0.1, 100)

    ax = plot_taylor(obs, mod)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_taylor_xarray():
    """Test Taylor diagram with xarray inputs."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    obs = xr.DataArray(np.random.rand(100), dims="time")
    mod = obs + np.random.normal(0, 0.1, 100)

    ax = plot_taylor(obs, mod)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_taylor_multi_model():
    """Test Taylor diagram with multiple models."""
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    obs = np.random.rand(100)
    models = {"Model A": obs + np.random.normal(0, 0.1, 100), "Model B": obs + np.random.normal(0, 0.2, 100)}

    ax = plot_taylor(obs, models)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_plot_spatial_dataset():
    """Test plot_spatial with xr.Dataset."""
    pytest.importorskip("cartopy")
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    lon = np.linspace(-130, -60, 10)
    lat = np.linspace(20, 50, 10)
    ds = xr.Dataset({"test": (("lat", "lon"), np.random.rand(10, 10))}, coords={"lat": lat, "lon": lon})

    ax = plot_spatial(ds, method="static")
    assert isinstance(ax, plt.Axes)
    plt.close()
