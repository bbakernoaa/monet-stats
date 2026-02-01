"""
Unit tests for the Taylor Diagram plotting function (Aero Protocol).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.visualize import plot_taylor


@pytest.fixture
def sample_data():
    """Create sample data for Taylor Diagram testing."""
    np.random.seed(42)
    obs = np.random.rand(100)
    mod1 = obs + np.random.randn(100) * 0.1
    mod2 = obs + np.random.randn(100) * 0.5

    obs_da = xr.DataArray(obs, dims="time", name="obs", attrs={"history": "initial"})
    mod1_da = xr.DataArray(mod1, dims="time", name="mod1")
    mod2_da = xr.DataArray(mod2, dims="time", name="mod2")

    return obs, mod1, mod2, obs_da, mod1_da, mod2_da


def test_plot_taylor_numpy(sample_data):
    """Test plot_taylor with NumPy array inputs."""
    plt = pytest.importorskip("matplotlib.pyplot")
    obs, mod1, _, _, _, _ = sample_data

    ax = plot_taylor(obs, mod1, title="NumPy Taylor")

    assert ax.name == "polar"
    assert ax.get_title() == "NumPy Taylor"
    plt.close()


def test_plot_taylor_xarray(sample_data):
    """Test plot_taylor with Xarray DataArray inputs."""
    plt = pytest.importorskip("matplotlib.pyplot")
    _, _, _, obs_da, mod1_da, _ = sample_data

    ax = plot_taylor(obs_da, mod1_da, title="Xarray Taylor")

    assert ax.name == "polar"
    assert "Generated Taylor Diagram" in obs_da.attrs["history"]
    plt.close()


def test_plot_taylor_multi_model(sample_data):
    """Test plot_taylor with multiple models in a dictionary."""
    plt = pytest.importorskip("matplotlib.pyplot")
    _, _, _, obs_da, mod1_da, mod2_da = sample_data

    models = {"Model A": mod1_da, "Model B": mod2_da}
    ax = plot_taylor(obs_da, models, title="Multi-Model Taylor")

    assert ax.name == "polar"
    assert len(ax.get_legend().get_texts()) == 3  # Obs + 2 models
    plt.close()


def test_plot_taylor_custom_ax(sample_data):
    """Test plot_taylor with a pre-provided axis."""
    plt = pytest.importorskip("matplotlib.pyplot")
    obs, mod1, _, _, _, _ = sample_data

    fig = plt.figure()
    custom_ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax = plot_taylor(obs, mod1, ax=custom_ax)

    assert ax is custom_ax
    plt.close()


def test_plot_taylor_missing_matplotlib(sample_data, monkeypatch):
    """Test handling of missing matplotlib dependency."""
    import builtins
    import sys

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("No module named 'matplotlib.pyplot'")
        return real_import(name, *args, **kwargs)

    # Remove from sys.modules to force re-import attempt
    saved_plt = sys.modules.get("matplotlib.pyplot")
    if "matplotlib.pyplot" in sys.modules:
        del sys.modules["matplotlib.pyplot"]

    monkeypatch.setattr(builtins, "__import__", mock_import)

    try:
        obs, mod1, _, _, _, _ = sample_data
        with pytest.raises(ImportError, match="Taylor Diagram requires 'matplotlib'"):
            plot_taylor(obs, mod1)
    finally:
        # Restore sys.modules
        if saved_plt:
            sys.modules["matplotlib.pyplot"] = saved_plt
