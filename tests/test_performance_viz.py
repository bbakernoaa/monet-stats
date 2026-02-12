import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from monet_stats.visualize import plot_performance_diagram


def test_plot_performance_diagram_numpy():
    """Test performance diagram with numpy arrays."""
    sr = np.array([0.6, 0.7, 0.8])
    pod = np.array([0.5, 0.6, 0.7])

    ax = plot_performance_diagram(sr, pod, title="Test Plot")
    assert ax is not None
    assert ax.get_title() == "Test Plot"
    assert ax.get_xlabel() == "Success Ratio (1 - FAR)"
    assert ax.get_ylabel() == "Probability of Detection (POD)"
    plt.close()


def test_plot_performance_diagram_xarray():
    """Test performance diagram with xarray objects."""
    sr = xr.DataArray([0.6, 0.7, 0.8], coords={"model": ["A", "B", "C"]}, dims="model")
    pod = xr.DataArray([0.5, 0.6, 0.7], coords={"model": ["A", "B", "C"]}, dims="model")

    ax = plot_performance_diagram(sr, pod, label="Models")
    assert ax is not None
    plt.close()


def test_performance_diagram_accessor_dataarray():
    """Test performance diagram via DataArray accessor."""
    sr = xr.DataArray([0.6, 0.7, 0.8], coords={"model": ["A", "B", "C"]}, dims="model")
    pod = xr.DataArray([0.5, 0.6, 0.7], coords={"model": ["A", "B", "C"]}, dims="model")

    # Call via accessor
    ax = sr.monet_stats.performance_diagram(pod=pod, title="Accessor Test")
    assert ax.get_title() == "Accessor Test"
    plt.close()


def test_performance_diagram_accessor_dataset():
    """Test performance diagram via Dataset accessor."""
    ds = xr.Dataset(
        {"SR": (["model"], [0.6, 0.7, 0.8]), "POD": (["model"], [0.5, 0.6, 0.7])}, coords={"model": ["A", "B", "C"]}
    )

    # Call via accessor
    ax = ds.monet_stats.performance_diagram(sr_name="SR", pod_name="POD")
    assert ax is not None
    plt.close()
