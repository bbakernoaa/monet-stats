import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats import stats
from monet_stats.plugin_system import plugin_manager


def test_plugin_decorator_registration():
    """Test that the @register decorator correctly registers a plugin."""

    @plugin_manager.register(name="CustomAdd", description="Adds 1 to the mean of mod")
    def custom_add(obs, mod, axis=None):
        if isinstance(mod, xr.DataArray):
            return mod.mean(dim=axis) + 1
        return np.mean(mod, axis=axis) + 1

    # Verify original function is preserved
    assert callable(custom_add)
    assert custom_add.__name__ == "custom_add"

    assert "CustomAdd" in plugin_manager.list_plugins()
    plugin = plugin_manager.get_plugin("CustomAdd")
    assert plugin.name() == "CustomAdd"
    assert plugin.description() == "Adds 1 to the mean of mod"


def test_stats_with_plugins_pandas():
    """Test that stats() correctly includes plugin results for Pandas DataFrames."""
    df = pd.DataFrame({"Obs": [1, 2, 3], "Mod": [2, 4, 6]})

    # We already have WMAPE registered as a built-in plugin
    results = stats(df, plugins=["WMAPE"])

    assert "WMAPE" in results
    # WMAPE = (sum|mod-obs| / sum|obs|) * 100 = ( (1+2+3) / (1+2+3) ) * 100 = 100.0
    assert np.isclose(results["WMAPE"], 100.0)


def test_stats_with_plugins_xarray():
    """Test that stats() correctly includes plugin results for Xarray Datasets."""
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="x", name="Obs")
    mod = xr.DataArray([2.0, 4.0, 6.0], dims="x", name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    results = stats(ds, plugins=["WMAPE", "MAPE_Bias"])

    assert "WMAPE" in results
    assert "MAPE_Bias" in results
    assert np.isclose(results["WMAPE"], 100.0)

    # MAPE_Bias: pe = (mod-obs)/obs = [1, 1, 1]. All positive.
    # pos_errors = mean([1, 1, 1]) = 1.0. neg_errors = 0.0. Result = 1.0.
    assert np.isclose(results["MAPE_Bias"], 1.0)


def test_stats_with_plugins_dask():
    """Test that plugins in stats() work with Dask and preserve laziness."""
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    obs = xr.DataArray(da.from_array([1.0, 2.0, 3.0], chunks=2), dims="x", name="Obs")
    mod = xr.DataArray(da.from_array([2.0, 4.0, 6.0], chunks=2), dims="x", name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    # This should trigger a single compute call
    results = stats(ds, plugins=["WMAPE"])

    assert "WMAPE" in results
    assert np.isclose(results["WMAPE"], 100.0)


def test_stats_with_missing_plugin():
    """Test that stats() handles non-existent plugins gracefully."""
    df = pd.DataFrame({"Obs": [1, 2, 3], "Mod": [2, 4, 6]})
    results = stats(df, plugins=["NonExistentPlugin"])

    assert "NonExistentPlugin" in results
    assert np.isnan(results["NonExistentPlugin"])


def test_plugin_zero_division_handling():
    """Test that refactored plugins handle zero division correctly."""
    # Obs has a zero
    obs = np.array([0.0, 1.0, 2.0])
    mod = np.array([1.0, 2.0, 3.0])

    # WMAPE should handle it (sum|obs| = 3.0, not zero)
    wmape = plugin_manager.get_plugin("WMAPE")
    res = wmape.compute(obs, mod)
    # sum|mod-obs| = 1+1+1 = 3. sum|obs| = 3. Result = 100.0
    assert np.isclose(res, 100.0)

    # MAPE_Bias has pe = (mod-obs)/obs. For the first element, it's 1.0/0.0 -> inf or nan.
    # Our implementation uses .where to put nan there.
    mape_bias = plugin_manager.get_plugin("MAPE_Bias")
    res = mape_bias.compute(obs, mod)
    # pe = [nan, 1.0, 0.5]. Mean of [1.0, 0.5] = 0.75.
    assert np.isclose(res, 0.75)


def test_stats_plugin_honors_axis_xarray():
    """
    Regression test: stats() must forward ``axis`` to plugins.

    A duplicated plugin loop in ``stats()`` once recomputed each plugin
    without ``axis``, clobbering the axis-aware result with a scalar reduced
    over every dimension. The two results only differ for multi-dimensional
    input, so 1-D tests could not catch it.
    """
    obs = xr.DataArray([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dims=["y", "x"], name="Obs")
    mod = xr.DataArray([[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]], dims=["y", "x"], name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    results = stats(ds, plugins=["WMAPE"], axis="x")

    # Reducing along "x" keeps the "y" dimension: one value per row.
    wmape = results["WMAPE"]
    assert hasattr(wmape, "dims") and "y" in wmape.dims
    assert np.shape(wmape) == (2,)

    # Row 0: sum|mod-obs| / sum|obs| = (1+2+3)/(1+2+3) = 100%
    # Row 1: (1+3+5)/(1+1+1)       = 300%
    # A missing axis would instead give the global 166.67% for both rows.
    np.testing.assert_allclose(np.asarray(wmape), [100.0, 300.0], rtol=1e-10)
    assert not np.isclose(float(np.asarray(wmape)[0]), 166.6667, rtol=1e-3)


def test_stats_plugin_honors_axis_pandas():
    """
    DataFrame input reduces obs/mod to 1-D Series, so ``axis`` cannot change
    plugin results there; this guards that forwarding a numeric axis does not
    break the pandas path.
    """
    df = pd.DataFrame({"Obs": [1.0, 2.0, 3.0], "Mod": [2.0, 4.0, 6.0]})

    results = stats(df, plugins=["WMAPE"], axis=0)

    # 1-D data: axis=0 reduces to the same scalar as axis=None.
    assert np.isclose(float(np.asarray(results["WMAPE"])), 100.0)


def test_stats_plugin_computed_once_with_axis(monkeypatch):
    """
    Regression test for the pandas branch of ``stats()``.

    A duplicated plugin loop once computed every plugin twice, and the second
    call omitted ``axis``, overwriting the first result. Because pandas input
    is 1-D, the numeric result is identical either way, so value-based tests
    cannot observe the defect. This test spies on ``compute_metric`` and pins
    the two observable facts: exactly one call per plugin, with ``axis``
    forwarded.
    """
    import monet_stats as ms

    calls = []
    real_compute = ms.plugin_manager.compute_metric

    def spy(name, obs, mod, **kwargs):
        calls.append((name, dict(kwargs)))
        return real_compute(name, obs, mod, **kwargs)

    monkeypatch.setattr(ms.plugin_manager, "compute_metric", spy)

    df = pd.DataFrame({"Obs": [1.0, 2.0, 3.0], "Mod": [2.0, 4.0, 6.0]})
    results = ms.stats(df, plugins=["WMAPE", "MAPE_Bias"], axis=0)

    for name in ("WMAPE", "MAPE_Bias"):
        plugin_calls = [c for c in calls if c[0] == name]
        assert len(plugin_calls) == 1, f"{name} computed {len(plugin_calls)} times: {plugin_calls}"
        assert plugin_calls[0][1].get("axis", "MISSING") == 0, plugin_calls[0][1]

    # Results still correct.
    assert np.isclose(float(np.asarray(results["WMAPE"])), 100.0)


def test_stats_plugin_honors_axis_dim_y_xarray():
    """The axis-aware plugin result must follow whichever dim is requested."""
    obs = xr.DataArray([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dims=["y", "x"], name="Obs")
    mod = xr.DataArray([[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]], dims=["y", "x"], name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    results = stats(ds, plugins=["WMAPE"], axis="y")

    # Reducing along "y" keeps "x": per-column values.
    # Col 0: (1+1)/(1+1)=100%, Col 1: (2+3)/(2+1)=166.67%, Col 2: (3+5)/(3+1)=200%
    assert "x" in results["WMAPE"].dims
    np.testing.assert_allclose(np.asarray(results["WMAPE"]), [100.0, 500.0 / 3.0, 200.0], rtol=1e-10)


def test_stats_plugin_axis_dask_lazy():
    """The axis-aware plugin path must also work with Dask-backed arrays."""
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    obs = xr.DataArray(da.from_array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], chunks=2), dims=["y", "x"], name="Obs")
    mod = xr.DataArray(da.from_array([[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]], chunks=2), dims=["y", "x"], name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    results = stats(ds, plugins=["WMAPE"], axis="x")

    assert np.shape(np.asarray(results["WMAPE"])) == (2,)
    np.testing.assert_allclose(np.asarray(results["WMAPE"]), [100.0, 300.0], rtol=1e-10)
